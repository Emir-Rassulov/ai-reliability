"""Run a suite: build requests, call the provider with bounded retries, evaluate, report."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from rag_app.interface import RAGSystem, RAGSystemError
from reliability import __version__
from reliability.checks.base import CheckContext
from reliability.checks.provider_call import ProviderCallCheck
from reliability.checks.registry import build_checks
from reliability.config import AppConfig, LimitsConfig
from reliability.evaluation.evaluator import Evaluator
from reliability.evaluation.outcome import CallOutcome
from reliability.evaluation.release import decide
from reliability.evaluation.scoring import (
    latency_stats,
    rag_metric_means,
    status_counts,
    total_usage,
    weighted_score,
)
from reliability.models import (
    Document,
    RetrievedRef,
    Role,
    RunReport,
    Scenario,
    ScenarioResult,
    SelftestSummary,
)
from reliability.providers.base import (
    ContextDocument,
    GenerationParams,
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    RateLimited,
)
from reliability.reporting.redaction import Redactor

log = logging.getLogger("reliability.orchestration")

Sleep = Callable[[float], None]


def utc_now() -> datetime:
    return datetime.now(UTC)


def context_documents(
    scenario: Scenario, documents: Mapping[str, Document]
) -> list[ContextDocument]:
    """Retrieved documents in rank order; a duplicated id is passed once, at its best rank."""
    seen: set[str] = set()
    out: list[ContextDocument] = []
    for ref in sorted(scenario.retrieved, key=lambda r: (r.rank, r.doc_id)):
        if ref.doc_id in seen:
            continue
        seen.add(ref.doc_id)
        doc = documents[ref.doc_id]
        out.append(
            ContextDocument(doc_id=doc.doc_id, title=doc.title, text=doc.text, rank=ref.rank)
        )
    return out


def build_request(
    scenario: Scenario, documents: Mapping[str, Document], config: AppConfig
) -> ProviderRequest:
    return ProviderRequest(
        scenario_id=scenario.id,
        system=scenario.input.system,
        messages=list(scenario.input.messages),
        documents=context_documents(scenario, documents),
        context=scenario.input.context,
        fields=dict(scenario.input.fields),
        params=GenerationParams(timeout_s=config.limits.request_timeout_s),
        session_id=scenario.session_id,
    )


def backoff_delay(attempt: int, limits: LimitsConfig) -> float:
    """Deterministic exponential backoff: base * 2**(attempt-1), capped (no jitter)."""
    return float(min(limits.backoff_max_s, limits.backoff_base_s * 2 ** (attempt - 1)))


def call_with_retries(
    provider: Provider,
    request: ProviderRequest,
    limits: LimitsConfig,
    sleep: Sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> CallOutcome:
    """Retry only timeouts, rate limits and unavailability, at most `max_retries` times."""
    waits: list[float] = []
    attempt = 0
    while True:
        attempt += 1
        started = clock()
        try:
            response = provider.generate(request)
        except ProviderError as exc:
            if not exc.retryable:
                return CallOutcome(
                    None, exc, attempt, tuple(waits), stop_reason="non-retryable error"
                )
            if attempt > limits.max_retries:
                return CallOutcome(
                    None,
                    exc,
                    attempt,
                    tuple(waits),
                    stop_reason=f"retries exhausted ({attempt} attempts)",
                )
            delay = backoff_delay(attempt, limits)
            if isinstance(exc, RateLimited) and exc.retry_after is not None:
                if exc.retry_after > limits.max_retry_after_s:
                    return CallOutcome(
                        None,
                        exc,
                        attempt,
                        tuple(waits),
                        stop_reason=f"Retry-After {exc.retry_after:g}s exceeds "
                        f"max_retry_after_s {limits.max_retry_after_s:g}s",
                    )
                delay = exc.retry_after
            waits.append(delay)
            log.info("retrying %s after %s (%.2fs)", request.scenario_id, exc.kind, delay)
            sleep(delay)
            continue
        if response.latency_ms is None:
            response = response.model_copy(
                update={
                    "latency_ms": round((clock() - started) * 1000, 3),
                    "latency_source": "measured",
                }
            )
        truncated = len(response.text) > limits.max_response_chars
        if truncated:
            response = response.model_copy(
                update={"text": response.text[: limits.max_response_chars]}
            )
        return CallOutcome(response, None, attempt, tuple(waits), truncated_by_limit=truncated)


def make_evaluator(
    config: AppConfig, documents: Mapping[str, Document], redactor: Redactor
) -> Evaluator:
    ctx = CheckContext(config=config, documents=documents)
    return Evaluator(
        build_checks(ctx),
        ProviderCallCheck(config),
        redactor,
        excerpt_chars=config.checks.response_excerpt_chars,
    )


def assemble_report(
    *,
    results: Sequence[ScenarioResult],
    config: AppConfig,
    provider_name: str,
    model: str | None,
    run_id: str,
    started: datetime,
    finished: datetime,
    target: str,
    defects: Sequence[str],
    scenario_files: Mapping[str, str],
    dataset_hash: str | None,
    warnings: Sequence[str] = (),
) -> RunReport:
    latency = latency_stats(results)
    selftests = [s for s in results if s.role is Role.SELFTEST]
    errors = [
        f"{s.scenario_id}: {s.error['kind']}: {s.error['message']}" for s in results if s.error
    ]
    return RunReport(
        run_id=run_id,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        tool_version=__version__,
        target="rag-app" if target == "rag-app" else "scenarios",
        defects=list(defects),
        provider=provider_name,
        model=model,
        scenario_files=dict(scenario_files),
        dataset_hash=dataset_hash,
        config=config.model_dump(mode="json"),
        counts=status_counts(results),
        selftest=SelftestSummary(
            total=len(selftests),
            matched=sum(1 for s in selftests if s.selftest_match),
            mismatched=[s.scenario_id for s in selftests if s.selftest_match is False],
        ),
        score=weighted_score(results, config.scoring.weights),
        decision=decide(results, config.release, latency.p95_ms),
        rag_metrics=rag_metric_means(results),
        latency=latency,
        token_usage=total_usage(results),
        errors=errors,
        warnings=list(warnings),
        scenarios=list(results),
    )


def new_run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def run_suite(
    *,
    scenarios: Sequence[Scenario],
    provider: Provider,
    documents: Mapping[str, Document],
    config: AppConfig,
    redactor: Redactor,
    scenario_files: Mapping[str, str] | None = None,
    dataset_hash: str | None = None,
    sleep: Sleep = time.sleep,
    now: Callable[[], datetime] = utc_now,
    run_id: str | None = None,
) -> RunReport:
    started = now()
    evaluator = make_evaluator(config, documents, redactor)
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        request = build_request(scenario, documents, config)
        outcome = call_with_retries(provider, request, config.limits, sleep)
        result = evaluator.evaluate(scenario, outcome)
        log.info("%s -> %s", scenario.id, result.status.value)
        results.append(result)
    return assemble_report(
        results=results,
        config=config,
        provider_name=provider.name,
        model=provider.model,
        run_id=run_id or new_run_id(started),
        started=started,
        finished=now(),
        target="scenarios",
        defects=(),
        scenario_files=scenario_files or {},
        dataset_hash=dataset_hash,
    )


# --------------------------------------------------------------------------- RAG system target


class TargetSystemError(ProviderError):
    """The RAG system under test raised instead of answering."""

    kind = "target_error"


def call_rag_system(
    system: RAGSystem,
    scenario: Scenario,
    limits: LimitsConfig,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Scenario, CallOutcome]:
    """Ask the system under test. Returns the scenario with `retrieved` replaced by the
    documents that actually reached the generator, plus the call outcome."""
    started = clock()
    try:
        result = system.answer(scenario.query, scenario.user_access, trace_id=scenario.id)
    except RAGSystemError as exc:
        error = TargetSystemError(str(exc))
        return scenario, CallOutcome(None, error, 1, stop_reason="target system error")
    latency = round((clock() - started) * 1000, 3)
    effective = scenario.model_copy(
        update={"retrieved": [RetrievedRef(doc_id=d.doc_id, rank=d.rank) for d in result.context]}
    )
    text = result.answer
    truncated = len(text) > limits.max_response_chars
    response = ProviderResponse(
        text=text[: limits.max_response_chars],
        provider="rag-app",
        model=result.generator,
        latency_ms=latency,
        latency_source="measured",
        is_simulated=result.simulated,
    )
    return effective, CallOutcome(response, None, 1, truncated_by_limit=truncated)


def run_rag_app(
    *,
    scenarios: Sequence[Scenario],
    system: RAGSystem,
    documents: Mapping[str, Document],
    config: AppConfig,
    redactor: Redactor,
    defects: Sequence[str] = (),
    generator_name: str | None = None,
    scenario_files: Mapping[str, str] | None = None,
    dataset_hash: str | None = None,
    now: Callable[[], datetime] = utc_now,
    run_id: str | None = None,
) -> RunReport:
    """Evaluate a RAG system through the `RAGSystem` protocol. Annotations stay in the
    scenarios; retrieval comes from the system instead of the precomputed `retrieved` list."""
    started = now()
    evaluator = make_evaluator(config, documents, redactor)
    results: list[ScenarioResult] = []
    model = generator_name
    for scenario in scenarios:
        effective, outcome = call_rag_system(system, scenario, config.limits)
        if outcome.response is not None:
            model = outcome.response.model
        results.append(evaluator.evaluate(effective, outcome))
    return assemble_report(
        results=results,
        config=config,
        provider_name="rag-app",
        model=model,
        run_id=run_id or new_run_id(started),
        started=started,
        finished=now(),
        target="rag-app",
        defects=defects,
        scenario_files=scenario_files or {},
        dataset_hash=dataset_hash,
    )
