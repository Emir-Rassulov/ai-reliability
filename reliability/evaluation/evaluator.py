"""Evaluate one scenario's call outcome with all of its listed checks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

from reliability.checks.base import Check
from reliability.checks.provider_call import ProviderCallCheck
from reliability.evaluation.outcome import CallOutcome
from reliability.models import (
    CheckResult,
    Role,
    Scenario,
    ScenarioResult,
    Status,
    aggregate_status,
)
from reliability.reporting.redaction import Redactor

log = logging.getLogger("reliability.evaluator")


class Evaluator:
    def __init__(
        self,
        checks: Mapping[str, Check],
        call_check: ProviderCallCheck,
        redactor: Redactor,
        *,
        excerpt_chars: int = 300,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.checks = checks
        self.call_check = call_check
        self.redactor = redactor
        self.excerpt_chars = excerpt_chars
        self.timer = timer

    def _error_result(self, scenario: Scenario, name: str, why: str) -> CheckResult:
        return CheckResult(
            check=name,
            status=Status.ERROR,
            severity=scenario.severity_of(name),
            method="not evaluated",
            explanation=why,
        )

    def _run_one(self, scenario: Scenario, name: str, outcome: CallOutcome) -> CheckResult:
        if name == self.call_check.name:
            return self.call_check.run_call(scenario, outcome)
        if outcome.response is None:
            kind = getattr(outcome.error, "kind", "unknown")
            return self._error_result(
                scenario, name, f"not evaluated: provider call failed ({kind})"
            )
        check = self.checks.get(name)
        if check is None:
            return self._error_result(
                scenario, name, f"check '{name}' is not available in this build"
            )
        if not check.applies_to(scenario):
            return CheckResult(
                check=name,
                status=Status.NOT_APPLICABLE,
                severity=scenario.severity_of(name),
                method="applies_to",
                explanation="check does not apply to this scenario",
            )
        try:
            return check.run(scenario, outcome.response)
        # A crashing check must never count as passed (principle 5): record it as ERROR.
        except Exception as exc:  # noqa: BLE001
            log.warning("check %s crashed on %s: %s", name, scenario.id, type(exc).__name__)
            return self._error_result(scenario, name, f"check raised {type(exc).__name__}: {exc}")

    def evaluate(self, scenario: Scenario, outcome: CallOutcome) -> ScenarioResult:
        results: list[CheckResult] = []
        for name in scenario.checks:
            started = self.timer()
            result = self._run_one(scenario, name, outcome)
            result.duration_ms = round((self.timer() - started) * 1000, 3)
            result.explanation = self.redactor.redact(result.explanation)
            result.evidence = self.redactor.redact_obj(result.evidence)
            result.expected = self.redactor.redact_obj(result.expected)
            results.append(result)

        mismatches: list[str] = []
        match: bool | None = None
        if scenario.role is Role.SELFTEST:
            observed = {r.check: r.status for r in results}
            for name, status in observed.items():
                want = scenario.expected_statuses.get(name)
                if want is None and status is not Status.NOT_APPLICABLE:
                    mismatches.append(f"{name}: observed {status.value}, not declared")
                elif want is not None and want is not status:
                    mismatches.append(f"{name}: expected {want.value}, observed {status.value}")
            match = not mismatches

        response = outcome.response
        excerpt = ""
        if response is not None and self.excerpt_chars:
            excerpt = self.redactor.redact(response.text[: self.excerpt_chars])
        error = None
        if outcome.error is not None:
            error = {
                "kind": str(getattr(outcome.error, "kind", "unknown")),
                "message": self.redactor.redact(str(outcome.error)),
                "stop_reason": outcome.stop_reason,
            }
        return ScenarioResult(
            scenario_id=scenario.id,
            name=scenario.name,
            category=scenario.category,
            severity=scenario.severity,
            role=scenario.role,
            tags=list(scenario.tags),
            status=aggregate_status([r.status for r in results]),
            checks=results,
            latency_ms=response.latency_ms if response is not None else None,
            latency_source=response.latency_source if response is not None else None,
            attempts=outcome.attempts,
            error=error,
            usage=response.usage if response is not None else None,
            response_excerpt=excerpt,
            retrieved_doc_ids=[r.doc_id for r in sorted(scenario.retrieved, key=lambda r: r.rank)],
            selftest_match=match,
            selftest_mismatches=mismatches,
        )
