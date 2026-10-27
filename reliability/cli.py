"""Command-line interface.

Exit codes: 0 PASS, 1 FAIL, 2 REVIEW_REQUIRED, 3 run status ERROR,
4 configuration / argument / scenario validation error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

# The CLI is the composition root: the only module that wires the reference app in.
from rag_app import build_reference_app
from rag_app.callable_generator import CallableGenerator
from rag_app.defects import DEFECTS as DEFECT_NAMES
from rag_app.interface import RAGSystem, RAGSystemError
from reliability import PROJECT_ROOT, __version__
from reliability.checks.base import CheckContext
from reliability.config import AppConfig, load_config
from reliability.errors import ConfigError, ScenarioValidationError
from reliability.evaluation.check_quality import evaluate_labeled, render_quality
from reliability.evaluation.live import (
    compare_live,
    load_record,
    model_metamorphic,
    render_live,
    render_metamorphic,
    repeated_run,
)
from reliability.evaluation.mutation import matrix_table, mutation_test, render_mutation
from reliability.evaluation.orchestration import call_with_retries, run_rag_app, run_suite
from reliability.evaluation.regression import compare_reports, render_comparison
from reliability.models import Category, Decision, Document, Message, Role, RunReport, Scenario
from reliability.providers.base import GenerationParams, Provider, ProviderRequest
from reliability.providers.factory import SUPPORTED_PROVIDERS, build_provider
from reliability.providers.mock_provider import load_mock_responses
from reliability.reporting.history import append_history
from reliability.reporting.html_report import HtmlTable, render_html
from reliability.reporting.json_report import load_report, write_report
from reliability.reporting.redaction import RedactingFilter, Redactor
from reliability.reporting.summary import render_summary
from reliability.scenarios import (
    MOCK_RESPONSES_FILE,
    LoadedScenarios,
    check_mock_coverage,
    collect_sensitive_values,
    load_documents,
    load_scenarios,
    sha256_file,
)

EXIT_PASS, EXIT_FAIL, EXIT_REVIEW, EXIT_ERROR, EXIT_CONFIG = 0, 1, 2, 3, 4
DECISION_EXIT = {
    Decision.PASS: EXIT_PASS,
    Decision.FAIL: EXIT_FAIL,
    Decision.REVIEW_REQUIRED: EXIT_REVIEW,
    Decision.ERROR: EXIT_ERROR,
}
DEFAULT_SCENARIOS = PROJECT_ROOT / "scenarios"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "synthetic_documents.json"
DEFAULT_LABELED = PROJECT_ROOT / "tests" / "fixtures" / "labeled_cases.json"

log = logging.getLogger("reliability")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--path", type=Path, default=DEFAULT_SCENARIOS, help="scenario directory or file"
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="synthetic document set")
    p.add_argument(
        "--config", type=Path, default=None, help="TOML file merged over config/default.toml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reliability.cli",
        description="Offline-first reliability evaluator for LLM and RAG systems.",
        epilog="Exit codes: 0 PASS, 1 FAIL, 2 REVIEW_REQUIRED, 3 ERROR, 4 config/validation error.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress (redacted)")
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser(
        "validate-scenarios", help="validate scenarios, documents and mock responses"
    )
    _add_common(val)

    run = sub.add_parser("run", help="run scenarios and write a JSON report")
    _add_common(run)
    run.add_argument("--provider", default="mock", choices=SUPPORTED_PROVIDERS)
    run.add_argument(
        "--model", default=None, help="model name, e.g. ollama/llama3.2 (litellm only)"
    )
    run.add_argument("--api-base", default=None, help="API base URL for local models")
    run.add_argument(
        "--target-config", type=Path, default=None, help="TOML file describing an HTTP app (http)"
    )
    run.add_argument("--allow-network", action="store_true", help="permit network access")
    run.add_argument("--allow-paid", action="store_true", help="permit paid hosted models")
    run.add_argument("--category", choices=[c.value for c in Category], default=None)
    run.add_argument("--scenario", default=None, help="run a single scenario by id")
    run.add_argument("--report-dir", type=Path, default=Path("reports"))
    run.add_argument(
        "--target",
        choices=("scenarios", "rag-app"),
        default="scenarios",
        help="scenarios: precomputed retrieval (default); rag-app: the reference RAG app",
    )
    run.add_argument(
        "--defect",
        action="append",
        default=[],
        choices=sorted(DEFECT_NAMES),
        help="inject a defect into the reference RAG app (repeatable; --target rag-app)",
    )
    run.add_argument("--html", action="store_true", help="also write <report-dir>/<run_id>.html")
    run.add_argument(
        "--history-file", type=Path, default=None, help="append a compact summary (JSONL)"
    )

    summ = sub.add_parser("summary", help="print a human-readable summary of a JSON report")
    summ.add_argument("--input", type=Path, required=True)

    mut = sub.add_parser(
        "mutation-test", help="run the reference RAG app once per injected defect (exit 1 on gaps)"
    )
    _add_common(mut)
    mut.add_argument("--target", choices=("rag-app",), default="rag-app")
    mut.add_argument("--report-dir", type=Path, default=Path("reports"))
    mut.add_argument("--html", action="store_true")
    _live_args(mut)

    lr = sub.add_parser(
        "live-regression",
        help="repeat each scenario N times and compare pass rates with a baseline (exit 1)",
    )
    _add_common(lr)
    _live_args(lr)
    lr.add_argument("--category", choices=[c.value for c in Category], default=None)
    lr.add_argument("--scenario", default=None)
    lr.add_argument("--runs", type=int, default=None, help="repeats per scenario (default: config)")
    lr.add_argument("--baseline", type=Path, default=None, help="compare with this record")
    lr.add_argument("--output", type=Path, default=None, help="write the record here")

    mm = sub.add_parser(
        "model-metamorphic",
        help="paraphrase / reorder / irrelevant-document variants; consistency per scenario",
    )
    _add_common(mm)
    _live_args(mm)
    mm.add_argument("--scenario", default=None)
    mm.add_argument("--output", type=Path, default=None)

    cq = sub.add_parser("check-quality", help="precision/recall of each check on a labeled set")
    cq.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    cq.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    cq.add_argument("--config", type=Path, default=None)

    cmp_ = sub.add_parser("compare", help="compare two JSON reports (exit 1 on regression)")
    cmp_.add_argument("--baseline", type=Path, required=True)
    cmp_.add_argument("--current", type=Path, required=True)
    return parser


def _load(args: argparse.Namespace) -> tuple[AppConfig, dict[str, Document], LoadedScenarios]:
    config = load_config(args.config)
    documents = load_documents(args.dataset, config.limits)
    loaded = load_scenarios(args.path, documents, config.limits)
    return config, documents, loaded


def _mock_path(path: Path) -> Path:
    return (path if path.is_dir() else path.parent) / MOCK_RESPONSES_FILE


def _select(
    scenarios: list[Scenario], category: str | None, scenario_id: str | None
) -> list[Scenario]:
    """Filter scenarios. A selected scenario that shares a session also pulls in the earlier
    scenarios of that session, because cross-scenario fixtures depend on prior turns."""
    chosen = {
        s.id
        for s in scenarios
        if (category is None or s.category.value == category)
        and (scenario_id is None or s.id == scenario_id)
    }
    if not chosen:
        raise ConfigError("no scenarios match the given --category/--scenario filters")
    for index, scenario in enumerate(scenarios):
        if scenario.id in chosen and scenario.session_id:
            chosen.update(s.id for s in scenarios[:index] if s.session_id == scenario.session_id)
    return [s for s in scenarios if s.id in chosen]


def cmd_validate(args: argparse.Namespace) -> int:
    _config, _documents, loaded = _load(args)
    mock = load_mock_responses(_mock_path(args.path))
    check_mock_coverage(loaded.scenarios, mock)
    gate = sum(1 for s in loaded.scenarios if s.role is Role.GATE)
    print(
        f"OK: {len(loaded.scenarios)} scenarios in {len(loaded.file_hashes)} files "
        f"({gate} gate, {len(loaded.scenarios) - gate} selftest); "
        f"{len(mock)} canned responses; documents valid."
    )
    return EXIT_PASS


def _print_outcome(report: RunReport, json_path: Path) -> None:
    counts = report.counts
    print(f"Decision: {report.decision.decision.value}")
    for rule in report.decision.fired_rules:
        print(f"  - {rule.rule}: {rule.detail}")
    print(
        "Gate checks: "
        + ", ".join(f"{k} {v}" for k, v in counts.items() if v)
        + f" | evaluator self-tests: {report.selftest.matched}/{report.selftest.total} matched"
    )
    print(f"Report: {json_path}")


def _rag_app_scenarios(scenarios: list[Scenario], args: argparse.Namespace) -> list[Scenario]:
    eligible = [s for s in scenarios if s.rag_app]
    category = getattr(args, "category", None)
    scenario_id = getattr(args, "scenario", None)
    selected = [
        s
        for s in eligible
        if (category is None or s.category.value == category)
        and (scenario_id is None or s.id == scenario_id)
    ]
    if not selected:
        raise ConfigError("no rag_app-eligible scenarios match the given filters")
    return selected


def _live_args(p: argparse.ArgumentParser, default_provider: str = "mock") -> None:
    p.add_argument("--provider", default=default_provider, choices=SUPPORTED_PROVIDERS)
    p.add_argument("--model", default=None, help="model name, e.g. ollama/llama3.2 (litellm)")
    p.add_argument("--api-base", default=None, help="API base URL for local models")
    p.add_argument(
        "--target-config", type=Path, default=None, help="TOML file describing an HTTP app (http)"
    )
    p.add_argument("--allow-network", action="store_true", help="permit network access")
    p.add_argument("--allow-paid", action="store_true", help="permit paid hosted models")


def _provider_from(args: argparse.Namespace, config: AppConfig) -> Provider:
    return build_provider(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        allow_network=args.allow_network,
        allow_paid=args.allow_paid,
        config=config,
        mock_responses_path=_mock_path(args.path),
        target_config=args.target_config,
    )


def _generator_factory(
    args: argparse.Namespace, config: AppConfig
) -> Callable[[frozenset[str]], RAGSystem]:
    """Reference app factory: offline mock generator, or a live model as the generator."""
    if args.provider == "mock":
        return build_reference_app
    provider = _provider_from(args, config)

    def complete(messages: list[dict[str, str]]) -> str:
        request = ProviderRequest(
            scenario_id="rag-app",
            system=messages[0]["content"],
            messages=[Message(role="user", content=messages[-1]["content"])],
            params=GenerationParams(timeout_s=config.limits.request_timeout_s),
        )
        outcome = call_with_retries(provider, request, config.limits, _sleep_for(provider))
        if outcome.response is None:
            raise RAGSystemError(f"generator failed: {outcome.error} ({outcome.stop_reason})")
        return outcome.response.text

    name = f"{provider.name}:{provider.model}"
    return lambda defects: build_reference_app(defects, generator=CallableGenerator(name, complete))


def _run_rag_app(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    documents = load_documents(args.dataset, config.limits)
    loaded = load_scenarios(args.path, documents, config.limits)
    selected = _rag_app_scenarios(loaded.scenarios, args)
    redactor = Redactor(collect_sensitive_values(loaded.scenarios))
    _install_log_redaction(redactor)
    defects = sorted(set(args.defect))
    report = run_rag_app(
        scenarios=selected,
        system=_generator_factory(args, config)(frozenset(defects)),
        documents=documents,
        config=config,
        redactor=redactor,
        defects=defects,
        scenario_files=loaded.file_hashes,
        dataset_hash=sha256_file(args.dataset),
    )
    _write_outputs(report, args, redactor)
    return DECISION_EXIT[report.decision.decision]


def cmd_mutation_test(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    documents = load_documents(args.dataset, config.limits)
    loaded = load_scenarios(args.path, documents, config.limits)
    selected = _rag_app_scenarios(loaded.scenarios, args)
    redactor = Redactor(collect_sensitive_values(loaded.scenarios))
    _install_log_redaction(redactor)
    result = mutation_test(
        scenarios=selected,
        factory=_generator_factory(args, config),
        defects=sorted(DEFECT_NAMES),
        documents=documents,
        config=config,
        redactor=redactor,
    )
    print(render_mutation(result), end="")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / f"mutation-{result.baseline.run_id}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Matrix: {out}")
    if args.html:
        html_path = args.report_dir / f"mutation-{result.baseline.run_id}.html"
        html_path.write_text(
            render_html(result.baseline, redactor, [matrix_table(result)]), encoding="utf-8"
        )
        print(f"HTML: {html_path}")
    return EXIT_PASS if result.criteria_met else EXIT_FAIL


def cmd_run(args: argparse.Namespace) -> int:
    if args.target == "rag-app":
        return _run_rag_app(args)
    if args.defect:
        raise ConfigError("--defect requires --target rag-app")
    config = load_config(args.config)
    provider = build_provider(
        provider=args.provider,
        model=args.model,
        api_base=args.api_base,
        allow_network=args.allow_network,
        allow_paid=args.allow_paid,
        config=config,
        mock_responses_path=_mock_path(args.path),
        target_config=args.target_config,
    )
    documents = load_documents(args.dataset, config.limits)
    loaded = load_scenarios(args.path, documents, config.limits)
    if args.provider == "mock":
        check_mock_coverage(loaded.scenarios, load_mock_responses(_mock_path(args.path)))
    selected = _select(loaded.scenarios, args.category, args.scenario)
    redactor = Redactor(collect_sensitive_values(loaded.scenarios))
    _install_log_redaction(redactor)
    report = run_suite(
        scenarios=selected,
        provider=provider,
        documents=documents,
        config=config,
        redactor=redactor,
        scenario_files=loaded.file_hashes,
        dataset_hash=sha256_file(args.dataset),
        sleep=_sleep_for(provider),
    )
    _write_outputs(report, args, redactor)
    return DECISION_EXIT[report.decision.decision]


def _sleep_for(provider: object) -> Callable[[float], None]:
    """Simulated providers record backoff waits without sleeping; real ones really wait."""
    if getattr(provider, "simulated_time", False):
        return lambda _seconds: None
    return time.sleep


def _write_outputs(
    report: RunReport,
    args: argparse.Namespace,
    redactor: Redactor,
    tables: Sequence[HtmlTable] = (),
) -> None:
    json_path = write_report(report, args.report_dir, redactor)
    _print_outcome(report, json_path)
    if args.html:
        html_path = args.report_dir / f"{report.run_id}.html"
        html_path.write_text(render_html(report, redactor, tables), encoding="utf-8")
        print(f"HTML: {html_path}")
    if args.history_file is not None:
        append_history(report, args.history_file)
        print(f"History: appended to {args.history_file}")


def cmd_summary(args: argparse.Namespace) -> int:
    print(render_summary(load_report(args.input)), end="")
    return EXIT_PASS


def cmd_check_quality(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    documents = load_documents(args.dataset, config.limits)
    if not args.labeled.exists():
        raise ConfigError(f"labeled set not found: {args.labeled}")
    results = evaluate_labeled(args.labeled, CheckContext(config=config, documents=documents))
    print(render_quality(results), end="")
    unexpected = [
        c
        for q in results.values()
        for c in q.cases
        if c.truth is not c.observed and not c.known_miss
    ]
    return EXIT_FAIL if unexpected else EXIT_PASS


def cmd_compare(args: argparse.Namespace) -> int:
    comparison = compare_reports(load_report(args.baseline), load_report(args.current))
    print(render_comparison(comparison), end="")
    return EXIT_FAIL if comparison.has_regression else EXIT_PASS


def _install_log_redaction(redactor: Redactor) -> None:
    for handler in logging.getLogger().handlers:
        handler.filters = [f for f in handler.filters if not isinstance(f, RedactingFilter)]
        handler.addFilter(RedactingFilter(redactor))


def _gate_selection(args: argparse.Namespace, scenarios: list[Scenario]) -> list[Scenario]:
    selected = _select(scenarios, getattr(args, "category", None), args.scenario)
    return [s for s in selected if s.role is Role.GATE] or selected


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_live_regression(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider = _provider_from(args, config)
    documents = load_documents(args.dataset, config.limits)
    loaded = load_scenarios(args.path, documents, config.limits)
    selected = _gate_selection(args, loaded.scenarios)
    redactor = Redactor(collect_sensitive_values(loaded.scenarios))
    _install_log_redaction(redactor)
    runs = args.runs if args.runs is not None else config.live.runs
    if runs < 1:
        raise ConfigError("--runs must be at least 1")
    record = repeated_run(
        scenarios=selected,
        provider=provider,
        documents=documents,
        config=config,
        redactor=redactor,
        runs=runs,
        sleep=_sleep_for(provider),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, record)
    if args.baseline is None:
        print(render_live(record), end="")
        return EXIT_PASS
    margin = config.live.regression_margin
    regressions, warnings = compare_live(load_record(args.baseline), record, margin)
    print(render_live(record, regressions, warnings, margin), end="")
    return EXIT_FAIL if regressions else EXIT_PASS


def cmd_model_metamorphic(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider = _provider_from(args, config)
    documents = load_documents(args.dataset, config.limits)
    loaded = load_scenarios(args.path, documents, config.limits)
    pool = [s for s in loaded.scenarios if s.role is Role.GATE and s.paraphrases]
    if args.scenario is not None:
        pool = [s for s in loaded.scenarios if s.id == args.scenario]
    if not pool:
        raise ConfigError("no scenarios with paraphrases match the filters")
    redactor = Redactor(collect_sensitive_values(loaded.scenarios))
    _install_log_redaction(redactor)
    result = model_metamorphic(
        scenarios=pool,
        provider=provider,
        documents=documents,
        config=config,
        redactor=redactor,
        sleep=_sleep_for(provider),
    )
    print(render_metamorphic(result), end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, result)
    return EXIT_PASS


COMMANDS = {
    "validate-scenarios": cmd_validate,
    "run": cmd_run,
    "summary": cmd_summary,
    "compare": cmd_compare,
    "check-quality": cmd_check_quality,
    "mutation-test": cmd_mutation_test,
    "live-regression": cmd_live_regression,
    "model-metamorphic": cmd_model_metamorphic,
}


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments; our contract says 4
        return EXIT_PASS if exc.code in (0, None) else EXIT_CONFIG
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return COMMANDS[args.command](args)
    except ScenarioValidationError as exc:
        print("Validation failed:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_CONFIG
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
