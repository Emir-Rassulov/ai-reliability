"""Human-readable run summary, written for a QA engineer."""

from __future__ import annotations

from collections.abc import Iterable

from reliability.models import (
    Category,
    CheckResult,
    Decision,
    Role,
    RunReport,
    ScenarioResult,
    Severity,
    Status,
)

SECURITY_CHECKS = frozenset({"prompt_injection", "data_leakage", "forbidden_content"})
RAG_CHECKS = frozenset({"retrieval_metrics", "context_quality", "grounding", "citations"})

CHECK_ADVICE: dict[str, str] = {
    "required_facts": "compare the answer with the annotated facts; check prompt and source data",
    "forbidden_content": "find why forbidden text reached the output; tighten filters",
    "structured_output": "inspect the raw output; enforce the schema (JSON mode, validation retry)",
    "refusal": "over-refusal or unsafe compliance: review safety instructions and examples",
    "instruction_following": "make format and length constraints explicit in the prompt",
    "retrieval_metrics": "inspect the ranking: query, index contents, top-K and scoring",
    "context_quality": "check access filtering, version handling and evidence coverage",
    "grounding": "check that the answer uses only retrieved evidence; review the grounding prompt",
    "citations": "check citation format instructions and that cited ids come from the context",
    "prompt_injection": "treat retrieved/user content as data; add isolation and output filtering",
    "data_leakage": "remove secrets from context, enforce access control, add output redaction",
    "provider_call": "check provider health, credentials, timeouts and retry configuration",
}

RULE_ADVICE: dict[str, str] = {
    "R0_evaluator_selftest": "Fix the evaluator first: a self-test fixture no longer reproduces "
    "its verdict, so this run's verdicts are not trustworthy.",
    "R1_critical_or_mandatory": "Block the release and fix the critical/mandatory failures listed.",
    "R2_min_pass_rate": "Block the release; triage the failing checks by severity.",
    "R3_max_error_rate": "No application verdict: fix infrastructure/provider errors and rerun.",
    "R4_incomplete_severe_check": "Manually review inconclusive or errored critical/high checks.",
    "R4_max_inconclusive_rate": "Too many undecidable checks: add annotations or review manually.",
    "R4_no_gate_checks": "Nothing was assessed as a gate: run the full suite before deciding.",
    "R5_latency": "Investigate latency regressions before release.",
    "R6_all_gates_passed": "Archive this report as release evidence; compare it with the previous "
    "run (`compare`) to spot drift.",
}


def _findings(
    scenarios: Iterable[ScenarioResult], role: Role
) -> list[tuple[ScenarioResult, CheckResult]]:
    return [
        (s, c)
        for s in scenarios
        if s.role is role
        for c in s.checks
        if c.status not in (Status.PASS, Status.NOT_APPLICABLE)
    ]


def _line(s: ScenarioResult, c: CheckResult) -> str:
    return f"  - [{c.status.value}] {s.scenario_id}/{c.check} ({c.severity.value}): {c.explanation}"


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.3f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def render_summary(report: RunReport) -> str:
    d = report.decision
    gate_findings = _findings(report.scenarios, Role.GATE)
    out: list[str] = [
        f"AI reliability run {report.run_id}",
        f"Provider {report.provider} ({report.model or 'model unknown'}), target {report.target}"
        + (f", defects: {', '.join(report.defects)}" if report.defects else ""),
        "",
        f"DECISION: {d.decision.value}",
        "Why:",
        *[f"  - {r.rule}: {r.detail}" for r in d.fired_rules],
        "",
        "Gate checks: " + " | ".join(f"{k} {v}" for k, v in report.counts.items()),
        f"Pass rate {_fmt(d.pass_rate, True)} | error rate {_fmt(d.error_rate, True)} | "
        f"inconclusive rate {_fmt(d.inconclusive_rate, True)} | "
        f"score {_fmt(report.score)} (informational)",
        f"Evaluator self-tests: {report.selftest.matched}/{report.selftest.total} reproduced "
        "their declared verdicts"
        + (
            f"; MISMATCHED: {', '.join(report.selftest.mismatched)}"
            if report.selftest.mismatched
            else ""
        ),
        "",
        "Critical and high findings (gate):",
    ]
    severe = [(s, c) for s, c in gate_findings if c.severity in (Severity.CRITICAL, Severity.HIGH)]
    out += [_line(s, c) for s, c in severe] or ["  - none"]

    out += ["", "RAG (gate means):"]
    rag = report.rag_metrics
    out.append(
        "  precision@k {p} | recall@k {r} | MRR {m} | context recall {cr} | "
        "citation accuracy {ca}".format(
            p=_fmt(rag.get("retrieval_metrics.precision_at_k")),
            r=_fmt(rag.get("retrieval_metrics.recall_at_k")),
            m=_fmt(rag.get("retrieval_metrics.mrr")),
            cr=_fmt(rag.get("context_quality.context_recall")),
            ca=_fmt(rag.get("citations.citation_accuracy")),
        )
    )
    rag_fail = [(s, c) for s, c in gate_findings if c.check in RAG_CHECKS]
    out += [_line(s, c) for s, c in rag_fail] or ["  - no RAG failures in gate scenarios"]

    out += ["", "Security findings (gate):"]
    sec = [(s, c) for s, c in gate_findings if c.check in SECURITY_CHECKS]
    out += [_line(s, c) for s, c in sec] or ["  - none"]
    detected = [
        (s, c)
        for s, c in _findings(report.scenarios, Role.SELFTEST)
        if c.check in SECURITY_CHECKS and c.status is Status.FAIL
    ]
    out.append(f"  ({len(detected)} known-bad security fixtures detected by the evaluator)")

    out += ["", "Operational:"]
    retried = [s for s in report.scenarios if s.role is Role.GATE and s.attempts > 1]
    out += [f"  - {s.scenario_id}: succeeded after {s.attempts} attempts" for s in retried]
    errored = [s for s in report.scenarios if s.role is Role.GATE and s.error]
    out += [
        f"  - {s.scenario_id}: {s.error['kind']} ({s.error['stop_reason']})"
        for s in errored
        if s.error
    ]
    lat = report.latency
    out.append(
        f"  latency median {_ms(lat.median_ms)}, p95 {_ms(lat.p95_ms)} over {lat.count} "
        f"gate calls (source: {', '.join(lat.sources) or 'n/a'})"
    )
    if report.token_usage is not None:
        out.append(
            f"  tokens reported: input {report.token_usage.input_tokens}, "
            f"output {report.token_usage.output_tokens}"
        )
    else:
        out.append("  tokens: not reported by the provider")

    out += ["", "Recommended next actions:"]
    out += [f"  - {RULE_ADVICE[r.rule]}" for r in d.fired_rules if r.rule in RULE_ADVICE]
    for check in sorted({c.check for _, c in gate_findings}):
        out.append(f"  - {check}: {CHECK_ADVICE.get(check, 'review the evidence in the report')}")
    if d.decision is Decision.PASS and any(s.category is Category.OPERATIONAL for s in retried):
        out.append("  - retries were needed: watch provider stability")
    return "\n".join(out) + "\n"
