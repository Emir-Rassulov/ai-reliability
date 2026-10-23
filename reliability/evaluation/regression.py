"""Compare two run reports: changed statuses, changed decision, metric deltas."""

from __future__ import annotations

from dataclasses import dataclass, field

from reliability.models import Decision, RunReport, Status

STATUS_RANK = {
    Status.PASS: 0,
    Status.NOT_APPLICABLE: 0,
    Status.INCONCLUSIVE: 1,
    Status.ERROR: 1,
    Status.FAIL: 2,
}
DECISION_RANK = {Decision.PASS: 0, Decision.REVIEW_REQUIRED: 1, Decision.ERROR: 1, Decision.FAIL: 2}


@dataclass(frozen=True)
class StatusChange:
    scenario_id: str
    check: str
    before: str
    after: str
    regression: bool


@dataclass
class Comparison:
    baseline_id: str
    current_id: str
    decision_before: Decision
    decision_after: Decision
    changes: list[StatusChange] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    deltas: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    @property
    def decision_regressed(self) -> bool:
        return DECISION_RANK[self.decision_after] > DECISION_RANK[self.decision_before]

    @property
    def has_regression(self) -> bool:
        return self.decision_regressed or any(c.regression for c in self.changes)


def _statuses(report: RunReport) -> dict[tuple[str, str], Status]:
    return {(s.scenario_id, c.check): c.status for s in report.scenarios for c in s.checks}


def compare_reports(baseline: RunReport, current: RunReport) -> Comparison:
    before, after = _statuses(baseline), _statuses(current)
    result = Comparison(
        baseline_id=baseline.run_id,
        current_id=current.run_id,
        decision_before=baseline.decision.decision,
        decision_after=current.decision.decision,
    )
    for key in sorted(before.keys() & after.keys()):
        if before[key] is not after[key]:
            result.changes.append(
                StatusChange(
                    scenario_id=key[0],
                    check=key[1],
                    before=before[key].value,
                    after=after[key].value,
                    regression=STATUS_RANK[after[key]] > STATUS_RANK[before[key]],
                )
            )
    base_ids = {s.scenario_id for s in baseline.scenarios}
    cur_ids = {s.scenario_id for s in current.scenarios}
    result.added = sorted(cur_ids - base_ids)
    result.removed = sorted(base_ids - cur_ids)
    pairs: dict[str, tuple[float | None, float | None]] = {
        "pass_rate": (baseline.decision.pass_rate, current.decision.pass_rate),
        "error_rate": (baseline.decision.error_rate, current.decision.error_rate),
        "score": (baseline.score, current.score),
        "latency_p95_ms": (baseline.latency.p95_ms, current.latency.p95_ms),
    }
    for metric in sorted(baseline.rag_metrics.keys() | current.rag_metrics.keys()):
        pairs[metric] = (baseline.rag_metrics.get(metric), current.rag_metrics.get(metric))
    result.deltas = {k: v for k, v in pairs.items() if v != (None, None)}
    return result


def _delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return f"{before} -> {after}"
    return f"{before:.4g} -> {after:.4g} ({after - before:+.4g})"


def render_comparison(cmp: Comparison) -> str:
    lines = [
        f"Baseline {cmp.baseline_id}  ->  current {cmp.current_id}",
        f"Decision: {cmp.decision_before.value} -> {cmp.decision_after.value}"
        + ("  (REGRESSION)" if cmp.decision_regressed else ""),
        "",
        "Changed check statuses:",
    ]
    lines += [
        f"  {'!' if c.regression else ' '} {c.scenario_id}/{c.check}: {c.before} -> {c.after}"
        for c in cmp.changes
    ] or ["  none"]
    if cmp.added or cmp.removed:
        lines += ["", f"Scenarios added: {', '.join(cmp.added) or '-'}"]
        lines += [f"Scenarios removed: {', '.join(cmp.removed) or '-'}"]
    lines += ["", "Metric deltas:"]
    lines += [f"  {k}: {_delta(*v)}" for k, v in cmp.deltas.items() if v[0] != v[1]] or ["  none"]
    lines += ["", "Result: " + ("REGRESSION" if cmp.has_regression else "no regression")]
    return "\n".join(lines) + "\n"
