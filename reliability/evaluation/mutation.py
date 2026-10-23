"""Mutation testing of the evaluator against the reference RAG app (§7a).

Run the eligible scenarios once without defects and once per injected defect. A check
"detects" a defect when, for at least one scenario, it was PASS/NOT_APPLICABLE in the
baseline and FAIL/INCONCLUSIVE/ERROR with the defect. Acceptance criteria:

* the baseline decision is PASS;
* every defect is detected by at least one of its expected checks, and turns the decision
  into FAIL or REVIEW_REQUIRED.

A defect that misses either criterion is an evaluator gap and is reported, not hidden.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_app.interface import RAGSystem
from reliability.config import AppConfig
from reliability.evaluation.orchestration import run_rag_app
from reliability.models import Decision, Document, RunReport, Scenario, Status
from reliability.reporting.html_report import HtmlTable
from reliability.reporting.redaction import Redactor

#: Expected detection per defect (spec §7a), keyed by defect name.
EXPECTED_DETECTION: dict[str, frozenset[str]] = {
    "no_access_filter": frozenset({"context_quality", "data_leakage"}),
    "stale_first": frozenset({"context_quality", "grounding"}),
    "no_grounding_instruction": frozenset({"grounding", "refusal"}),
    "truncated_context": frozenset({"grounding", "context_quality"}),
    "no_untrusted_marking": frozenset({"prompt_injection"}),
}
_OK = (Status.PASS, Status.NOT_APPLICABLE)
_BAD = (Status.FAIL, Status.INCONCLUSIVE, Status.ERROR)

SystemFactory = Callable[[frozenset[str]], RAGSystem]


@dataclass
class MutationResult:
    baseline: RunReport
    runs: dict[str, RunReport]
    checks: list[str]
    #: defect -> check -> scenario ids where the check flipped to a bad status
    detections: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    @property
    def baseline_ok(self) -> bool:
        return self.baseline.decision.decision is Decision.PASS

    @property
    def criteria_met(self) -> bool:
        return self.baseline_ok and not self.gaps

    def matrix(self) -> dict[str, dict[str, bool]]:
        return {
            defect: {check: bool(self.detections[defect].get(check)) for check in self.checks}
            for defect in self.runs
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline.run_id,
            "baseline_decision": self.baseline.decision.decision.value,
            "decisions": {d: r.decision.decision.value for d, r in self.runs.items()},
            "matrix": self.matrix(),
            "detections": self.detections,
            "expected": {d: sorted(c) for d, c in EXPECTED_DETECTION.items() if d in self.runs},
            "gaps": self.gaps,
            "criteria_met": self.criteria_met,
        }


def _statuses(report: RunReport) -> dict[tuple[str, str], Status]:
    return {(s.scenario_id, c.check): c.status for s in report.scenarios for c in s.checks}


def mutation_test(
    *,
    scenarios: Sequence[Scenario],
    factory: SystemFactory,
    defects: Iterable[str],
    documents: Mapping[str, Document],
    config: AppConfig,
    redactor: Redactor,
) -> MutationResult:
    def run(active: frozenset[str]) -> RunReport:
        return run_rag_app(
            scenarios=scenarios,
            system=factory(active),
            documents=documents,
            config=config,
            redactor=redactor,
            defects=sorted(active),
        )

    baseline = run(frozenset())
    base = _statuses(baseline)
    checks = sorted({check for _, check in base})
    result = MutationResult(baseline=baseline, runs={}, checks=checks)
    for defect in defects:
        report = run(frozenset({defect}))
        result.runs[defect] = report
        flipped: dict[str, list[str]] = {}
        for (scenario_id, check), status in sorted(_statuses(report).items()):
            if base.get((scenario_id, check)) in _OK and status in _BAD:
                flipped.setdefault(check, []).append(scenario_id)
        result.detections[defect] = flipped
        expected = EXPECTED_DETECTION.get(defect, frozenset())
        caught = expected & flipped.keys()
        decision = report.decision.decision
        if not caught:
            result.gaps.append(
                f"{defect}: none of the expected checks ({', '.join(sorted(expected))}) "
                "changed verdict"
            )
        if decision not in (Decision.FAIL, Decision.REVIEW_REQUIRED):
            result.gaps.append(f"{defect}: decision stayed {decision.value}")
    return result


def render_mutation(result: MutationResult) -> str:
    first = max(len(d) for d in [*result.runs, "defect"]) + 2
    widths = [max(len(c), len("DETECTED*")) + 2 for c in result.checks]
    header = (
        "defect".ljust(first)
        + "".join(c.ljust(w) for c, w in zip(result.checks, widths, strict=True))
        + "decision"
    )
    lines = [
        f"Baseline (no defects): {result.baseline.decision.decision.value}",
        "",
        header,
        "-" * len(header),
    ]
    for defect, row in result.matrix().items():
        expected = EXPECTED_DETECTION.get(defect, frozenset())
        cells = []
        for check, width in zip(result.checks, widths, strict=True):
            mark = "DETECTED" if row[check] else "-"
            if check in expected:
                mark = f"{mark}*" if row[check] else "missed*"
            cells.append(mark.ljust(width))
        lines.append(
            defect.ljust(first) + "".join(cells) + result.runs[defect].decision.decision.value
        )
    lines += ["", "* = expected detection for that defect (spec section 7a)"]
    lines += ["", "Evaluator gaps:"]
    lines += [f"  - {g}" for g in result.gaps] or ["  none"]
    lines += ["", "Acceptance criteria: " + ("MET" if result.criteria_met else "NOT MET")]
    return "\n".join(lines) + "\n"


def matrix_table(result: MutationResult) -> HtmlTable:
    rows = []
    for defect, row in result.matrix().items():
        expected = EXPECTED_DETECTION.get(defect, frozenset())
        cells = [
            ("detected" if row[c] else "-") + (" (expected)" if c in expected else "")
            for c in result.checks
        ]
        rows.append([defect, *cells, result.runs[defect].decision.decision.value])
    return HtmlTable(
        title="Defect detection matrix (reference RAG app)",
        columns=["defect", *result.checks, "decision"],
        rows=rows,
        note=f"Baseline without defects: {result.baseline.decision.decision.value}. "
        + (
            "All acceptance criteria met."
            if result.criteria_met
            else "Gaps: " + "; ".join(result.gaps)
        ),
    )
