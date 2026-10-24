"""Data loading and aggregation for the dashboard. Plain functions, testable without Streamlit.

Rule 1: presentation only. Nothing here evaluates, scores or decides; every number comes from
stored reports (`reports/*.json`, `reports/mutation-*.json`) or `history/runs.jsonl`, and
comparisons reuse `reliability.evaluation.regression`.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reliability import PROJECT_ROOT
from reliability.evaluation.regression import Comparison, compare_reports
from reliability.models import Decision, Role, RunReport, Status

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_HISTORY = PROJECT_ROOT / "history" / "runs.jsonl"
MAX_REPORT_BYTES = 20_000_000


def public_mode(env: dict[str, str] | None = None) -> bool:
    """Public mode is ON unless AIREL_PUBLIC_MODE is explicitly false/0/no/off."""
    raw = (env if env is not None else os.environ).get("AIREL_PUBLIC_MODE", "true")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def report_dir(env: dict[str, str] | None = None) -> Path:
    raw = (env if env is not None else os.environ).get("AIREL_REPORT_DIR", "")
    return Path(raw) if raw.strip() else DEFAULT_REPORT_DIR


def history_path(env: dict[str, str] | None = None) -> Path:
    raw = (env if env is not None else os.environ).get("AIREL_HISTORY_FILE", "")
    return Path(raw) if raw.strip() else DEFAULT_HISTORY


# --------------------------------------------------------------------------- loading


@dataclass
class Loaded:
    reports: list[RunReport] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def load_reports(directory: Path) -> Loaded:
    """All run reports in `directory`, oldest first. Malformed files are listed, not fatal."""
    out = Loaded()
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("mutation-"):
            continue
        try:
            if path.stat().st_size > MAX_REPORT_BYTES:
                out.problems.append(f"{path.name}: larger than {MAX_REPORT_BYTES} bytes, skipped")
                continue
            out.reports.append(RunReport.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, ValidationError, ValueError) as exc:
            kind = "invalid report" if isinstance(exc, ValidationError) else type(exc).__name__
            out.problems.append(f"{path.name}: {kind}, skipped")
    out.reports.sort(key=lambda r: (r.started_at, r.run_id))
    return out


def latest_mutation(directory: Path) -> dict[str, Any] | None:
    """The newest mutation-test result, or None. Only well-formed files are accepted."""
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob("mutation-*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("matrix"), dict):
            return data
    return None


def read_history(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """History entries (JSONL), oldest first. Malformed lines are reported and skipped."""
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    if not path.is_file():
        return entries, problems
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            problems.append(f"{path.name}:{number}: not valid JSON, skipped")
            continue
        if not isinstance(item, dict) or "run_id" not in item or "decision" not in item:
            problems.append(f"{path.name}:{number}: missing run_id/decision, skipped")
            continue
        entries.append(item)
    entries.sort(key=lambda e: (str(e.get("timestamp", "")), str(e["run_id"])))
    return entries, problems


# --------------------------------------------------------------------------- overview


@dataclass(frozen=True)
class Card:
    label: str
    value: str
    delta: str | None
    #: Streamlit delta colour: "normal" (up is good), "inverse" (up is bad) or "off".
    delta_color: str
    help: str


def _num(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:,.0f}"


def _delta(cur: float | None, prev: float | None, pct: bool = False) -> str | None:
    if cur is None or prev is None:
        return None
    diff = cur - prev
    if diff == 0:
        return "0"
    return f"{diff:+.1%}" if pct else f"{diff:+,.0f}"


def previous_comparable(reports: list[RunReport], current: RunReport) -> RunReport | None:
    """The most recent earlier run against the same target (scenario suite vs RAG app)."""
    earlier = [r for r in reports[: reports.index(current)] if r.target == current.target]
    return earlier[-1] if earlier else None


def metric_cards(current: RunReport, previous: RunReport | None) -> list[Card]:
    def count(report: RunReport | None, status: Status) -> float | None:
        return None if report is None else float(report.counts.get(status.value, 0))

    cur_d, prev_d = current.decision, previous.decision if previous else None
    cards = [
        Card(
            "Release decision",
            cur_d.decision.value,
            None
            if prev_d is None or prev_d.decision is cur_d.decision
            else f"was {prev_d.decision.value}",
            "off",
            "First fired rule of the release policy (README: Scoring and release decision).",
        ),
        Card(
            "Pass rate",
            _num(cur_d.pass_rate, pct=True),
            _delta(cur_d.pass_rate, prev_d.pass_rate if prev_d else None, pct=True),
            "normal",
            "PASS / (PASS + FAIL) over gate checks.",
        ),
    ]
    for status, label in (
        (Status.FAIL, "Failures"),
        (Status.INCONCLUSIVE, "Inconclusive"),
        (Status.ERROR, "Errors"),
    ):
        cur, prev = count(current, status), count(previous, status)
        cards.append(
            Card(label, _num(cur), _delta(cur, prev), "inverse", f"{status.value} gate checks.")
        )
    cur_lat = current.latency.median_ms
    prev_lat = previous.latency.median_ms if previous else None
    if previous is not None and previous.latency.sources != current.latency.sources:
        prev_lat = None  # simulated and measured latency are not comparable
    # Only known values reach the tooltip, which Streamlit renders as Markdown.
    source = ", ".join(s for s in current.latency.sources if s in ("simulated", "measured"))
    source = source or "n/a"
    cards.append(
        Card(
            "Median latency (ms)",
            _num(cur_lat),
            _delta(cur_lat, prev_lat),
            "inverse",
            f"Median over gate calls; source: {source}.",
        )
    )
    return cards


@dataclass(frozen=True)
class Finding:
    scenario_id: str
    name: str
    role: str
    category: str
    check: str
    severity: str
    status: str
    explanation: str
    evidence: str
    excerpt: str


def findings(
    report: RunReport,
    *,
    statuses: set[str] | None = None,
    categories: set[str] | None = None,
    severities: set[str] | None = None,
    roles: set[str] | None = None,
    include_passing: bool = False,
) -> list[Finding]:
    out: list[Finding] = []
    for s in report.scenarios:
        for c in s.checks:
            if not include_passing and c.status in (Status.PASS, Status.NOT_APPLICABLE):
                continue
            if statuses is not None and c.status.value not in statuses:
                continue
            if categories is not None and s.category.value not in categories:
                continue
            if severities is not None and c.severity.value not in severities:
                continue
            if roles is not None and s.role.value not in roles:
                continue
            out.append(
                Finding(
                    scenario_id=s.scenario_id,
                    name=s.name,
                    role=s.role.value,
                    category=s.category.value,
                    check=c.check,
                    severity=c.severity.value,
                    status=c.status.value,
                    explanation=c.explanation,
                    evidence=json.dumps(c.evidence, indent=1, sort_keys=True),
                    excerpt=s.response_excerpt,
                )
            )
    return out


SECURITY_TYPES = {
    "prompt_injection": "prompt injection",
    "data_leakage": "data leakage",
    "forbidden_content": "forbidden content",
}


def security_breakdown(report: RunReport) -> dict[str, dict[str, int]]:
    """FAILs by security type, split into gate findings and detected self-test fixtures."""
    out: dict[str, Counter[str]] = {"gate": Counter(), "selftest": Counter()}
    for s in report.scenarios:
        for c in s.checks:
            if c.status is not Status.FAIL:
                continue
            if c.check in SECURITY_TYPES:
                out[s.role.value][SECURITY_TYPES[c.check]] += 1
            elif c.check == "context_quality" and "unauthorized" in c.flags:
                out[s.role.value]["unauthorized document"] += 1
    return {role: dict(counter) for role, counter in out.items()}


def rag_quality(report: RunReport) -> dict[str, dict[str, float | None]]:
    """Retrieval and generation means already stored in the report, grouped for display."""
    m = report.rag_metrics
    return {
        "retrieval": {
            "Precision@K": m.get("retrieval_metrics.precision_at_k"),
            "Recall@K": m.get("retrieval_metrics.recall_at_k"),
            "MRR": m.get("retrieval_metrics.mrr"),
            "Context recall": m.get("context_quality.context_recall"),
        },
        "generation": {
            "Citation accuracy": m.get("citations.citation_accuracy"),
            "Unverified content ratio": m.get("grounding.unverified_content_ratio"),
        },
    }


GROUNDING_RESULTS = frozenset(
    {
        "supported",
        "unsupported",
        "contradicted",
        "omitted",
        "hedged",
        "unattributed",
        "false_claim",
        "hedged_false_claim",
    }
)


def grounding_outcomes(report: RunReport) -> dict[str, int]:
    """Counts of grounding evidence results (supported / contradicted / omitted / ...)."""
    counter: Counter[str] = Counter()
    for s in report.scenarios:
        if s.role is not Role.GATE:
            continue
        for c in s.checks:
            if c.check != "grounding":
                continue
            for item in c.evidence:
                result = item.get("result")
                if result in GROUNDING_RESULTS:  # chart labels: known values only
                    counter[str(result)] += 1
    return dict(sorted(counter.items()))


# --------------------------------------------------------------------------- history / compare

TREND_METRICS: dict[str, str] = {
    "pass_rate": "Pass rate",
    "score": "Score (informational)",
    "latency_p95_ms": "p95 latency (ms)",
    "rag_metrics.retrieval_metrics.mrr": "MRR",
    "rag_metrics.retrieval_metrics.recall_at_k": "Recall@K",
    "rag_metrics.context_quality.context_recall": "Context recall",
    "rag_metrics.citations.citation_accuracy": "Citation accuracy",
    "counts.FAIL": "Failures",
    "counts.ERROR": "Errors",
}


def _dig(entry: dict[str, Any], dotted: str) -> float | None:
    head, _, rest = dotted.partition(".")
    if rest and head in ("rag_metrics", "counts"):
        container = entry.get(head)
        value = container.get(rest) if isinstance(container, dict) else None
    else:
        value = entry.get(dotted)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def trend(entries: list[dict[str, Any]], metric: str) -> list[tuple[str, float | None, str]]:
    """(run label, value, decision) per history entry for one metric."""
    if metric not in TREND_METRICS:
        raise KeyError(f"unknown trend metric '{metric}'")
    return [
        (_safe_label(e, i), _dig(e, metric), _safe_decision(e)) for i, e in enumerate(entries, 1)
    ]


_LABEL_OK = re.compile(r"^[0-9A-Za-z:._+\-]{1,40}$")


def _safe_label(entry: dict[str, Any], index: int) -> str:
    """Chart labels come from a file; Plotly renders some HTML in labels, so only allow a
    timestamp-like token and fall back to a neutral label otherwise."""
    raw = str(entry.get("timestamp", ""))[:19]
    return raw if _LABEL_OK.match(raw) else f"run {index}"


def safe_decision(value: object) -> str:
    """A decision value from a file, or 'unknown' - safe to show in Markdown-rendered text."""
    text = str(value)
    return text if text in _DECISION_RANK else "unknown"


def _safe_decision(entry: dict[str, Any]) -> str:
    value = str(entry.get("decision"))
    return value if value in _DECISION_RANK else "unknown"


_DECISION_RANK = {
    d.value: i
    for i, d in enumerate((Decision.PASS, Decision.REVIEW_REQUIRED, Decision.ERROR, Decision.FAIL))
}


def history_regressions(entries: list[dict[str, Any]]) -> list[str]:
    """Runs whose decision got worse, or whose FAIL count grew, compared with the run before."""
    alerts: list[str] = []
    for prev, cur in itertools.pairwise(entries):
        before, after = str(prev["decision"]), str(cur["decision"])
        if _DECISION_RANK.get(after, 0) > _DECISION_RANK.get(before, 0):
            alerts.append(f"{cur['run_id']}: decision {before} -> {after}")
        fails_before, fails_after = _dig(prev, "counts.FAIL"), _dig(cur, "counts.FAIL")
        if fails_before is not None and fails_after is not None and fails_after > fails_before:
            alerts.append(f"{cur['run_id']}: failures {fails_before:.0f} -> {fails_after:.0f}")
    return alerts


def compare(baseline: RunReport, current: RunReport) -> Comparison:
    return compare_reports(baseline, current)


def report_label(report: RunReport) -> str:
    extra = f", defects: {', '.join(report.defects)}" if report.defects else ""
    return f"{report.started_at[:19]} - {report.decision.decision.value} ({report.target}{extra})"


# --------------------------------------------------------------------------- RAG pipeline

PIPELINE_STAGES: list[dict[str, Any]] = [
    {
        "stage": "Search (BM25)",
        "checks": ["retrieval_metrics", "context_quality"],
        "defects": ["stale_first"],
    },
    {
        "stage": "Access filter",
        "checks": ["context_quality", "data_leakage"],
        "defects": ["no_access_filter"],
    },
    {
        "stage": "Prompt assembly",
        "checks": ["grounding", "prompt_injection", "citations"],
        "defects": ["no_grounding_instruction", "no_untrusted_marking", "truncated_context"],
    },
    {
        "stage": "Generation",
        "checks": ["grounding", "citations", "refusal", "data_leakage"],
        "defects": [],
    },
]


def pipeline_rows(mutation: dict[str, Any] | None) -> list[dict[str, str]]:
    matrix: dict[str, dict[str, bool]] = (mutation or {}).get("matrix", {})
    rows = []
    for stage in PIPELINE_STAGES:
        caught = []
        for defect in stage["defects"]:
            hits = [c for c, ok in matrix.get(defect, {}).items() if ok]
            if mutation is None:
                caught.append(f"{defect}: no mutation-test data")
            else:
                caught.append(f"{defect}: " + (", ".join(hits) if hits else "NOT DETECTED"))
        rows.append(
            {
                "Stage": stage["stage"],
                "Covering checks": ", ".join(stage["checks"]),
                "Injected defects and the checks that caught them": "; ".join(caught) or "-",
            }
        )
    return rows


def matrix_rows(mutation: dict[str, Any]) -> list[dict[str, str]]:
    expected: dict[str, list[str]] = mutation.get("expected", {})
    decisions: dict[str, str] = mutation.get("decisions", {})
    rows = []
    for defect, row in sorted(mutation["matrix"].items()):
        cells = {
            check: ("detected" if ok else "-")
            + (" (expected)" if check in expected.get(defect, []) else "")
            for check, ok in sorted(row.items())
        }
        rows.append({"defect": defect, **cells, "decision": decisions.get(defect, "n/a")})
    return rows
