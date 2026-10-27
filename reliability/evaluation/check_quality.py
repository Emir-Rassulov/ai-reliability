"""Testing the tester: precision and recall of each check on a labeled set.

Each labeled case has a `truth` (what a perfect evaluator would say). A "positive" is a
true failure (truth FAIL); a detection is an observed FAIL.

* precision = observed FAIL that are truly FAIL / all observed FAIL
* recall    = observed FAIL that are truly FAIL / all truly FAIL
* agreement = observed status equals truth, over all cases (including INCONCLUSIVE/NA)

`known_miss` marks a documented limitation: the check is expected to disagree with truth.
Known misses are counted in the metrics, never hidden.

A labeled set may name a scenario suite (`scenarios_path`, relative to the project root); a
case with `"scenario": "<id>"` is then judged against that real scenario's annotations, so
labels on real application replies use exactly the configuration the run used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reliability import PROJECT_ROOT
from reliability.checks.base import CheckContext
from reliability.checks.registry import build_checks
from reliability.models import Scenario, Status
from reliability.providers.base import ProviderResponse
from reliability.scenarios import load_scenarios

BASE: dict[str, Any] = {
    "id": "labeled",
    "name": "labeled case",
    "category": "llm",
    "severity": "high",
    "input": {"messages": [{"role": "user", "content": "question"}]},
}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    check: str
    kind: str
    truth: Status
    observed: Status
    known_miss: bool
    note: str


@dataclass
class CheckQuality:
    check: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return sum(1 for c in self.cases if c.truth is Status.FAIL and c.observed is Status.FAIL)

    @property
    def fp(self) -> int:
        return sum(
            1 for c in self.cases if c.truth is not Status.FAIL and c.observed is Status.FAIL
        )

    @property
    def fn(self) -> int:
        return sum(
            1 for c in self.cases if c.truth is Status.FAIL and c.observed is not Status.FAIL
        )

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return round(self.tp / denom, 3) if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return round(self.tp / denom, 3) if denom else None

    @property
    def agreement(self) -> float:
        return round(sum(1 for c in self.cases if c.truth is c.observed) / len(self.cases), 3)


def _scenario(base: dict[str, Any], check: str, overrides: dict[str, Any]) -> Scenario:
    data = {**BASE, **base, **overrides, "checks": [check]}
    return Scenario.model_validate(data)


def evaluate_labeled(path: Path, ctx: CheckContext) -> dict[str, CheckQuality]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    checks = build_checks(ctx)
    suite: dict[str, Scenario] = {}
    if "scenarios_path" in raw:
        loaded = load_scenarios(
            PROJECT_ROOT / raw["scenarios_path"], ctx.documents, ctx.config.limits
        )
        suite = {s.id: s for s in loaded.scenarios}
    out: dict[str, CheckQuality] = {}
    for case in raw["cases"]:
        name = case["check"]
        if "scenario" in case:
            scenario = suite[case["scenario"]].model_copy(update={"checks": [name]})
        else:
            scenario = _scenario(
                raw.get("bases", {}).get(name, {}), name, case.get("overrides", {})
            )
        response = ProviderResponse(text=case["response"], provider="labeled", is_simulated=True)
        observed = checks[name].run(scenario, response).status
        out.setdefault(name, CheckQuality(name)).cases.append(
            CaseResult(
                case_id=case["id"],
                check=name,
                kind=case["kind"],
                truth=Status(case["truth"]),
                observed=observed,
                known_miss=bool(case.get("known_miss", False)),
                note=case.get("note", ""),
            )
        )
    return out


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_quality(results: dict[str, CheckQuality]) -> str:
    header = (
        f"{'check':<22} {'cases':>5} {'TP':>3} {'FP':>3} {'FN':>3} "
        f"{'precision':>9} {'recall':>6} {'agree':>6}"
    )
    lines = [header, "-" * len(header)]
    for name in sorted(results):
        q = results[name]
        lines.append(
            f"{name:<22} {len(q.cases):>5} {q.tp:>3} {q.fp:>3} {q.fn:>3} "
            f"{_fmt(q.precision):>9} {_fmt(q.recall):>6} {_fmt(q.agreement):>6}"
        )
    misses = [c for q in results.values() for c in q.cases if c.truth is not c.observed]
    lines += ["", "Disagreements with truth:"]
    lines += [
        f"  {c.case_id} ({c.check}, {c.kind}): truth {c.truth.value}, observed {c.observed.value}"
        + (f" - known limitation: {c.note}" if c.known_miss else " - UNEXPECTED")
        for c in misses
    ] or ["  none"]
    return "\n".join(lines) + "\n"
