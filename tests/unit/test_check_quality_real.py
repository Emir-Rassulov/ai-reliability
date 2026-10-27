"""Agent quality on REAL replies (llama3.2:1b behind the demo bot), labeled by hand.

These numbers were measured on the same replies that motivated the latest fixes, so they are
optimistic; a fresh run is the honest test (README: "Measured quality on real replies").
"""

from __future__ import annotations

import pytest

from reliability import PROJECT_ROOT
from reliability.checks.base import CheckContext
from reliability.evaluation.check_quality import CheckQuality, evaluate_labeled
from reliability.models import Status

REAL = PROJECT_ROOT / "tests" / "fixtures" / "labeled_real_demo_bot.json"


@pytest.fixture(scope="module")
def results(ctx: CheckContext) -> dict[str, CheckQuality]:
    return evaluate_labeled(REAL, ctx)


def cases(results: dict[str, CheckQuality]) -> list:  # type: ignore[type-arg]
    return [c for q in results.values() for c in q.cases]


def test_no_unexpected_disagreement(results: dict[str, CheckQuality]) -> None:
    assert [
        c.case_id for c in cases(results) if c.truth is not c.observed and not c.known_miss
    ] == []


def test_known_misses_still_miss(results: dict[str, CheckQuality]) -> None:
    assert sorted(c.case_id for c in cases(results) if c.known_miss) == [
        "b3-nb-010-prompt_injection",
        "v2-nb-009-prompt_injection",
        "v3-nb-011-refusal",
    ]
    assert [c.case_id for c in cases(results) if c.known_miss and c.truth is c.observed] == []


def test_overall_numbers_hand_checked(results: dict[str, CheckQuality]) -> None:
    all_cases = cases(results)
    tp = sum(q.tp for q in results.values())
    fp = sum(q.fp for q in results.values())
    fn = sum(q.fn for q in results.values())
    assert (len(all_cases), tp, fp, fn) == (104, 33, 0, 3)
    assert sum(1 for c in all_cases if c.truth is c.observed) == 101


def test_over_refusal_is_not_reported_as_injection(results: dict[str, CheckQuality]) -> None:
    for run in ("v1", "v2", "v3"):  # b3 summarised the note correctly
        pi = next(
            c
            for c in results["prompt_injection"].cases
            if c.case_id == f"{run}-nb-010-prompt_injection"
        )
        rf = next(
            c
            for c in results["required_facts"].cases
            if c.case_id == f"{run}-nb-010-required_facts"
        )
        assert pi.observed is Status.PASS and rf.observed is Status.FAIL
