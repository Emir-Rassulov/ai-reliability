"""Evaluator quality on REAL replies of llama3.2 (3B, via LiteLLM/Ollama) to the main gate
scenarios, labeled blind. The run's own verdicts (before the fixes) are recorded in README:
precision 0.56. These replay numbers come after fixes made on the same replies (optimistic)."""

from __future__ import annotations

import pytest

from reliability import PROJECT_ROOT
from reliability.checks.base import CheckContext
from reliability.evaluation.check_quality import CheckQuality, evaluate_labeled

REAL = PROJECT_ROOT / "tests" / "fixtures" / "labeled_real_main_llama.json"


@pytest.fixture(scope="module")
def results(ctx: CheckContext) -> dict[str, CheckQuality]:
    return evaluate_labeled(REAL, ctx)


def test_only_known_misses_disagree(results: dict[str, CheckQuality]) -> None:
    cases = [c for q in results.values() for c in q.cases]
    assert [c.case_id for c in cases if c.truth is not c.observed and not c.known_miss] == []
    assert sorted(c.case_id for c in cases if c.known_miss) == [
        "l3-llm-001-required_facts",
        "l3-rag-004-grounding",
    ]
    assert all(c.truth is not c.observed for c in cases if c.known_miss)


def test_numbers_hand_checked(results: dict[str, CheckQuality]) -> None:
    cases = [c for q in results.values() for c in q.cases]
    tp = sum(q.tp for q in results.values())
    fp = sum(q.fp for q in results.values())
    fn = sum(q.fn for q in results.values())
    assert (len(cases), tp, fp, fn) == (58, 12, 2, 0)


def test_real_citation_forms_are_accepted(results: dict[str, CheckQuality]) -> None:
    cit = {c.case_id: c for c in results["citations"].cases}
    for sid in ("rag-001", "rag-004", "rag-008", "rag-009", "rag-013"):
        assert cit[f"l3-{sid}-citations"].observed.value == "PASS", sid
    assert (
        cit["l3-rag-011-citations"].observed.value == "FAIL"
    )  # misattributed, for the right reason
