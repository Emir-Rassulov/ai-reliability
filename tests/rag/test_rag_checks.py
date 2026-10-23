"""RAG checks against hand-calculated values and annotated cases."""

from __future__ import annotations

from typing import Any

import pytest

from reliability.checks.base import CheckContext
from reliability.models import RetrievedRef, Status
from reliability.rag.citation_validation import CitationCheck
from reliability.rag.context_quality import ContextQualityCheck
from reliability.rag.grounding import GroundingCheck, unverified_ratio
from reliability.rag.retrieval_metrics import RetrievalMetricsCheck, compute_metrics, dedupe
from tests.conftest import make_scenario, resp


def refs(*pairs: tuple[str, int]) -> list[RetrievedRef]:
    return [RetrievedRef(doc_id=d, rank=r) for d, r in pairs]


# ----------------------------------------------------------------------------- retrieval metrics


def test_metrics_hand_calculated_basic() -> None:
    # relevant: a, c ; retrieved a(1) b(2) c(3) d(4) ; K=3
    m = compute_metrics(refs(("a", 1), ("b", 2), ("c", 3), ("d", 4)), ["a", "c"], 3)
    assert (m.precision_at_k, m.recall_at_k, m.mrr) == (round(2 / 3, 4), 1.0, 1.0)


def test_metrics_first_relevant_at_rank_three() -> None:
    m = compute_metrics(refs(("x", 1), ("y", 2), ("a", 3)), ["a", "b"], 2)
    # top-2 has no relevant doc; MRR still sees rank 3
    assert (m.precision_at_k, m.recall_at_k, m.mrr) == (0.0, 0.0, round(1 / 3, 4))


def test_metrics_duplicates_count_once_at_best_rank() -> None:
    m = compute_metrics(refs(("a", 2), ("a", 1), ("b", 3), ("a", 4)), ["a"], 3)
    assert m.duplicate_count == 2
    assert (m.precision_at_k, m.recall_at_k, m.mrr) == (round(1 / 3, 4), 1.0, 1.0)


def test_metrics_k_larger_than_result_list() -> None:
    m = compute_metrics(refs(("a", 1), ("b", 2)), ["a", "b", "c"], 10)
    assert (m.precision_at_k, m.recall_at_k) == (0.2, round(2 / 3, 4))


def test_metrics_empty_retrieval() -> None:
    m = compute_metrics([], ["a"], 3)
    assert m.empty_retrieval and (m.precision_at_k, m.recall_at_k, m.mrr) == (0.0, 0.0, 0.0)


def test_metrics_rank_gaps_are_kept() -> None:
    m = compute_metrics(refs(("x", 1), ("a", 5)), ["a"], 3)
    assert (m.recall_at_k, m.mrr) == (0.0, 0.2)


def test_dedupe_tie_break_by_id() -> None:
    unique, dup = dedupe(refs(("b", 1), ("a", 1)))
    assert [r.doc_id for r in unique] == ["a", "b"] and dup == 0


def _retrieval(ctx: CheckContext, **kw: Any) -> Any:
    scenario = make_scenario(category="rag", checks=["retrieval_metrics"], **kw)
    return RetrievalMetricsCheck(ctx).run(scenario, resp("x"))


def test_retrieval_check_no_labels_not_applicable(ctx: CheckContext) -> None:
    assert (
        _retrieval(ctx, retrieved=[{"doc_id": "policy-2025", "rank": 1}]).status
        is Status.NOT_APPLICABLE
    )
    assert _retrieval(ctx, relevant_doc_ids=[]).status is Status.NOT_APPLICABLE


def test_retrieval_check_thresholds_from_scenario(ctx: CheckContext) -> None:
    kw = {
        "retrieved": [{"doc_id": "recipe-pasta", "rank": 1}, {"doc_id": "policy-2025", "rank": 2}],
        "relevant_doc_ids": ["policy-2025"],
    }
    assert _retrieval(ctx, **kw).status is Status.PASS  # mrr 0.5 >= default 0.5
    strict = _retrieval(ctx, retrieval_thresholds={"min_mrr": 0.9}, **kw)
    assert strict.status is Status.FAIL
    assert strict.evidence == [{"metric": "mrr", "value": 0.5, "threshold": 0.9}]
    k1 = _retrieval(ctx, retrieval_thresholds={"k": 1}, **kw)
    assert k1.status is Status.FAIL and k1.metrics["recall_at_k"] == 0.0


def test_retrieval_check_flags(ctx: CheckContext) -> None:
    empty = _retrieval(ctx, relevant_doc_ids=["policy-2025"])
    assert empty.status is Status.FAIL and empty.flags == ["empty_retrieval"]
    dup = _retrieval(
        ctx,
        retrieved=[{"doc_id": "policy-2025", "rank": 1}, {"doc_id": "policy-2025", "rank": 2}],
        relevant_doc_ids=["policy-2025"],
    )
    assert dup.status is Status.PASS and dup.flags == ["duplicates"]


# ----------------------------------------------------------------------------- context quality


def _context(ctx: CheckContext, **kw: Any) -> Any:
    scenario = make_scenario(category="rag", checks=["context_quality"], **kw)
    return ContextQualityCheck(ctx).run(scenario, resp("x"))


def test_context_precision_and_recall(ctx: CheckContext) -> None:
    result = _context(
        ctx,
        retrieved=[{"doc_id": "policy-2025", "rank": 1}, {"doc_id": "recipe-pasta", "rank": 2}],
        relevant_doc_ids=["policy-2025"],
        required_evidence_ids=["policy-2025", "returns-faq"],
    )
    assert result.metrics["context_precision"] == 0.5
    assert result.metrics["context_recall"] == 0.5
    assert result.status is Status.FAIL  # recall below default 1.0


def test_context_outdated_below_current_is_flag_only(ctx: CheckContext) -> None:
    result = _context(
        ctx, retrieved=[{"doc_id": "policy-2025", "rank": 1}, {"doc_id": "policy-2023", "rank": 2}]
    )
    assert result.status is Status.PASS
    assert result.flags == ["outdated"]
    assert result.evidence[0]["superseded_by"] == "policy-2025"


def test_context_outdated_above_current_fails(ctx: CheckContext) -> None:
    result = _context(
        ctx, retrieved=[{"doc_id": "policy-2023", "rank": 1}, {"doc_id": "policy-2025", "rank": 2}]
    )
    assert result.status is Status.FAIL
    assert "outdated_ranked_above_current" in result.flags


def test_context_unauthorized_and_conflicts(ctx: CheckContext) -> None:
    result = _context(
        ctx,
        retrieved=[
            {"doc_id": "warranty-main", "rank": 1},
            {"doc_id": "warranty-promo", "rank": 2},
            {"doc_id": "salary-bands", "rank": 3},
        ],
        conflicts=[{"doc_ids": ["warranty-main", "warranty-promo"]}],
        user_access=["public"],
    )
    assert result.status is Status.FAIL
    assert set(result.flags) == {"conflicting", "unauthorized"}
    allowed = _context(
        ctx,
        retrieved=[{"doc_id": "salary-bands", "rank": 1}],
        user_access=["public", "confidential"],
    )
    assert allowed.status is Status.PASS


def test_context_relevant_not_supporting(ctx: CheckContext) -> None:
    result = _context(
        ctx,
        retrieved=[{"doc_id": "returns-faq", "rank": 1}, {"doc_id": "policy-2025", "rank": 2}],
        relevant_doc_ids=["returns-faq", "policy-2025"],
        expected_claims=[{"id": "w", "phrases": ["30 days"], "supported_by": ["policy-2025"]}],
    )
    assert result.status is Status.PASS
    assert result.flags == ["relevant_not_supporting"]


def test_context_empty_with_labels_and_nothing(ctx: CheckContext) -> None:
    assert _context(ctx).status is Status.NOT_APPLICABLE
    empty = _context(ctx, relevant_doc_ids=["policy-2025"])
    assert empty.metrics["context_precision"] == 0.0 and empty.status is Status.PASS
    custom = _context(
        ctx,
        retrieved=[{"doc_id": "policy-2025", "rank": 1}],
        required_evidence_ids=["policy-2025", "returns-faq"],
        retrieval_thresholds={"min_context_recall": 0.5},
    )
    assert custom.status is Status.PASS


def test_context_valid_until_tiebreak(ctx: CheckContext, documents: Any) -> None:
    from reliability.models import Document

    docs = dict(documents)
    docs["v-old"] = Document(
        doc_id="v-old", title="t", text="x", topic="same", version=1, valid_until="2024-01-01"
    )
    docs["v-new"] = Document(doc_id="v-new", title="t", text="x", topic="same", version=1)
    local = CheckContext(config=ctx.config, documents=docs)
    scenario = make_scenario(
        checks=["context_quality"],
        retrieved=[{"doc_id": "v-old", "rank": 1}, {"doc_id": "v-new", "rank": 2}],
    )
    result = ContextQualityCheck(local).run(scenario, resp("x"))
    assert result.status is Status.FAIL and "outdated" in result.flags


# ----------------------------------------------------------------------------- grounding

WINDOW = [
    {
        "id": "window",
        "phrases": ["30 days"],
        "number": 30,
        "supported_by": ["policy-2025"],
        "mandatory": True,
    }
]
FALSE = [{"id": "old", "phrases": ["14 days"]}]


def _ground(ctx: CheckContext, text: str, *, live: bool = False, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "category": "rag",
        "checks": ["grounding"],
        "retrieved": [{"doc_id": "policy-2025", "rank": 1}],
        "expected_claims": WINDOW,
        "false_claims": FALSE,
    }
    base.update(kw)
    scenario = make_scenario(**base)
    response = resp(text).model_copy(update={"is_simulated": not live})
    return GroundingCheck(ctx).run(scenario, response)


@pytest.mark.parametrize(
    ("text", "status", "kind"),
    [
        ("You have 30 days [policy-2025].", Status.PASS, "supported"),
        ("You have 14 days.", Status.FAIL, "false_claim"),
        ("It is not 30 days.", Status.FAIL, "contradicted"),
        ("Contact support.", Status.FAIL, "omitted"),
        ("Maybe 30 days.", Status.INCONCLUSIVE, "hedged"),
        ("30 days. Perhaps 14 days.", Status.INCONCLUSIVE, "hedged_false_claim"),
    ],
)
def test_grounding_verdicts(ctx: CheckContext, text: str, status: Status, kind: str) -> None:
    result = _ground(ctx, text)
    assert result.status is status
    assert kind in {e.get("result") for e in result.evidence}


def test_grounding_unsupported_when_support_not_retrieved(ctx: CheckContext) -> None:
    result = _ground(ctx, "You have 30 days.", retrieved=[{"doc_id": "returns-faq", "rank": 1}])
    assert result.status is Status.FAIL
    assert result.evidence[0]["result"] == "unsupported"
    assert result.mandatory_violation is False


def test_grounding_mandatory_violation(ctx: CheckContext) -> None:
    assert _ground(ctx, "Contact support.").mandatory_violation is True


def test_grounding_unattributed_claim_is_inconclusive(ctx: CheckContext) -> None:
    claims = [{"id": "w", "phrases": ["30 days"]}]
    assert _ground(ctx, "30 days.", expected_claims=claims).status is Status.INCONCLUSIVE


def test_grounding_unverified_ratio_mock_vs_live(ctx: CheckContext) -> None:
    text = "You have 30 days [policy-2025]. Our shop was founded in 1999."
    mock = _ground(ctx, text)
    assert mock.status is Status.PASS and mock.metrics["unverified_content_ratio"] == 0.5
    live = _ground(ctx, text, live=True)
    assert live.status is Status.INCONCLUSIVE
    tolerant = _ground(ctx, text, live=True, max_unverified_ratio=0.5)
    assert tolerant.status is Status.PASS


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("The provided documents do not contain this information.", Status.PASS),
        ("You have 14 days.", Status.FAIL),
        ("You have 30 days.", Status.FAIL),  # invented an answer
        ("I can't help with that.", Status.INCONCLUSIVE),
    ],
)
def test_grounding_abstain(ctx: CheckContext, text: str, status: Status) -> None:
    assert _ground(ctx, text, expected_behavior="abstain", expected_claims=[]).status is status


def test_grounding_not_applicable(ctx: CheckContext) -> None:
    assert _ground(ctx, "x", expected_behavior="refuse").status is Status.NOT_APPLICABLE
    assert _ground(ctx, "x", expected_claims=[], false_claims=[]).status is Status.NOT_APPLICABLE


def test_unverified_ratio_counts_sentences() -> None:
    assert unverified_ratio("", ["a"]) == 0.0
    assert unverified_ratio("Alpha. Beta. The documents do not mention gamma.", ["alpha"]) == round(
        1 / 3, 4
    )


# ----------------------------------------------------------------------------- citations


def _cite(ctx: CheckContext, text: str, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "category": "rag",
        "checks": ["citations"],
        "citations_required": True,
        "retrieved": [{"doc_id": "policy-2025", "rank": 1}, {"doc_id": "returns-faq", "rank": 2}],
        "expected_claims": WINDOW,
    }
    base.update(kw)
    return CitationCheck(ctx).run(make_scenario(**base), resp(text))


@pytest.mark.parametrize(
    ("text", "status", "problems"),
    [
        ("You have 30 days [policy-2025].", Status.PASS, set()),
        ("You have 30 days [POLICY-2025].", Status.PASS, set()),
        ("You have 30 days. [policy-2025]", Status.PASS, set()),  # citation-only next sentence
        ("You have 30 days [policy-2099].", Status.FAIL, {"fabricated", "misattributed"}),
        ("You have 30 days [policy-2023].", Status.FAIL, {"not_retrieved", "misattributed"}),
        ("You have 30 days [returns-faq].", Status.FAIL, {"misattributed"}),
        ("You have 30 days. Also see [policy-2025].", Status.FAIL, {"uncited_claim"}),
        ("You have 30 days.", Status.FAIL, {"uncited_claim", "no_citations"}),
    ],
)
def test_citations(ctx: CheckContext, text: str, status: Status, problems: set[str]) -> None:
    result = _cite(ctx, text)
    assert result.status is status
    assert {e["problem"] for e in result.evidence} == problems


def test_citation_accuracy_metric(ctx: CheckContext) -> None:
    result = _cite(ctx, "30 days [policy-2025] and [nope-1].")
    assert result.metrics["citation_accuracy"] == 0.5


def test_citations_not_applicable_and_abstain(ctx: CheckContext) -> None:
    assert _cite(ctx, "x", citations_required=False).status is Status.NOT_APPLICABLE
    assert _cite(ctx, "x", expected_behavior="refuse").status is Status.NOT_APPLICABLE
    abstain = _cite(
        ctx, "The documents do not say.", expected_behavior="abstain", expected_claims=[]
    )
    assert abstain.status is Status.PASS


def test_citations_claim_without_support_annotation_is_skipped(ctx: CheckContext) -> None:
    claims = [{"id": "w", "phrases": ["30 days"]}]
    assert _cite(ctx, "30 days [returns-faq].", expected_claims=claims).status is Status.PASS
