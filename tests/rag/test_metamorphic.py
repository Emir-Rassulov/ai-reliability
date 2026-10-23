"""Evaluator metamorphic relations (§13a) over the full starter dataset.

A violation is an evaluator defect: the test names the scenario, transformation and check.
"""

from __future__ import annotations

import pytest

from reliability import PROJECT_ROOT
from reliability.checks.base import CheckContext
from reliability.checks.provider_call import ProviderCallCheck
from reliability.checks.registry import build_checks
from reliability.checks.text import match_phrases
from reliability.config import load_config
from reliability.evaluation.evaluator import Evaluator
from reliability.evaluation.metamorphic import (
    NEUTRAL_SENTENCE,
    TEXT_SHAPE_SENSITIVE_CHECKS,
    TEXT_TRANSFORMS,
    comparable,
    insert_negation,
    promote,
    reverse_ranks,
    verdicts,
)
from reliability.models import Scenario, Status
from reliability.providers.mock_provider import load_mock_responses
from reliability.rag.retrieval_metrics import compute_metrics
from reliability.reporting.redaction import Redactor
from reliability.scenarios import load_documents, load_scenarios

CONFIG = load_config(env={})
DOCS = load_documents(PROJECT_ROOT / "datasets" / "synthetic_documents.json", CONFIG.limits)
SCENARIOS = load_scenarios(PROJECT_ROOT / "scenarios", DOCS, CONFIG.limits).scenarios
MOCK = load_mock_responses(PROJECT_ROOT / "scenarios" / "mock_responses.json")
EVALUATOR = Evaluator(
    build_checks(CheckContext(config=CONFIG, documents=DOCS)), ProviderCallCheck(CONFIG), Redactor()
)

#: Scenarios whose canned response is plain text (error and echo entries have none).
TEXT_CASES = [
    (s, MOCK[s.id].text) for s in SCENARIOS if MOCK[s.id].text and MOCK[s.id].error is None
]
IDS = [s.id for s, _ in TEXT_CASES]


def _shape_insensitive(v: dict[str, Status]) -> dict[str, Status]:
    return {k: s for k, s in v.items() if k not in TEXT_SHAPE_SENSITIVE_CHECKS}


def test_dataset_has_enough_text_cases() -> None:
    assert len(TEXT_CASES) >= 40


#: Documented exclusion (see evaluation/metamorphic.py): case transforms are not applied to
#: case-sensitive payloads such as base64. The pairs are filtered here, not skipped at runtime.
M1_CASES = [
    (s, t, name) for s, t in TEXT_CASES for name in sorted(TEXT_TRANSFORMS) if comparable(s, name)
]


@pytest.mark.parametrize(
    ("scenario", "text", "transform"), M1_CASES, ids=[f"{s.id}-{n}" for s, _, n in M1_CASES]
)
def test_m1_case_and_whitespace_do_not_change_verdicts(
    scenario: Scenario, text: str, transform: str
) -> None:
    before = _shape_insensitive(verdicts(EVALUATOR, scenario, text))
    after = _shape_insensitive(verdicts(EVALUATOR, scenario, TEXT_TRANSFORMS[transform](text)))
    assert after == before


def test_m1_exclusions_are_few_and_explicit() -> None:
    excluded = len(TEXT_CASES) * len(TEXT_TRANSFORMS) - len(M1_CASES)
    assert excluded == 2  # sec-010 upper/lower: base64 secret in the response


RAG_CASES = [(s, t) for s, t in TEXT_CASES if len({r.doc_id for r in s.retrieved}) >= 2]


@pytest.mark.parametrize(("scenario", "text"), RAG_CASES, ids=[s.id for s, _ in RAG_CASES])
def test_m2_reordering_keeps_grounding_and_citations(scenario: Scenario, text: str) -> None:
    keep = ("grounding", "citations")
    before = {k: v for k, v in verdicts(EVALUATOR, scenario, text).items() if k in keep}
    after = {
        k: v for k, v in verdicts(EVALUATOR, reverse_ranks(scenario), text).items() if k in keep
    }
    assert after == before


LABELLED = [s for s in SCENARIOS if s.relevant_doc_ids and s.retrieved]


@pytest.mark.parametrize("scenario", LABELLED, ids=[s.id for s in LABELLED])
def test_m2_promoting_relevant_doc_never_lowers_rank_metrics(scenario: Scenario) -> None:
    assert scenario.relevant_doc_ids is not None
    k = CONFIG.retrieval.k
    before = compute_metrics(scenario.retrieved, scenario.relevant_doc_ids, k)
    for doc_id in scenario.relevant_doc_ids:
        moved = promote(scenario, doc_id)
        after = compute_metrics(moved.retrieved, scenario.relevant_doc_ids, k)
        assert after.mrr >= before.mrr
        assert after.precision_at_k >= before.precision_at_k
        assert after.recall_at_k >= before.recall_at_k


@pytest.mark.parametrize(("scenario", "text"), TEXT_CASES, ids=IDS)
def test_m3_neutral_sentence_never_turns_pass_into_fail(scenario: Scenario, text: str) -> None:
    before = _shape_insensitive(verdicts(EVALUATOR, scenario, text))
    after = _shape_insensitive(verdicts(EVALUATOR, scenario, text + NEUTRAL_SENTENCE))
    for check, status in before.items():
        if status is Status.PASS:
            assert after[check] is not Status.FAIL, check


FACT_CASES = [
    (s, t, fact)
    for s, t in TEXT_CASES
    for fact in [*s.required_facts, *s.expected_claims]
    if match_phrases(t, fact.phrases, number=fact.number, unit=fact.unit).status == "present"
]


@pytest.mark.parametrize(
    ("scenario", "text", "fact"), FACT_CASES, ids=[f"{s.id}:{f.id}" for s, _, f in FACT_CASES]
)
def test_m4_negation_before_fact_is_never_present(
    scenario: Scenario, text: str, fact: object
) -> None:
    from reliability.models import Fact

    assert isinstance(fact, Fact)
    found = match_phrases(text, fact.phrases, number=fact.number, unit=fact.unit)
    assert found.offset is not None
    negated = insert_negation(text, found.offset)
    assert (
        match_phrases(negated, fact.phrases, number=fact.number, unit=fact.unit).status != "present"
    )


def test_fact_cases_exist() -> None:
    assert len(FACT_CASES) >= 15
