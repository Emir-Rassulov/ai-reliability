"""Metamorphic relations for testing the evaluator itself (§13a, deterministic part).

A metamorphic relation states how a verdict must (not) change when the input is
transformed, which lets us test the evaluator without a single "correct" expected output:

M1  Case, extra whitespace and line breaks do not change any verdict.
M2  Reordering the retrieved documents does not change grounding or citation verdicts;
    moving a relevant document to the top never lowers precision, recall or MRR.
M3  Appending a neutral sentence to a passing response never produces FAIL.
M4  Inserting a negation directly before a matched fact never keeps it `present`.
M5  Replacing a canary/secret with its base64 form is still detected as leakage.

Documented exclusions: `structured_output` is excluded from M1 and M3 because case, line
breaks and trailing prose change the meaning of a JSON document. Scenarios tagged
`case-sensitive-payload` (e.g. a base64 secret in the response) are excluded from the
case transformations of M1, because base64 is case-sensitive by definition.
"""

from __future__ import annotations

import base64
from collections.abc import Callable

from reliability.evaluation.evaluator import Evaluator
from reliability.evaluation.outcome import CallOutcome
from reliability.models import RetrievedRef, Scenario, Status
from reliability.providers.base import ProviderResponse

NEUTRAL_SENTENCE = " Thank you for your patience."
TEXT_SHAPE_SENSITIVE_CHECKS = frozenset({"structured_output", "provider_call"})
CASE_SENSITIVE_TAG = "case-sensitive-payload"


def _wrap_every(n: int) -> Callable[[str], str]:
    def wrap(text: str) -> str:
        out: list[str] = []
        count = 0
        for ch in text:
            if ch == " ":
                count += 1
                if count % n == 0:
                    out.append("\n")
                    continue
            out.append(ch)
        return "".join(out)

    return wrap


TEXT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "upper": str.upper,
    "lower": str.lower,
    "double_spaces": lambda t: t.replace(" ", "  "),
    "line_wrap": _wrap_every(5),
    "outer_whitespace": lambda t: "\n  " + t + " \t\n",
}
CASE_TRANSFORMS = frozenset({"upper", "lower"})


def verdicts(evaluator: Evaluator, scenario: Scenario, text: str) -> dict[str, Status]:
    response = ProviderResponse(text=text, provider="mock", is_simulated=True)
    result = evaluator.evaluate(scenario, CallOutcome(response, None, 1))
    return {c.check: c.status for c in result.checks}


def comparable(scenario: Scenario, transform: str) -> bool:
    return not (transform in CASE_TRANSFORMS and CASE_SENSITIVE_TAG in scenario.tags)


def insert_negation(text: str, offset: int) -> str:
    return f"{text[:offset]}not {text[offset:]}"


def reverse_ranks(scenario: Scenario) -> Scenario:
    if not scenario.retrieved:
        return scenario
    top = max(r.rank for r in scenario.retrieved)
    reordered = [RetrievedRef(doc_id=r.doc_id, rank=top + 1 - r.rank) for r in scenario.retrieved]
    return scenario.model_copy(update={"retrieved": reordered})


def promote(scenario: Scenario, doc_id: str) -> Scenario:
    """Move `doc_id` to rank 1 and shift every document ranked above it down by one."""
    ranks = [r.rank for r in scenario.retrieved if r.doc_id == doc_id]
    if not ranks:
        return scenario
    best = min(ranks)
    moved = [
        RetrievedRef(
            doc_id=r.doc_id,
            rank=1 if r.doc_id == doc_id else (r.rank + 1 if r.rank < best else r.rank),
        )
        for r in scenario.retrieved
    ]
    return scenario.model_copy(update={"retrieved": moved})


def to_base64(text: str, value: str) -> str:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return text.replace(value, encoded)
