"""Switchable injected defects. All are off by default.

Each defect breaks exactly one stage of the pipeline, so the evaluator can be shown to
catch real bugs in a real system rather than only label prepared fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFECTS: dict[str, str] = {
    "no_access_filter": "the access filter is skipped",
    "stale_first": "superseded document versions are kept and ranked before current ones",
    "no_grounding_instruction": "the system prompt no longer restricts answers to the documents",
    "truncated_context": "only the first N characters of the context reach the generator",
    "no_untrusted_marking": "retrieved documents are inserted as instructions, not marked as data",
}


def validate_defects(names: Iterable[str]) -> frozenset[str]:
    chosen = frozenset(names)
    unknown = sorted(chosen - DEFECTS.keys())
    if unknown:
        raise ValueError(f"unknown defect(s): {', '.join(unknown)}; known: {', '.join(DEFECTS)}")
    return chosen
