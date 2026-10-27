"""Tolerant citation extraction (real llama3.2 output used `[doc-ID: x]` and `(doc-ID "x")`)."""

from __future__ import annotations

import re

from reliability.rag.citation_parsing import Citation, find_citations, strip_citations

PATTERN = re.compile(r"\[(?P<id>[A-Za-z0-9][A-Za-z0-9._:-]*)\]")
KNOWN = {"policy-2025", "shipping-standard", "warranty-main"}


def ids(text: str) -> list[str]:
    return [c.doc_id for c in find_citations(text, PATTERN, KNOWN)]


def test_configured_form() -> None:
    assert ids("30 days [policy-2025].") == ["policy-2025"]


def test_labelled_forms_seen_in_real_replies() -> None:
    assert ids("5 business days. [doc-ID: shipping-standard]") == ["shipping-standard"]
    assert ids('within 2 days. [doc-ID: "policy-2025"]') == ["policy-2025"]
    assert ids('2-year warranty (doc-ID "warranty-main").') == ["warranty-main"]
    assert ids("see doc_id=policy-2025") == ["policy-2025"]


def test_labelled_unknown_id_still_counts_so_it_is_reported_as_fabricated() -> None:
    assert ids("[doc-ID: made-up-doc]") == ["made-up-doc"]


def test_parentheses_count_only_for_known_ids() -> None:
    assert ids("within 30 days (policy-2025).") == ["policy-2025"]
    assert ids("a fee (optional) applies (see below)") == []


def test_no_double_counting_and_order() -> None:
    text = "[doc-ID: policy-2025] and [shipping-standard]"
    assert ids(text) == ["policy-2025", "shipping-standard"]


def test_strip() -> None:
    text = "Five days. [doc-ID: shipping-standard]"
    found = find_citations(text, PATTERN, KNOWN)
    assert strip_citations(text, found).strip() == "Five days."
    assert strip_citations("abc", [Citation("x", 0, 0)]) == "abc"
