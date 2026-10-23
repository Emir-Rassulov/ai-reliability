"""Lexical matching primitives: boundaries, sentences, negation scope, number conflicts."""

from __future__ import annotations

import pytest

from reliability.checks.text import (
    clause_before,
    infer_unit,
    match_phrases,
    numbers_with_unit,
    sentence_spans,
)


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("You have 30 days.", "present"),
        ("You have 30  DAYS.", "present"),  # case and whitespace tolerant
        ("You have 130 days.", "missing"),  # word boundary: false-positive trap
        ("You have 2.30 days.", "missing"),  # decimal boundary
        ("You have 30 dayss.", "missing"),
        ("It is not 30 days.", "contradicted"),
        ("It isn't 30 days.", "contradicted"),
        ("It is never 30 days.", "contradicted"),
        ("No, it is 30 days.", "present"),  # comma ends the negation scope
        ("It is probably 30 days.", "hedged"),
        ("", "missing"),
    ],
)
def test_match_phrase_status(text: str, status: str) -> None:
    assert match_phrases(text, ["30 days"]).status == status


def test_negation_outside_window_is_ignored() -> None:
    text = "I do not know much about this, the window is 30 days."
    assert match_phrases(text, ["30 days"], window=3).status == "present"


def test_negation_in_previous_sentence_is_ignored() -> None:
    assert match_phrases("That is not right. It is 30 days.", ["30 days"]).status == "present"


def test_any_contradicting_occurrence_wins() -> None:
    text = "It is 30 days. Actually it is not 30 days."
    assert match_phrases(text, ["30 days"]).status == "contradicted"


def test_number_conflict_same_sentence() -> None:
    found = match_phrases("Return within 30 days, sorry, 14 days.", ["30 days"], number=30)
    assert found.status == "contradicted"
    assert "14" in found.reason


def test_negated_conflicting_number_is_not_a_conflict() -> None:
    text = "Not 14 days as before — now you have 30 days."
    assert match_phrases(text, ["30 days"], number=30).status == "present"


def test_number_in_other_sentence_is_not_a_conflict() -> None:
    text = "You have 30 days. Archive after 130 days."
    assert match_phrases(text, ["30 days"], number=30).status == "present"


def test_tolerance_allows_close_numbers() -> None:
    text = "Delivery takes 5 business days, at most 6 business days."
    assert match_phrases(text, ["5 business days"], number=5, tolerance=1).status == "present"
    assert match_phrases(text, ["5 business days"], number=5).status == "contradicted"


def test_missing_reports_other_values_with_unit() -> None:
    found = match_phrases("You have 14 days.", ["30 days"], number=30)
    assert found.status == "missing"
    assert "14" in found.reason


def test_infer_unit() -> None:
    assert infer_unit(["thirty days", "30 days"]) == "days"
    assert infer_unit(["two years"]) is None


def test_numbers_with_unit_singular_and_plural() -> None:
    values = [v for v, _, _ in numbers_with_unit("1 day or 3 days", "days", 3)]
    assert values == [1.0, 3.0]


def test_sentence_spans_do_not_split_on_single_newline() -> None:
    text = "The window is not\n30 days. Next one!\n\nNew paragraph\n- item one\n- item two"
    pieces = [text[a:b] for a, b in sentence_spans(text)]
    assert pieces == [
        "The window is not\n30 days.",
        "Next one!",
        "New paragraph",
        "- item one",
        "- item two",
    ]


def test_clause_before_cuts_at_punctuation() -> None:
    assert clause_before("if it does not stop, use kill", 0, 29) == " use kill"
    assert clause_before("no break here", 0, 13) == "no break here"


def test_typographic_apostrophe_negation() -> None:
    text = "It isn" + chr(0x2019) + "t 30 days."
    assert match_phrases(text, ["30 days"]).status == "contradicted"
