"""Shared lexical matching: phrases, sentences, negation, hedging, number conflicts.

This is lexical verification, not semantic understanding. It is deliberately simple and
documented so that every verdict can be explained by pointing at characters in the text.

* Matching is case-insensitive and whitespace-tolerant (any run of whitespace matches).
* Phrases must sit on word boundaries: "30 days" does not match "130 days" or "2.30 days".
* Sentences end at ., ! or ? followed by whitespace, at blank lines, and before list items.
  A single line break inside a sentence does not end it.
* A match is negated when a negator ("not", "no", "never", "...n't", ...) occurs within the
  last `window` tokens before it in the same sentence; hedged likewise for hedge words.
* A fact with a number is contradicted when the same sentence states a different,
  non-negated number with the same unit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

_QUOTES = str.maketrans({0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"'})
_TOKEN = re.compile(r"[a-z0-9']+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n[ \t]*\n\s*|\n(?=[ \t]*(?:[-*•]|\d+[.)])\s)")

NEGATORS = frozenset(
    {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "nothing", "nobody"}
)
HEDGES = frozenset({"maybe", "might", "possibly", "perhaps", "probably", "likely", "unclear"})


def prep(text: str) -> str:
    """Length-preserving normalisation (typographic quotes -> ASCII), so offsets stay valid."""
    return text.translate(_QUOTES)


def collapse_ws(text: str) -> str:
    return " ".join(text.split())


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        if text[start : match.start()].strip():
            spans.append((start, match.start()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def sentence_of(spans: Sequence[tuple[int, int]], offset: int) -> tuple[int, int]:
    for start, end in spans:
        if start <= offset < end:
            return start, end
    return offset, offset


@lru_cache(maxsize=4096)
def phrase_pattern(phrase: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(word) for word in prep(phrase).split())
    return re.compile(rf"(?<!\w)(?<!\d[.,]){body}(?!\w)", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def is_negator(token: str) -> bool:
    return token in NEGATORS or token.endswith("n't")


_CLAUSE_BREAKS = (",", ";", ":", "(", ")", "—", " - ", " but ")


def clause_before(text: str, start: int, offset: int) -> str:
    """Text from the start of the clause containing `offset` (punctuation ends negation scope)."""
    segment = text[start:offset]
    lowered = segment.lower()
    cut = 0
    for brk in _CLAUSE_BREAKS:
        pos = lowered.rfind(brk)
        if pos >= 0:
            cut = max(cut, pos + len(brk))
    return segment[cut:]


def preceding_tokens(text: str, sentence_start: int, offset: int, window: int) -> list[str]:
    return tokens(clause_before(text, sentence_start, offset))[-window:]


def infer_unit(phrases: Sequence[str]) -> str | None:
    for phrase in phrases:
        match = re.search(r"\d+(?:\.\d+)?\s+([A-Za-z]+)", phrase)
        if match:
            return match.group(1).lower()
    return None


@lru_cache(maxsize=1024)
def _number_unit_pattern(unit: str) -> re.Pattern[str]:
    stem = unit[:-1] if unit.endswith("s") and len(unit) > 2 else unit
    return re.compile(
        rf"(?<![\w.,])(\d+(?:\.\d+)?)\s+{re.escape(stem)}s?(?!\w)",
        re.IGNORECASE,
    )


def numbers_with_unit(text: str, unit: str, window: int) -> list[tuple[float, int, bool]]:
    """(value, offset, negated) for every `<number> <unit>` in text."""
    found: list[tuple[float, int, bool]] = []
    for match in _number_unit_pattern(unit).finditer(text):
        before = preceding_tokens(text, 0, match.start(), window)
        found.append((float(match.group(1)), match.start(), any(is_negator(t) for t in before)))
    return found


FactStatus = Literal["present", "missing", "contradicted", "hedged"]


@dataclass(frozen=True)
class FactMatch:
    status: FactStatus
    phrase: str | None = None
    offset: int | None = None
    reason: str = ""


def match_phrases(
    text: str,
    phrases: Sequence[str],
    *,
    number: float | None = None,
    unit: str | None = None,
    tolerance: float = 0.0,
    window: int = 3,
) -> FactMatch:
    """Classify a fact against a response. Any contradicting occurrence wins (conservative)."""
    prepared = prep(text)
    spans = sentence_spans(prepared)
    unit = unit or (infer_unit(phrases) if number is not None else None)
    present: FactMatch | None = None
    hedged: FactMatch | None = None
    for phrase in phrases:
        for match in phrase_pattern(phrase).finditer(prepared):
            s_start, s_end = sentence_of(spans, match.start())
            before = preceding_tokens(prepared, s_start, match.start(), window)
            if any(is_negator(t) for t in before):
                return FactMatch("contradicted", phrase, match.start(), "negated")
            if any(t in HEDGES for t in before):
                hedged = hedged or FactMatch("hedged", phrase, match.start(), "hedged wording")
                continue
            if number is not None and unit:
                sentence = prepared[s_start:s_end]
                for value, _, negated in numbers_with_unit(sentence, unit, window):
                    if not negated and abs(value - number) > tolerance:
                        return FactMatch(
                            "contradicted",
                            phrase,
                            match.start(),
                            f"conflicting number in the same sentence: {value:g} {unit}",
                        )
            present = present or FactMatch("present", phrase, match.start())
    if present is not None:
        return present
    if hedged is not None:
        return hedged
    reason = ""
    if number is not None and unit:
        others = [v for v, _, neg in numbers_with_unit(prepared, unit, window) if not neg]
        if others:
            reason = "other values with the same unit: " + ", ".join(f"{v:g}" for v in others)
    return FactMatch("missing", reason=reason)


# --------------------------------------------------------------------------- system prompt

SHINGLE_WORDS = 8
MIN_PROMPT_WORDS = 5


def _allowed_mask(words: list[str], allowed: Sequence[str]) -> list[bool]:
    """Mark the words of `system` that belong to a phrase the app is meant to say verbatim."""
    covered = [False] * len(words)
    for phrase in allowed:
        target = tokens(phrase)
        if not target:
            continue
        for i in range(len(words) - len(target) + 1):
            if words[i : i + len(target)] == target:
                for j in range(i, i + len(target)):
                    covered[j] = True
    return covered


def system_prompt_leak(system: str, response: str, allowed: Sequence[str] = ()) -> str | None:
    """Return the first leaked word run, or None.

    Word runs touching an `allowed` phrase (a fixed reply the prompt tells the app to use,
    such as its "I don't know" sentence) never count: saying them is the intended behavior.
    Found in a real run, where the bot's own abstain sentence was reported as a disclosure.
    """
    words = tokens(system)
    if len(words) < MIN_PROMPT_WORDS:
        return None
    covered = _allowed_mask(words, allowed)
    n = min(SHINGLE_WORDS, len(words))
    haystack = " " + " ".join(tokens(response)) + " "
    for i in range(len(words) - n + 1):
        if any(covered[i : i + n]):
            continue
        shingle = " ".join(words[i : i + n])
        if f" {shingle} " in haystack:
            return shingle
    return None
