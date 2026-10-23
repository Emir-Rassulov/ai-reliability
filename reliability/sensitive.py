"""Masking and detection of sensitive values (canaries, synthetic secrets).

Detection is lexical. It covers the exact value (case-insensitive), the value with its
separators removed, replaced or spread out (up to three separator characters), base64 of
the value (standard and URL-safe, with or without padding) and, optionally, the last four
alphanumeric characters as a standalone token.
It does not cover paraphrase, other encodings (hex, rot13), base64 of a longer string that
contains the value, or a value split across sentences.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Literal

HitKind = Literal["direct", "compact", "base64", "last4"]


def mask(value: str) -> str:
    """`CANARY-7F3A9B` -> `CANARY-****9B`; `4111-5555-8888-1234` -> `****34`.

    Only an alphabetic label before the first dash is kept ("CANARY-", "sk-"), and only if at
    least two secret characters stay hidden. Short values become `****`.
    """
    if len(value) <= 6:
        return "****"
    dash = value.find("-")
    hidden_after_dash = len(value) - (dash + 1) - 2
    label = value[:dash]
    keep = dash > 0 and label.isalpha() and hidden_after_dash >= 2
    prefix = value[: dash + 1] if keep else ""
    return f"{prefix}****{value[-2:]}"


def base64_variants(value: str) -> list[str]:
    raw = value.encode("utf-8")
    variants: set[str] = set()
    for enc in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        text = enc.decode("ascii")
        variants.add(text)
        variants.add(text.rstrip("="))
    return sorted(variants, key=len, reverse=True)


def _alnum(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def compact_pattern(value: str) -> re.Pattern[str] | None:
    """Matches the value's alphanumerics with up to three separators between them
    ("4111 5555 8888 1234", "C A N A R Y - 7 F 3 A")."""
    chars = _alnum(value)
    if len(chars) < 8:
        return None
    body = r"[\s\-_.:/]{0,3}".join(re.escape(c) for c in chars)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class Hit:
    kind: HitKind
    offset: int


def find_sensitive(text: str, value: str, *, include_last4: bool = False) -> list[Hit]:
    hits: list[Hit] = []
    for match in re.finditer(re.escape(value), text, re.IGNORECASE):
        hits.append(Hit("direct", match.start()))
    pattern = compact_pattern(value)
    if pattern is not None:
        direct_offsets = {h.offset for h in hits}
        for match in pattern.finditer(text):
            if match.start() not in direct_offsets:
                hits.append(Hit("compact", match.start()))
    for variant in base64_variants(value):
        start = text.find(variant)
        if start >= 0:
            hits.append(Hit("base64", start))
            break
    if include_last4 and not hits:
        tail = _alnum(value)[-4:]
        if len(_alnum(value)) >= 8:
            tail_re = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(tail)}(?![A-Za-z0-9])", re.IGNORECASE
            )
            hits.extend(Hit("last4", m.start()) for m in tail_re.finditer(text))
    return sorted(hits, key=lambda h: h.offset)
