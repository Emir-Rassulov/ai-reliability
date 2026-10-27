"""Citation extraction shared by the citation and grounding checks.

Three forms are recognised:

* the configured pattern (`checks.citation_pattern`, default `[doc-id]`): any id it matches
  counts, so an unknown id is reported as fabricated;
* an explicit label - `[doc-ID: policy-2025]`, `[doc-ID: "policy-2025"]`,
  `(doc-ID "policy-2025")`, `doc_id=policy-2025` - any id counts;
* a bare id in parentheses - `(policy-2025)` - only when it is a known document id, so
  ordinary parenthesised words ("(optional)") are never read as fabricated citations.

Found with real models: asked to "cite sources as [doc-ID]", llama3.2 wrote
`[doc-ID: shipping-standard]` and `(doc-ID "warranty-main")`, which the strict pattern
missed, so correct citations were reported as "no_citations".
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

_ID = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_LABELLED = re.compile(
    r"[\[(]?\s*doc(?:ument)?[-_ ]?id\s*[:=]?\s*[\"']?(?P<id>" + _ID + r")[\"']?\s*[\])]?",
    re.IGNORECASE,
)
_PARENTHESISED = re.compile(r"\(\s*[\"']?(?P<id>" + _ID + r")[\"']?\s*\)")


@dataclass(frozen=True)
class Citation:
    doc_id: str  # lower case
    start: int
    end: int


def find_citations(text: str, pattern: re.Pattern[str], known: Collection[str]) -> list[Citation]:
    """All citations in `text`, ordered by position, without overlaps."""
    known_lower = {k.lower() for k in known}
    found: list[Citation] = []

    def add(doc_id: str, start: int, end: int) -> None:
        if any(start < c.end and c.start < end for c in found):
            return
        found.append(Citation(doc_id.lower(), start, end))

    for m in _LABELLED.finditer(text):
        add(m.group("id"), m.start(), m.end())
    for m in pattern.finditer(text):
        add(m.group("id"), m.start(), m.end())
    for m in _PARENTHESISED.finditer(text):
        if m.group("id").lower() in known_lower:
            add(m.group("id"), m.start(), m.end())
    return sorted(found, key=lambda c: c.start)


def strip_citations(text: str, citations: Collection[Citation]) -> str:
    """`text` with the given citation spans removed."""
    out, pos = [], 0
    for c in sorted(citations, key=lambda c: c.start):
        out.append(text[pos : c.start])
        pos = c.end
    out.append(text[pos:])
    return "".join(out)
