"""Pure-Python Okapi BM25.

Tokenisation: lowercase, split on anything that is not [a-z0-9], drop a small English
stopword list, then a minimal suffix stripper (-ing, -ed, plural -s; words ending in -ss are
kept) so that "returned"/"return" and "items"/"item" match. No other stemming.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)        (always positive)

Defaults k1 = 1.5, b = 0.75. Ties are broken by doc_id ascending, so results are
deterministic. Documents with score 0 are not returned.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "how", "i", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what", "when",
        "which", "who", "with", "you", "your", "have", "has", "me", "we",
    }
)  # fmt: skip


def stem(token: str) -> str:
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: Mapping[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tf: dict[str, Counter[str]] = {d: Counter(tokenize(text)) for d, text in docs.items()}
        self.length = {d: sum(c.values()) for d, c in self.tf.items()}
        self.n_docs = len(docs)
        self.avgdl = (sum(self.length.values()) / self.n_docs) if self.n_docs else 0.0
        self.df: Counter[str] = Counter()
        for counts in self.tf.values():
            self.df.update(counts.keys())

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str, doc_id: str) -> float:
        counts = self.tf[doc_id]
        norm = self.k1 * (1 - self.b + self.b * self.length[doc_id] / self.avgdl)
        total = 0.0
        for term in tokenize(query):
            tf = counts.get(term, 0)
            if tf:
                total += self.idf(term) * tf * (self.k1 + 1) / (tf + norm)
        return total

    def search(self, query: str) -> list[tuple[str, float]]:
        scored = [(d, self.score(query, d)) for d in self.tf]
        return sorted(((d, s) for d, s in scored if s > 0), key=lambda x: (-x[1], x[0]))
