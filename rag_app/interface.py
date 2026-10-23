"""The only contract between the evaluator and a RAG system under test.

Any RAG system (this reference app, one built on a vector database, an HTTP service) can be
evaluated by implementing `RAGSystem`. The evaluator imports nothing else from `rag_app`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RAGResult:
    #: Search results after access filtering, in rank order (what retrieval returned).
    retrieved: tuple[RetrievedDoc, ...]
    #: Documents whose full text reached the generator, in rank order (the real context).
    context: tuple[RetrievedDoc, ...]
    answer: str
    generator: str
    #: True when the generator is canned/mock output rather than a model.
    simulated: bool


class RAGSystemError(Exception):
    """The system under test failed to produce an answer."""


@runtime_checkable
class RAGSystem(Protocol):
    def answer(
        self, query: str, user_access: Sequence[str], *, trace_id: str | None = None
    ) -> RAGResult: ...
