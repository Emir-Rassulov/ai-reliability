"""Reference RAG application: the system under test for phase 5.

Kept separate from the evaluator: nothing here imports `reliability`, and the evaluator
depends only on `rag_app.interface`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rag_app.interface import RAGSystem
from rag_app.mock_generator import MockGenerator
from rag_app.pipeline import ReferenceRAG, load_corpus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "datasets" / "synthetic_documents.json"
DEFAULT_ANSWERS = Path(__file__).resolve().parent / "mock_answers.json"


def build_reference_app(
    defects: Iterable[str] = (),
    corpus_path: Path = DEFAULT_CORPUS,
    answers_path: Path = DEFAULT_ANSWERS,
    top_k: int = 3,
) -> RAGSystem:
    return ReferenceRAG(
        load_corpus(corpus_path),
        MockGenerator.from_file(answers_path),
        top_k=top_k,
        defects=defects,
    )
