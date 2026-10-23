"""Reference RAG app (phase 5): BM25, filters, defect switches, architecture, detection matrix."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_app import DEFAULT_CORPUS, build_reference_app
from rag_app.bm25 import BM25, stem, tokenize
from rag_app.defects import DEFECTS, validate_defects
from rag_app.interface import RAGResult, RAGSystem, RAGSystemError
from rag_app.mock_generator import FALLBACK, MockGenerator
from rag_app.pipeline import (
    GROUNDED_SYSTEM,
    UNGROUNDED_SYSTEM,
    AssembledPrompt,
    ReferenceRAG,
    load_corpus,
)
from reliability import PROJECT_ROOT
from reliability.cli import main
from reliability.config import load_config
from reliability.evaluation.mutation import mutation_test, render_mutation
from reliability.evaluation.orchestration import run_rag_app
from reliability.models import Status
from reliability.reporting.redaction import Redactor
from reliability.scenarios import collect_sensitive_values, load_documents, load_scenarios

CONFIG = load_config(env={})
DOCS = load_documents(PROJECT_ROOT / "datasets" / "synthetic_documents.json", CONFIG.limits)
SCENARIOS = load_scenarios(PROJECT_ROOT / "scenarios", DOCS, CONFIG.limits).scenarios
ELIGIBLE = [s for s in SCENARIOS if s.rag_app]
REDACTOR = Redactor(collect_sensitive_values(SCENARIOS))


# ----------------------------------------------------------------------------- BM25


def test_tokenize_and_stem() -> None:
    assert tokenize("How many Items were RETURNED?") == ["many", "item", "were", "return"]
    assert [stem(w) for w in ("shipping", "business", "days", "is", "used")] == [
        "shipp",
        "business",
        "day",
        "is",
        "used",
    ]


def test_bm25_matches_hand_calculation() -> None:
    # corpus: d1 = [apple, banana], d2 = [apple, apple, cherry], d3 = [banana]; avgdl = 2
    index = BM25({"d1": "apple banana", "d2": "apple apple cherry", "d3": "banana"})
    idf_apple = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)  # ln(1.6)
    norm_d1 = 1.5 * (1 - 0.75 + 0.75 * 2 / 2)  # 1.5
    norm_d2 = 1.5 * (1 - 0.75 + 0.75 * 3 / 2)  # 2.0625
    assert index.score("apple", "d1") == pytest.approx(idf_apple * 1 * 2.5 / (1 + norm_d1))
    assert index.score("apple", "d2") == pytest.approx(idf_apple * 2 * 2.5 / (2 + norm_d2))
    assert index.score("apple", "d2") == pytest.approx(0.5784660052)
    idf_cherry = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    assert index.score("cherry apples", "d2") == pytest.approx(
        idf_apple * 5 / 4.0625 + idf_cherry * 2.5 / (1 + norm_d2)
    )
    assert [d for d, _ in index.search("apple")] == ["d2", "d1"]
    assert index.search("durian") == []


def test_bm25_tie_break_by_doc_id() -> None:
    index = BM25({"b": "same text", "a": "same text"})
    assert [d for d, _ in index.search("same")] == ["a", "b"]


# ----------------------------------------------------------------------------- pipeline


def app(*defects: str, answers: dict[str, list[dict[str, object]]] | None = None) -> ReferenceRAG:
    return ReferenceRAG(load_corpus(DEFAULT_CORPUS), MockGenerator(answers or {}), defects=defects)


def last_prompt(system: ReferenceRAG) -> AssembledPrompt:
    generator = system.generator
    assert isinstance(generator, MockGenerator)
    return generator.prompts[-1]


def test_access_filter_applied_before_prompt_assembly() -> None:
    safe = app()
    safe.answer("What is the salary band for level L3?", ["public"])
    prompt = last_prompt(safe)
    assert prompt.delivered == () and "61000" not in prompt.documents_block + prompt.system
    insider = app()
    result = insider.answer("What is the salary band for level L3?", ["public", "confidential"])
    assert [d.doc_id for d in result.context] == ["salary-bands"]


def test_freshness_drops_superseded_versions() -> None:
    result = app().answer("How many days do I have to return an item?", ["public"])
    assert "policy-2023" not in [d.doc_id for d in result.retrieved]


def _snapshot(
    system: ReferenceRAG, query: str, access: Sequence[str] = ("public",)
) -> tuple[list[str], AssembledPrompt]:
    result = system.answer(query, access)
    return [d.doc_id for d in result.retrieved], last_prompt(system)


QUERY_RETURNS = "How many days do I have to return an item?"


def test_defect_no_access_filter_changes_only_retrieval() -> None:
    base_ids, base_prompt = _snapshot(app(), "What is the salary band for level L3?")
    ids, prompt = _snapshot(app("no_access_filter"), "What is the salary band for level L3?")
    assert base_ids == [] and ids == ["salary-bands"]
    assert prompt.system == base_prompt.system and prompt.grounded and prompt.untrusted_marked


def test_defect_stale_first_changes_only_ranking() -> None:
    base_ids, base_prompt = _snapshot(app(), QUERY_RETURNS)
    ids, prompt = _snapshot(app("stale_first"), QUERY_RETURNS)
    assert ids.index("policy-2023") < ids.index("policy-2025")
    assert "policy-2023" not in base_ids
    assert prompt.system == base_prompt.system


def test_defect_no_grounding_instruction_changes_only_system_prompt() -> None:
    base_ids, base_prompt = _snapshot(app(), QUERY_RETURNS)
    ids, prompt = _snapshot(app("no_grounding_instruction"), QUERY_RETURNS)
    assert ids == base_ids and prompt.documents_block == base_prompt.documents_block
    assert base_prompt.system == GROUNDED_SYSTEM and prompt.system == UNGROUNDED_SYSTEM


def test_defect_truncated_context_changes_only_delivery() -> None:
    base_ids, base_prompt = _snapshot(app(), QUERY_RETURNS)
    ids, prompt = _snapshot(app("truncated_context"), QUERY_RETURNS)
    assert ids == base_ids and prompt.system == base_prompt.system
    assert len(prompt.delivered) < len(base_prompt.delivered)
    assert len(prompt.documents_block) < len(base_prompt.documents_block)


def test_defect_no_untrusted_marking_moves_documents_into_instructions() -> None:
    base_ids, base_prompt = _snapshot(app(), QUERY_RETURNS)
    ids, prompt = _snapshot(app("no_untrusted_marking"), QUERY_RETURNS)
    assert ids == base_ids and prompt.delivered == base_prompt.delivered
    assert "untrusted" in base_prompt.documents_block and prompt.documents_block == ""
    assert "IMPORTANT SYSTEM NOTE" in prompt.system and not prompt.untrusted_marked


def test_unknown_defect_rejected() -> None:
    with pytest.raises(ValueError, match="unknown defect"):
        validate_defects(["teleport"])
    assert set(DEFECTS) == {
        "no_access_filter",
        "stale_first",
        "no_grounding_instruction",
        "truncated_context",
        "no_untrusted_marking",
    }


def test_mock_generator_matching_and_fallback() -> None:
    answers: dict[str, list[dict[str, object]]] = {
        "t": [
            {"docs": ["a"], "text": "plain"},
            {"docs": ["a"], "grounded": False, "text": "ungrounded"},
        ]
    }
    gen = MockGenerator(answers)

    def prompt(delivered: tuple[str, ...], grounded: bool = True) -> AssembledPrompt:
        return AssembledPrompt("s", "d", "q", delivered, grounded, True)

    assert gen.generate(prompt(("a",)), "t") == "plain"
    assert gen.generate(prompt(("a",), grounded=False), "t") == "ungrounded"
    assert gen.generate(prompt(("b",)), "t") == FALLBACK
    assert gen.generate(prompt(("a",)), None) == FALLBACK


def test_reference_app_satisfies_protocol() -> None:
    assert isinstance(build_reference_app(), RAGSystem)


# ------------------------------------------------------------------------ evaluator integration


def test_baseline_rag_app_run_passes() -> None:
    report = run_rag_app(
        scenarios=ELIGIBLE,
        system=build_reference_app(),
        documents=DOCS,
        config=CONFIG,
        redactor=REDACTOR,
    )
    assert report.decision.decision.value == "PASS"
    assert report.target == "rag-app" and report.provider == "rag-app"


class _Broken:
    def answer(
        self, query: str, user_access: Sequence[str], *, trace_id: str | None = None
    ) -> RAGResult:
        raise RAGSystemError("index offline")


def test_rag_system_error_is_an_error_never_a_pass() -> None:
    report = run_rag_app(
        scenarios=ELIGIBLE[:1], system=_Broken(), documents=DOCS, config=CONFIG, redactor=REDACTOR
    )
    statuses = {c.status for s in report.scenarios for c in s.checks}
    assert statuses == {Status.ERROR}
    assert report.decision.decision.value in ("ERROR", "REVIEW_REQUIRED")


def test_detection_matrix_reproduces_snapshot() -> None:
    result = mutation_test(
        scenarios=ELIGIBLE,
        factory=build_reference_app,
        defects=sorted(DEFECTS),
        documents=DOCS,
        config=CONFIG,
        redactor=REDACTOR,
    )
    expected = json.loads(
        (PROJECT_ROOT / "tests" / "regression" / "expected_detection_matrix.json").read_text()
    )
    observed = {
        "baseline_decision": result.baseline.decision.decision.value,
        "decisions": {k: v.decision.decision.value for k, v in result.runs.items()},
        "detections": result.detections,
        "gaps": result.gaps,
    }
    assert observed == expected
    assert result.criteria_met
    assert "Acceptance criteria: MET" in render_mutation(result)


def test_mutation_reports_gaps_honestly() -> None:
    """A defect the scenarios cannot see must be reported as a gap, not silently pass."""
    blind = [s for s in ELIGIBLE if s.id == "rag-001"]
    result = mutation_test(
        scenarios=blind,
        factory=build_reference_app,
        defects=["no_access_filter"],
        documents=DOCS,
        config=CONFIG,
        redactor=REDACTOR,
    )
    assert not result.criteria_met
    assert any(g.startswith("no_access_filter") for g in result.gaps)
    assert "NOT MET" in render_mutation(result)


# ----------------------------------------------------------------------------- CLI


def test_cli_rag_app_and_mutation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--target", "rag-app", "--report-dir", str(tmp_path)]) == 0
    assert (
        main(
            [
                "run",
                "--target",
                "rag-app",
                "--defect",
                "no_access_filter",
                "--report-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert main(["run", "--defect", "stale_first", "--report-dir", str(tmp_path)]) == 4
    assert (
        main(["run", "--target", "rag-app", "--provider", "litellm", "--report-dir", str(tmp_path)])
        == 4
    )
    assert (
        main(["run", "--target", "rag-app", "--scenario", "llm-001", "--report-dir", str(tmp_path)])
        == 4
    )
    capsys.readouterr()
    assert main(["mutation-test", "--report-dir", str(tmp_path), "--html"]) == 0
    out = capsys.readouterr().out
    assert "Acceptance criteria: MET" in out
    html = next(tmp_path.glob("mutation-*.html")).read_text(encoding="utf-8")
    assert "Defect detection matrix" in html


# ----------------------------------------------------------------------------- architecture


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_rag_app_never_imports_the_evaluator() -> None:
    for path in (PROJECT_ROOT / "rag_app").rglob("*.py"):
        assert not any(m.startswith("reliability") for m in _imports(path)), path.name


def test_evaluator_uses_only_the_rag_interface() -> None:
    for path in (PROJECT_ROOT / "reliability").rglob("*.py"):
        if path.name == "cli.py":  # composition root
            continue
        rag = {m for m in _imports(path) if m.startswith("rag_app")}
        assert rag <= {"rag_app.interface"}, f"{path.name}: {rag}"
