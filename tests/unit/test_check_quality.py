"""Testing the tester: every labeled case behaves as recorded; metrics are computed honestly."""

from __future__ import annotations

from pathlib import Path

import pytest

from reliability import PROJECT_ROOT
from reliability.checks.base import CheckContext
from reliability.cli import main
from reliability.evaluation.check_quality import CheckQuality, evaluate_labeled, render_quality
from reliability.models import Status

LABELED = PROJECT_ROOT / "tests" / "fixtures" / "labeled_cases.json"


@pytest.fixture(scope="module")
def results(ctx: CheckContext) -> dict[str, CheckQuality]:
    return evaluate_labeled(LABELED, ctx)


def test_every_response_check_has_labeled_cases(results: dict[str, CheckQuality]) -> None:
    from reliability.checks.registry import CHECK_CLASSES

    assert {cls.name for cls in CHECK_CLASSES} == set(results)


def test_each_check_covers_required_case_kinds(results: dict[str, CheckQuality]) -> None:
    for name, quality in results.items():
        kinds = {c.kind for c in quality.cases}
        assert {"good", "bad"} <= kinds or {"good", "insufficient"} <= kinds, name
        assert kinds & {"fp_trap", "fn_trap", "misleading", "borderline"}, name


def test_no_unexpected_disagreement(results: dict[str, CheckQuality]) -> None:
    unexpected = [
        c.case_id
        for q in results.values()
        for c in q.cases
        if c.truth is not c.observed and not c.known_miss
    ]
    assert unexpected == []


def test_known_misses_still_miss(results: dict[str, CheckQuality]) -> None:
    """If a limitation gets fixed, this fails so the label and README are updated honestly."""
    stale = [
        c.case_id
        for q in results.values()
        for c in q.cases
        if c.known_miss and c.truth is c.observed
    ]
    assert stale == []


def test_metrics_hand_checked(results: dict[str, CheckQuality]) -> None:
    leak = results["data_leakage"]
    assert (leak.tp, leak.fp, leak.fn) == (4, 1, 1)
    assert leak.precision == 0.8 and leak.recall == 0.8
    assert results["structured_output"].precision == 1.0


def test_metric_edge_cases() -> None:
    from reliability.evaluation.check_quality import CaseResult

    empty = CheckQuality("x", [CaseResult("a", "x", "good", Status.PASS, Status.PASS, False, "")])
    assert empty.precision is None and empty.recall is None and empty.agreement == 1.0


def test_render_and_cli(
    results: dict[str, CheckQuality], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    text = render_quality(results)
    assert "known limitation" in text and "UNEXPECTED" not in text
    assert main(["check-quality"]) == 0
    assert "precision" in capsys.readouterr().out
    assert main(["check-quality", "--labeled", str(tmp_path / "none.json")]) == 4
