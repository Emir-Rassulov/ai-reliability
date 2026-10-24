"""Phase 6 dashboard. Runs only where the [ui] extra is installed (its own CI job)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest

from dashboard import components, data, runner
from reliability import PROJECT_ROOT
from reliability.models import RunReport, Status
from reliability.reporting.history import append_history
from reliability.reporting.json_report import write_report
from reliability.reporting.redaction import Redactor
from tests.regression.outcomes import run_full_suite

DASH = PROJECT_ROOT / "dashboard"
PAGES = ["overview", "history", "compare", "rag_pipeline", "run"]
XSS = "<script>alert('x')</script> [click me](https://evil.example) **bold** <img src=x onerror=1>"


@pytest.fixture(scope="module")
def base_report() -> RunReport:
    return run_full_suite()


def _variant(report: RunReport, run_id: str, started: str, *, fail: bool = False) -> RunReport:
    clone = report.model_copy(deep=True)
    clone.run_id = run_id
    clone.started_at = started
    if fail:
        scenario = next(s for s in clone.scenarios if s.scenario_id == "rag-001")
        scenario.checks[0].status = Status.FAIL
        scenario.response_excerpt = XSS
        scenario.checks[0].explanation = XSS
        clone.counts["FAIL"] = 1
    return clone


@pytest.fixture
def populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_report: RunReport) -> Path:
    reports = tmp_path / "reports"
    first = _variant(base_report, "20260101T000000Z-aaaaaa", "2026-01-01T00:00:00+00:00")
    second = _variant(
        base_report, "20260102T000000Z-bbbbbb", "2026-01-02T00:00:00+00:00", fail=True
    )
    for report in (first, second):
        write_report(report, reports, Redactor())
    (reports / "mutation-x.json").write_text(
        json.dumps(
            {
                "matrix": {"stale_first": {"context_quality": True, "citations": False}},
                "expected": {"stale_first": ["context_quality"]},
                "decisions": {"stale_first": "FAIL"},
                "baseline_decision": "PASS",
                "gaps": [],
            }
        ),
        encoding="utf-8",
    )
    history = tmp_path / "history" / "runs.jsonl"
    append_history(first, history, "c1")
    append_history(second, history, "c2")
    monkeypatch.setenv("AIREL_REPORT_DIR", str(reports))
    monkeypatch.setenv("AIREL_HISTORY_FILE", str(history))
    monkeypatch.setenv("AIREL_PUBLIC_MODE", "true")
    return tmp_path


@pytest.fixture
def empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AIREL_REPORT_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("AIREL_HISTORY_FILE", str(tmp_path / "none.jsonl"))
    monkeypatch.setenv("AIREL_PUBLIC_MODE", "true")
    return tmp_path


def page(name: str) -> AppTest:
    return AppTest.from_file(str(DASH / "pages" / f"{name}.py"), default_timeout=60).run()


# ----------------------------------------------------------------------------- data.py


def test_load_reports_sorted_and_tolerant(populated: Path) -> None:
    (populated / "reports" / "broken.json").write_text("{", encoding="utf-8")
    (populated / "reports" / "other.json").write_text("{}", encoding="utf-8")
    loaded = data.load_reports(populated / "reports")
    assert [r.run_id for r in loaded.reports] == [
        "20260101T000000Z-aaaaaa",
        "20260102T000000Z-bbbbbb",
    ]
    assert sorted(loaded.problems) == [
        "broken.json: invalid report, skipped",
        "other.json: invalid report, skipped",
    ]
    assert data.load_reports(populated / "missing").reports == []


def test_metric_cards_deltas(populated: Path) -> None:
    first, second = data.load_reports(populated / "reports").reports
    cards = {c.label: c for c in data.metric_cards(second, first)}
    assert cards["Failures"].value == "1" and cards["Failures"].delta == "+1"
    assert cards["Failures"].delta_color == "inverse"
    assert cards["Release decision"].delta is None  # decision unchanged in the copy
    alone = {c.label: c for c in data.metric_cards(first, None)}
    assert alone["Pass rate"].delta is None and alone["Pass rate"].value == "100.0%"


def test_findings_filters(base_report: RunReport) -> None:
    every = data.findings(base_report)
    assert every and all(f.role == "selftest" for f in every)  # the gate suite passes
    only_sec = data.findings(base_report, categories={"security"}, statuses={"FAIL"})
    assert only_sec and {f.category for f in only_sec} == {"security"}
    assert data.findings(base_report, roles={"gate"}) == []
    assert data.findings(base_report, severities={"low"}) == []
    assert len(data.findings(base_report, include_passing=True)) > len(every)


def test_security_breakdown_and_rag(base_report: RunReport) -> None:
    breakdown = data.security_breakdown(base_report)
    assert breakdown["gate"] == {}
    assert {"prompt injection", "data leakage", "unauthorized document"} <= set(
        breakdown["selftest"]
    )
    assert data.rag_quality(base_report)["retrieval"]["Recall@K"] == 1.0
    assert set(data.grounding_outcomes(base_report)) <= data.GROUNDING_RESULTS


def test_history_trend_and_regressions(populated: Path) -> None:
    entries, problems = data.read_history(populated / "history" / "runs.jsonl")
    assert problems == [] and len(entries) == 2
    points = data.trend(entries, "counts.FAIL")
    assert [p[1] for p in points] == [0.0, 1.0]
    assert data.history_regressions(entries) == ["20260102T000000Z-bbbbbb: failures 0 -> 1"]
    with pytest.raises(KeyError):
        data.trend(entries, "nonsense")


def test_history_handles_malformed_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text(
        '\n{"run_id": "a", "decision": "PASS"}\nnot json\n{"x": 1}\n[1]\n', encoding="utf-8"
    )
    entries, problems = data.read_history(path)
    assert [e["run_id"] for e in entries] == ["a"] and len(problems) == 3
    assert data.read_history(tmp_path / "missing.jsonl") == ([], [])
    assert data.history_regressions([]) == []


def test_chart_labels_from_files_are_sanitised() -> None:
    entries = [
        {"run_id": "a", "decision": "<b>x</b>", "timestamp": "<a href='x'>t</a>", "pass_rate": 1},
        {"run_id": "b", "decision": "FAIL", "timestamp": "2026-01-01T00:00:00", "pass_rate": True},
    ]
    points = data.trend(entries, "pass_rate")
    assert points == [("run 1", 1.0, "unknown"), ("2026-01-01T00:00:00", None, "FAIL")]
    assert data.safe_decision("<script>") == "unknown" and data.safe_decision("PASS") == "PASS"


def test_compare_reuses_cli_logic(populated: Path) -> None:
    first, second = data.load_reports(populated / "reports").reports
    result = data.compare(first, second)
    assert result.has_regression and result.changes[0].scenario_id == "rag-001"


def test_pipeline_and_matrix_rows(populated: Path) -> None:
    mutation = data.latest_mutation(populated / "reports")
    assert mutation is not None
    rows = data.pipeline_rows(mutation)
    assert rows[0]["Stage"] == "Search (BM25)"
    assert (
        "stale_first: context_quality"
        in rows[0]["Injected defects and the checks that caught them"]
    )
    assert (
        "no_access_filter: NOT DETECTED"
        in rows[1]["Injected defects and the checks that caught them"]
    )
    assert data.matrix_rows(mutation) == [
        {
            "defect": "stale_first",
            "citations": "-",
            "context_quality": "detected (expected)",
            "decision": "FAIL",
        }
    ]
    assert (
        "no mutation-test data"
        in data.pipeline_rows(None)[0]["Injected defects and the checks that caught them"]
    )


def test_latest_mutation_skips_bad_files(tmp_path: Path) -> None:
    (tmp_path / "mutation-b.json").write_text("{", encoding="utf-8")
    (tmp_path / "mutation-a.json").write_text('{"no": "matrix"}', encoding="utf-8")
    assert data.latest_mutation(tmp_path) is None
    assert data.latest_mutation(tmp_path / "missing") is None


def test_env_helpers() -> None:
    assert data.public_mode({}) is True
    assert data.public_mode({"AIREL_PUBLIC_MODE": "false"}) is False
    assert data.public_mode({"AIREL_PUBLIC_MODE": "anything"}) is True
    assert data.report_dir({"AIREL_REPORT_DIR": "/x"}) == Path("/x")
    assert data.report_dir({}) == data.DEFAULT_REPORT_DIR
    assert runner.max_scenarios({"AIREL_UI_MAX_SCENARIOS": "oops"}) == 60
    assert runner.max_scenarios({"AIREL_UI_MAX_SCENARIOS": "5"}) == 5


# ----------------------------------------------------------------------------- figures


def test_figures_build() -> None:
    assert components.bar_figure({"a": 0.5, "b": None}, "t").data[0].y == ("a",)
    assert components.count_figure({"supported": 2, "omitted": 1}, "t").data[0].x == (
        "supported",
        "omitted",
    )
    assert components.donut_figure({"x": 1}, "t").data[0].hole == 0.6
    assert components.trend_figure([("a", 1.0, "PASS")], "t").data[0].mode == "lines+markers"
    assert components.bar_figure({"n": 3.0}, "t", percent=False).data[0].text == ("3",)


# ----------------------------------------------------------------------------- pages (AppTest)


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_on_real_reports(populated: Path, name: str) -> None:
    at = page(name)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_on_empty_data(empty: Path, name: str) -> None:
    at = page(name)
    assert not at.exception, [e.value for e in at.exception]
    if name in ("overview", "compare"):
        assert any("No reports found yet" in i.value for i in at.info)


def test_app_entry_point_renders(populated: Path) -> None:
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Overview"]


def test_overview_filters_change_visible_findings(populated: Path) -> None:
    at = page("overview")
    assert at.selectbox[0].value == "20260102T000000Z-bbbbbb"  # newest run by default
    total = len(at.dataframe[0].value)
    at.multiselect[3].set_value(["gate"]).run()
    assert len(at.dataframe[0].value) == 1 < total
    at.multiselect[0].set_value(["ERROR"]).run()
    assert not at.dataframe  # nothing matches: the table disappears
    assert any("0 finding(s)" in c.value for c in at.caption)


def test_report_content_is_inert_text(populated: Path) -> None:
    """Model output containing HTML/Markdown is shown via code/dataframe, never rendered."""
    at = page("overview")
    at.multiselect[3].set_value(["gate"]).run()
    rendered = [m.value for m in at.markdown] + [c.value for c in at.caption]
    rendered += [s.value for s in at.subheader] + [t.value for t in at.title]
    assert not any("<script>" in r or "evil.example" in r for r in rendered)
    assert XSS in at.dataframe[0].value["explanation"].tolist()
    assert any(c.value == XSS for c in at.code)


def test_dashboard_never_uses_unsafe_html() -> None:
    for path in DASH.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                assert node.arg != "unsafe_allow_html", path.name
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "html":
                    assert path.name == "components.py", path.name
                    assert ast.unparse(node.args[0]) == "theme.CSS"
                if node.func.attr == "markdown":
                    assert all(isinstance(a, ast.Constant) for a in node.args), path.name


def test_public_mode_offers_no_live_provider(populated: Path) -> None:
    at = page("run")
    labels = [s.label for s in at.selectbox]
    assert "Provider" not in labels and not at.text_input
    assert labels == [
        "Target",
        "Category",
        "Single scenario",
        "Injected defect (reference RAG app only)",
    ]


def test_private_mode_offers_local_provider(
    populated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIREL_PUBLIC_MODE", "false")
    at = page("run")
    assert "Provider" in [s.label for s in at.selectbox] and at.text_input


def test_run_page_starts_a_mock_run(populated: Path) -> None:
    at = page("run")
    at.selectbox[1].set_value("llm")
    at.button[0].click().run()
    assert not at.exception and any("Run finished" in s.value for s in at.success)
    assert len(data.load_reports(populated / "reports").reports) == 3


def test_run_page_rejects_defect_outside_rag_app(populated: Path) -> None:
    at = page("run")
    at.selectbox[3].set_value("stale_first")
    at.button[0].click().run()
    assert any("only to the reference RAG app" in e.value for e in at.error)


# ----------------------------------------------------------------------------- runner


def test_runner_bounds_and_modes(populated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = populated / "out"
    with pytest.raises(runner.RunRejected, match="mock provider only"):
        runner.start_run(
            runner.RunRequest(provider="litellm", model="gpt-4o"),
            report_dir=out,
            history_file=None,
            public=True,
        )
    with pytest.raises(runner.RunRejected, match="PAID"):
        runner.start_run(
            runner.RunRequest(provider="litellm", model="gpt-4o"),
            report_dir=out,
            history_file=None,
            public=False,
        )
    with pytest.raises(runner.RunRejected, match="unknown defect"):
        runner.start_run(
            runner.RunRequest(target="rag-app", defect="x"),
            report_dir=out,
            history_file=None,
            public=True,
        )
    with pytest.raises(runner.RunRejected, match="no scenarios"):
        runner.start_run(
            runner.RunRequest(scenario_id="missing"), report_dir=out, history_file=None, public=True
        )
    monkeypatch.setenv("AIREL_UI_MAX_SCENARIOS", "3")
    with pytest.raises(runner.RunRejected, match="exceed the UI limit"):
        runner.start_run(runner.RunRequest(), report_dir=out, history_file=None, public=True)


def test_runner_one_run_at_a_time(populated: Path) -> None:
    assert runner._LOCK.acquire(blocking=False)
    try:
        with pytest.raises(runner.RunRejected, match="in progress"):
            runner.start_run(
                runner.RunRequest(), report_dir=populated, history_file=None, public=True
            )
    finally:
        runner._LOCK.release()


@pytest.mark.parametrize(
    ("request_kwargs", "decision"),
    [
        ({"scenario_id": "sec-013"}, "PASS"),  # pulls in sec-012 from the same session
        ({"target": "rag-app", "defect": "no_access_filter"}, "FAIL"),
        ({"target": "mutation-test"}, "PASS"),
    ],
)
def test_runner_targets(populated: Path, request_kwargs: dict[str, Any], decision: str) -> None:
    hist = populated / "runs.jsonl"
    result = runner.start_run(
        runner.RunRequest(**request_kwargs),
        report_dir=populated / "o",
        history_file=hist,
        public=True,
    )
    assert result.report.decision.decision.value == decision
    assert result.path.exists() and hist.exists()
    if request_kwargs.get("scenario_id") == "sec-013":
        assert [s.scenario_id for s in result.report.scenarios] == ["sec-012", "sec-013"]
    if request_kwargs.get("target") == "mutation-test":
        assert result.mutation_path is not None and result.mutation_path.exists()


def test_available_scenarios_lists_dataset() -> None:
    items = runner.available_scenarios()
    assert len(items) >= 59 and ("rag-001", "rag", True) in items


def test_previous_comparable_and_latency_sources(populated: Path, base_report: RunReport) -> None:
    first, second = data.load_reports(populated / "reports").reports
    app_run = _variant(base_report, "20260103T000000Z-cccccc", "2026-01-03T00:00:00+00:00")
    app_run.target = "rag-app"
    app_run.latency.sources = ["measured"]
    reports = [first, second, app_run]
    assert data.previous_comparable(reports, app_run) is None  # no earlier rag-app run
    assert data.previous_comparable(reports, second) is first
    cards = {c.label: c for c in data.metric_cards(app_run, second)}
    assert cards["Median latency (ms)"].delta is None  # measured vs simulated: no delta
