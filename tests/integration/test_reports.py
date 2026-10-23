"""JSON, summary, compare, HTML and history outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from reliability.cli import main
from reliability.errors import ConfigError
from reliability.evaluation.regression import compare_reports, render_comparison
from reliability.models import RunReport, Status
from reliability.reporting.history import append_history, history_entry
from reliability.reporting.html_report import SCRIPT, HtmlTable, render_html
from reliability.reporting.json_report import load_report
from reliability.reporting.redaction import Redactor
from reliability.reporting.summary import render_summary
from tests.conftest import BASE_SCENARIO
from tests.integration.test_cli import make_suite
from tests.regression.outcomes import run_full_suite

XSS = "<script>alert('xss')</script><img src=x onerror=alert(1)>"


@pytest.fixture(scope="module")
def report() -> RunReport:
    return run_full_suite()


def _with_text(report: RunReport, text: str) -> RunReport:
    clone = report.model_copy(deep=True)
    target = next(s for s in clone.scenarios if any(c.status is Status.FAIL for c in s.checks))
    target.response_excerpt = text
    target.checks[0].explanation = text
    target.checks[0].evidence = [{"observed": text}]
    return clone


def test_html_escapes_model_output(report: RunReport) -> None:
    html = render_html(_with_text(report, XSS), Redactor())
    assert XSS not in html
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html
    assert html.count("<script>") == 1  # only our own, hash-pinned script
    assert "onerror=alert(1)>" not in html


def test_html_is_self_contained_with_csp(report: RunReport) -> None:
    html = render_html(report, Redactor())
    assert not re.search(r"(src|href)=['\"]?https?://", html)
    assert "@import" not in html and "url(" not in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    import base64
    import hashlib

    digest = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
    assert f"sha256-{digest}" in html


def test_html_contains_decision_findings_and_tables(report: RunReport) -> None:
    table = HtmlTable(
        title="Detection matrix", columns=["defect", "check"], rows=[["d<1>", "x"]], note="n"
    )
    html = render_html(report, Redactor(), [table])
    assert "R6_all_gates_passed" in html
    assert "data-role='selftest'" in html and "data-status='FAIL'" in html
    assert "Detection matrix" in html and "d&lt;1&gt;" in html


def test_html_redacts(report: RunReport) -> None:
    secret = "CANARY-HTML-1234X"
    html = render_html(_with_text(report, f"code {secret}"), Redactor([secret]))
    assert secret not in html


def test_summary_sections(report: RunReport) -> None:
    text = render_summary(report)
    for heading in (
        "DECISION: PASS",
        "Critical and high findings (gate):",
        "RAG (gate means):",
        "Security findings (gate):",
        "Operational:",
        "Recommended next actions:",
    ):
        assert heading in text
    assert "succeeded after 2 attempts" in text


def test_summary_of_failing_run_lists_findings_and_advice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario = {
        **BASE_SCENARIO,
        "severity": "critical",
        "required_facts": [{"id": "w", "phrases": ["30 days"]}],
    }
    suite = make_suite(tmp_path, [scenario], {"t-001": {"text": "no idea"}})
    assert main(["run", "--path", str(suite), "--report-dir", str(tmp_path / "r")]) == 1
    path = next((tmp_path / "r").glob("*.json"))
    capsys.readouterr()
    assert main(["summary", "--input", str(path)]) == 0
    out = capsys.readouterr().out
    assert "DECISION: FAIL" in out
    assert "[FAIL] t-001/required_facts (critical)" in out
    assert "Block the release" in out and "required_facts: compare the answer" in out


def test_compare_detects_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = make_suite(
        tmp_path / "a",
        [{**BASE_SCENARIO, "required_facts": [{"id": "w", "phrases": ["30 days"]}]}],
        {"t-001": {"text": "30 days"}},
    )
    bad = make_suite(
        tmp_path / "b",
        [{**BASE_SCENARIO, "required_facts": [{"id": "w", "phrases": ["30 days"]}]}],
        {"t-001": {"text": "no"}},
    )
    assert main(["run", "--path", str(good), "--report-dir", str(tmp_path / "ra")]) == 0
    assert main(["run", "--path", str(bad), "--report-dir", str(tmp_path / "rb")]) == 1
    base = next((tmp_path / "ra").glob("*.json"))
    cur = next((tmp_path / "rb").glob("*.json"))
    capsys.readouterr()
    assert main(["compare", "--baseline", str(base), "--current", str(cur)]) == 1
    out = capsys.readouterr().out
    assert "PASS -> FAIL  (REGRESSION)" in out and "! t-001/required_facts: PASS -> FAIL" in out
    assert main(["compare", "--baseline", str(base), "--current", str(base)]) == 0


def test_compare_metric_deltas_and_scenario_sets(report: RunReport) -> None:
    other = report.model_copy(deep=True)
    other.scenarios = other.scenarios[1:]
    other.score = 0.5
    cmp = compare_reports(report, other)
    assert cmp.removed == [report.scenarios[0].scenario_id]
    assert cmp.deltas["score"] == (report.score, 0.5)
    assert not cmp.has_regression
    assert "Scenarios removed" in render_comparison(cmp)


def test_load_report_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_report(tmp_path / "x.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a valid run report"):
        load_report(bad)


def test_cli_html_and_history(tmp_path: Path) -> None:
    hist = tmp_path / "history" / "runs.jsonl"
    for _ in range(2):
        assert (
            main(["run", "--html", "--report-dir", str(tmp_path), "--history-file", str(hist)]) == 0
        )
    assert len(list(tmp_path.glob("*.html"))) == 2
    lines = [json.loads(x) for x in hist.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["decision"] == "PASS"
    assert set(lines[0]) >= {"run_id", "timestamp", "commit", "counts", "pass_rate", "rag_metrics"}
    assert "evidence" not in hist.read_text(encoding="utf-8")


def test_history_entry_commit(report: RunReport, tmp_path: Path) -> None:
    assert history_entry(report, "abc123")["commit"] == "abc123"
    append_history(report, tmp_path / "h.jsonl", "abc")
    assert (tmp_path / "h.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_html_and_summary_edge_paths(report: RunReport) -> None:
    """No findings, unknown metrics, a table without a note, reported token usage."""
    from reliability.models import Role, TokenUsage

    clean = report.model_copy(deep=True)
    clean.scenarios = [s for s in clean.scenarios if s.role is Role.GATE]
    clean.score = None
    clean.token_usage = TokenUsage(input_tokens=12, output_tokens=7)
    html = render_html(clean, Redactor(), [HtmlTable(title="t", columns=["a"], rows=[["1"]])])
    assert "No findings." in html and "n/a" in html
    assert "tokens reported: input 12, output 7" in render_summary(clean)


def test_redactor_short_value_without_compact_form() -> None:
    assert Redactor(["ab-12x9"]).redact("value ab-12x9 here") == "value ab-****x9 here"
