"""CLI behaviour: every exit code, filters, help, and no secrets on stdout/stderr."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from reliability import PROJECT_ROOT
from reliability.cli import main
from tests.conftest import BASE_SCENARIO, write_json


def make_suite(tmp_path: Path, scenarios: list[dict[str, Any]], responses: dict[str, Any]) -> Path:
    suite = tmp_path / "suite"
    suite.mkdir(parents=True)
    write_json(
        suite / "cases.json", {"schema_version": "1.0", "synthetic": True, "scenarios": scenarios}
    )
    write_json(
        suite / "mock_responses.json",
        {"schema_version": "1.0", "synthetic": True, "responses": responses},
    )
    return suite


FACT = [{"id": "w", "phrases": ["30 days"], "mandatory": True}]


def run_cli(*args: str) -> int:
    return main(list(args))


def test_exit_0_pass_on_starter_suite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run_cli("run", "--report-dir", str(tmp_path))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Decision: PASS" in out
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_exit_1_fail(tmp_path: Path) -> None:
    suite = make_suite(
        tmp_path, [{**BASE_SCENARIO, "required_facts": FACT}], {"t-001": {"text": "no idea"}}
    )
    assert run_cli("run", "--path", str(suite), "--report-dir", str(tmp_path / "r")) == 1


def test_exit_2_review_required(tmp_path: Path) -> None:
    suite = make_suite(
        tmp_path, [{**BASE_SCENARIO, "required_facts": FACT}], {"t-001": {"text": "maybe 30 days"}}
    )
    assert run_cli("run", "--path", str(suite), "--report-dir", str(tmp_path / "r")) == 2


def test_exit_3_infrastructure_error(tmp_path: Path) -> None:
    suite = make_suite(
        tmp_path, [{**BASE_SCENARIO, "required_facts": FACT}], {"t-001": {"error": "auth"}}
    )
    assert run_cli("run", "--path", str(suite), "--report-dir", str(tmp_path / "r")) == 3


def test_exit_4_invalid_scenario(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = make_suite(tmp_path, [{**BASE_SCENARIO, "severity": "extreme"}], {})
    assert run_cli("validate-scenarios", "--path", str(suite)) == 4
    assert "scenario 't-001' field 'severity'" in capsys.readouterr().err


def test_exit_4_missing_mock_response(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = make_suite(tmp_path, [BASE_SCENARIO], {})
    assert run_cli("run", "--path", str(suite), "--report-dir", str(tmp_path / "r")) == 4
    assert "no canned response" in capsys.readouterr().err


def test_exit_4_bad_arguments() -> None:
    assert run_cli("run", "--category", "nonsense") == 4
    assert run_cli("frobnicate") == 4


def test_exit_4_paid_model_without_unlock(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_cli("run", "--provider", "litellm", "--model", "gpt-4o", "--allow-network")
    assert code == 4
    assert "--allow-paid" in capsys.readouterr().err


def test_exit_4_missing_config(tmp_path: Path) -> None:
    assert run_cli("run", "--config", str(tmp_path / "missing.toml")) == 4


def test_exit_4_no_matching_scenarios(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli("run", "--scenario", "does-not-exist", "--report-dir", str(tmp_path)) == 4
    assert "no scenarios match" in capsys.readouterr().err


def test_validate_scenarios_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli("validate-scenarios") == 0
    assert capsys.readouterr().out.startswith("OK: ")


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli("--help") == 0
    assert "Exit codes" in capsys.readouterr().out


def test_single_scenario_and_category_filters(tmp_path: Path) -> None:
    assert run_cli("run", "--scenario", "llm-001", "--report-dir", str(tmp_path)) == 0
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert [s["scenario_id"] for s in report["scenarios"]] == ["llm-001"]


def test_single_selftest_scenario_is_review_not_pass(tmp_path: Path) -> None:
    """A run with no gate checks never produces PASS (R4_no_gate_checks)."""
    assert run_cli("run", "--scenario", "llm-003", "--report-dir", str(tmp_path)) == 2


def test_module_entry_point_subprocess(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "reliability.cli", "validate-scenarios"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_live_run_skips_selftests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the first real `run --provider litellm` sent the 31 self-tests (canned
    known-bad answers) to the model and ended in R0 ERROR. A live run uses gate scenarios."""
    from reliability import cli
    from reliability.providers.mock_provider import StatefulMockProvider, load_mock_responses

    canned = StatefulMockProvider(
        load_mock_responses(PROJECT_ROOT / "scenarios" / "mock_responses.json")
    )
    monkeypatch.setattr(cli, "build_provider", lambda **_kw: canned)  # stands in for a live model
    code = run_cli(
        "run",
        "--provider",
        "litellm",
        "--model",
        "ollama/x",
        "--allow-network",
        "--report-dir",
        str(tmp_path),
    )
    captured = capsys.readouterr()
    assert code == 0, captured.out
    assert "31 self-test scenario(s) skipped" in captured.err
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["selftest"]["total"] == 0
    assert {s["role"] for s in report["scenarios"]} == {"gate"}


def test_live_run_of_a_selftest_only_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from reliability import cli

    monkeypatch.setattr(cli, "build_provider", lambda **_kw: object())
    code = run_cli(
        "run",
        "--provider",
        "litellm",
        "--model",
        "ollama/x",
        "--allow-network",
        "--scenario",
        "llm-006",
        "--report-dir",
        str(tmp_path),
    )
    assert code != 0
    assert "need --provider mock" in capsys.readouterr().err
