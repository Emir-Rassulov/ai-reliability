"""The full starter dataset on the mock provider must reproduce the reviewed snapshot exactly."""

from __future__ import annotations

import json

from reliability.config import load_config
from reliability.evaluation.release import decide
from reliability.models import Role, RunReport, Status
from reliability.reporting.json_report import report_to_json
from reliability.reporting.redaction import Redactor
from tests.regression.outcomes import EXPECTED, outcomes, run_full_suite


def test_full_dataset_reproduces_expected_outcomes() -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert outcomes(run_full_suite()) == expected


def test_every_selftest_matches_its_declared_statuses() -> None:
    report = run_full_suite()
    assert report.selftest.mismatched == []
    assert report.selftest.matched == report.selftest.total > 0


def test_gate_suite_passes_and_selftests_contain_failures() -> None:
    report = run_full_suite()
    assert report.decision.decision.value == "PASS"
    gate_statuses = {c.status for s in report.scenarios if s.role is Role.GATE for c in s.checks}
    assert gate_statuses <= {Status.PASS, Status.NOT_APPLICABLE}
    selftest_statuses = {
        c.status for s in report.scenarios if s.role is Role.SELFTEST for c in s.checks
    }
    assert Status.FAIL in selftest_statuses


def test_decision_reproducible_from_stored_report() -> None:
    report = run_full_suite()
    stored = RunReport.model_validate_json(report_to_json(report, Redactor()))
    cfg = load_config(env={})
    again = decide(stored.scenarios, cfg.release, stored.latency.p95_ms)
    assert again == stored.decision


def test_report_json_is_stable() -> None:
    report = run_full_suite()
    assert report_to_json(report, Redactor()) == report_to_json(report, Redactor())
