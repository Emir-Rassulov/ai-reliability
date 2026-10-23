"""Evaluator behaviour: failed calls, crashing checks, self-test matching, redaction."""

from __future__ import annotations

from reliability.checks.base import CheckContext
from reliability.checks.provider_call import ProviderCallCheck
from reliability.checks.registry import build_checks
from reliability.config import AppConfig
from reliability.evaluation.evaluator import Evaluator
from reliability.evaluation.outcome import CallOutcome
from reliability.models import CheckResult, Scenario, Status
from reliability.providers.base import AuthenticationError, ProviderResponse
from reliability.reporting.redaction import Redactor
from tests.conftest import make_scenario, resp

FACT = [{"id": "w", "phrases": ["30 days"], "mandatory": True}]


def evaluator(ctx: CheckContext, config: AppConfig, values: list[str] | None = None) -> Evaluator:
    return Evaluator(build_checks(ctx), ProviderCallCheck(config), Redactor(values or []))


def ok(text: str) -> CallOutcome:
    return CallOutcome(resp(text), None, 1)


def test_failed_call_marks_every_check_error(ctx: CheckContext, config: AppConfig) -> None:
    scenario = make_scenario(required_facts=FACT, checks=["required_facts", "provider_call"])
    outcome = CallOutcome(None, AuthenticationError("401"), 1, stop_reason="non-retryable error")
    result = evaluator(ctx, config).evaluate(scenario, outcome)
    assert [c.status for c in result.checks] == [Status.ERROR, Status.ERROR]
    assert result.status is Status.ERROR
    assert result.error is not None and result.error["kind"] == "authentication"


class _Crashing:
    name = "required_facts"

    def applies_to(self, scenario: Scenario) -> bool:
        return True

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        raise ZeroDivisionError("boom")


def test_crashing_check_is_error_never_pass(config: AppConfig) -> None:
    ev = Evaluator({"required_facts": _Crashing()}, ProviderCallCheck(config), Redactor())
    result = ev.evaluate(make_scenario(required_facts=FACT), ok("30 days"))
    assert result.checks[0].status is Status.ERROR
    assert "ZeroDivisionError" in result.checks[0].explanation


def test_unavailable_check_is_error(config: AppConfig) -> None:
    ev = Evaluator({}, ProviderCallCheck(config), Redactor())
    assert ev.evaluate(make_scenario(), ok("x")).checks[0].status is Status.ERROR


class _NeverApplies(_Crashing):
    def applies_to(self, scenario: Scenario) -> bool:
        return False


def test_non_applicable_check(config: AppConfig) -> None:
    ev = Evaluator({"required_facts": _NeverApplies()}, ProviderCallCheck(config), Redactor())
    assert ev.evaluate(make_scenario(), ok("x")).checks[0].status is Status.NOT_APPLICABLE


def test_selftest_match_and_mismatch(ctx: CheckContext, config: AppConfig) -> None:
    scenario = make_scenario(
        role="selftest", required_facts=FACT, expected_statuses={"required_facts": "FAIL"}
    )
    ev = evaluator(ctx, config)
    assert ev.evaluate(scenario, ok("no idea")).selftest_match is True
    mismatch = ev.evaluate(scenario, ok("30 days"))
    assert mismatch.selftest_match is False
    assert mismatch.selftest_mismatches == ["required_facts: expected FAIL, observed PASS"]


def test_selftest_undeclared_applicable_check_is_mismatch(
    ctx: CheckContext, config: AppConfig
) -> None:
    scenario = make_scenario(
        role="selftest",
        required_facts=FACT,
        checks=["required_facts", "refusal"],
        expected_statuses={"required_facts": "PASS"},
    )
    result = evaluator(ctx, config).evaluate(scenario, ok("30 days"))
    assert result.selftest_mismatches == ["refusal: observed PASS, not declared"]


def test_gate_has_no_selftest_verdict(ctx: CheckContext, config: AppConfig) -> None:
    result = evaluator(ctx, config).evaluate(make_scenario(required_facts=FACT), ok("30 days"))
    assert result.selftest_match is None


def test_excerpt_and_evidence_are_redacted(ctx: CheckContext, config: AppConfig) -> None:
    secret = "CANARY-ZZ9QX7"
    scenario = make_scenario(forbidden_markers=[{"value": secret}], checks=["forbidden_content"])
    result = evaluator(ctx, config, [secret]).evaluate(scenario, ok(f"the code is {secret}"))
    dumped = result.model_dump_json()
    assert secret not in dumped
    assert "CANARY-****X7" in result.response_excerpt


def test_durations_recorded(ctx: CheckContext, config: AppConfig) -> None:
    result = evaluator(ctx, config).evaluate(make_scenario(required_facts=FACT), ok("30 days"))
    assert result.checks[0].duration_ms >= 0
