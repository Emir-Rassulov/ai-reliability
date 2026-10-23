"""Release decision (§10), computed only from stored scenario results.

Rules are evaluated in order; every rule that fires is recorded, and the first one decides:

R0  evaluator self-test mismatch                          -> ERROR   (addition, see README)
R1  FAIL in a critical check, or a violated mandatory fact -> FAIL
R2  pass rate PASS/(PASS+FAIL) below min_pass_rate         -> FAIL
R3  ERROR rate above max_error_rate                        -> ERROR (infrastructure)
R4  INCONCLUSIVE/ERROR in a critical or high check,
    inconclusive rate above max_inconclusive_rate, or no
    applicable gate checks at all                          -> REVIEW_REQUIRED
R5  p95 latency above max_p95_latency_ms (optional)        -> REVIEW_REQUIRED
R6  otherwise                                              -> PASS

The pass rate excludes ERROR and INCONCLUSIVE so that an infrastructure outage reaches R3
instead of being mislabelled as an application failure by R2.
"""

from __future__ import annotations

from collections.abc import Sequence

from reliability.config import ReleaseConfig
from reliability.evaluation.scoring import gate_checks
from reliability.models import (
    Decision,
    FiredRule,
    ReleaseDecision,
    Role,
    ScenarioResult,
    Severity,
    Status,
)

_SEVERE = (Severity.CRITICAL, Severity.HIGH)


def _ref(scenario: ScenarioResult, check: str) -> str:
    return f"{scenario.scenario_id}/{check}"


def decide(
    results: Sequence[ScenarioResult],
    cfg: ReleaseConfig,
    latency_p95_ms: float | None,
) -> ReleaseDecision:
    fired: list[FiredRule] = []

    mismatched = [
        s.scenario_id for s in results if s.role is Role.SELFTEST and s.selftest_match is False
    ]
    if mismatched:
        fired.append(
            FiredRule(
                rule="R0_evaluator_selftest",
                outcome=Decision.ERROR,
                detail="evaluator self-test fixtures did not reproduce their declared verdicts: "
                + ", ".join(mismatched),
            )
        )

    pairs = gate_checks(results)
    applicable = [(s, c) for s, c in pairs if c.status is not Status.NOT_APPLICABLE]
    n = len(applicable)
    passed = sum(1 for _, c in applicable if c.status is Status.PASS)
    failed = sum(1 for _, c in applicable if c.status is Status.FAIL)
    errors = sum(1 for _, c in applicable if c.status is Status.ERROR)
    inconclusive = sum(1 for _, c in applicable if c.status is Status.INCONCLUSIVE)
    decided = passed + failed
    pass_rate = round(passed / decided, 4) if decided else None
    error_rate = round(errors / n, 4) if n else None
    inconclusive_rate = round(inconclusive / n, 4) if n else None

    critical = [
        _ref(s, c.check)
        for s, c in applicable
        if (c.status is Status.FAIL and c.severity is Severity.CRITICAL) or c.mandatory_violation
    ]
    if critical:
        fired.append(
            FiredRule(
                rule="R1_critical_or_mandatory",
                outcome=Decision.FAIL,
                detail="critical failure or violated mandatory fact/claim: " + ", ".join(critical),
            )
        )
    if pass_rate is not None and pass_rate < cfg.min_pass_rate:
        fired.append(
            FiredRule(
                rule="R2_min_pass_rate",
                outcome=Decision.FAIL,
                detail=f"pass rate {pass_rate:.2%} < min_pass_rate {cfg.min_pass_rate:.2%}",
            )
        )
    if error_rate is not None and error_rate > cfg.max_error_rate:
        fired.append(
            FiredRule(
                rule="R3_max_error_rate",
                outcome=Decision.ERROR,
                detail=f"error rate {error_rate:.2%} > max_error_rate {cfg.max_error_rate:.2%}",
            )
        )
    incomplete = [
        _ref(s, c.check)
        for s, c in applicable
        if c.status in (Status.INCONCLUSIVE, Status.ERROR) and c.severity in _SEVERE
    ]
    if incomplete:
        fired.append(
            FiredRule(
                rule="R4_incomplete_severe_check",
                outcome=Decision.REVIEW_REQUIRED,
                detail="critical/high checks INCONCLUSIVE or ERROR: " + ", ".join(incomplete),
            )
        )
    if inconclusive_rate is not None and inconclusive_rate > cfg.max_inconclusive_rate:
        fired.append(
            FiredRule(
                rule="R4_max_inconclusive_rate",
                outcome=Decision.REVIEW_REQUIRED,
                detail=f"inconclusive rate {inconclusive_rate:.2%} > "
                f"max_inconclusive_rate {cfg.max_inconclusive_rate:.2%}",
            )
        )
    if n == 0:
        fired.append(
            FiredRule(
                rule="R4_no_gate_checks",
                outcome=Decision.REVIEW_REQUIRED,
                detail="no applicable gate checks: nothing supports a PASS",
            )
        )
    limit = cfg.max_p95_latency_ms
    if limit is not None and latency_p95_ms is not None and latency_p95_ms > limit:
        fired.append(
            FiredRule(
                rule="R5_latency",
                outcome=Decision.REVIEW_REQUIRED,
                detail=f"p95 latency {latency_p95_ms:.0f} ms > max_p95_latency_ms {limit:.0f} ms",
            )
        )
    if not fired:
        fired.append(
            FiredRule(rule="R6_all_gates_passed", outcome=Decision.PASS, detail="no rule fired")
        )
    return ReleaseDecision(
        decision=fired[0].outcome,
        fired_rules=fired,
        applicable_checks=n,
        pass_rate=pass_rate,
        error_rate=error_rate,
        inconclusive_rate=inconclusive_rate,
        latency_p95_ms=latency_p95_ms,
    )
