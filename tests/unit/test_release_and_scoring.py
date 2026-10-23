"""Every release rule (§10) has an explicit test; score and aggregates are hand-checked."""

from __future__ import annotations

from reliability.config import ReleaseConfig, Weights
from reliability.evaluation.release import decide
from reliability.evaluation.scoring import (
    latency_stats,
    nearest_rank,
    rag_metric_means,
    status_counts,
    total_usage,
    weighted_score,
)
from reliability.models import (
    Category,
    CheckResult,
    Decision,
    Role,
    ScenarioResult,
    Severity,
    Status,
    TokenUsage,
    aggregate_status,
)

CFG = ReleaseConfig()


def chk(status: Status, severity: Severity = Severity.MEDIUM, **kw: object) -> CheckResult:
    return CheckResult(
        check=str(kw.pop("check", "required_facts")),
        status=status,
        severity=severity,
        method="t",
        explanation="t",
        **kw,  # type: ignore[arg-type]
    )


def scen(
    checks: list[CheckResult],
    *,
    sid: str = "s",
    role: Role = Role.GATE,
    match: bool | None = None,
    latency: float | None = None,
    usage: TokenUsage | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        name=sid,
        category=Category.LLM,
        severity=Severity.MEDIUM,
        role=role,
        tags=[],
        status=aggregate_status([c.status for c in checks]),
        checks=checks,
        latency_ms=latency,
        latency_source="simulated" if latency is not None else None,
        usage=usage,
        selftest_match=match,
    )


def passes(n: int) -> list[CheckResult]:
    return [chk(Status.PASS) for _ in range(n)]


def rules(decision: object) -> list[str]:
    return [r.rule for r in decision.fired_rules]  # type: ignore[attr-defined]


def test_r6_all_pass() -> None:
    d = decide([scen(passes(5))], CFG, None)
    assert d.decision is Decision.PASS
    assert rules(d) == ["R6_all_gates_passed"]
    assert d.pass_rate == 1.0


def test_r1_critical_fail_cannot_be_averaged_away() -> None:
    checks = [*passes(99), chk(Status.FAIL, Severity.CRITICAL)]
    d = decide([scen(checks)], CFG, None)
    assert d.decision is Decision.FAIL
    assert rules(d)[0] == "R1_critical_or_mandatory"
    assert d.pass_rate == 0.99  # above min_pass_rate, still FAIL


def test_r1_mandatory_violation_on_low_severity() -> None:
    checks = [*passes(50), chk(Status.FAIL, Severity.LOW, mandatory_violation=True)]
    assert decide([scen(checks)], CFG, None).decision is Decision.FAIL


def test_r2_min_pass_rate() -> None:
    checks = [*passes(8), chk(Status.FAIL), chk(Status.FAIL)]
    d = decide([scen(checks)], CFG, None)
    assert d.decision is Decision.FAIL
    assert rules(d) == ["R2_min_pass_rate"]
    assert d.pass_rate == 0.8


def test_r2_boundary_exactly_at_threshold_passes() -> None:
    checks = [*passes(9), chk(Status.FAIL)]
    assert decide([scen(checks)], CFG, None).decision is Decision.PASS


def test_r3_error_rate_is_run_error_not_fail() -> None:
    checks = [*passes(8), chk(Status.ERROR, Severity.LOW), chk(Status.ERROR, Severity.LOW)]
    d = decide([scen(checks)], CFG, None)
    assert d.decision is Decision.ERROR
    assert rules(d)[0] == "R3_max_error_rate"
    assert d.pass_rate == 1.0  # errors excluded from the pass rate


def test_r4_incomplete_critical_check_never_passes() -> None:
    checks = [*passes(40), chk(Status.INCONCLUSIVE, Severity.CRITICAL)]
    d = decide([scen(checks)], CFG, None)
    assert d.decision is Decision.REVIEW_REQUIRED
    assert rules(d) == ["R4_incomplete_severe_check"]


def test_r4_error_in_high_check() -> None:
    checks = [*passes(40), chk(Status.ERROR, Severity.HIGH)]
    assert decide([scen(checks)], CFG, None).decision is Decision.REVIEW_REQUIRED


def test_r4_inconclusive_rate() -> None:
    checks = [
        *passes(8),
        chk(Status.INCONCLUSIVE, Severity.LOW),
        chk(Status.INCONCLUSIVE, Severity.LOW),
    ]
    d = decide([scen(checks)], CFG, None)
    assert rules(d) == ["R4_max_inconclusive_rate"]


def test_r4_low_inconclusive_below_rate_passes() -> None:
    checks = [*passes(19), chk(Status.INCONCLUSIVE, Severity.LOW)]
    assert decide([scen(checks)], CFG, None).decision is Decision.PASS


def test_r4_no_gate_checks_is_never_pass() -> None:
    na = [chk(Status.NOT_APPLICABLE)]
    d = decide([scen(na)], CFG, None)
    assert d.decision is Decision.REVIEW_REQUIRED
    assert rules(d) == ["R4_no_gate_checks"]
    assert decide([], CFG, None).decision is Decision.REVIEW_REQUIRED


def test_r5_latency() -> None:
    cfg = ReleaseConfig(max_p95_latency_ms=1000)
    assert rules(decide([scen(passes(3))], cfg, 1500.0)) == ["R5_latency"]
    assert decide([scen(passes(3))], cfg, 900.0).decision is Decision.PASS
    assert decide([scen(passes(3))], CFG, 99999.0).decision is Decision.PASS  # disabled


def test_r0_selftest_mismatch_is_run_error() -> None:
    results = [scen(passes(3)), scen([chk(Status.PASS)], sid="st", role=Role.SELFTEST, match=False)]
    d = decide(results, CFG, None)
    assert d.decision is Decision.ERROR
    assert rules(d) == ["R0_evaluator_selftest"]


def test_selftest_results_do_not_count_toward_gate() -> None:
    results = [
        scen(passes(3)),
        scen([chk(Status.FAIL, Severity.CRITICAL)], sid="st", role=Role.SELFTEST, match=True),
    ]
    d = decide(results, CFG, None)
    assert d.decision is Decision.PASS
    assert d.applicable_checks == 3


def test_rule_order_first_fired_decides() -> None:
    checks = [chk(Status.FAIL, Severity.CRITICAL), chk(Status.ERROR, Severity.HIGH), *passes(2)]
    d = decide([scen(checks)], CFG, None)
    assert d.decision is Decision.FAIL
    assert rules(d) == [
        "R1_critical_or_mandatory",
        "R2_min_pass_rate",
        "R3_max_error_rate",
        "R4_incomplete_severe_check",
    ]


# ----------------------------------------------------------------------------- scoring


def test_weighted_score_hand_calculated() -> None:
    checks = [
        chk(Status.PASS, Severity.CRITICAL),  # 10
        chk(Status.FAIL, Severity.HIGH),  # 5
        chk(Status.PASS, Severity.LOW),  # 1
        chk(Status.INCONCLUSIVE, Severity.CRITICAL),  # excluded
        chk(Status.ERROR, Severity.HIGH),  # excluded
        chk(Status.NOT_APPLICABLE, Severity.HIGH),  # excluded
    ]
    assert weighted_score([scen(checks)], Weights()) == round(11 / 16, 4)


def test_weighted_score_none_when_nothing_decided() -> None:
    assert weighted_score([scen([chk(Status.ERROR)])], Weights()) is None


def test_status_counts_cover_all_statuses() -> None:
    counts = status_counts([scen([chk(Status.PASS), chk(Status.FAIL)])])
    assert counts == {"PASS": 1, "FAIL": 1, "INCONCLUSIVE": 0, "ERROR": 0, "NOT_APPLICABLE": 0}


def test_nearest_rank_percentile() -> None:
    assert nearest_rank([], 95) is None
    assert nearest_rank([5.0], 95) == 5.0
    assert nearest_rank([float(i) for i in range(1, 21)], 95) == 19.0
    assert nearest_rank([float(i) for i in range(1, 101)], 95) == 95.0


def test_latency_stats_gate_only() -> None:
    results = [
        scen(passes(1), sid="a", latency=100.0),
        scen(passes(1), sid="b", latency=300.0),
        scen(passes(1), sid="c", role=Role.SELFTEST, match=True, latency=9999.0),
        scen(passes(1), sid="d"),
    ]
    stats = latency_stats(results)
    assert stats.count == 2
    assert stats.median_ms == 200.0
    assert stats.p95_ms == 300.0
    assert stats.sources == ["simulated"]


def test_total_usage_only_when_returned() -> None:
    assert total_usage([scen(passes(1))]) is None
    usage = total_usage(
        [
            scen(passes(1), usage=TokenUsage(input_tokens=10, output_tokens=5)),
            scen(passes(1), usage=TokenUsage(input_tokens=7)),
        ]
    )
    assert usage == TokenUsage(input_tokens=17, output_tokens=5)


def test_rag_metric_means() -> None:
    checks = [
        chk(Status.PASS, check="retrieval_metrics", metrics={"mrr": 1.0, "recall_at_k": 1.0}),
        chk(Status.FAIL, check="retrieval_metrics", metrics={"mrr": 0.5, "recall_at_k": None}),
    ]
    means = rag_metric_means([scen(checks)])
    assert means["retrieval_metrics.mrr"] == 0.75
    assert means["retrieval_metrics.recall_at_k"] == 1.0
    assert means["context_quality.context_recall"] is None


def test_aggregate_status_precedence() -> None:
    assert aggregate_status([Status.PASS, Status.ERROR, Status.FAIL]) is Status.FAIL
    assert aggregate_status([Status.PASS, Status.INCONCLUSIVE]) is Status.INCONCLUSIVE
    assert aggregate_status([Status.NOT_APPLICABLE]) is Status.NOT_APPLICABLE
    assert aggregate_status([]) is Status.NOT_APPLICABLE
