"""Informational score and aggregate statistics. The score never decides a release."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import median

from reliability.config import Weights
from reliability.models import (
    CheckResult,
    LatencyStats,
    Role,
    ScenarioResult,
    Status,
    TokenUsage,
)


def gate_checks(results: Iterable[ScenarioResult]) -> list[tuple[ScenarioResult, CheckResult]]:
    return [(s, c) for s in results if s.role is Role.GATE for c in s.checks]


def status_counts(results: Iterable[ScenarioResult]) -> dict[str, int]:
    counts = Counter(c.status.value for _, c in gate_checks(results))
    return {status.value: counts.get(status.value, 0) for status in Status}


def weighted_score(results: Iterable[ScenarioResult], weights: Weights) -> float | None:
    """sum(weight * passed) / sum(weight) over gate checks with PASS or FAIL."""
    num = den = 0.0
    for _, check in gate_checks(results):
        if check.status not in (Status.PASS, Status.FAIL):
            continue
        weight = float(getattr(weights, check.severity.value))
        den += weight
        if check.status is Status.PASS:
            num += weight
    return round(num / den, 4) if den else None


def nearest_rank(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def latency_stats(results: Iterable[ScenarioResult]) -> LatencyStats:
    gate = [s for s in results if s.role is Role.GATE and s.latency_ms is not None]
    values = [s.latency_ms for s in gate if s.latency_ms is not None]
    return LatencyStats(
        count=len(values),
        median_ms=round(median(values), 3) if values else None,
        p95_ms=nearest_rank(values, 95),
        sources=sorted({s.latency_source for s in gate if s.latency_source}),
    )


def total_usage(results: Iterable[ScenarioResult]) -> TokenUsage | None:
    """Sum token usage only over values a provider actually returned; None if none did."""
    inputs = [s.usage.input_tokens for s in results if s.usage and s.usage.input_tokens is not None]
    outputs = [
        s.usage.output_tokens for s in results if s.usage and s.usage.output_tokens is not None
    ]
    if not inputs and not outputs:
        return None
    return TokenUsage(
        input_tokens=sum(inputs) if inputs else None,
        output_tokens=sum(outputs) if outputs else None,
    )


RAG_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "retrieval_metrics": ("precision_at_k", "recall_at_k", "mrr"),
    "context_quality": ("context_precision", "context_recall"),
    "grounding": ("unverified_content_ratio",),
    "citations": ("citation_accuracy",),
}


def rag_metric_means(results: Iterable[ScenarioResult]) -> dict[str, float | None]:
    sums: dict[str, list[float]] = {
        f"{check}.{key}": [] for check, keys in RAG_METRIC_KEYS.items() for key in keys
    }
    for _, check in gate_checks(results):
        for key in RAG_METRIC_KEYS.get(check.check, ()):
            value = check.metrics.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                sums[f"{check.check}.{key}"].append(float(value))
    return {k: (round(sum(v) / len(v), 4) if v else None) for k, v in sums.items()}
