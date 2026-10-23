"""Retrieval metrics against `relevant_doc_ids` labels.

* A duplicated id counts once, at its best (lowest) rank; `duplicate_count` reports extras.
* "Top K" = unique documents whose best rank is <= K (ranks as returned by the system;
  gaps in the ranking are kept, not compacted).
* Precision@K = relevant in top K / K (K, not the number returned).
* Recall@K   = relevant in top K / total relevant.
* MRR        = 1 / best rank of the first relevant document in the whole list, 0 if none.
* Empty retrieval: all metrics 0, flag `empty_retrieval`.
* No relevance labels (`relevant_doc_ids` is null or empty): NOT_APPLICABLE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reliability.checks.base import BaseCheck
from reliability.models import CheckResult, RetrievedRef, Scenario, Status
from reliability.providers.base import ProviderResponse


def dedupe(retrieved: Sequence[RetrievedRef]) -> tuple[list[RetrievedRef], int]:
    best: dict[str, RetrievedRef] = {}
    for ref in retrieved:
        if ref.doc_id not in best or ref.rank < best[ref.doc_id].rank:
            best[ref.doc_id] = ref
    unique = sorted(best.values(), key=lambda r: (r.rank, r.doc_id))
    return unique, len(retrieved) - len(unique)


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    duplicate_count: int
    relevant_in_top_k: int
    relevant_total: int
    empty_retrieval: bool


def compute_metrics(
    retrieved: Sequence[RetrievedRef], relevant: Sequence[str], k: int
) -> RetrievalMetrics:
    unique, duplicates = dedupe(retrieved)
    rel = set(relevant)
    if not unique:
        return RetrievalMetrics(k, 0.0, 0.0, 0.0, duplicates, 0, len(rel), True)
    hits = sum(1 for r in unique if r.rank <= k and r.doc_id in rel)
    first = next((r.rank for r in unique if r.doc_id in rel), None)
    return RetrievalMetrics(
        k=k,
        precision_at_k=round(hits / k, 4),
        recall_at_k=round(hits / len(rel), 4) if rel else 0.0,
        mrr=round(1 / first, 4) if first else 0.0,
        duplicate_count=duplicates,
        relevant_in_top_k=hits,
        relevant_total=len(rel),
        empty_retrieval=False,
    )


class RetrievalMetricsCheck(BaseCheck):
    name = "retrieval_metrics"
    method = "precision@K, recall@K, MRR against relevance labels"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        if not scenario.relevant_doc_ids:
            return self.not_applicable(scenario, "no relevance labels: metrics are not computed")
        cfg = self.ctx.config.retrieval
        t = scenario.retrieval_thresholds
        k = (t.k if t and t.k else None) or cfg.k
        min_p = t.min_precision_at_k if t and t.min_precision_at_k is not None else None
        min_r = t.min_recall_at_k if t and t.min_recall_at_k is not None else None
        min_m = t.min_mrr if t and t.min_mrr is not None else None
        thresholds = {
            "precision_at_k": cfg.min_precision_at_k if min_p is None else min_p,
            "recall_at_k": cfg.min_recall_at_k if min_r is None else min_r,
            "mrr": cfg.min_mrr if min_m is None else min_m,
        }
        m = compute_metrics(scenario.retrieved, scenario.relevant_doc_ids, k)
        values = {"precision_at_k": m.precision_at_k, "recall_at_k": m.recall_at_k, "mrr": m.mrr}
        below = [
            {"metric": name, "value": values[name], "threshold": limit}
            for name, limit in thresholds.items()
            if values[name] < limit
        ]
        flags = ["empty_retrieval"] if m.empty_retrieval else []
        if m.duplicate_count:
            flags.append("duplicates")
        metrics: dict[str, float | int | bool | None] = {
            **values,
            "k": k,
            "duplicate_count": m.duplicate_count,
            "relevant_in_top_k": m.relevant_in_top_k,
            "relevant_total": m.relevant_total,
            "empty_retrieval": m.empty_retrieval,
        }
        if below:
            names = ", ".join(str(b["metric"]) for b in below)
            return self.result(
                scenario,
                Status.FAIL,
                f"below threshold: {names}",
                expected=thresholds,
                evidence=below,
                metrics=metrics,
                flags=flags,
            )
        return self.result(
            scenario,
            Status.PASS,
            "retrieval metrics meet thresholds",
            expected=thresholds,
            metrics=metrics,
            flags=flags,
        )
