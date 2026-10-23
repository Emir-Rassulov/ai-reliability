"""Compact run summaries appended to a JSONL file for trends (no evidence, no secrets)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from reliability.models import RunReport


def history_entry(report: RunReport, commit_sha: str | None = None) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "timestamp": report.started_at,
        "commit": commit_sha if commit_sha is not None else os.environ.get("GITHUB_SHA"),
        "target": report.target,
        "provider": report.provider,
        "decision": report.decision.decision.value,
        "fired_rules": [r.rule for r in report.decision.fired_rules],
        "counts": report.counts,
        "pass_rate": report.decision.pass_rate,
        "score": report.score,
        "selftest": {"matched": report.selftest.matched, "total": report.selftest.total},
        "latency_p95_ms": report.latency.p95_ms,
        "rag_metrics": report.rag_metrics,
    }


def append_history(report: RunReport, path: Path, commit_sha: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_entry(report, commit_sha), sort_keys=True) + "\n")
