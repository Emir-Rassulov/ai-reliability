"""Helpers to compute and (deliberately, by hand) refresh the regression snapshot.

Refreshing the snapshot is a reviewed action, not something tests do automatically:
    python -m tests.regression.outcomes --write
Review the diff before committing: every changed verdict needs a reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reliability import PROJECT_ROOT
from reliability.config import load_config
from reliability.evaluation.orchestration import run_suite
from reliability.models import RunReport
from reliability.providers.mock_provider import StatefulMockProvider, load_mock_responses
from reliability.reporting.redaction import Redactor
from reliability.scenarios import collect_sensitive_values, load_documents, load_scenarios

EXPECTED = Path(__file__).with_name("expected_outcomes.json")
SCENARIOS = PROJECT_ROOT / "scenarios"
DATASET = PROJECT_ROOT / "datasets" / "synthetic_documents.json"


def run_full_suite() -> RunReport:
    config = load_config(env={})
    documents = load_documents(DATASET, config.limits)
    loaded = load_scenarios(SCENARIOS, documents, config.limits)
    provider = StatefulMockProvider(load_mock_responses(SCENARIOS / "mock_responses.json"))
    return run_suite(
        scenarios=loaded.scenarios,
        provider=provider,
        documents=documents,
        config=config,
        redactor=Redactor(collect_sensitive_values(loaded.scenarios)),
        sleep=lambda _s: None,
        run_id="regression",
    )


def outcomes(report: RunReport) -> dict[str, Any]:
    return {
        "decision": report.decision.decision.value,
        "fired_rules": [r.rule for r in report.decision.fired_rules],
        "scenarios": {
            s.scenario_id: {c.check: c.status.value for c in s.checks} for s in report.scenarios
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = outcomes(run_full_suite())
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.write:
        EXPECTED.write_text(text, encoding="utf-8")
    print(text)
