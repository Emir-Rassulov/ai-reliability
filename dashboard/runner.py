"""Bounded run launcher for the dashboard. Runs go through the existing orchestration API only.

* One run at a time per process (non-blocking lock): a public deployment cannot be flooded.
* At most `max_scenarios()` scenarios per run.
* Public mode: the offline mock provider only. Private mode additionally allows a *local*
  model through the normal factory and cost guard; paid models are never unlocked from
  the UI (`allow_paid=False` is hard-coded).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rag_app import build_reference_app
from rag_app.defects import DEFECTS
from reliability import PROJECT_ROOT
from reliability.config import load_config
from reliability.errors import ConfigError
from reliability.evaluation.mutation import mutation_test
from reliability.evaluation.orchestration import run_rag_app, run_suite
from reliability.models import RunReport
from reliability.providers.factory import build_provider
from reliability.providers.mock_provider import load_mock_responses
from reliability.reporting.history import append_history
from reliability.reporting.json_report import write_report
from reliability.reporting.redaction import Redactor
from reliability.scenarios import (
    check_mock_coverage,
    collect_sensitive_values,
    load_documents,
    load_scenarios,
    sha256_file,
)

SCENARIOS = PROJECT_ROOT / "scenarios"
DATASET = PROJECT_ROOT / "datasets" / "synthetic_documents.json"
MOCK_RESPONSES = SCENARIOS / "mock_responses.json"
_LOCK = threading.Lock()


class RunRejected(Exception):
    """The request was refused (busy, too large, or not allowed in this mode)."""


def max_scenarios(env: dict[str, str] | None = None) -> int:
    raw = (env if env is not None else os.environ).get("AIREL_UI_MAX_SCENARIOS", "60")
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


@dataclass(frozen=True)
class RunRequest:
    target: str = "scenarios"  # "scenarios" | "rag-app" | "mutation-test"
    category: str | None = None
    scenario_id: str | None = None
    defect: str | None = None
    provider: str = "mock"
    model: str | None = None


@dataclass(frozen=True)
class RunResult:
    report: RunReport
    path: Path
    mutation_path: Path | None = None


def available_scenarios() -> list[tuple[str, str, bool]]:
    """(id, category, rag_app-eligible) for the run form."""
    config = load_config(env={})
    documents = load_documents(DATASET, config.limits)
    loaded = load_scenarios(SCENARIOS, documents, config.limits)
    return [(s.id, s.category.value, s.rag_app) for s in loaded.scenarios]


def start_run(
    request: RunRequest, *, report_dir: Path, history_file: Path | None, public: bool
) -> RunResult:
    if request.provider != "mock" and public:
        raise RunRejected("public mode allows the offline mock provider only")
    if request.defect is not None and request.defect not in DEFECTS:
        raise RunRejected(f"unknown defect '{request.defect}'")
    if not _LOCK.acquire(blocking=False):
        raise RunRejected("another run is in progress; try again when it has finished")
    try:
        return _run(request, report_dir=report_dir, history_file=history_file)
    except ConfigError as exc:
        raise RunRejected(str(exc)) from exc
    finally:
        _LOCK.release()


def _no_sleep(_seconds: float) -> None:
    """The mock provider simulates time: backoff waits are recorded, not slept."""


def _run(request: RunRequest, *, report_dir: Path, history_file: Path | None) -> RunResult:
    config = load_config()
    documents = load_documents(DATASET, config.limits)
    loaded = load_scenarios(SCENARIOS, documents, config.limits)
    pool = loaded.scenarios
    if request.target in ("rag-app", "mutation-test"):
        pool = [s for s in pool if s.rag_app]
    if request.category:
        pool = [s for s in pool if s.category.value == request.category]
    if request.scenario_id:
        index = next((i for i, s in enumerate(pool) if s.id == request.scenario_id), None)
        if index is None:
            raise RunRejected("no scenarios match the selection")
        target = pool[index]
        # Cross-session fixtures depend on earlier turns of the same session (as in the CLI).
        earlier = [
            s for s in pool[:index] if target.session_id and s.session_id == target.session_id
        ]
        pool = [*earlier, target]
    if not pool:
        raise RunRejected("no scenarios match the selection")
    if len(pool) > max_scenarios():
        raise RunRejected(f"{len(pool)} scenarios exceed the UI limit of {max_scenarios()}")
    redactor = Redactor(collect_sensitive_values(loaded.scenarios))
    hashes, dataset_hash = loaded.file_hashes, sha256_file(DATASET)

    mutation_path: Path | None = None
    if request.target == "scenarios":
        provider = build_provider(
            provider=request.provider,
            model=request.model,
            api_base=None,
            allow_network=request.provider != "mock",
            allow_paid=False,
            config=config,
            mock_responses_path=MOCK_RESPONSES,
        )
        if request.provider == "mock":
            check_mock_coverage(pool, load_mock_responses(MOCK_RESPONSES))
        report = run_suite(
            scenarios=pool,
            provider=provider,
            documents=documents,
            config=config,
            redactor=redactor,
            scenario_files=hashes,
            dataset_hash=dataset_hash,
            sleep=_no_sleep if getattr(provider, "simulated_time", False) else time.sleep,
        )
    elif request.target == "rag-app":
        defects = [request.defect] if request.defect else []
        report = run_rag_app(
            scenarios=pool,
            system=build_reference_app(defects),
            documents=documents,
            config=config,
            redactor=redactor,
            defects=defects,
            scenario_files=hashes,
            dataset_hash=dataset_hash,
        )
    elif request.target == "mutation-test":
        result = mutation_test(
            scenarios=pool,
            factory=build_reference_app,
            defects=sorted(DEFECTS),
            documents=documents,
            config=config,
            redactor=redactor,
        )
        report = result.baseline
        report_dir.mkdir(parents=True, exist_ok=True)
        mutation_path = report_dir / f"mutation-{report.run_id}.json"
        mutation_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        raise RunRejected(f"unknown target '{request.target}'")
    path = write_report(report, report_dir, redactor)
    if history_file is not None:
        append_history(report, history_file)
    return RunResult(report=report, path=path, mutation_path=mutation_path)
