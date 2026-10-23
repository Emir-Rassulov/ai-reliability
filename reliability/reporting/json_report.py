"""JSON report: stable key ordering so two runs can be diffed line by line."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from reliability.errors import ConfigError
from reliability.models import RunReport
from reliability.reporting.redaction import Redactor


def report_to_json(report: RunReport, redactor: Redactor) -> str:
    data = redactor.redact_obj(report.model_dump(mode="json"))
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_report(report: RunReport, report_dir: Path, redactor: Redactor) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{report.run_id}.json"
    path.write_text(report_to_json(report, redactor), encoding="utf-8")
    return path


def load_report(path: Path) -> RunReport:
    try:
        return RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"report not found: {path}") from exc
    except ValidationError as exc:
        raise ConfigError(
            f"{path} is not a valid run report: {exc.error_count()} problem(s)"
        ) from exc
