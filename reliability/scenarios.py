"""Load and validate scenario files and the synthetic document set.

Error messages name the file, scenario ID, field and problem. Validation runs before any
execution; the whole suite is rejected if anything is wrong (exit code 4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reliability.checks.registry import KNOWN_CHECKS
from reliability.config import LimitsConfig
from reliability.errors import ScenarioValidationError
from reliability.models import Document, DocumentSet, Scenario, ScenarioFile

MOCK_RESPONSES_FILE = "mock_responses.json"


@dataclass(frozen=True)
class LoadedScenarios:
    scenarios: list[Scenario]
    file_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, limits: LimitsConfig, problems: list[str]) -> Any:
    size = path.stat().st_size
    if size > limits.max_scenario_file_bytes:
        problems.append(
            f"{path.name}: file is {size} bytes, above max_scenario_file_bytes "
            f"{limits.max_scenario_file_bytes}"
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{path.name}: invalid JSON at line {exc.lineno}: {exc.msg}")
        return None


def _format_errors(file: str, raw: Any, exc: ValidationError) -> list[str]:
    out: list[str] = []
    for err in exc.errors():
        loc = list(err["loc"])
        where = ""
        if len(loc) >= 2 and loc[0] in ("scenarios", "documents") and isinstance(loc[1], int):
            items = raw.get(loc[0], []) if isinstance(raw, dict) else []
            item = items[loc[1]] if loc[1] < len(items) else {}
            key = "id" if loc[0] == "scenarios" else "doc_id"
            ident = item.get(key, f"#{loc[1]}") if isinstance(item, dict) else f"#{loc[1]}"
            label = "scenario" if loc[0] == "scenarios" else "document"
            where = f" {label} '{ident}'"
            loc = loc[2:]
        field = ".".join(str(p) for p in loc) or "<root>"
        out.append(f"{file}:{where} field '{field}': {err['msg']}")
    return out


def load_documents(path: Path, limits: LimitsConfig) -> dict[str, Document]:
    problems: list[str] = []
    raw = _read_json(path, limits, problems) if path.exists() else None
    if not path.exists():
        problems.append(f"{path}: document set not found")
    if problems:
        raise ScenarioValidationError(problems)
    try:
        doc_set = DocumentSet.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioValidationError(_format_errors(path.name, raw, exc)) from exc
    docs: dict[str, Document] = {}
    for doc in doc_set.documents:
        if doc.doc_id in docs:
            problems.append(f"{path.name}: document '{doc.doc_id}' is duplicated")
        if len(doc.text) > limits.max_document_chars:
            problems.append(
                f"{path.name}: document '{doc.doc_id}' text exceeds max_document_chars "
                f"{limits.max_document_chars}"
            )
        docs[doc.doc_id] = doc
    if problems:
        raise ScenarioValidationError(problems)
    return docs


def scenario_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ScenarioValidationError([f"{path}: scenario path not found"])
    files = sorted(p for p in path.glob("*.json") if p.name != MOCK_RESPONSES_FILE)
    if not files:
        raise ScenarioValidationError([f"{path}: no scenario files (*.json) found"])
    return files


def _semantic_problems(
    file: str, scenario: Scenario, documents: Mapping[str, Document], limits: LimitsConfig
) -> list[str]:
    prefix = f"{file}: scenario '{scenario.id}'"
    problems: list[str] = []
    for name in scenario.checks:
        if name not in KNOWN_CHECKS:
            problems.append(f"{prefix} field 'checks': unknown check '{name}'")
    for msg in scenario.input.messages:
        if len(msg.content) > limits.max_message_chars:
            problems.append(f"{prefix} field 'input.messages': message exceeds max_message_chars")
    referenced: list[tuple[str, str]] = [("retrieved", r.doc_id) for r in scenario.retrieved]
    referenced += [("relevant_doc_ids", d) for d in scenario.relevant_doc_ids or []]
    referenced += [("required_evidence_ids", d) for d in scenario.required_evidence_ids]
    referenced += [
        ("expected_claims.supported_by", d)
        for c in scenario.expected_claims
        for d in c.supported_by
    ]
    referenced += [("conflicts", d) for pair in scenario.conflicts for d in pair.doc_ids]
    for field, doc_id in referenced:
        if doc_id not in documents:
            problems.append(f"{prefix} field '{field}': unknown document '{doc_id}'")
    return problems


def load_scenarios(
    path: Path, documents: Mapping[str, Document], limits: LimitsConfig
) -> LoadedScenarios:
    problems: list[str] = []
    scenarios: list[Scenario] = []
    hashes: dict[str, str] = {}
    seen: dict[str, str] = {}
    for file in scenario_files(path):
        raw = _read_json(file, limits, problems)
        if raw is None:
            continue
        hashes[file.name] = sha256_file(file)
        try:
            parsed = ScenarioFile.model_validate(raw)
        except ValidationError as exc:
            problems.extend(_format_errors(file.name, raw, exc))
            continue
        for scenario in parsed.scenarios:
            if scenario.id in seen:
                problems.append(
                    f"{file.name}: scenario '{scenario.id}' duplicates an id "
                    f"from {seen[scenario.id]}"
                )
            seen[scenario.id] = file.name
            problems.extend(_semantic_problems(file.name, scenario, documents, limits))
            scenarios.append(scenario)
    if problems:
        raise ScenarioValidationError(problems)
    return LoadedScenarios(scenarios=scenarios, file_hashes=hashes)


def check_mock_coverage(scenarios: Iterable[Scenario], mock_ids: Iterable[str]) -> None:
    missing = sorted({s.id for s in scenarios} - set(mock_ids))
    if missing:
        raise ScenarioValidationError(
            [f"{MOCK_RESPONSES_FILE}: no canned response for scenario '{sid}'" for sid in missing]
        )


def collect_sensitive_values(scenarios: Iterable[Scenario]) -> list[str]:
    values: set[str] = set()
    for scenario in scenarios:
        values.update(scenario.all_sensitive_values())
    return sorted(values)
