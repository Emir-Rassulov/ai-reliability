"""Shared fixtures. The offline suite must never open a network connection."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reliability import PROJECT_ROOT
from reliability.checks.base import CheckContext
from reliability.config import AppConfig, load_config
from reliability.models import Document, Scenario
from reliability.providers.base import ProviderResponse
from reliability.scenarios import load_documents

SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
DATASET = PROJECT_ROOT / "datasets" / "synthetic_documents.json"


class NetworkAccessError(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """Fail any test that tries to open a socket connection (§8 control)."""
    attempts: list[Any] = []

    def guard(*args: Any, **kwargs: Any) -> None:
        attempts.append(args)
        raise NetworkAccessError(f"network access attempted during offline tests: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)
    yield attempts


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return load_config(env={})


@pytest.fixture(scope="session")
def documents(config: AppConfig) -> dict[str, Document]:
    return load_documents(DATASET, config.limits)


@pytest.fixture(scope="session")
def ctx(config: AppConfig, documents: dict[str, Document]) -> CheckContext:
    return CheckContext(config=config, documents=documents)


BASE_SCENARIO: dict[str, Any] = {
    "id": "t-001",
    "name": "test scenario",
    "category": "llm",
    "severity": "high",
    "input": {"messages": [{"role": "user", "content": "question?"}]},
    "checks": ["required_facts"],
}


def make_scenario(**overrides: Any) -> Scenario:
    data = {**BASE_SCENARIO, **overrides}
    return Scenario.model_validate(data)


def resp(text: str, **kw: Any) -> ProviderResponse:
    return ProviderResponse(text=text, provider="mock", is_simulated=True, **kw)


def write_json(path: Path, data: Any) -> Path:
    import json

    path.write_text(json.dumps(data), encoding="utf-8")
    return path
