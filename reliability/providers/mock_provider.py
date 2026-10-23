"""Offline providers that return canned responses and simulate failures.

`MockProvider` serves `scenarios/mock_responses.json`. Each entry is either a canned text,
or an error to raise (optionally only for the first `fail_times` attempts, after which the
text is returned), with an optional simulated latency.

`StatefulMockProvider` additionally remembers user messages per `session_id`, so an entry
with `mode: "echo_history"` can model a system that leaks earlier conversation turns.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from reliability.errors import ConfigError
from reliability.models import TokenUsage
from reliability.providers.base import (
    AuthenticationError,
    MalformedResponse,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)

ErrorKind = Literal["timeout", "rate_limited", "unavailable", "auth", "malformed"]


class MockEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str | None = None
    error: ErrorKind | None = None
    retry_after: float | None = Field(default=None, ge=0)
    #: With `error`: raise only for the first N attempts, then return `text`.
    fail_times: int | None = Field(default=None, ge=1)
    latency_ms: float | None = Field(default=None, ge=0)
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    mode: Literal["static", "echo_history"] = "static"

    @model_validator(mode="after")
    def _shape(self) -> MockEntry:
        if self.mode == "echo_history":
            if self.error is not None or self.text is not None:
                raise ValueError("echo_history entries take neither text nor error")
            return self
        if self.error is None and self.text is None:
            raise ValueError("entry needs 'text' or 'error'")
        if self.fail_times is not None and (self.error is None or self.text is None):
            raise ValueError("'fail_times' requires both 'error' and 'text'")
        if self.error is not None and self.fail_times is None and self.text is not None:
            raise ValueError("an always-failing entry must not define 'text'")
        return self


class MockResponsesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    synthetic: Literal[True]
    responses: dict[str, MockEntry]


def load_mock_responses(path: Path) -> dict[str, MockEntry]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"mock responses file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    try:
        return MockResponsesFile.model_validate(raw).responses
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: {problems}") from exc


def _error_for(entry: MockEntry, scenario_id: str) -> ProviderError:
    message = f"simulated {entry.error} for scenario {scenario_id}"
    match entry.error:
        case "timeout":
            return ProviderTimeout(message)
        case "rate_limited":
            return RateLimited(message, retry_after=entry.retry_after)
        case "unavailable":
            return ProviderUnavailable(message + " (HTTP 503)")
        case "auth":
            return AuthenticationError(message + " (HTTP 401)")
        case _:
            return MalformedResponse(message)


class MockProvider:
    name = "mock"
    model: str | None = "mock-canned-v1"

    def __init__(self, responses: dict[str, MockEntry]) -> None:
        self._responses = responses
        self._attempts: defaultdict[str, int] = defaultdict(int)

    def attempts_for(self, scenario_id: str) -> int:
        return self._attempts[scenario_id]

    def _entry(self, request: ProviderRequest) -> MockEntry:
        try:
            return self._responses[request.scenario_id]
        except KeyError as exc:
            raise ConfigError(
                f"no canned mock response for scenario '{request.scenario_id}'; "
                "add it to scenarios/mock_responses.json"
            ) from exc

    def _text(self, entry: MockEntry, request: ProviderRequest) -> str:
        if entry.text is None:  # echo_history is handled by StatefulMockProvider
            raise MalformedResponse(
                f"echo_history needs StatefulMockProvider ({request.scenario_id})"
            )
        return entry.text

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        entry = self._entry(request)
        self._attempts[request.scenario_id] += 1
        attempt = self._attempts[request.scenario_id]
        if entry.error is not None and (entry.fail_times is None or attempt <= entry.fail_times):
            raise _error_for(entry, request.scenario_id)
        return ProviderResponse(
            text=self._text(entry, request),
            provider=self.name,
            model=self.model,
            latency_ms=entry.latency_ms if entry.latency_ms is not None else 0.0,
            latency_source="simulated",
            usage=entry.usage,
            finish_reason=entry.finish_reason,
            is_simulated=True,
        )


class StatefulMockProvider(MockProvider):
    """Keeps user messages per session so `echo_history` can leak earlier turns."""

    name = "mock"

    def __init__(self, responses: dict[str, MockEntry]) -> None:
        super().__init__(responses)
        self._history: defaultdict[str, list[str]] = defaultdict(list)

    def _text(self, entry: MockEntry, request: ProviderRequest) -> str:
        if entry.mode == "echo_history":
            previous = self._history[request.session_id] if request.session_id else []
            if not previous:
                return "There is no earlier conversation in this session."
            return "Earlier in this session: " + " | ".join(previous)
        return super()._text(entry, request)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        response = super().generate(request)
        if request.session_id:
            self._history[request.session_id].extend(
                m.content for m in request.messages if m.role == "user"
            )
        return response
