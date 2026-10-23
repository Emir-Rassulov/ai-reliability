"""Failure paths of provider calls. Retries use an injectable sleep and clock: tests are instant."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from reliability.config import LimitsConfig
from reliability.evaluation.orchestration import backoff_delay, call_with_retries
from reliability.models import Message
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

LIMITS = LimitsConfig()
REQ = ProviderRequest(scenario_id="x", system="", messages=[Message(role="user", content="q")])


class Scripted:
    """Raises the scripted errors in order, then returns `text`."""

    name = "scripted"
    model: str | None = None

    def __init__(
        self, *errors: ProviderError, text: str = "ok", latency: float | None = 5.0
    ) -> None:
        self.errors = list(errors)
        self.text = text
        self.latency = latency
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return ProviderResponse(text=self.text, provider=self.name, latency_ms=self.latency)


class FakeSleep:
    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def fake_clock(*ticks: float) -> Iterator[float]:
    yield from ticks


def call(provider: Scripted, limits: LimitsConfig = LIMITS) -> tuple[object, FakeSleep]:
    sleep = FakeSleep()
    return call_with_retries(provider, REQ, limits, sleep), sleep


def test_success_first_try() -> None:
    outcome, sleep = call(Scripted())
    assert outcome.response is not None and outcome.attempts == 1  # type: ignore[attr-defined]
    assert sleep.waits == []


def test_timeout_then_success() -> None:
    outcome, sleep = call(Scripted(ProviderTimeout("t")))
    assert outcome.attempts == 2 and outcome.response.text == "ok"  # type: ignore[attr-defined]
    assert sleep.waits == [0.5]


def test_429_respects_retry_after() -> None:
    outcome, sleep = call(Scripted(RateLimited("r", retry_after=3)))
    assert sleep.waits == [3]
    assert outcome.waits_s == (3,)  # type: ignore[attr-defined]


def test_429_without_retry_after_uses_exponential_backoff() -> None:
    outcome, sleep = call(Scripted(RateLimited("r"), RateLimited("r")))
    assert sleep.waits == [0.5, 1.0]
    assert outcome.attempts == 3  # type: ignore[attr-defined]


def test_5xx_then_success() -> None:
    _, sleep = call(Scripted(ProviderUnavailable("503"), ProviderUnavailable("503")))
    assert sleep.waits == [0.5, 1.0]


@pytest.mark.parametrize("error", [AuthenticationError("401"), MalformedResponse("bad")])
def test_non_retryable_errors_stop_immediately(error: ProviderError) -> None:
    provider = Scripted(error)
    outcome, sleep = call(provider)
    assert outcome.response is None and outcome.error is error  # type: ignore[attr-defined]
    assert provider.calls == 1 and sleep.waits == []
    assert outcome.stop_reason == "non-retryable error"  # type: ignore[attr-defined]


def test_retries_exhausted() -> None:
    provider = Scripted(*[ProviderTimeout("t") for _ in range(10)])
    outcome, sleep = call(provider)
    assert provider.calls == LIMITS.max_retries + 1 == 4
    assert sleep.waits == [0.5, 1.0, 2.0]
    assert "retries exhausted" in outcome.stop_reason  # type: ignore[attr-defined]


def test_retry_after_above_limit_stops() -> None:
    outcome, sleep = call(Scripted(RateLimited("r", retry_after=120)))
    assert outcome.response is None and sleep.waits == []  # type: ignore[attr-defined]
    assert "exceeds max_retry_after_s" in outcome.stop_reason  # type: ignore[attr-defined]


def test_backoff_is_capped() -> None:
    limits = LimitsConfig(backoff_base_s=1, backoff_max_s=3)
    assert [backoff_delay(n, limits) for n in range(1, 5)] == [1, 2, 3, 3]


def test_zero_retries() -> None:
    outcome, sleep = call(Scripted(ProviderTimeout("t")), LimitsConfig(max_retries=0))
    assert outcome.attempts == 1 and outcome.response is None and sleep.waits == []  # type: ignore[attr-defined]


def test_latency_measured_with_injected_clock_when_not_reported() -> None:
    ticks = fake_clock(10.0, 10.25)
    outcome = call_with_retries(
        Scripted(latency=None), REQ, LIMITS, FakeSleep(), clock=lambda: next(ticks)
    )
    assert outcome.response is not None
    assert outcome.response.latency_ms == 250.0
    assert outcome.response.latency_source == "measured"


def test_response_longer_than_limit_is_truncated_and_flagged() -> None:
    outcome, _ = call(Scripted(text="x" * 50), LimitsConfig(max_response_chars=10))
    assert outcome.truncated_by_limit  # type: ignore[attr-defined]
    assert len(outcome.response.text) == 10  # type: ignore[attr-defined]


def test_operational_suite_end_to_end(tmp_path: object) -> None:
    from pathlib import Path

    from reliability.cli import main

    out = Path(str(tmp_path))
    assert main(["run", "--category", "operational", "--report-dir", str(out)]) == 0
