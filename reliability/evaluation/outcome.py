"""Outcome of one provider call including retries."""

from __future__ import annotations

from dataclasses import dataclass, field

from reliability.providers.base import ProviderError, ProviderResponse


@dataclass(frozen=True)
class CallOutcome:
    response: ProviderResponse | None
    error: ProviderError | None
    attempts: int
    waits_s: tuple[float, ...] = field(default_factory=tuple)
    truncated_by_limit: bool = False
    stop_reason: str = ""
