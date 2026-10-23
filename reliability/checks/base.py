"""Check protocol and a small base class with shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from reliability.config import AppConfig
from reliability.models import CheckResult, Document, Scenario, Status
from reliability.providers.base import ProviderResponse


@dataclass(frozen=True)
class CheckContext:
    config: AppConfig
    documents: Mapping[str, Document]


@runtime_checkable
class Check(Protocol):
    @property
    def name(self) -> str: ...

    def applies_to(self, scenario: Scenario) -> bool: ...

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult: ...


class BaseCheck(ABC):
    """Convenience base: `applies_to` = listed in the scenario's `checks`."""

    name: ClassVar[str]
    method: ClassVar[str]

    def __init__(self, ctx: CheckContext) -> None:
        self.ctx = ctx

    def applies_to(self, scenario: Scenario) -> bool:
        return self.name in scenario.checks

    @abstractmethod
    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        """Evaluate one response; implemented by every concrete check."""

    def result(
        self,
        scenario: Scenario,
        status: Status,
        explanation: str,
        *,
        expected: Any = None,
        evidence: list[dict[str, Any]] | None = None,
        metrics: dict[str, float | int | bool | None] | None = None,
        flags: list[str] | None = None,
        mandatory_violation: bool = False,
    ) -> CheckResult:
        return CheckResult(
            check=self.name,
            status=status,
            severity=scenario.severity_of(self.name),
            method=self.method,
            explanation=explanation,
            expected=expected,
            evidence=evidence or [],
            metrics=metrics or {},
            flags=flags or [],
            mandatory_violation=mandatory_violation,
        )

    def not_applicable(self, scenario: Scenario, why: str) -> CheckResult:
        return self.result(scenario, Status.NOT_APPLICABLE, why)
