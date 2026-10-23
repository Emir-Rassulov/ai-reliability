"""Errors that map to CLI exit code 4 (configuration / validation problems)."""

from __future__ import annotations


class ConfigError(Exception):
    """Invalid configuration, arguments, or a refused provider selection."""


class CostGuardError(ConfigError):
    """A paid model was requested without the explicit unlock, or a budget was exceeded."""


class ScenarioValidationError(ConfigError):
    """One or more scenario / dataset files are invalid."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("\n".join(problems))
