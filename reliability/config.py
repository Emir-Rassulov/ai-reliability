"""Configuration: weights, thresholds, limits, cost mode. TOML + environment overrides."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from reliability import PROJECT_ROOT
from reliability.errors import ConfigError

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Weights(_Cfg):
    critical: float = Field(default=10, gt=0)
    high: float = Field(default=5, gt=0)
    medium: float = Field(default=2, gt=0)
    low: float = Field(default=1, gt=0)


class ScoringConfig(_Cfg):
    weights: Weights = Weights()


class ReleaseConfig(_Cfg):
    min_pass_rate: float = Field(default=0.90, ge=0, le=1)
    max_error_rate: float = Field(default=0.05, ge=0, le=1)
    max_inconclusive_rate: float = Field(default=0.10, ge=0, le=1)
    max_p95_latency_ms: float | None = Field(default=None, gt=0)


class RetrievalConfig(_Cfg):
    k: int = Field(default=3, ge=1)
    min_precision_at_k: float = Field(default=0.0, ge=0, le=1)
    min_recall_at_k: float = Field(default=1.0, ge=0, le=1)
    min_mrr: float = Field(default=0.5, ge=0, le=1)
    min_context_recall: float = Field(default=1.0, ge=0, le=1)


class ChecksConfig(_Cfg):
    strip_code_fences: bool = False
    citation_pattern: str = r"\[(?P<id>[A-Za-z0-9][A-Za-z0-9._:-]*)\]"
    negation_window: int = Field(default=3, ge=1, le=10)
    response_excerpt_chars: int = Field(default=300, ge=0, le=5000)

    @field_validator("citation_pattern")
    @classmethod
    def _pattern_has_id_group(cls, v: str) -> str:
        try:
            compiled = re.compile(v)
        except re.error as exc:
            raise ValueError(f"citation_pattern is not a valid regex: {exc}") from exc
        if "id" not in compiled.groupindex:
            raise ValueError("citation_pattern must define a named group 'id'")
        return v


class LimitsConfig(_Cfg):
    max_scenario_file_bytes: int = Field(default=2_000_000, ge=1)
    max_document_chars: int = Field(default=20_000, ge=1)
    max_message_chars: int = Field(default=8_000, ge=1)
    max_response_chars: int = Field(default=8_000, ge=1)
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    backoff_base_s: float = Field(default=0.5, ge=0)
    backoff_max_s: float = Field(default=8.0, ge=0)
    max_retry_after_s: float = Field(default=30.0, ge=0)


class CostConfig(_Cfg):
    cost_mode: Literal["zero", "allow_paid"] = "zero"
    max_cost_usd: float | None = Field(default=None, gt=0)
    pricing_file: str | None = None


class AppConfig(_Cfg):
    scoring: ScoringConfig = ScoringConfig()
    release: ReleaseConfig = ReleaseConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    checks: ChecksConfig = ChecksConfig()
    limits: LimitsConfig = LimitsConfig()
    cost: CostConfig = CostConfig()


_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "AIREL_COST_MODE": ("cost", "cost_mode"),
    "AIREL_MAX_COST_USD": ("cost", "max_cost_usd"),
    "AIREL_REQUEST_TIMEOUT_S": ("limits", "request_timeout_s"),
    "AIREL_MAX_RETRIES": ("limits", "max_retries"),
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid TOML: {exc}") from exc


def load_config(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
    default_path: Path = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    """Load defaults, deep-merge an optional user file, then apply environment overrides."""
    data: dict[str, Any] = _read_toml(default_path) if default_path.exists() else {}
    if path is not None:
        data = _deep_merge(data, _read_toml(path))
    environ = os.environ if env is None else env
    for var, (section, key) in _ENV_OVERRIDES.items():
        raw = environ.get(var, "").strip()
        if raw:
            data = _deep_merge(data, {section: {key: raw}})
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"invalid configuration: {problems}") from exc
