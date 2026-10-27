"""Zero-cost guard: classify models into cost tiers and refuse paid ones unless unlocked.

The classification is an explicit allowlist, not a guess:

* ``mock``  - the offline mock provider.
* ``local`` - a LiteLLM model name with the ``ollama/`` or ``ollama_chat/`` prefix whose API
  base resolves to a loopback host (default ``http://localhost:11434``), or the generic
  ``http`` provider pointed at a loopback URL (an app running on this machine).
* ``paid``  - everything else, including unknown providers and unknown model names.

All decisions happen before any provider object is created, so a refused model can never
open a network connection.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reliability.config import AppConfig
from reliability.errors import ConfigError, CostGuardError
from reliability.providers.base import Provider, ProviderRequest, ProviderResponse

LOCAL_MODEL_PREFIXES = ("ollama/", "ollama_chat/")
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_OLLAMA_API_BASE = "http://localhost:11434"


class CostTier(StrEnum):
    MOCK = "mock"
    LOCAL = "local"
    PAID = "paid"


def classify_model(provider: str, model: str | None, api_base: str | None = None) -> CostTier:
    if provider == "mock":
        return CostTier.MOCK
    if provider == "http":
        host = urlparse(api_base or "").hostname
        return CostTier.LOCAL if host in LOCAL_HOSTS else CostTier.PAID
    if provider == "litellm" and model and model.startswith(LOCAL_MODEL_PREFIXES):
        host = urlparse(api_base or DEFAULT_OLLAMA_API_BASE).hostname
        if host in LOCAL_HOSTS:
            return CostTier.LOCAL
    return CostTier.PAID


def check_cost_guard(
    *,
    provider: str,
    model: str | None,
    api_base: str | None,
    allow_network: bool,
    allow_paid: bool,
    config: AppConfig,
) -> CostTier:
    """Raise CostGuardError/ConfigError if the selection is not allowed. No network access."""
    tier = classify_model(provider, model, api_base)
    if tier is CostTier.PAID:
        unlocked = allow_paid or config.cost.cost_mode == "allow_paid"
        if not unlocked:
            raise CostGuardError(
                f"model '{model}' via provider '{provider}' is classified as PAID and "
                "zero-cost mode is active. To unlock it, pass --allow-paid "
                '(or set cost_mode = "allow_paid") '
                "together with --allow-network. Local models: --provider litellm "
                "--model ollama/<name> --allow-network."
            )
        if not allow_network:
            raise CostGuardError("paid models also require --allow-network")
    elif tier is CostTier.LOCAL and not allow_network:
        raise ConfigError(
            "local models and apps are served over HTTP on localhost; pass --allow-network"
        )
    return tier


# --------------------------------------------------------------------------- budget


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_1k_usd: float = Field(ge=0)
    output_per_1k_usd: float = Field(ge=0)


class PricingFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: dict[str, ModelPrice]


def load_pricing(path: Path) -> dict[str, ModelPrice]:
    """Prices come only from a user-supplied file; nothing is ever guessed."""
    try:
        with path.open("rb") as fh:
            return PricingFile.model_validate(tomllib.load(fh)).models
    except FileNotFoundError as exc:
        raise ConfigError(f"pricing file not found: {path}") from exc
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError(f"invalid pricing file {path}: {exc}") from exc


def upper_bound_input_tokens(request: ProviderRequest) -> int:
    """Conservative token bound: byte-level tokenizers never emit more tokens than UTF-8 bytes."""
    parts = [request.system, request.context or ""]
    parts += [m.content for m in request.messages]
    parts += [d.title + d.text for d in request.documents]
    parts += list(request.fields.values())
    return sum(len(p.encode("utf-8")) for p in parts)


class BudgetExceeded(CostGuardError):
    pass


class BudgetGuard:
    """Wraps a provider and stops before a request could push spend above the budget."""

    def __init__(self, inner: Provider, price: ModelPrice, max_cost_usd: float) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.price = price
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1000) * self.price.input_per_1k_usd + (
            output_tokens / 1000
        ) * self.price.output_per_1k_usd

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        estimate = self._cost(upper_bound_input_tokens(request), request.params.max_tokens)
        if self.spent_usd + estimate > self.max_cost_usd:
            raise BudgetExceeded(
                f"budget stop: worst-case cost {estimate:.4f} USD would exceed the remaining "
                f"budget {self.max_cost_usd - self.spent_usd:.4f} USD "
                f"(max_cost_usd={self.max_cost_usd})"
            )
        response = self.inner.generate(request)
        usage = response.usage
        if usage and usage.input_tokens is not None and usage.output_tokens is not None:
            self.spent_usd += self._cost(usage.input_tokens, usage.output_tokens)
        else:
            self.spent_usd += estimate
        return response
