"""Provider factory. Applies the cost guard before constructing anything."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reliability.config import AppConfig
from reliability.errors import ConfigError
from reliability.providers.base import Provider
from reliability.providers.cost_guard import BudgetGuard, CostTier, check_cost_guard, load_pricing
from reliability.providers.litellm_provider import LiteLLMProvider
from reliability.providers.mock_provider import StatefulMockProvider, load_mock_responses

SUPPORTED_PROVIDERS = ("mock", "litellm")


def build_provider(
    *,
    provider: str,
    model: str | None,
    api_base: str | None,
    allow_network: bool,
    allow_paid: bool,
    config: AppConfig,
    mock_responses_path: Path,
    litellm_module: Any = None,
) -> Provider:
    tier = check_cost_guard(
        provider=provider,
        model=model,
        api_base=api_base,
        allow_network=allow_network,
        allow_paid=allow_paid,
        config=config,
    )
    if provider == "mock":
        return StatefulMockProvider(load_mock_responses(mock_responses_path))
    if provider == "litellm":
        if not model:
            raise ConfigError("--provider litellm needs --model, e.g. ollama/llama3.2")
        live: Provider = LiteLLMProvider(model, api_base=api_base, module=litellm_module)
        if tier is CostTier.PAID and config.cost.max_cost_usd is not None:
            if not config.cost.pricing_file:
                raise ConfigError(
                    "max_cost_usd is set but no pricing_file is configured; prices are never "
                    "guessed, so the budget cannot be enforced"
                )
            prices = load_pricing(Path(config.cost.pricing_file))
            if model not in prices:
                raise ConfigError(f"pricing file has no entry for model '{model}'")
            live = BudgetGuard(live, prices[model], config.cost.max_cost_usd)
        return live
    raise ConfigError(f"unknown provider '{provider}'; supported: {', '.join(SUPPORTED_PROVIDERS)}")
