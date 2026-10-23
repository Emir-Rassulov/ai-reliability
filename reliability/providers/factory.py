"""Provider factory. Applies the cost guard before constructing anything."""

from __future__ import annotations

from pathlib import Path

from reliability.config import AppConfig
from reliability.errors import ConfigError
from reliability.providers.base import Provider
from reliability.providers.cost_guard import check_cost_guard
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
) -> Provider:
    check_cost_guard(
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
        raise ConfigError(
            "provider 'litellm' belongs to phase 7 (live provider), which is not implemented in "
            "this build. Use --provider mock. See README > Roadmap."
        )
    raise ConfigError(f"unknown provider '{provider}'; supported: {', '.join(SUPPORTED_PROVIDERS)}")
