"""Live provider through LiteLLM (phase 7). Local Ollama by default; hosted only when unlocked.

The `litellm` package is an optional extra (`pip install -e ".[live]"`) and is imported only
when this provider is constructed. The core never imports it.

LiteLLM's own retries are disabled (`num_retries=0`) so that retries stay bounded and visible
in the evaluator's retry loop; LiteLLM exceptions are mapped onto the evaluator's error
hierarchy by class name, which keeps the mapping testable without the package installed.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Callable
from typing import Any

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
from reliability.providers.prompting import render_messages

#: LiteLLM exception class name -> evaluator error class. Checked in this order (subclasses
#: first), using the exception's MRO, so e.g. a Timeout subclass of APIConnectionError maps
#: to ProviderTimeout.
ERROR_MAP: tuple[tuple[str, type[ProviderError]], ...] = (
    ("Timeout", ProviderTimeout),
    ("RateLimitError", RateLimited),
    ("AuthenticationError", AuthenticationError),
    ("PermissionDeniedError", AuthenticationError),
    ("ServiceUnavailableError", ProviderUnavailable),
    ("InternalServerError", ProviderUnavailable),
    ("APIConnectionError", ProviderUnavailable),
    ("BadRequestError", MalformedResponse),
    ("NotFoundError", MalformedResponse),
)

INSTALL_HINT = 'the live provider needs the optional extra: pip install -e ".[live]"'


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def map_error(exc: BaseException) -> ProviderError:
    names = [cls.__name__ for cls in type(exc).__mro__]
    message = f"{type(exc).__name__}: {str(exc)[:300]}"
    for name, target in ERROR_MAP:
        if name in names:
            if target is RateLimited:
                return RateLimited(message, retry_after=_retry_after(exc))
            return target(message)
    return ProviderUnavailable(f"unexpected provider exception {message}")


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class LiteLLMProvider:
    name = "litellm"
    simulated_time = False

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        module: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if module is None:
            # LiteLLM downloads a model price list from GitHub when it is imported. That is an
            # unrequested network call (found by the offline test's socket guard), so the
            # bundled local copy is forced unless the user configured this explicitly.
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            try:
                module = importlib.import_module("litellm")
            except ImportError as exc:
                raise ConfigError(INSTALL_HINT) from exc
        self._litellm = module
        self.model: str | None = model
        self.api_base = api_base
        self.clock = clock

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": render_messages(request),
            "temperature": request.params.temperature,
            "max_tokens": request.params.max_tokens,
            "timeout": request.params.timeout_s,
            "num_retries": 0,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        started = self.clock()
        try:
            raw = self._litellm.completion(**kwargs)
        # Third-party exceptions are mapped by class name; anything unknown becomes a
        # retryable "unavailable" error rather than crashing the run or counting as a pass.
        except Exception as exc:
            raise map_error(exc) from exc
        latency = round((self.clock() - started) * 1000, 3)
        choices = _field(raw, "choices")
        if not choices:
            raise MalformedResponse("provider returned no choices")
        first = choices[0]
        message = _field(first, "message")
        content = _field(message, "content") if message is not None else None
        if content is not None and not isinstance(content, str):
            raise MalformedResponse(f"message content is {type(content).__name__}, not text")
        usage_raw = _field(raw, "usage")
        usage = None
        if usage_raw is not None:
            prompt_tokens = _field(usage_raw, "prompt_tokens")
            completion_tokens = _field(usage_raw, "completion_tokens")
            if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
                usage = TokenUsage(
                    input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                    output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                )
        model = _field(raw, "model")
        finish = _field(first, "finish_reason")
        return ProviderResponse(
            text=content or "",
            provider=self.name,
            model=model if isinstance(model, str) else self.model,
            latency_ms=latency,
            latency_source="measured",
            usage=usage,
            finish_reason=finish if isinstance(finish, str) else None,
            is_simulated=False,
        )
