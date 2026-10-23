"""Provider protocol, request/response models, and the provider error hierarchy."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from reliability.models import Message, TokenUsage


class ContextDocument(BaseModel):
    """A retrieved document passed to the model. Always untrusted data, never instructions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    title: str
    text: str
    rank: int


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = 0.0
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=30.0, gt=0)


class ProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    system: str
    messages: list[Message]
    documents: list[ContextDocument] = Field(default_factory=list)
    context: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)
    params: GenerationParams = GenerationParams()
    session_id: str | None = None


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    provider: str
    model: str | None = None
    latency_ms: float | None = None
    latency_source: Literal["simulated", "measured"] = "measured"
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    #: True for canned/mock output: the evaluator, not a model, is under test.
    is_simulated: bool = False


class ProviderError(Exception):
    """Base class. `retryable` decides whether the retry loop may try again."""

    retryable = False
    kind = "provider_error"


class ProviderTimeout(ProviderError):
    retryable = True
    kind = "timeout"


class RateLimited(ProviderError):
    retryable = True
    kind = "rate_limited"

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    retryable = True
    kind = "unavailable"


class AuthenticationError(ProviderError):
    kind = "authentication"


class MalformedResponse(ProviderError):
    kind = "malformed_response"


@runtime_checkable
class Provider(Protocol):
    name: str
    model: str | None

    def generate(self, request: ProviderRequest) -> ProviderResponse: ...
