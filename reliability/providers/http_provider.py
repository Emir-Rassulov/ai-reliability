"""Generic HTTP provider: test any chatbot or AI app that exposes a JSON-over-HTTP API.

The application is a black box. A small TOML "target" file says where to send the request,
what the JSON body looks like and where the answer is in the JSON response:

    [http]
    url = "http://127.0.0.1:8765/chat"
    response_text_path = "reply"           # dotted path; list indexes allowed: "choices.0.text"

    [http.body]                            # any JSON shape; strings may use placeholders
    message = "{last_user_message}"
    session_id = "{session_id}"

    [http.headers]
    "Content-Type" = "application/json"

    [http.secret_headers]                  # values come from environment variables only
    "Authorization" = "MY_BOT_TOKEN"

Placeholders: {last_user_message}, {system}, {session_id}, {scenario_id}, {context}
(reference context plus retrieved documents as plain text). A string that is exactly one
placeholder is replaced; placeholders inside longer strings are substituted as text.

Sessions: a scenario without `session_id` gets a fresh session on every call, so earlier
questions cannot influence the answer through the app's memory. Scenarios that share a
`session_id` share one app session within a run; a random per-run suffix keeps a rerun
from seeing the previous run's conversation.

Only the standard library is used (no extra dependency). The transport is injectable, so
tests never open a socket. Error mapping: 401/403 -> AuthenticationError, 429 ->
RateLimited (honouring Retry-After), 5xx and connection failures -> ProviderUnavailable,
timeouts -> ProviderTimeout, other 4xx / non-JSON / missing answer -> MalformedResponse.
"""

from __future__ import annotations

import itertools
import json
import os
import secrets
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from reliability.errors import ConfigError
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

PLACEHOLDERS = ("last_user_message", "system", "session_id", "scenario_id", "context")
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
MAX_RESPONSE_BYTES = 2_000_000


class HttpTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    method: Literal["POST", "PUT"] = "POST"
    name: str = "http-app"
    body: dict[str, Any] = Field(default_factory=lambda: {"message": "{last_user_message}"})
    response_text_path: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    #: header name -> environment variable that holds the value (keeps secrets out of files)
    secret_headers: dict[str, str] = Field(default_factory=dict)
    #: prefix of the session ids sent to the app
    session_prefix: str = "airel"

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("url must be an absolute http:// or https:// URL")
        return v

    @property
    def is_loopback(self) -> bool:
        return urlparse(self.url).hostname in LOOPBACK_HOSTS


def load_target(path: Path) -> HttpTargetConfig:
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"target config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"target config {path} is not valid TOML: {exc}") from exc
    if not isinstance(raw.get("http"), dict):
        raise ConfigError(f"target config {path} needs an [http] section")
    try:
        return HttpTargetConfig.model_validate(raw["http"])
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"invalid target config {path}: {problems}") from exc


# --------------------------------------------------------------------------- request body


def _values(request: ProviderRequest, session: str) -> dict[str, str]:
    context_parts = [request.context] if request.context else []
    context_parts += [f"[{d.doc_id}] {d.title}\n{d.text}" for d in request.documents]
    return {
        "last_user_message": request.messages[-1].content,
        "system": request.system,
        "session_id": session,
        "scenario_id": request.scenario_id,
        "context": "\n\n".join(context_parts),
    }


def render_body(template: Any, values: Mapping[str, str]) -> Any:
    """Fill placeholders in every string of a JSON-like template. Values are never evaluated."""
    if isinstance(template, str):
        for key in PLACEHOLDERS:
            if template == "{" + key + "}":
                return values[key]
        for key in PLACEHOLDERS:
            template = template.replace("{" + key + "}", values[key])
        return template
    if isinstance(template, dict):
        return {k: render_body(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render_body(v, values) for v in template]
    return template


def extract(data: Any, path: str) -> Any:
    """Follow a dotted path through dicts and lists ("choices.0.message.content")."""
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise MalformedResponse(f"response has no field '{path}' (stopped at '{part}')")
    return current


# --------------------------------------------------------------------------- transport


@dataclass(frozen=True)
class HttpReply:
    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, str, dict[str, str], bytes, float], HttpReply]


def urllib_transport(
    url: str, method: str, headers: dict[str, str], body: bytes, timeout: float
) -> HttpReply:
    """Standard-library transport. Loopback URLs bypass any configured HTTP proxy."""
    handlers: list[urllib.request.BaseHandler] = []
    if urlparse(url).hostname in LOOPBACK_HOSTS:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310 - scheme validated in HttpTargetConfig
    try:
        with opener.open(req, timeout=timeout) as resp:
            return HttpReply(resp.status, dict(resp.headers), resp.read(MAX_RESPONSE_BYTES + 1))
    except urllib.error.HTTPError as exc:
        return HttpReply(exc.code, dict(exc.headers or {}), exc.read(MAX_RESPONSE_BYTES + 1))
    except TimeoutError as exc:
        raise ProviderTimeout(f"no response within {timeout:g}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise ProviderTimeout(f"no response within {timeout:g}s") from exc
        raise ProviderUnavailable(f"cannot reach {url}: {exc.reason}") from exc


def _retry_after(headers: Mapping[str, str]) -> float | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return float(value)
            except ValueError:
                return None
    return None


def map_status(reply: HttpReply) -> ProviderError | None:
    status = reply.status
    if 200 <= status < 300:
        return None
    if status in (401, 403):
        return AuthenticationError(f"HTTP {status}")
    if status == 429:
        return RateLimited("HTTP 429", retry_after=_retry_after(reply.headers))
    if status in (408, 504):
        return ProviderTimeout(f"HTTP {status}")
    if status >= 500:
        return ProviderUnavailable(f"HTTP {status}")
    return MalformedResponse(f"HTTP {status}")


class HttpProvider:
    simulated_time = False

    def __init__(
        self,
        target: HttpTargetConfig,
        *,
        transport: Transport = urllib_transport,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.target = target
        self.name = "http"
        self.model: str | None = target.name
        self.transport = transport
        self.clock = clock
        self._run_tag = secrets.token_hex(4)
        self._calls = itertools.count(1)
        environ = os.environ if env is None else env
        self.headers = {"Content-Type": "application/json", **target.headers}
        for header, var in target.secret_headers.items():
            value = environ.get(var, "")
            if not value:
                raise ConfigError(
                    f"target config expects environment variable {var} for header '{header}'"
                )
            self.headers[header] = value

    def session_for(self, request: ProviderRequest) -> str:
        prefix = f"{self.target.session_prefix}-{self._run_tag}"
        if request.session_id:
            return f"{prefix}-{request.session_id}"
        return f"{prefix}-{request.scenario_id}-{next(self._calls)}"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        body = render_body(self.target.body, _values(request, self.session_for(request)))
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = self.clock()
        reply = self.transport(
            self.target.url,
            self.target.method,
            dict(self.headers),
            payload,
            request.params.timeout_s,
        )
        latency = round((self.clock() - started) * 1000, 3)
        error = map_status(reply)
        if error is not None:
            raise error
        if len(reply.body) > MAX_RESPONSE_BYTES:
            raise MalformedResponse(f"response larger than {MAX_RESPONSE_BYTES} bytes")
        try:
            data = json.loads(reply.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedResponse(f"response is not JSON: {exc}") from exc
        text = extract(data, self.target.response_text_path)
        if not isinstance(text, str):
            raise MalformedResponse(
                f"'{self.target.response_text_path}' is {type(text).__name__}, not text"
            )
        return ProviderResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency,
            latency_source="measured",
            is_simulated=False,
        )
