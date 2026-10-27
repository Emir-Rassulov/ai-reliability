"""Generic HTTP provider: config, body templates, response paths, errors, sessions, cost tier.

No socket is opened: every test injects a fake transport.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

import pytest

from reliability.config import AppConfig
from reliability.errors import ConfigError, CostGuardError
from reliability.models import Message
from reliability.providers.base import (
    AuthenticationError,
    ContextDocument,
    MalformedResponse,
    ProviderRequest,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from reliability.providers.cost_guard import CostTier, classify_model
from reliability.providers.factory import build_provider
from reliability.providers.http_provider import (
    HttpProvider,
    HttpReply,
    HttpTargetConfig,
    extract,
    load_target,
    map_status,
    render_body,
    urllib_transport,
)

TARGET = HttpTargetConfig(
    url="http://127.0.0.1:8765/chat",
    response_text_path="reply",
    body={"message": "{last_user_message}", "session_id": "{session_id}"},
)


def req(text: str = "hello", *, session: str | None = None, sid: str = "s-1") -> ProviderRequest:
    return ProviderRequest(
        scenario_id=sid,
        system="sys",
        messages=[Message(role="user", content="earlier"), Message(role="user", content=text)],
        session_id=session,
    )


class FakeTransport:
    def __init__(self, *replies: HttpReply) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, method: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> HttpReply:
        self.calls.append(
            {
                "url": url,
                "method": method,
                "headers": headers,
                "body": json.loads(body),
                "timeout": timeout,
            }
        )
        return self.replies.pop(0)


def ok(data: Any) -> HttpReply:
    return HttpReply(200, {}, json.dumps(data).encode())


# ----------------------------------------------------------------------------- config


def test_load_target_and_errors(tmp_path: Path) -> None:
    good = tmp_path / "t.toml"
    good.write_text(
        '[http]\nurl = "https://bot.example/api"\nresponse_text_path = "a.b"\n', encoding="utf-8"
    )
    assert load_target(good).url == "https://bot.example/api"
    cases = {
        "missing.toml": "not found",
        "bad.toml": "not valid TOML",
        "nosection.toml": r"needs an \[http\] section",
        "badurl.toml": "absolute http",
        "extra.toml": "Extra inputs",
    }
    contents = {
        "bad.toml": "[http\n",
        "nosection.toml": "x = 1\n",
        "badurl.toml": '[http]\nurl = "ftp://x"\nresponse_text_path = "a"\n',
        "extra.toml": '[http]\nurl = "http://x"\nresponse_text_path = "a"\nsurprise = 1\n',
    }
    for name, message in cases.items():
        path = tmp_path / name
        if name in contents:
            path.write_text(contents[name], encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            load_target(path)


def test_example_target_file_is_valid() -> None:
    from reliability import PROJECT_ROOT

    target = load_target(PROJECT_ROOT / "examples" / "demo_bot" / "target.toml")
    assert target.is_loopback and target.response_text_path == "reply"


# ----------------------------------------------------------------------------- templates and paths


def test_render_body_placeholders() -> None:
    values = {
        "last_user_message": "q",
        "system": "s",
        "session_id": "x",
        "scenario_id": "id",
        "context": "c",
    }
    template = {
        "m": "{last_user_message}",
        "nested": [{"t": "id={scenario_id}; s={session_id}"}],
        "n": 3,
        "b": True,
    }
    assert render_body(template, values) == {
        "m": "q",
        "nested": [{"t": "id=id; s=x"}],
        "n": 3,
        "b": True,
    }
    assert render_body("{unknown}", values) == "{unknown}"  # never evaluated


def test_context_placeholder_includes_documents() -> None:
    transport = FakeTransport(ok({"reply": "ok"}))
    target = TARGET.model_copy(update={"body": {"q": "{last_user_message}", "ctx": "{context}"}})
    request = req().model_copy(
        update={
            "context": "note",
            "documents": [ContextDocument(doc_id="d1", title="T", text="body", rank=1)],
        }
    )
    HttpProvider(target, transport=transport).generate(request)
    assert transport.calls[0]["body"]["ctx"] == "note\n\n[d1] T\nbody"


def test_extract_paths() -> None:
    data = {"choices": [{"message": {"content": "hi"}}]}
    assert extract(data, "choices.0.message.content") == "hi"
    for bad in ("choices.1.message", "choices.x", "nope"):
        with pytest.raises(MalformedResponse, match="has no field"):
            extract(data, bad)


# ----------------------------------------------------------------------------- generate


def test_generate_sends_last_message_and_parses_reply() -> None:
    ticks = iter([1.0, 1.05])
    transport = FakeTransport(ok({"reply": "Answer"}))
    provider = HttpProvider(TARGET, transport=transport, clock=lambda: next(ticks))
    response = provider.generate(req("question?"))
    call = transport.calls[0]
    assert call["url"] == TARGET.url and call["method"] == "POST" and call["timeout"] == 30.0
    assert call["body"]["message"] == "question?"
    assert call["headers"]["Content-Type"] == "application/json"
    assert response.text == "Answer" and response.latency_ms == 50.0
    assert (
        response.provider == "http" and response.model == "http-app" and not response.is_simulated
    )


def test_sessions_are_fresh_or_shared_by_scenario_session() -> None:
    transport = FakeTransport(*[ok({"reply": "x"}) for _ in range(4)])
    provider = HttpProvider(TARGET, transport=transport)
    provider.generate(req(sid="a"))
    provider.generate(req(sid="a"))
    provider.generate(req(session="cust-1"))
    provider.generate(req(session="cust-1"))
    sessions = [c["body"]["session_id"] for c in transport.calls]
    assert sessions[0] != sessions[1]  # no session: a fresh one per call
    assert sessions[2] == sessions[3] and sessions[2].endswith("-cust-1")
    other_run = HttpProvider(TARGET, transport=FakeTransport(ok({"reply": "x"})))
    assert other_run.session_for(req(session="cust-1")) != sessions[2]  # reruns do not share memory


def test_secret_headers_come_from_environment() -> None:
    target = TARGET.model_copy(
        update={"secret_headers": {"Authorization": "BOT_TOKEN"}, "headers": {"X-App": "1"}}
    )
    transport = FakeTransport(ok({"reply": "x"}))
    HttpProvider(target, transport=transport, env={"BOT_TOKEN": "Bearer s3cr3t"}).generate(req())
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer s3cr3t"
    assert transport.calls[0]["headers"]["X-App"] == "1"
    with pytest.raises(ConfigError, match="BOT_TOKEN"):
        HttpProvider(target, transport=transport, env={})


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (HttpReply(401, {}, b""), AuthenticationError),
        (HttpReply(403, {}, b""), AuthenticationError),
        (HttpReply(429, {}, b""), RateLimited),
        (HttpReply(408, {}, b""), ProviderTimeout),
        (HttpReply(504, {}, b""), ProviderTimeout),
        (HttpReply(502, {}, b""), ProviderUnavailable),
        (HttpReply(400, {}, b""), MalformedResponse),
        (HttpReply(200, {}, b"not json"), MalformedResponse),
        (HttpReply(200, {}, b"\xff\xfe"), MalformedResponse),
        (ok({"other": "x"}), MalformedResponse),
        (ok({"reply": ["list"]}), MalformedResponse),
        (HttpReply(200, {}, b"x" * 2_000_001), MalformedResponse),
    ],
)
def test_error_mapping(reply: HttpReply, error: type[Exception]) -> None:
    with pytest.raises(error):
        HttpProvider(TARGET, transport=FakeTransport(reply)).generate(req())


def test_retry_after_header() -> None:
    mapped = map_status(HttpReply(429, {"Retry-After": "4"}, b""))
    assert isinstance(mapped, RateLimited) and mapped.retry_after == 4.0
    assert map_status(HttpReply(429, {"retry-after": "soon"}, b"")).retry_after is None  # type: ignore[union-attr]
    assert map_status(HttpReply(204, {}, b"")) is None


class _FakeOpener:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.handlers: tuple[Any, ...] = ()

    def open(self, req: Any, timeout: float) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Resp:
    status = 200
    headers: ClassVar[dict[str, str]] = {"X": "1"}

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def read(self, _n: int) -> bytes:
        return b'{"reply": "ok"}'


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (urllib.error.URLError(ConnectionRefusedError(61, "refused")), ProviderUnavailable),
        (urllib.error.URLError(TimeoutError("slow")), ProviderTimeout),
        (TimeoutError("slow"), ProviderTimeout),
    ],
)
def test_urllib_transport_maps_network_errors(
    monkeypatch: pytest.MonkeyPatch, outcome: BaseException, expected: type[Exception]
) -> None:
    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _FakeOpener(outcome))
    with pytest.raises(expected):
        urllib_transport("http://127.0.0.1:9/chat", "POST", {}, b"{}", 1.0)


def test_urllib_transport_success_http_error_and_proxy_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Any, ...]] = []

    def build(*handlers: Any) -> _FakeOpener:
        seen.append(handlers)
        return _FakeOpener(_Resp())

    monkeypatch.setattr(urllib.request, "build_opener", build)
    reply = urllib_transport("http://127.0.0.1:8765/chat", "POST", {}, b"{}", 1.0)
    assert reply.status == 200 and reply.body == b'{"reply": "ok"}'
    assert isinstance(seen[0][0], urllib.request.ProxyHandler)  # loopback skips proxies
    urllib_transport("https://bot.example.com/chat", "POST", {}, b"{}", 1.0)
    assert seen[1] == ()  # remote hosts keep the user's proxy settings

    error = urllib.error.HTTPError(
        "http://x", 503, "busy", {"Retry-After": "2"}, io.BytesIO(b"down")
    )  # type: ignore[arg-type]
    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: _FakeOpener(error))
    reply = urllib_transport("http://127.0.0.1:8765/chat", "POST", {}, b"{}", 1.0)
    assert (reply.status, reply.body) == (503, b"down")


# ----------------------------------------------------------------------------- factory + cost tier


@pytest.mark.parametrize(
    ("url", "tier"),
    [
        ("http://127.0.0.1:8765/chat", CostTier.LOCAL),
        ("http://localhost/x", CostTier.LOCAL),
        ("https://bot.example.com/api", CostTier.PAID),
        (None, CostTier.PAID),
    ],
)
def test_http_cost_tier(url: str | None, tier: CostTier) -> None:
    assert classify_model("http", "x", url) is tier


def _target_file(tmp_path: Path, url: str) -> Path:
    path = tmp_path / "target.toml"
    path.write_text(f'[http]\nurl = "{url}"\nresponse_text_path = "reply"\n', encoding="utf-8")
    return path


def test_factory_http(tmp_path: Path, config: AppConfig) -> None:
    common: dict[str, Any] = {
        "provider": "http",
        "model": None,
        "api_base": None,
        "config": config,
        "mock_responses_path": tmp_path / "x.json",
    }
    local = _target_file(tmp_path, "http://127.0.0.1:8765/chat")
    provider = build_provider(**common, allow_network=True, allow_paid=False, target_config=local)
    assert isinstance(provider, HttpProvider)
    with pytest.raises(ConfigError, match="allow-network"):
        build_provider(**common, allow_network=False, allow_paid=False, target_config=local)
    with pytest.raises(ConfigError, match="needs --target-config"):
        build_provider(**common, allow_network=True, allow_paid=False)
    remote = _target_file(tmp_path, "https://bot.example.com/api")
    with pytest.raises(CostGuardError, match="PAID"):
        build_provider(**common, allow_network=True, allow_paid=False, target_config=remote)
    assert isinstance(
        build_provider(**common, allow_network=True, allow_paid=True, target_config=remote),
        HttpProvider,
    )
