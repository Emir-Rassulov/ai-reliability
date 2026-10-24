"""Scenario validation, configuration loading, mock providers and the zero-cost guard."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from reliability.config import AppConfig, CostConfig, load_config
from reliability.errors import ConfigError, CostGuardError, ScenarioValidationError
from reliability.models import Message
from reliability.providers.base import (
    AuthenticationError,
    GenerationParams,
    MalformedResponse,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from reliability.providers.cost_guard import (
    BudgetExceeded,
    BudgetGuard,
    CostTier,
    ModelPrice,
    check_cost_guard,
    classify_model,
    load_pricing,
    upper_bound_input_tokens,
)
from reliability.providers.factory import build_provider
from reliability.providers.mock_provider import (
    MockEntry,
    MockProvider,
    StatefulMockProvider,
    load_mock_responses,
)
from reliability.scenarios import (
    check_mock_coverage,
    load_documents,
    load_scenarios,
)
from tests.conftest import BASE_SCENARIO, DATASET, make_scenario, write_json

# ----------------------------------------------------------------------------- scenario model


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_scenario(surprise=True)


def test_selftest_requires_expected_statuses() -> None:
    with pytest.raises(ValidationError, match="expected_statuses"):
        make_scenario(role="selftest")


def test_gate_must_not_declare_expected_statuses() -> None:
    with pytest.raises(ValidationError, match="gate scenarios"):
        make_scenario(expected_statuses={"required_facts": "PASS"})


def test_expected_statuses_must_name_listed_checks() -> None:
    with pytest.raises(ValidationError, match="not listed"):
        make_scenario(role="selftest", expected_statuses={"refusal": "PASS"})


def test_invalid_regex_marker_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid regex"):
        make_scenario(forbidden_markers=[{"value": "(", "kind": "regex"}])


def test_invalid_json_schema_rejected() -> None:
    with pytest.raises(ValidationError, match="not a valid JSON Schema"):
        make_scenario(output_schema={"type": 12})


def test_last_message_must_be_user() -> None:
    with pytest.raises(ValidationError, match="role 'user'"):
        make_scenario(input={"messages": [{"role": "assistant", "content": "hi"}]})


def test_duplicate_fact_ids_rejected() -> None:
    fact = {"id": "a", "phrases": ["x"]}
    with pytest.raises(ValidationError, match="duplicate"):
        make_scenario(required_facts=[fact, fact])


def _scenario_file(
    tmp_path: Path, scenarios: list[dict[str, Any]], name: str = "cases.json"
) -> Path:
    data = {"schema_version": "1.0", "synthetic": True, "scenarios": scenarios}
    return write_json(tmp_path / name, data)


def test_loader_error_names_file_scenario_and_field(
    tmp_path: Path, config: AppConfig, documents: Any
) -> None:
    bad = {**BASE_SCENARIO, "id": "bad-1", "severity": "extreme"}
    _scenario_file(tmp_path, [bad])
    with pytest.raises(ScenarioValidationError) as err:
        load_scenarios(tmp_path, documents, config.limits)
    assert len(err.value.problems) == 1
    assert err.value.problems[0].startswith("cases.json: scenario 'bad-1' field 'severity': ")
    assert "Input should be" in err.value.problems[0]


def test_loader_semantic_checks(tmp_path: Path, config: AppConfig, documents: Any) -> None:
    s1 = {**BASE_SCENARIO, "id": "a", "checks": ["telepathy"]}
    s2 = {**BASE_SCENARIO, "id": "b", "retrieved": [{"doc_id": "nope", "rank": 1}]}
    s3 = {**BASE_SCENARIO, "id": "a"}
    _scenario_file(tmp_path, [s1, s2, s3])
    with pytest.raises(ScenarioValidationError) as err:
        load_scenarios(tmp_path, documents, config.limits)
    joined = "\n".join(err.value.problems)
    assert "unknown check 'telepathy'" in joined
    assert "unknown document 'nope'" in joined
    assert "duplicates an id" in joined


def test_loader_rejects_invalid_json_and_oversize(tmp_path: Path, documents: Any) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    cfg = AppConfig()
    with pytest.raises(ScenarioValidationError, match="invalid JSON"):
        load_scenarios(tmp_path, documents, cfg.limits)
    small = load_config(env={}).limits.model_copy(update={"max_scenario_file_bytes": 5})
    (tmp_path / "broken.json").unlink()
    _scenario_file(tmp_path, [BASE_SCENARIO])
    with pytest.raises(ScenarioValidationError, match="max_scenario_file_bytes"):
        load_scenarios(tmp_path, documents, small)


def test_loader_missing_path(tmp_path: Path, documents: Any, config: AppConfig) -> None:
    with pytest.raises(ScenarioValidationError, match="not found"):
        load_scenarios(tmp_path / "missing", documents, config.limits)
    with pytest.raises(ScenarioValidationError, match="no scenario files"):
        load_scenarios(tmp_path, documents, config.limits)


def test_documents_validation(tmp_path: Path, config: AppConfig) -> None:
    doc = {"doc_id": "d", "title": "t", "text": "x" * 50, "topic": "t", "version": 1}
    path = write_json(
        tmp_path / "docs.json",
        {"schema_version": "1.0", "synthetic": True, "documents": [doc, doc]},
    )
    with pytest.raises(ScenarioValidationError, match="duplicated"):
        load_documents(path, config.limits)
    tiny = config.limits.model_copy(update={"max_document_chars": 10})
    write_json(path, {"schema_version": "1.0", "synthetic": True, "documents": [doc]})
    with pytest.raises(ScenarioValidationError, match="max_document_chars"):
        load_documents(path, tiny)
    write_json(path, {"schema_version": "1.0", "synthetic": False, "documents": [doc]})
    with pytest.raises(ScenarioValidationError, match="field 'synthetic'"):
        load_documents(path, config.limits)
    with pytest.raises(ScenarioValidationError, match="not found"):
        load_documents(tmp_path / "missing.json", config.limits)


def test_mock_coverage() -> None:
    with pytest.raises(ScenarioValidationError, match="no canned response for scenario 't-001'"):
        check_mock_coverage([make_scenario()], [])


def test_real_dataset_documents_load(config: AppConfig) -> None:
    assert len(load_documents(DATASET, config.limits)) >= 15


# ----------------------------------------------------------------------------- config


def test_default_config_values(config: AppConfig) -> None:
    assert config.cost.cost_mode == "zero"
    assert config.release.min_pass_rate == 0.90
    assert config.scoring.weights.critical == 10
    assert config.limits.request_timeout_s == 30.0


def test_env_overrides_and_user_file(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text("[release]\nmin_pass_rate = 0.5\n", encoding="utf-8")
    cfg = load_config(user, env={"AIREL_MAX_RETRIES": "1", "AIREL_COST_MODE": "allow_paid"})
    assert cfg.release.min_pass_rate == 0.5
    assert cfg.release.max_error_rate == 0.05  # deep merge keeps defaults
    assert cfg.limits.max_retries == 1
    assert cfg.cost.cost_mode == "allow_paid"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[release]\nmin_pass_rate = 7\n", "min_pass_rate"),
        ("[checks]\ncitation_pattern = '(['\n", "not a valid regex"),
        ("[checks]\ncitation_pattern = '\\\\[x\\\\]'\n", "named group 'id'"),
        ("[unknown]\nx = 1\n", "unknown"),
        ("this is = = not toml", "not valid TOML"),
    ],
)
def test_invalid_config(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "c.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path, env={})


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml", env={})


def test_config_without_default_file(tmp_path: Path) -> None:
    assert load_config(env={}, default_path=tmp_path / "absent.toml") == AppConfig()


# ----------------------------------------------------------------------------- mock providers


def _req(sid: str, session: str | None = None, text: str = "hi") -> ProviderRequest:
    return ProviderRequest(
        scenario_id=sid,
        system="",
        messages=[Message(role="user", content=text)],
        session_id=session,
    )


def test_mock_returns_canned_text() -> None:
    provider = MockProvider({"a": MockEntry(text="hello", latency_ms=12)})
    response = provider.generate(_req("a"))
    assert response.text == "hello"
    assert response.latency_ms == 12
    assert response.latency_source == "simulated"
    assert response.is_simulated


@pytest.mark.parametrize(
    ("kind", "exc"),
    [
        ("timeout", ProviderTimeout),
        ("rate_limited", RateLimited),
        ("unavailable", ProviderUnavailable),
        ("auth", AuthenticationError),
        ("malformed", MalformedResponse),
    ],
)
def test_mock_simulates_errors(kind: Any, exc: type[Exception]) -> None:
    provider = MockProvider({"a": MockEntry(error=kind, retry_after=2)})
    with pytest.raises(exc):
        provider.generate(_req("a"))


def test_mock_fail_times_then_succeeds() -> None:
    provider = MockProvider({"a": MockEntry(error="unavailable", fail_times=2, text="ok")})
    for _ in range(2):
        with pytest.raises(ProviderUnavailable):
            provider.generate(_req("a"))
    assert provider.generate(_req("a")).text == "ok"
    assert provider.attempts_for("a") == 3


def test_retryability_of_error_classes() -> None:
    assert ProviderTimeout.retryable and RateLimited.retryable and ProviderUnavailable.retryable
    assert not AuthenticationError.retryable and not MalformedResponse.retryable


def test_mock_missing_response_is_config_error() -> None:
    with pytest.raises(ConfigError, match="no canned mock response"):
        MockProvider({}).generate(_req("zzz"))


def test_echo_history_requires_stateful_provider() -> None:
    with pytest.raises(MalformedResponse):
        MockProvider({"a": MockEntry(mode="echo_history")}).generate(_req("a"))


def test_stateful_mock_isolates_sessions() -> None:
    provider = StatefulMockProvider(
        {"a": MockEntry(text="noted"), "b": MockEntry(mode="echo_history")}
    )
    provider.generate(_req("a", session="s1", text="my card is 4111"))
    leaked = provider.generate(_req("b", session="s1", text="what was said?"))
    assert "my card is 4111" in leaked.text
    isolated = provider.generate(_req("b", session="s2"))
    assert isolated.text == "There is no earlier conversation in this session."
    no_session = provider.generate(_req("b"))
    assert "no earlier conversation" in no_session.text


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"mode": "echo_history", "text": "x"},
        {"fail_times": 1, "text": "x"},
        {"error": "timeout", "text": "x"},
    ],
)
def test_mock_entry_shape_validation(entry: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        MockEntry.model_validate(entry)


def test_load_mock_responses_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_mock_responses(tmp_path / "x.json")
    bad = tmp_path / "m.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_mock_responses(bad)
    bad.write_text(
        json.dumps({"schema_version": "1.0", "synthetic": True, "responses": {"a": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"responses\.a"):
        load_mock_responses(bad)


# ----------------------------------------------------------------------------- zero-cost guard


@pytest.mark.parametrize(
    ("provider", "model", "api_base", "tier"),
    [
        ("mock", None, None, CostTier.MOCK),
        ("litellm", "ollama/llama3.2", None, CostTier.LOCAL),
        ("litellm", "ollama_chat/qwen2.5", "http://127.0.0.1:11434", CostTier.LOCAL),
        ("litellm", "ollama/llama3.2", "https://ollama.example.com", CostTier.PAID),
        ("litellm", "gpt-4o", None, CostTier.PAID),
        ("litellm", "some-unknown-model", None, CostTier.PAID),
        ("litellm", None, None, CostTier.PAID),
        ("somevendor", "ollama/x", None, CostTier.PAID),
    ],
)
def test_classify_model(
    provider: str, model: str | None, api_base: str | None, tier: CostTier
) -> None:
    assert classify_model(provider, model, api_base) is tier


def _spy_sockets(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    opened: list[Any] = []
    real_init = socket.socket.__init__

    def spy(self: socket.socket, *a: Any, **k: Any) -> None:
        opened.append(a)
        real_init(self, *a, **k)

    monkeypatch.setattr(socket.socket, "__init__", spy)
    return opened


def test_paid_model_rejected_before_any_socket(
    monkeypatch: pytest.MonkeyPatch, config: AppConfig, tmp_path: Path
) -> None:
    opened = _spy_sockets(monkeypatch)
    with pytest.raises(CostGuardError, match="classified as PAID"):
        build_provider(
            provider="litellm",
            model="gpt-4o",
            api_base=None,
            allow_network=True,
            allow_paid=False,
            config=config,
            mock_responses_path=tmp_path / "unused.json",
        )
    assert opened == []


def test_paid_requires_network_even_when_unlocked(config: AppConfig) -> None:
    with pytest.raises(CostGuardError, match="allow-network"):
        check_cost_guard(
            provider="litellm",
            model="gpt-4o",
            api_base=None,
            allow_network=False,
            allow_paid=True,
            config=config,
        )


def test_paid_unlocked_by_config_mode() -> None:
    cfg = AppConfig(cost=CostConfig(cost_mode="allow_paid"))
    tier = check_cost_guard(
        provider="litellm",
        model="gpt-4o",
        api_base=None,
        allow_network=True,
        allow_paid=False,
        config=cfg,
    )
    assert tier is CostTier.PAID


def test_local_requires_network_flag(config: AppConfig) -> None:
    with pytest.raises(ConfigError, match="allow-network"):
        check_cost_guard(
            provider="litellm",
            model="ollama/x",
            api_base=None,
            allow_network=False,
            allow_paid=False,
            config=config,
        )
    tier = check_cost_guard(
        provider="litellm",
        model="ollama/x",
        api_base=None,
        allow_network=True,
        allow_paid=False,
        config=config,
    )
    assert tier is CostTier.LOCAL


def test_mock_passes_guard_and_builds(config: AppConfig) -> None:
    from reliability import PROJECT_ROOT

    provider = build_provider(
        provider="mock",
        model=None,
        api_base=None,
        allow_network=False,
        allow_paid=False,
        config=config,
        mock_responses_path=PROJECT_ROOT / "scenarios" / "mock_responses.json",
    )
    assert isinstance(provider, StatefulMockProvider)


def test_litellm_without_extra_gives_actionable_error(
    config: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The [live] extra is never installed in CI; requesting it must fail clearly (exit 4)."""
    import builtins

    real_import = builtins.__import__

    def no_litellm(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_litellm)
    monkeypatch.delitem(__import__("sys").modules, "litellm", raising=False)
    with pytest.raises(ConfigError, match=r'pip install -e "\.\[live\]"'):
        build_provider(
            provider="litellm",
            model="ollama/llama3.2",
            api_base=None,
            allow_network=True,
            allow_paid=False,
            config=config,
            mock_responses_path=tmp_path / "x.json",
        )


def test_unknown_provider(config: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown provider"):
        build_provider(
            provider="acme",
            model="x",
            api_base=None,
            allow_network=True,
            allow_paid=True,
            config=config,
            mock_responses_path=tmp_path / "x.json",
        )


class _FakePaid:
    name = "fake-paid"
    model: str | None = "fake-model"

    def __init__(self, usage: Any = None) -> None:
        self.calls = 0
        self.usage = usage

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(text="ok", provider=self.name, usage=self.usage)


def _budget_req(chars: int = 1000, max_tokens: int = 500) -> ProviderRequest:
    return ProviderRequest(
        scenario_id="x",
        system="",
        messages=[Message(role="user", content="a" * chars)],
        params=GenerationParams(max_tokens=max_tokens),
    )


def test_upper_bound_tokens_counts_utf8_bytes() -> None:
    req = _budget_req(chars=10)
    assert upper_bound_input_tokens(req) == 10
    multibyte = ProviderRequest(
        scenario_id="x", system="é", messages=[Message(role="user", content="€")]
    )
    assert upper_bound_input_tokens(multibyte) == 2 + 3


def test_budget_stops_run_before_exceeding() -> None:
    price = ModelPrice(input_per_1k_usd=1.0, output_per_1k_usd=2.0)
    inner = _FakePaid()
    guard = BudgetGuard(inner, price, max_cost_usd=4.0)
    # worst case per request: 1000 bytes -> 1.0 USD + 500 tokens -> 1.0 USD = 2.0 USD
    guard.generate(_budget_req())
    guard.generate(_budget_req())
    with pytest.raises(BudgetExceeded, match="budget stop"):
        guard.generate(_budget_req())
    assert inner.calls == 2
    assert guard.spent_usd == pytest.approx(4.0)


def test_budget_uses_reported_usage_when_available() -> None:
    from reliability.models import TokenUsage

    price = ModelPrice(input_per_1k_usd=1.0, output_per_1k_usd=1.0)
    guard = BudgetGuard(_FakePaid(TokenUsage(input_tokens=100, output_tokens=100)), price, 10.0)
    guard.generate(_budget_req())
    assert guard.spent_usd == pytest.approx(0.2)


def test_load_pricing(tmp_path: Path) -> None:
    path = tmp_path / "pricing.toml"
    path.write_text(
        '[models."gpt-4o"]\ninput_per_1k_usd = 0.1\noutput_per_1k_usd = 0.2\n', encoding="utf-8"
    )
    assert load_pricing(path)["gpt-4o"].output_per_1k_usd == 0.2
    path.write_text(
        '[models."x"]\ninput_per_1k_usd = -1\noutput_per_1k_usd = 0\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="invalid pricing"):
        load_pricing(path)
    with pytest.raises(ConfigError, match="not found"):
        load_pricing(tmp_path / "none.toml")
