"""LLM adapter: no real network calls, no real keys."""
from types import SimpleNamespace

import httpx
import openai
import pytest

import llm_client as L

FAKE_KEY = "test-only-fake-key-0123456789"  # deliberately not a real key format


class FakeCompletions:
    def __init__(self, replies=None, exc=None):
        self.replies, self.exc, self.calls = list(replies or []), exc, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        text = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))


def fake_client(**kw):
    comp = FakeCompletions(**kw)
    models = SimpleNamespace(list=lambda: [SimpleNamespace(id="m-b"), SimpleNamespace(id="m-a")])
    return SimpleNamespace(chat=SimpleNamespace(completions=comp), models=models), comp


def cfg(provider="Groq", **kw):
    return L.LLMConfig(provider=provider, api_key=FAKE_KEY, **kw)


def api_error(cls, status):
    req = httpx.Request("POST", "https://example.test")
    resp = httpx.Response(status, request=req)
    return cls(f"failure mentioning {FAKE_KEY}", response=resp, body=None)


def test_all_providers_configured():
    assert set(L.PROVIDERS) == {"Groq", "OpenAI", "Google Gemini", "Anthropic Claude", "Custom (OpenAI-compatible)"}
    assert cfg().resolved_model()  # Groq has a default model
    assert cfg("Anthropic Claude").resolved_base_url() == "https://api.anthropic.com/v1/"


def test_key_never_in_repr_and_env_fallback(monkeypatch):
    assert FAKE_KEY not in repr(cfg())
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    c = L.LLMConfig(provider="Google Gemini", model="x")
    assert c.resolved_key() == "env-key" and c.is_ready()


def test_not_ready_reasons(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "No API key" in L.LLMConfig(provider="OpenAI").missing()
    assert "model" in L.LLMConfig(provider="OpenAI", api_key="k").missing().lower()
    assert "base URL" in L.LLMConfig(provider="Custom (OpenAI-compatible)", api_key="k", model="m").missing()
    with pytest.raises(L.LLMError):
        L.chat(L.LLMConfig(provider="OpenAI"), "s", "u")


def test_chat_returns_text_and_reported_usage():
    client, comp = fake_client(replies=["hello"])
    r = L.chat(cfg(), "sys", "user", client=client)
    assert r["text"] == "hello" and r["input_tokens"] == 11 and r["output_tokens"] == 7
    assert comp.calls[0]["model"] == L.PROVIDERS["Groq"]["default_model"]


def test_json_mode_only_where_supported():
    client, comp = fake_client(replies=['{"a": 1}', '{"a": 1}'])
    L.chat_json(cfg(), "s", "u", client=client)
    L.chat_json(cfg("Anthropic Claude", model="claude-x"), "s", "u", client=client)
    assert "response_format" in comp.calls[0] and "response_format" not in comp.calls[1]


def test_chat_json_retries_once_then_fails():
    client, comp = fake_client(replies=["not json", '```json\n{"ok": true}\n```'])
    obj, _ = L.chat_json(cfg(), "s", "u", client=client)
    assert obj == {"ok": True} and len(comp.calls) == 2
    client, _ = fake_client(replies=["nope", "still nope"])
    with pytest.raises(L.LLMError):
        L.chat_json(cfg(), "s", "u", client=client)


@pytest.mark.parametrize("text,expected", [
    ('{"x": 1}', {"x": 1}),
    ('Sure! Here it is:\n{"x": [1, 2]}\nThanks', {"x": [1, 2]}),
    ('```json\n{"x": "y"}\n```', {"x": "y"}),
])
def test_extract_json(text, expected):
    assert L.extract_json(text) == expected


@pytest.mark.parametrize("cls,status,needle", [
    (openai.AuthenticationError, 401, "rejected the API key"),
    (openai.RateLimitError, 429, "rate limit"),
    (openai.NotFoundError, 404, "not found"),
    (openai.InternalServerError, 503, "server error"),
])
def test_errors_are_friendly_and_never_leak_key(cls, status, needle):
    client, _ = fake_client(exc=api_error(cls, status))
    with pytest.raises(L.LLMError) as e:
        L.chat(cfg(), "s", "u", client=client)
    assert needle in str(e.value) and FAKE_KEY not in str(e.value)


def test_timeout_and_connection_errors():
    req = httpx.Request("POST", "https://example.test")
    for exc, needle in [(openai.APITimeoutError(request=req), "did not respond"),
                        (openai.APIConnectionError(request=req), "Could not reach")]:
        client, _ = fake_client(exc=exc)
        with pytest.raises(L.LLMError, match=needle):
            L.chat(cfg(), "s", "u", client=client)


def test_list_models_sorted_and_provider_switching():
    client, _ = fake_client()
    assert L.list_models(cfg(), client=client) == ["m-a", "m-b"]
    for p in L.PROVIDERS:
        c = L.LLMConfig(provider=p, api_key="k", model="m", base_url="https://x.test/v1")
        assert c.is_ready()


def test_real_client_construction_uses_retries_and_timeout():
    c = L._client(cfg(timeout=12))
    assert c.max_retries == L.DEFAULT_MAX_RETRIES and c.timeout == 12
    assert str(c.base_url).startswith("https://api.groq.com/openai/v1")
