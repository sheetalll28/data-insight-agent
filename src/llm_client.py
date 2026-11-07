"""
Provider-agnostic LLM access (bring your own key).

Every supported provider exposes an OpenAI-compatible Chat Completions
endpoint, so one small adapter built on the official `openai` SDK covers all
of them without extra dependencies:

    Groq, OpenAI, Google Gemini, Anthropic Claude, and any custom
    OpenAI-compatible host (OpenRouter, Together AI, a self-hosted server...).

Design rules
- The analytical code never imports a provider SDK; it calls `chat()` only.
- API keys come from the user (per session) or from environment variables.
  They are never logged, never written to disk by this module, and are
  scrubbed from any error message shown to the user.
- Only small, aggregated statistics are sent to the model - never raw rows.
- The math engines do not depend on this module at all.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

# Model IDs change often, so these are only editable starting suggestions.
# Users can type any model ID or fetch the live list from the provider.
PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_var": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "json_mode": True,
        "key_help": "console.groq.com/keys",
    },
    "OpenAI": {
        "base_url": None,  # SDK default
        "env_var": "OPENAI_API_KEY",
        "default_model": "",
        "json_mode": True,
        "key_help": "platform.openai.com/api-keys (API billing is separate from a ChatGPT subscription)",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_var": "GEMINI_API_KEY",
        "default_model": "",
        "json_mode": True,
        "key_help": "aistudio.google.com/apikey",
    },
    "Anthropic Claude": {
        "base_url": "https://api.anthropic.com/v1/",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "",
        "json_mode": False,  # compatibility layer: rely on prompt + JSON extraction
        "key_help": "platform.claude.com/settings/keys",
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "env_var": "LLM_API_KEY",
        "default_model": "",
        "json_mode": False,
        "key_help": "your provider's dashboard",
    },
}

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 3  # SDK retries 408/409/429/5xx with exponential backoff; never 401/403


class LLMError(Exception):
    """A user-presentable error that never contains the API key."""


@dataclass
class LLMConfig:
    provider: str = "Groq"
    api_key: str = field(default="", repr=False)   # repr=False: never printed
    model: str = ""
    base_url: str = ""
    timeout: float = DEFAULT_TIMEOUT_S

    def resolved_key(self) -> str:
        if self.api_key:
            return self.api_key
        env = PROVIDERS.get(self.provider, {}).get("env_var")
        return os.environ.get(env, "") if env else ""

    def resolved_base_url(self):
        if self.provider == "Custom (OpenAI-compatible)":
            return self.base_url or None
        return self.base_url or PROVIDERS[self.provider]["base_url"]

    def resolved_model(self) -> str:
        return self.model or PROVIDERS.get(self.provider, {}).get("default_model", "")

    def is_ready(self) -> bool:
        if self.provider not in PROVIDERS or not self.resolved_key() or not self.resolved_model():
            return False
        if self.provider == "Custom (OpenAI-compatible)" and not self.base_url:
            return False
        return True

    def missing(self) -> str:
        if self.provider not in PROVIDERS:
            return f"Unknown provider '{self.provider}'."
        if not self.resolved_key():
            return (f"No API key for {self.provider}. Enter one in AI Settings or set "
                    f"{PROVIDERS[self.provider]['env_var']} in your .env file.")
        if self.provider == "Custom (OpenAI-compatible)" and not self.base_url:
            return "Enter the base URL of your OpenAI-compatible endpoint."
        if not self.resolved_model():
            return "Choose a model (type a model ID or click 'Fetch models')."
        return ""


def _scrub(text: str, cfg: LLMConfig) -> str:
    key = cfg.resolved_key()
    if key:
        text = text.replace(key, "***")
    # Generic safety net for anything that looks like a secret token
    return re.sub(r"(sk|gsk|sk-ant|AIza)[-_A-Za-z0-9]{12,}", "***", text)


def _friendly_error(exc: Exception, cfg: LLMConfig) -> LLMError:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return LLMError(_scrub(str(exc), cfg))
    status = getattr(exc, "status_code", None)
    if isinstance(exc, openai.AuthenticationError) or status == 401:
        msg = f"{cfg.provider} rejected the API key (401). Check the key and that it belongs to this provider."
    elif isinstance(exc, openai.PermissionDeniedError) or status == 403:
        msg = f"{cfg.provider} denied access (403). The key may lack permission for this model or region."
    elif isinstance(exc, openai.NotFoundError) or status == 404:
        msg = f"Model '{cfg.resolved_model()}' or the endpoint was not found (404). The model may be retired - pick another."
    elif isinstance(exc, openai.RateLimitError) or status == 429:
        msg = f"{cfg.provider} rate limit or quota exceeded (429), even after retries. Wait, or check credits/billing."
    elif isinstance(exc, openai.APITimeoutError):
        msg = f"{cfg.provider} did not respond within {cfg.timeout:.0f}s."
    elif isinstance(exc, openai.APIConnectionError):
        msg = f"Could not reach {cfg.provider}. Check your network and the base URL."
    elif isinstance(exc, openai.BadRequestError) or status == 400:
        msg = f"{cfg.provider} rejected the request (400): {getattr(exc, 'message', exc)}"
    elif status and status >= 500:
        msg = f"{cfg.provider} had a server error ({status}). Try again later."
    else:
        msg = f"{cfg.provider} error: {exc}"
    return LLMError(_scrub(str(msg), cfg))


def _client(cfg: LLMConfig):
    # A model is not needed just to list models, but a key (and, for custom
    # hosts, a base URL) is.
    if cfg.provider not in PROVIDERS or not cfg.resolved_key() or (
        cfg.provider == "Custom (OpenAI-compatible)" and not cfg.base_url
    ):
        raise LLMError(cfg.missing())
    from openai import OpenAI
    return OpenAI(
        api_key=cfg.resolved_key(),
        base_url=cfg.resolved_base_url(),
        timeout=cfg.timeout,
        max_retries=DEFAULT_MAX_RETRIES,
    )


def chat(cfg: LLMConfig, system: str, user: str, json_mode: bool = False, client=None) -> dict:
    """
    Send one system+user exchange. Returns
    {"text", "provider", "model", "seconds", "input_tokens", "output_tokens"}.
    Token counts are whatever the provider reported (None if not reported).
    """
    if not cfg.is_ready():
        raise LLMError(cfg.missing())
    client = client or _client(cfg)
    kwargs = dict(
        model=cfg.resolved_model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if json_mode and PROVIDERS[cfg.provider]["json_mode"]:
        kwargs["response_format"] = {"type": "json_object"}
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - mapped to a safe message
        raise _friendly_error(exc, cfg) from None
    usage = getattr(resp, "usage", None)
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    return {
        "text": text,
        "provider": cfg.provider,
        "model": cfg.resolved_model(),
        "seconds": round(time.time() - t0, 2),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


def chat_json(cfg: LLMConfig, system: str, user: str, client=None, retries: int = 1) -> tuple[dict, dict]:
    """chat() + validated JSON parsing, with one corrective retry."""
    last = None
    for attempt in range(retries + 1):
        prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
        result = chat(cfg, system, prompt, json_mode=True, client=client)
        try:
            return extract_json(result["text"]), result
        except ValueError as exc:
            last = exc
    raise LLMError(f"The model did not return valid JSON ({last}).")


def extract_json(text: str) -> dict:
    """Parse a JSON object from a reply that may include code fences or prose."""
    if not text:
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.append(text)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found")


def list_models(cfg: LLMConfig, client=None) -> list[str]:
    """Live model list from the provider (where its API supports it)."""
    client = client or _client(cfg)
    try:
        return sorted(m.id for m in client.models.list())
    except Exception as exc:  # noqa: BLE001
        raise _friendly_error(exc, cfg) from None


def test_connection(cfg: LLMConfig, client=None) -> dict:
    """Tiny round-trip to confirm key + model work."""
    return chat(cfg, "You are a connectivity check.", "Reply with the single word: OK", client=client)
