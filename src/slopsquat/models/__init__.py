"""Model adapters: send a prompt to a provider, record exactly what came back.

Two disciplines carry over from the registry checker:

*   **A failed call is recorded, not raised.** One model error must not abort a
    5,500-call sweep, and — like a registry ``ERROR`` — a failed call is *excluded from
    denominators*, never counted as "the model produced no hallucination". ``ok=False``
    with ``error`` set is a distinct outcome, not a silent zero.
*   **Record what actually happened, not what we asked for.** Every response stores the
    ``resolved_model`` the API reports (e.g. ``claude-opus-5-2026xxxx``), not the alias we
    sent. A published claim cites the resolved version; the alias is only what a developer
    types when they don't think about pinning — which is the condition under study.

The real Anthropic and OpenAI SDKs are used directly. There is deliberately no
OpenAI-compatible shim in front of Claude: routing both providers through one wire format
would blur provider-specific behaviour, and that behaviour is exactly the variable.

``truncated`` is load-bearing. Thinking / reasoning tokens share the ``max_tokens``
budget, so a high-effort run can hit the ceiling mid-code-block and cut off a package
list. That would silently *under*-count hallucinations. The flag surfaces it per record
so truncated responses can be excluded or re-run rather than trusted.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

from slopsquat.config import ModelConfig

# Transient failures worth retrying. Anything else (401 auth, 400 bad request) is a
# terminal error: retrying would just repeat the same mistake and waste quota.
_TRANSIENT_TYPES = (
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ModelError(Exception):
    """Raised only for misconfiguration (unknown provider, missing key) — never for an
    API failure, which is captured in the ModelResponse instead."""


@dataclass(frozen=True)
class ModelResponse:
    model_id: str  # config id, e.g. "anthropic/opus-5" — the stable slicing key
    provider: str
    model_alias: str  # what we sent, e.g. "claude-opus-5"
    ok: bool
    created_at: str
    latency_s: float
    resolved_model: str | None = None  # what the API reports back
    text: str = ""
    thinking: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    truncated: bool = False
    error: str | None = None
    params_sent: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# A 429 usually means "slow down" and is worth retrying — but a quota- or
# credit-exhausted 429 will never succeed, and retrying it 5,500 times just burns
# wall-clock. These markers make it terminal.
_TERMINAL_CODES = {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}


def _is_transient(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code in _TERMINAL_CODES:
        return False
    # The code sometimes only appears in the message body, not as an attribute.
    if any(marker in str(exc) for marker in _TERMINAL_CODES):
        return False
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS


# --------------------------------------------------------------------------- clients

_clients: dict[str, Any] = {}


def get_client(provider: str, *, timeout: float = 120.0) -> Any:
    """Return a provider client with our own retry disabled.

    ``max_retries=0`` hands retry control entirely to :func:`complete`, so the recorded
    latency and attempt count reflect what actually happened rather than the SDK's hidden
    retries. Clients are cached per provider; ``with_options`` applies the per-call
    timeout to a lightweight copy.
    """
    base = _clients.get(provider)
    if base is None:
        if provider == "anthropic":
            base = anthropic.Anthropic(max_retries=0)
        elif provider == "openai":
            base = openai.OpenAI(max_retries=0)
        else:
            raise ModelError(f"unknown provider {provider!r}")
        _clients[provider] = base
    return base.with_options(timeout=timeout)


# --------------------------------------------------------------------------- adapters


def _call_anthropic(
    client: Any, model: ModelConfig, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    params = model.params
    kwargs: dict[str, Any] = {
        "model": model.model,
        "max_tokens": int(params.get("max_tokens", 4096)),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    # thinking: adaptive -> {"type": "adaptive"} ; disabled -> {"type": "disabled"}.
    # budget_tokens is rejected on Opus 5 / Sonnet 5, so it is never sent.
    thinking = params.get("thinking")
    if thinking == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
    elif thinking == "disabled":
        kwargs["thinking"] = {"type": "disabled"}

    # effort belongs in output_config on the Opus 5 / Sonnet 5 family, not top-level.
    effort = params.get("effort")
    if effort:
        kwargs["output_config"] = {"effort": effort}

    resp = client.messages.create(**kwargs)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")

    usage = resp.usage
    return {
        "resolved_model": resp.model,
        "text": "".join(text_parts),
        "thinking": "".join(thinking_parts) or None,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "stop_reason": resp.stop_reason,
        "truncated": resp.stop_reason == "max_tokens",
        "params_sent": {k: v for k, v in kwargs.items() if k not in ("system", "messages")},
    }


def _call_openai(
    client: Any, model: ModelConfig, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    params = model.params
    # GPT-5 rejects `max_tokens`; the budget is `max_completion_tokens`, and reasoning
    # tokens are drawn from it — see the truncated-flag note in the module docstring.
    kwargs: dict[str, Any] = {
        "model": model.model,
        "max_completion_tokens": int(params.get("max_tokens", 4096)),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    effort = params.get("reasoning_effort")
    if effort:
        kwargs["reasoning_effort"] = effort

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    usage = resp.usage
    return {
        "resolved_model": resp.model,
        "text": choice.message.content or "",
        "thinking": None,  # reasoning tokens are billed but not returned as text
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "stop_reason": choice.finish_reason,
        "truncated": choice.finish_reason == "length",
        "params_sent": {k: v for k, v in kwargs.items() if k != "messages"},
    }


_ADAPTERS = {"anthropic": _call_anthropic, "openai": _call_openai}


def complete(
    model: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    timeout: float = 120.0,
    retries: int = 2,
    client: Any = None,
) -> ModelResponse:
    """Send one prompt to one model and return a fully-populated record.

    Never raises on an API failure: transient errors are retried with backoff, and a
    terminal failure returns ``ok=False`` with the error captured. A ``client`` may be
    injected for testing.
    """
    adapter = _ADAPTERS.get(model.provider)
    if adapter is None:
        raise ModelError(f"unknown provider {model.provider!r}")

    created_at = _now()
    if client is None:
        client = get_client(model.provider, timeout=timeout)

    def base(**over: Any) -> dict[str, Any]:
        return {
            "model_id": model.id,
            "provider": model.provider,
            "model_alias": model.model,
            "created_at": created_at,
            **over,
        }

    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            data = adapter(client, model, system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 — deliberately captured, not raised
            latency = time.monotonic() - start
            if attempt < retries and _is_transient(exc):
                time.sleep(min(2**attempt, 8))
                continue
            return ModelResponse(
                **base(
                    ok=False,
                    latency_s=latency,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            )
        latency = time.monotonic() - start
        return ModelResponse(**base(ok=True, latency_s=latency, **data))

    # Unreachable: the loop either returns a success or, on the final attempt, an error.
    raise AssertionError("retry loop fell through")
