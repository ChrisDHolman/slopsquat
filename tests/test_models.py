"""Model adapter tests.

All API calls are faked — the suite never contacts a provider and needs no keys. The
behaviours under test are the ones that would quietly corrupt the dataset if wrong:

* a failed call is recorded (ok=False), never raised, so one error can't abort a sweep;
* a truncated response is flagged, so a cut-off code block isn't read as fewer packages;
* the *resolved* model version is recorded, not the alias we sent;
* provider params are mapped correctly (Anthropic effort -> output_config; OpenAI
  max_tokens -> max_completion_tokens).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slopsquat.config import ModelConfig
from slopsquat.models import complete


# ------------------------------------------------------------------ fake SDK shapes


def _anthropic_response(*, text, model, stop_reason="end_turn", thinking=None):
    content = []
    if thinking is not None:
        content.append(SimpleNamespace(type="thinking", thinking=thinking))
    content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=22),
    )


def _openai_response(*, text, model, finish_reason="stop"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=33, completion_tokens=44),
    )


class FakeAnthropic:
    """Stands in for anthropic.Anthropic(). Records the kwargs it was called with."""

    def __init__(self, response=None, raises=None, raise_times=0):
        self._response = response
        self._raises = raises
        self._raise_times = raise_times
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None and len(self.calls) <= self._raise_times:
            raise self._raises
        return self._response


class FakeOpenAI:
    def __init__(self, response=None, raises=None, raise_times=0):
        self._response = response
        self._raises = raises
        self._raise_times = raise_times
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None and len(self.calls) <= self._raise_times:
            raise self._raises
        return self._response


class Boom(Exception):
    """Fake API error carrying a status_code, like the real SDK errors."""

    def __init__(self, status_code):
        super().__init__(f"boom {status_code}")
        self.status_code = status_code


ANTH = ModelConfig(
    id="anthropic/opus-5",
    provider="anthropic",
    model="claude-opus-5",
    enabled=True,
    params={"max_tokens": 8192, "thinking": "adaptive", "effort": "high"},
)
OAI = ModelConfig(
    id="openai/gpt-5",
    provider="openai",
    model="gpt-5",
    enabled=True,
    params={"max_tokens": 8192},
)


# ------------------------------------------------------------------ anthropic mapping


def test_anthropic_happy_path_records_resolved_model_and_usage() -> None:
    client = FakeAnthropic(
        _anthropic_response(
            text="use `requests`", model="claude-opus-5-20260101", thinking="hmm"
        )
    )
    r = complete(ANTH, "sys", "how do I fetch a url", client=client)

    assert r.ok is True
    assert r.model_alias == "claude-opus-5"  # what we sent
    assert r.resolved_model == "claude-opus-5-20260101"  # what the API reported
    assert r.text == "use `requests`"
    assert r.thinking == "hmm"
    assert (r.input_tokens, r.output_tokens) == (11, 22)
    assert r.truncated is False


def test_anthropic_maps_thinking_and_effort() -> None:
    client = FakeAnthropic(_anthropic_response(text="x", model="claude-opus-5-x"))
    complete(ANTH, "sys", "hi", client=client)

    sent = client.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}  # NOT a top-level effort arg
    assert sent["max_tokens"] == 8192
    assert "budget_tokens" not in sent  # rejected on Opus 5


def test_anthropic_disabled_thinking() -> None:
    model = ModelConfig(
        id="a", provider="anthropic", model="claude-opus-5", enabled=True,
        params={"max_tokens": 100, "thinking": "disabled"},
    )
    client = FakeAnthropic(_anthropic_response(text="x", model="m"))
    complete(model, "sys", "hi", client=client)
    assert client.calls[0]["thinking"] == {"type": "disabled"}
    assert "output_config" not in client.calls[0]  # no effort configured


def test_anthropic_truncation_is_flagged() -> None:
    client = FakeAnthropic(
        _anthropic_response(text="half a code bl", model="m", stop_reason="max_tokens")
    )
    r = complete(ANTH, "sys", "hi", client=client)
    assert r.truncated is True
    assert r.stop_reason == "max_tokens"


# ------------------------------------------------------------------ openai mapping


def test_openai_happy_path_and_token_param_name() -> None:
    client = FakeOpenAI(_openai_response(text="npm i express", model="gpt-5-2026"))
    r = complete(OAI, "sys", "build an api", client=client)

    assert r.ok is True
    assert r.resolved_model == "gpt-5-2026"
    assert r.text == "npm i express"
    assert (r.input_tokens, r.output_tokens) == (33, 44)

    sent = client.calls[0]
    assert sent["max_completion_tokens"] == 8192  # NOT max_tokens
    assert "max_tokens" not in sent  # GPT-5 rejects it
    assert sent["messages"][0]["role"] == "system"


def test_openai_truncation_is_flagged() -> None:
    client = FakeOpenAI(_openai_response(text="cut", model="gpt-5", finish_reason="length"))
    r = complete(OAI, "sys", "hi", client=client)
    assert r.truncated is True


def test_openai_none_content_becomes_empty_string() -> None:
    client = FakeOpenAI(_openai_response(text=None, model="gpt-5"))
    r = complete(OAI, "sys", "hi", client=client)
    assert r.ok is True
    assert r.text == ""


# ------------------------------------------------------------------ failure handling


def test_terminal_error_is_recorded_not_raised() -> None:
    client = FakeAnthropic(raises=Boom(400), raise_times=99)
    r = complete(ANTH, "sys", "hi", client=client, retries=2)
    assert r.ok is False
    assert "400" in (r.error or "")
    assert len(client.calls) == 1  # 400 is terminal — not retried


def test_transient_error_is_retried_then_succeeds() -> None:
    client = FakeAnthropic(
        response=_anthropic_response(text="ok", model="m"),
        raises=Boom(503),
        raise_times=1,
    )
    r = complete(ANTH, "sys", "hi", client=client, retries=2)
    assert r.ok is True
    assert r.text == "ok"
    assert len(client.calls) == 2  # failed once, then succeeded


def test_transient_error_exhausts_retries_and_is_recorded() -> None:
    client = FakeOpenAI(raises=Boom(429), raise_times=99)
    r = complete(OAI, "sys", "hi", client=client, retries=1)
    assert r.ok is False
    assert "429" in (r.error or "")
    assert len(client.calls) == 2  # initial + 1 retry


def test_a_failed_call_still_carries_identity_fields() -> None:
    """Even an error record must be sliceable by model — it belongs in the dataset as an
    excluded call, not an anonymous blank."""
    client = FakeAnthropic(raises=Boom(401), raise_times=99)
    r = complete(ANTH, "sys", "hi", client=client)
    assert r.model_id == "anthropic/opus-5"
    assert r.provider == "anthropic"
    assert r.model_alias == "claude-opus-5"
    assert r.created_at  # timestamped
    assert r.latency_s >= 0


def test_quota_exhausted_429_is_terminal_not_retried() -> None:
    """A credit-exhausted 429 will never succeed; retrying it across a 5,500-call sweep
    would waste time and quota. It must be treated as terminal despite the 429 status."""

    class QuotaError(Exception):
        def __init__(self):
            super().__init__(
                "Error code: 429 - {'error': {'code': 'insufficient_quota'}}"
            )
            self.status_code = 429

    client = FakeOpenAI(raises=QuotaError(), raise_times=99)
    r = complete(OAI, "sys", "hi", client=client, retries=3)
    assert r.ok is False
    assert len(client.calls) == 1  # not retried


def test_unknown_provider_raises_config_error() -> None:
    bad = ModelConfig(id="x", provider="cohere", model="c", enabled=True, params={})
    with pytest.raises(Exception):
        complete(bad, "sys", "hi", client=object())
