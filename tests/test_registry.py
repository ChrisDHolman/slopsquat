"""Registry checker tests.

The rule under test throughout: **an error is not a negative.** A timeout, 429, or 5xx
must never become "does not exist", must never be cached, and must always be retried —
otherwise flaky networking manufactures hallucinations indistinguishable from real ones.

All HTTP is mocked. Tests do not touch the real registries.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from slopsquat.registry import Status, check_names, registry_url
from slopsquat.registry.cache import RegistryCache

PYPI = "https://pypi.org/pypi/{name}/json"
NPM = "https://registry.npmjs.org/{name}"


def transport(handler):
    return httpx.MockTransport(handler)


async def _check(names, ecosystem, handler, **kwargs):
    """Run check_names against a mocked transport."""
    import slopsquat.registry as reg

    real_client = httpx.AsyncClient

    def factory(*args, **kw):
        kw["transport"] = transport(handler)
        return real_client(*args, **kw)

    reg.httpx.AsyncClient = factory  # type: ignore[assignment]
    try:
        return await check_names(
            names,
            ecosystem,
            url_template=PYPI if ecosystem == "python" else NPM,
            user_agent="test/1.0",
            delay=0,
            retries=kwargs.pop("retries", 1),
            **kwargs,
        )
    finally:
        reg.httpx.AsyncClient = real_client  # type: ignore[assignment]


# ---------------------------------------------------------------- URL building


def test_scoped_npm_name_percent_encodes_slash() -> None:
    """A raw slash yields a 404 the registry never intended — which would read as a
    hallucination."""
    assert registry_url(NPM, "javascript", "@types/node") == (
        "https://registry.npmjs.org/@types%2Fnode"
    )


def test_unscoped_names_are_untouched() -> None:
    assert registry_url(NPM, "javascript", "express") == (
        "https://registry.npmjs.org/express"
    )
    assert registry_url(PYPI, "python", "requests") == (
        "https://pypi.org/pypi/requests/json"
    )


# ---------------------------------------------------------------- outcomes


def test_200_is_exists_and_404_is_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if "requests" in str(request.url) else 404)

    results = asyncio.run(_check(["requests", "notarealpkg"], "python", handler))
    by_name = {r.name: r for r in results}
    assert by_name["requests"].status == Status.EXISTS
    assert by_name["notarealpkg"].status == Status.NOT_FOUND


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_retryable_http_errors_end_as_error_not_not_found(code: int) -> None:
    """The single most important behaviour in this module."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code)

    results = asyncio.run(_check(["something"], "python", handler, retries=1))
    assert results[0].status == Status.ERROR
    assert results[0].status != Status.NOT_FOUND


def test_network_exception_is_error_not_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout")

    results = asyncio.run(_check(["something"], "python", handler, retries=1))
    assert results[0].status == Status.ERROR
    assert "Timeout" in (results[0].detail or "")


def test_transient_failure_then_success_is_recovered() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    results = asyncio.run(_check(["flaky"], "python", handler, retries=2))
    assert results[0].status == Status.EXISTS
    assert calls["n"] == 2


def test_non_retryable_status_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403)

    results = asyncio.run(_check(["forbidden"], "python", handler, retries=3))
    assert results[0].status == Status.ERROR
    assert calls["n"] == 1  # 403 is not transient — retrying would just be rude


# ---------------------------------------------------------------- de-duplication


def test_duplicate_names_are_requested_and_returned_once() -> None:
    """One result per distinct name, in first-seen order — a repeated name is one fact
    about the registry, not several."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    results = asyncio.run(_check(["b", "a", "b", "a"], "python", handler))
    assert len(seen) == 2
    assert [r.name for r in results] == ["b", "a"]


# ---------------------------------------------------------------- caching


def test_definitive_results_are_cached_and_reused(tmp_path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    cache = RegistryCache(tmp_path)
    asyncio.run(_check(["ghostpkg"], "python", handler, cache=cache))
    second = asyncio.run(_check(["ghostpkg"], "python", handler, cache=cache))

    assert calls["n"] == 1  # second lookup served from cache
    assert second[0].from_cache is True
    assert second[0].status == Status.NOT_FOUND


def test_errors_are_never_cached(tmp_path) -> None:
    """Caching a timeout would freeze a transient failure into the dataset."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("down")

    cache = RegistryCache(tmp_path)
    asyncio.run(_check(["x"], "python", handler, cache=cache, retries=0))
    asyncio.run(_check(["x"], "python", handler, cache=cache, retries=0))

    assert calls["n"] == 2  # re-tried, not served from a poisoned cache
    assert not list(tmp_path.rglob("*.json"))


def test_stale_negative_expires(tmp_path) -> None:
    """A name that does not exist today can be registered tomorrow — that is the whole
    attack. Negatives must not be trusted indefinitely."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    cache = RegistryCache(tmp_path, max_age_days=7)
    asyncio.run(_check(["ghost"], "python", handler, cache=cache))

    entry_path = next(tmp_path.rglob("*.json"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["checked_at_epoch"] = 0  # epoch 1970 — far outside any sane window
    entry_path.write_text(json.dumps(entry), encoding="utf-8")

    asyncio.run(_check(["ghost"], "python", handler, cache=cache))
    assert calls["n"] == 2


def test_refresh_bypasses_cache(tmp_path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200)

    cache = RegistryCache(tmp_path)
    asyncio.run(_check(["requests"], "python", handler, cache=cache))
    asyncio.run(_check(["requests"], "python", handler, cache=cache, refresh=True))
    assert calls["n"] == 2


def test_scoped_name_survives_cache_round_trip(tmp_path) -> None:
    """'@' and '/' are not safe in a filename; the cache must handle scoped packages."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    cache = RegistryCache(tmp_path)
    asyncio.run(_check(["@types/node"], "javascript", handler, cache=cache))
    second = asyncio.run(_check(["@types/node"], "javascript", handler, cache=cache))
    assert second[0].from_cache is True
    assert second[0].name == "@types/node"
