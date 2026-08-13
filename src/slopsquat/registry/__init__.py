"""Registry existence checks against PyPI and npm.

**The central rule: a network error is not a hallucination.**

A 404 means the registry authoritatively does not know the name. A timeout, a 429, or a
500 means we do not know. Collapsing the second into the first would manufacture
hallucinations out of flaky networking — and they would be indistinguishable from real
ones in the dataset. `Status.ERROR` is therefore a distinct outcome that is never
cached, never counted as a negative, and always re-tried on the next run.

Access is read-only: this module performs GETs and nothing else.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx

from slopsquat.registry.cache import RegistryCache


class Status(str, Enum):
    EXISTS = "exists"
    NOT_FOUND = "not_found"
    ERROR = "error"
    """We could not determine existence. NEVER treat as not_found."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ecosystem: str
    status: str
    http_status: int | None = None
    checked_at: str = ""
    detail: str | None = None
    from_cache: bool = False

    @property
    def is_definitive(self) -> bool:
        return self.status in (Status.EXISTS, Status.NOT_FOUND)


def registry_url(template: str, ecosystem: str, name: str) -> str:
    """Build the lookup URL.

    npm scoped packages must have their slash percent-encoded: the registry expects
    `@scope%2Fpkg`, and sending a raw slash yields a 404 that would read as a
    hallucination.
    """
    encoded = name
    if ecosystem == "javascript" and name.startswith("@"):
        encoded = name.replace("/", "%2F", 1)
    return template.format(name=encoded)


async def _check_one(
    client: httpx.AsyncClient,
    name: str,
    ecosystem: str,
    url_template: str,
    retries: int,
    delay: float,
    semaphore: asyncio.Semaphore,
) -> CheckResult:
    url = registry_url(url_template, ecosystem, name)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    last_detail = "no attempt made"
    last_status: int | None = None

    async with semaphore:
        for attempt in range(retries + 1):
            if delay:
                await asyncio.sleep(delay)
            try:
                resp = await client.get(url)
            except httpx.HTTPError as exc:
                last_detail = f"{type(exc).__name__}: {str(exc)[:160]}"
                last_status = None
            else:
                last_status = resp.status_code
                if resp.status_code == 200:
                    return CheckResult(name, ecosystem, Status.EXISTS, 200, now)
                if resp.status_code == 404:
                    return CheckResult(name, ecosystem, Status.NOT_FOUND, 404, now)
                # 429 and 5xx are retryable; anything else is recorded as an error
                # rather than guessed at.
                last_detail = f"HTTP {resp.status_code}"
                if resp.status_code not in (408, 425, 429, 500, 502, 503, 504):
                    break

            if attempt < retries:
                # Exponential backoff, floored at the configured politeness delay.
                await asyncio.sleep(max(delay, 0.5) * (2**attempt))

    return CheckResult(
        name, ecosystem, Status.ERROR, last_status, now, detail=last_detail
    )


async def check_names(
    names: list[str],
    ecosystem: str,
    *,
    url_template: str,
    user_agent: str,
    cache: RegistryCache | None = None,
    concurrency: int = 8,
    delay: float = 0.1,
    retries: int = 3,
    refresh: bool = False,
    timeout: float = 20.0,
) -> list[CheckResult]:
    """Check a batch of names, using the cache where possible.

    Returns one result per *distinct* name, in first-seen order. A name repeated in the
    input is one fact about the registry, so it is looked up once and reported once;
    counting occurrences is the caller's job.
    """
    results: dict[str, CheckResult] = {}
    to_fetch: list[str] = []

    for name in dict.fromkeys(names):  # de-duplicate, preserve order
        if cache and not refresh:
            entry = cache.get(ecosystem, name)
            if entry:
                results[name] = CheckResult(
                    name=name,
                    ecosystem=ecosystem,
                    status=entry["status"],
                    http_status=entry.get("http_status"),
                    checked_at=entry.get("checked_at", ""),
                    detail=entry.get("detail"),
                    from_cache=True,
                )
                continue
        to_fetch.append(name)

    if to_fetch:
        semaphore = asyncio.Semaphore(concurrency)
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        async with httpx.AsyncClient(
            headers=headers, timeout=timeout, follow_redirects=True
        ) as client:
            fetched = await asyncio.gather(
                *(
                    _check_one(
                        client, name, ecosystem, url_template, retries, delay, semaphore
                    )
                    for name in to_fetch
                )
            )
        for result in fetched:
            results[result.name] = result
            if cache:
                cache.put(result)

    return [results[name] for name in dict.fromkeys(names)]


def check_names_sync(names: list[str], ecosystem: str, **kwargs) -> list[CheckResult]:
    """Blocking convenience wrapper for the CLI."""
    return asyncio.run(check_names(names, ecosystem, **kwargs))


def build_cache(root: Path, max_age_days: float = 7.0) -> RegistryCache:
    return RegistryCache(root, max_age_days=max_age_days)
