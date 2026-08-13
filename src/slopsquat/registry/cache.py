"""On-disk cache for registry lookups.

Two reasons this matters beyond speed:

*   **Politeness.** A sweep re-asks about the same popular packages thousands of times.
    PyPI and npm are shared public infrastructure; caching is how this stays a good
    citizen at N=20 runs.
*   **Correctness.** Errors are deliberately **not** cached. Caching a timeout would
    freeze a transient failure into the dataset, and an error must never harden into a
    "does not exist".

A negative result is time-sensitive in a way a positive one is not: a name that does not
exist today can be registered tomorrow — that is the entire attack being studied. Every
entry therefore records when it was checked, and negatives expire.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from slopsquat.registry import CheckResult


def _safe_filename(ecosystem: str, name: str) -> str:
    """Filesystem-safe cache key for an *arbitrary* name.

    The name may be anything a model emitted — npm scopes (`@`, `/`) but also stray
    quotes, colons, or other characters that are illegal in a Windows filename. Percent-
    encoding everything outside a conservative safe set means the cache never crashes on
    unexpected input (a bad name is still a lookup we want to record, not a fatal error).
    Very long names are hashed to stay under the filesystem's per-component limit. Names
    are lowercased by the caller, so a case-insensitive filesystem cannot collide two
    distinct packages.
    """
    encoded = quote(name, safe="")  # encodes @ / " : and everything else non-alphanumeric
    if len(encoded) > 120:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
        encoded = encoded[:80] + "-" + digest
    return f"{ecosystem}__{encoded}.json"


class RegistryCache:
    def __init__(self, root: Path, max_age_days: float = 7.0) -> None:
        self.root = root
        self.max_age_seconds = max_age_days * 86400
        self.hits = 0
        self.misses = 0

    def _path(self, ecosystem: str, name: str) -> Path:
        filename = _safe_filename(ecosystem, name)
        # Shard on the first two characters to keep directory sizes sane.
        shard = filename[len(ecosystem) + 2 :][:2] or "_"
        return self.root / ecosystem / shard / filename

    def get(self, ecosystem: str, name: str) -> dict | None:
        path = self._path(ecosystem, name)
        if not path.is_file():
            self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.misses += 1
            return None

        # Errors are never written, but tolerate a hand-edited or older cache.
        if entry.get("status") == "error":
            self.misses += 1
            return None

        age = time.time() - float(entry.get("checked_at_epoch", 0))
        if age > self.max_age_seconds:
            self.misses += 1
            return None

        self.hits += 1
        return entry

    def put(self, result: CheckResult) -> None:
        """Persist a definitive result. Errors are silently ignored by design."""
        if result.status == "error":
            return
        path = self._path(result.ecosystem, result.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = asdict(result)
        entry["checked_at_epoch"] = time.time()
        entry.pop("from_cache", None)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
