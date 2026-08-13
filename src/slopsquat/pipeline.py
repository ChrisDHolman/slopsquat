"""The sweep: send every prompt to every enabled model, N times, and record what came
back — one immutable JSONL record per model call.

Two design commitments shape this file:

*   **Resumability.** A full sweep is thousands of paid calls. It must survive a crash,
    a rate-limit wall, or a Ctrl-C without redoing completed work. Records are appended
    the instant each call returns, and a resumed run skips any (prompt, model, run_index)
    that already has a successful record. A failed call is retried on resume.

*   **The model layer and the existence layer are separate.** A run record captures what
    the model *said* — it is immutable evidence, timestamped once. Whether a name exists
    in a registry is a *different* fact that changes over time: a hallucinated name absent
    today can be registered by an attacker tomorrow, and re-checking it later is a core
    part of the research. So existence is resolved into its own timestamped file
    (``registry.jsonl``) and joined at analysis time — never baked into the run record.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from slopsquat.config import ModelConfig, Prompt
from slopsquat.extract import extract as run_extraction
from slopsquat.models import ModelResponse, complete

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SweepTask:
    prompt: Prompt
    model: ModelConfig
    run_index: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.prompt.id, self.model.id, self.run_index)


def build_tasks(
    models: list[ModelConfig], prompts: list[Prompt], runs_per_prompt: int
) -> list[SweepTask]:
    """Every (enabled model × prompt × run) combination the sweep will execute."""
    tasks: list[SweepTask] = []
    for model in models:
        if not model.enabled:
            continue
        for prompt in prompts:
            for i in range(runs_per_prompt):
                tasks.append(SweepTask(prompt=prompt, model=model, run_index=i))
    return tasks


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    """Keys of calls already recorded as successful, so a resumed sweep skips them.

    A record with ``ok=False`` is deliberately *not* counted as done — a failed call is
    a gap to retry, not an observation. Malformed lines are ignored rather than aborting
    the resume.
    """
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("ok") and "prompt_id" in rec and "model_id" in rec:
                done.add((rec["prompt_id"], rec["model_id"], int(rec["run_index"])))
    return done


def build_record(
    task: SweepTask,
    resp: ModelResponse,
    stdlib: dict[str, set[str]],
    aliases: dict[str, str],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the JSONL record for one model call.

    Extraction runs here so the expensive part (the model call) and the cheap,
    improvable part (parsing its text) live in one record — and re-extraction later can
    always fall back to the stored ``response_text``. No registry lookups happen here.
    """
    packages: list[dict[str, Any]] = []
    extraction_notes: list[str] = []
    if resp.ok and resp.text:
        result = run_extraction(
            resp.text, ecosystem=task.prompt.ecosystem, stdlib=stdlib, aliases=aliases
        )
        packages = [p.to_dict() for p in result.packages]
        extraction_notes = list(result.notes)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        # --- experimental cell -------------------------------------------------
        "prompt_id": task.prompt.id,
        "ecosystem": task.prompt.ecosystem,
        "domain": task.prompt.domain,
        "specificity": task.prompt.specificity,
        "model_id": task.model.id,
        "model_alias": resp.model_alias,
        "resolved_model": resp.resolved_model,
        "run_index": task.run_index,
        # --- call outcome ------------------------------------------------------
        "ok": resp.ok,
        "error": resp.error,
        "created_at": resp.created_at,
        "latency_s": round(resp.latency_s, 3),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "stop_reason": resp.stop_reason,
        "truncated": resp.truncated,
        "params_sent": resp.params_sent,
        # --- what the model said ----------------------------------------------
        "packages": packages,
        "extraction_notes": extraction_notes,
        "response_text": resp.text,
    }


@dataclass
class SweepSummary:
    total: int
    skipped: int
    attempted: int
    ok: int
    failed: int
    truncated: int


def run_sweep(
    tasks: list[SweepTask],
    system_prompt: str,
    stdlib: dict[str, set[str]],
    aliases: dict[str, str],
    *,
    out_path: Path,
    run_id: str,
    concurrency: int,
    retries: int,
    completed: set[tuple[str, str, int]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    complete_fn: Callable[..., ModelResponse] = complete,
) -> SweepSummary:
    """Execute the sweep, appending one record per attempted call.

    ``complete_fn`` is injectable so tests can drive the whole pipeline without touching
    a provider. Records are flushed as they complete, so killing the process loses at
    most the in-flight calls.
    """
    completed = completed if completed is not None else set()
    pending = [t for t in tasks if t.key not in completed]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    summary = SweepSummary(
        total=len(tasks),
        skipped=len(tasks) - len(pending),
        attempted=0,
        ok=0,
        failed=0,
        truncated=0,
    )

    def worker(task: SweepTask) -> dict[str, Any]:
        resp = complete_fn(
            task.model, system_prompt, task.prompt.text, retries=retries
        )
        record = build_record(task, resp, stdlib, aliases, run_id=run_id)
        line = json.dumps(record, ensure_ascii=False)
        with write_lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            summary.attempted += 1
            if resp.ok:
                summary.ok += 1
            else:
                summary.failed += 1
            if resp.truncated:
                summary.truncated += 1
        return record

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for record in pool.map(worker, pending):
                if progress is not None:
                    progress(record)

    return summary


def collect_unique_names(
    out_path: Path,
) -> dict[str, list[str]]:
    """Read runs.jsonl and return every distinct extracted name, grouped by ecosystem.

    This is the input to the (separate) existence pass. Only names from successful calls
    are returned — a failed call produced no observation.
    """
    by_eco: dict[str, set[str]] = {}
    if not out_path.exists():
        return {}
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not rec.get("ok"):
                continue
            for pkg in rec.get("packages", []):
                eco = pkg.get("ecosystem")
                name = pkg.get("name")
                if eco and name:
                    by_eco.setdefault(eco, set()).add(name)
    return {eco: sorted(names) for eco, names in by_eco.items()}


def reextract(
    runs_path: Path,
    stdlib: dict[str, set[str]],
    aliases: dict[str, str],
) -> dict[str, int]:
    """Re-run extraction over stored responses, rewriting each record's ``packages``.

    The whole reason ``response_text`` is stored is so the cheap, improvable parsing step
    can be corrected without re-spending on the models. This reads runs.jsonl, re-extracts
    from each successful record's text, and writes the file back atomically. Failed
    records are passed through untouched.

    Returns counts of records processed and records whose package set changed.
    """
    if not runs_path.exists():
        return {"records": 0, "changed": 0, "packages_before": 0, "packages_after": 0}

    processed = changed = before = after = 0
    tmp = runs_path.with_suffix(".jsonl.tmp")
    with runs_path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            processed += 1
            if rec.get("ok") and rec.get("response_text"):
                before += len(rec.get("packages", []))
                old = rec.get("packages", [])
                result = run_extraction(
                    rec["response_text"],
                    ecosystem=rec.get("ecosystem"),
                    stdlib=stdlib,
                    aliases=aliases,
                )
                rec["packages"] = [p.to_dict() for p in result.packages]
                rec["extraction_notes"] = list(result.notes)
                after += len(rec["packages"])
                if rec["packages"] != old:
                    changed += 1
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(runs_path)
    return {"records": processed, "changed": changed,
            "packages_before": before, "packages_after": after}


def write_existence_snapshot(
    names_by_eco: dict[str, list[str]],
    out_path: Path,
    *,
    registries: dict[str, str],
    user_agent: str,
    cache: Any = None,
    concurrency: int = 8,
    delay: float = 0.1,
    retries: int = 3,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Resolve existence for every unique name and write a timestamped snapshot.

    This is deliberately a separate artefact from ``runs.jsonl``: existence is
    time-varying, so the snapshot records *when* each name was checked. Re-running it
    later (to see whether a hallucinated name has since been registered) produces a new
    snapshot without touching the immutable run records.
    """
    from slopsquat.registry import Status, check_names_sync

    snapshot_at = dt.datetime.now(dt.timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as fh:
        for eco, names in sorted(names_by_eco.items()):
            if not names:
                continue
            key = "pypi" if eco == "python" else "npm"
            results = check_names_sync(
                names,
                eco,
                url_template=registries[key],
                user_agent=user_agent,
                cache=cache,
                concurrency=concurrency,
                delay=delay,
                retries=retries,
                refresh=refresh,
            )
            for r in results:
                status = r.status.value if isinstance(r.status, Status) else str(r.status)
                rec = {
                    "ecosystem": eco,
                    "name": r.name,
                    "status": status,
                    "http_status": r.http_status,
                    "checked_at": r.checked_at,
                    "from_cache": r.from_cache,
                    "snapshot_at": snapshot_at,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
    return records


def now_run_id() -> str:
    """A sortable run id for one sweep invocation."""
    return dt.datetime.now(dt.timezone.utc).strftime("sweep-%Y%m%dT%H%M%SZ")
