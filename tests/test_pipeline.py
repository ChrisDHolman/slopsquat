"""Sweep pipeline tests.

No provider is contacted: a fake ``complete_fn`` stands in for the model call, so the
suite exercises task fan-out, record structure, resumability, and name collection for
free. The behaviours that matter for data integrity:

* one record is written per attempted call, the instant it completes (crash-safety);
* a resumed sweep skips successful calls and retries failed ones;
* only successful calls contribute extracted names to the existence pass.
"""

from __future__ import annotations

import json

from slopsquat.config import ModelConfig, Prompt
from slopsquat.models import ModelResponse
from slopsquat.pipeline import (
    SweepTask,
    build_record,
    build_tasks,
    collect_unique_names,
    load_completed,
    run_sweep,
)

MODEL = ModelConfig(
    id="anthropic/opus-5", provider="anthropic", model="claude-opus-5",
    enabled=True, params={"max_tokens": 8192},
)
DISABLED = ModelConfig(
    id="openai/gpt-5", provider="openai", model="gpt-5", enabled=False, params={},
)
PROMPT = Prompt(
    id="py-01", ecosystem="python", domain="web", specificity="detailed",
    text="scrape a page",
)
STDLIB = {"python": {"os", "sys"}, "javascript": set()}
ALIASES: dict[str, str] = {}


def _fake_response(text="use `requests`", *, ok=True, truncated=False, error=None):
    def complete_fn(model, system_prompt, user_prompt, *, retries=2):
        return ModelResponse(
            model_id=model.id,
            provider=model.provider,
            model_alias=model.model,
            ok=ok,
            created_at="2026-01-01T00:00:00+00:00",
            latency_s=0.01,
            resolved_model=f"{model.model}-resolved" if ok else None,
            text=text if ok else "",
            input_tokens=10 if ok else None,
            output_tokens=20 if ok else None,
            stop_reason="max_tokens" if truncated else ("end_turn" if ok else None),
            truncated=truncated,
            error=error,
        )

    return complete_fn


# ----------------------------------------------------------------- task fan-out


def test_build_tasks_skips_disabled_and_multiplies_out() -> None:
    tasks = build_tasks([MODEL, DISABLED], [PROMPT], runs_per_prompt=3)
    assert len(tasks) == 3  # disabled model contributes nothing
    assert {t.run_index for t in tasks} == {0, 1, 2}
    assert all(t.model.id == "anthropic/opus-5" for t in tasks)


# ----------------------------------------------------------------- record shape


def test_build_record_runs_extraction_and_carries_metadata() -> None:
    task = SweepTask(PROMPT, MODEL, 0)
    resp = _fake_response("install with `pip install requests faketestpkg`")(
        MODEL, "sys", "hi"
    )
    rec = build_record(task, resp, STDLIB, ALIASES, run_id="r1")

    assert rec["prompt_id"] == "py-01"
    assert rec["model_id"] == "anthropic/opus-5"
    assert rec["resolved_model"] == "claude-opus-5-resolved"
    assert rec["ecosystem"] == "python"
    assert rec["specificity"] == "detailed"
    names = {p["name"] for p in rec["packages"]}
    assert {"requests", "faketestpkg"} <= names
    # No registry status is baked into the run record — existence is a separate layer.
    assert all("registry_status" not in p for p in rec["packages"])


def test_failed_call_record_has_no_packages() -> None:
    task = SweepTask(PROMPT, MODEL, 0)
    resp = _fake_response(ok=False, error="Boom: 500")(MODEL, "sys", "hi")
    rec = build_record(task, resp, STDLIB, ALIASES, run_id="r1")
    assert rec["ok"] is False
    assert rec["packages"] == []
    assert rec["error"] == "Boom: 500"


# ----------------------------------------------------------------- sweep + resume


def test_sweep_writes_one_line_per_task(tmp_path) -> None:
    out = tmp_path / "runs.jsonl"
    tasks = build_tasks([MODEL], [PROMPT], runs_per_prompt=3)
    summary = run_sweep(
        tasks, "sys", STDLIB, ALIASES, out_path=out, run_id="r1",
        concurrency=2, retries=0, complete_fn=_fake_response(),
    )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert summary.ok == 3 and summary.failed == 0
    assert summary.attempted == 3 and summary.skipped == 0


def test_resume_skips_completed_and_retries_failed(tmp_path) -> None:
    out = tmp_path / "runs.jsonl"
    tasks = build_tasks([MODEL], [PROMPT], runs_per_prompt=3)

    # First pass: run_index 1 fails, the rest succeed.
    def mixed(model, system_prompt, user_prompt, *, retries=2):
        # run_index isn't passed in; simulate a failure for a specific call via a counter.
        raise AssertionError("unused")

    # Seed a completed OK record for run_index 0 and a FAILED record for run_index 2.
    seeded = [
        {"ok": True, "prompt_id": "py-01", "model_id": "anthropic/opus-5", "run_index": 0},
        {"ok": False, "prompt_id": "py-01", "model_id": "anthropic/opus-5", "run_index": 2},
    ]
    out.write_text("\n".join(json.dumps(r) for r in seeded) + "\n", encoding="utf-8")

    completed = load_completed(out)
    assert completed == {("py-01", "anthropic/opus-5", 0)}  # failed one not counted

    summary = run_sweep(
        tasks, "sys", STDLIB, ALIASES, out_path=out, run_id="r2",
        concurrency=2, retries=0, completed=completed, complete_fn=_fake_response(),
    )
    # run_index 0 skipped; 1 and 2 attempted (2 is a retry of the failed one).
    assert summary.skipped == 1
    assert summary.attempted == 2


def test_collect_unique_names_only_from_ok_records(tmp_path) -> None:
    out = tmp_path / "runs.jsonl"
    records = [
        {"ok": True, "packages": [
            {"ecosystem": "python", "name": "requests"},
            {"ecosystem": "python", "name": "requests"},  # dup
            {"ecosystem": "javascript", "name": "express"},
        ]},
        {"ok": False, "packages": [{"ecosystem": "python", "name": "ghost-from-error"}]},
    ]
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    names = collect_unique_names(out)
    assert names == {"python": ["requests"], "javascript": ["express"]}
    assert "ghost-from-error" not in names.get("python", [])
