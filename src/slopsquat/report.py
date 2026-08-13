"""Analysis: turn the two JSONL layers into defensible numbers.

Reads ``runs.jsonl`` (what each model said) and ``registry.jsonl`` (whether each name
exists) with DuckDB, joins them, and reports hallucination rates and — the headline —
*recurrence*: how reliably a model re-invents the same non-existent name across repeated
runs of one prompt. A name that recurs is the registerable, dangerous kind; a one-off is
noise.

Definitions, applied consistently and printed in the report so a reader can audit them:

* **mention** — one extracted package reference in one successful response.
* **checkable mention** — a mention whose name resolved to a *definitive* registry status
  (``exists`` or ``not_found``). ``error`` (undetermined) is excluded from both numerator
  and denominator: a network failure is not evidence either way.
* **hallucination** — a checkable mention with status ``not_found``.
* **needs-review** — a Python hallucination whose import name was never mapped to a
  distribution name (``alias_resolved = false``). It is genuinely ambiguous between an
  invented name and a real project with a differing distribution name, so it is reported
  separately rather than silently counted.

Niche prompts deliberately push toward obscure libraries and inflate the rate, so every
table is also broken down by specificity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb


def _lit(path: Path) -> str:
    """A single-quoted SQL string literal for a path (paths can't be bound parameters in
    a CREATE VIEW)."""
    return "'" + str(path).replace("'", "''") + "'"


class ReportError(Exception):
    pass


@dataclass
class Report:
    runs_path: Path
    registry_path: Path
    totals: dict[str, Any] = field(default_factory=dict)
    per_model: list[dict[str, Any]] = field(default_factory=list)
    by_specificity: list[dict[str, Any]] = field(default_factory=list)
    by_ecosystem: list[dict[str, Any]] = field(default_factory=list)
    recurrence: list[dict[str, Any]] = field(default_factory=list)
    recurrence_summary: dict[str, int] = field(default_factory=dict)
    needs_review: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def _connect(runs_path: Path, registry_path: Path) -> duckdb.DuckDBPyConnection:
    if not runs_path.exists():
        raise ReportError(f"no run data at {runs_path} — run a sweep first")
    if not registry_path.exists():
        raise ReportError(
            f"no existence snapshot at {registry_path} — run `slopsquat run` "
            "(or the existence pass) first"
        )

    c = duckdb.connect()
    c.execute(
        f"CREATE VIEW runs AS SELECT * FROM read_json({_lit(runs_path)}, "
        "format='newline_delimited')"
    )
    # Flatten packages to one row per mention, keeping only successful responses.
    c.execute(
        "CREATE VIEW mentions AS SELECT run_id, prompt_id, ecosystem, domain, "
        "specificity, model_id, run_index, pkg.name AS name, pkg.source AS source, "
        "pkg.origin AS origin, pkg.alias_resolved AS alias_resolved "
        "FROM (SELECT *, unnest(packages) AS pkg FROM runs WHERE ok)"
    )
    c.execute(
        f"CREATE VIEW names AS SELECT ecosystem, name, status FROM "
        f"read_json({_lit(registry_path)}, format='newline_delimited')"
    )
    # Every mention with its resolved status attached (LEFT JOIN: an unchecked name shows
    # NULL status and is treated as undetermined, never as a hallucination).
    c.execute(
        "CREATE VIEW m AS SELECT me.*, n.status AS status "
        "FROM mentions me LEFT JOIN names n USING (ecosystem, name)"
    )
    c.execute(
        "CREATE VIEW cell_runs AS SELECT model_id, prompt_id, "
        "count(DISTINCT run_index) AS runs FROM runs WHERE ok GROUP BY 1, 2"
    )
    return c


def analyse(runs_path: Path, registry_path: Path) -> Report:
    c = _connect(runs_path, registry_path)
    rep = Report(runs_path=runs_path, registry_path=registry_path)

    total, ok, failed, truncated = c.execute(
        "SELECT count(*), count(*) FILTER (WHERE ok), count(*) FILTER (WHERE NOT ok), "
        "count(*) FILTER (WHERE truncated) FROM runs"
    ).fetchone()
    undetermined = c.execute(
        "SELECT count(*) FROM m WHERE status='error' OR status IS NULL"
    ).fetchone()[0]
    rep.totals = {
        "records": total,
        "ok": ok,
        "failed": failed,
        "truncated": truncated,
        "undetermined_mentions": undetermined,
    }

    # --- per model -----------------------------------------------------------------
    rows = c.execute(
        """
        WITH resp AS (
          SELECT model_id, count(*) AS ok_responses FROM runs WHERE ok GROUP BY 1
        ),
        halluc_resp AS (
          SELECT model_id, count(*) AS n FROM (
            SELECT model_id, run_id, prompt_id, run_index,
                   max(CASE WHEN status='not_found' THEN 1 ELSE 0 END) AS has_h
            FROM m GROUP BY 1,2,3,4
          ) WHERE has_h=1 GROUP BY model_id
        ),
        agg AS (
          SELECT model_id,
            count(*) AS mentions,
            count(DISTINCT name) AS distinct_names,
            count(*) FILTER (WHERE status IN ('exists','not_found')) AS checkable,
            count(*) FILTER (WHERE status='not_found') AS halluc_mentions,
            count(DISTINCT name) FILTER (WHERE status='not_found') AS distinct_halluc
          FROM m GROUP BY model_id
        )
        SELECT resp.model_id, resp.ok_responses,
               COALESCE(agg.mentions,0), COALESCE(agg.distinct_names,0),
               COALESCE(agg.checkable,0), COALESCE(agg.halluc_mentions,0),
               COALESCE(agg.distinct_halluc,0), COALESCE(halluc_resp.n,0)
        FROM resp
        LEFT JOIN agg USING(model_id)
        LEFT JOIN halluc_resp USING(model_id)
        ORDER BY resp.model_id
        """
    ).fetchall()
    for (mid, ok_resp, mentions, distinct_names, checkable, halluc, distinct_h, h_resp) in rows:
        rep.per_model.append(
            {
                "model_id": mid,
                "ok_responses": ok_resp,
                "mentions": mentions,
                "distinct_names": distinct_names,
                "checkable": checkable,
                "halluc_mentions": halluc,
                "distinct_halluc": distinct_h,
                "mention_rate": (halluc / checkable) if checkable else None,
                "responses_with_halluc": h_resp,
                "response_rate": (h_resp / ok_resp) if ok_resp else None,
            }
        )

    # --- by specificity (niche is reported separately by construction) --------------
    for (spec, checkable, halluc, dn, dh) in c.execute(
        """
        SELECT specificity,
               count(*) FILTER (WHERE status IN ('exists','not_found')) AS checkable,
               count(*) FILTER (WHERE status='not_found') AS halluc,
               count(DISTINCT name) AS distinct_names,
               count(DISTINCT name) FILTER (WHERE status='not_found') AS distinct_halluc
        FROM m GROUP BY specificity ORDER BY specificity
        """
    ).fetchall():
        rep.by_specificity.append(
            {
                "specificity": spec,
                "checkable": checkable,
                "halluc_mentions": halluc,
                "mention_rate": (halluc / checkable) if checkable else None,
                "distinct_names": dn,
                "distinct_halluc": dh,
            }
        )

    # --- by ecosystem ---------------------------------------------------------------
    for (eco, checkable, halluc) in c.execute(
        """
        SELECT ecosystem,
               count(*) FILTER (WHERE status IN ('exists','not_found')) AS checkable,
               count(*) FILTER (WHERE status='not_found') AS halluc
        FROM m GROUP BY ecosystem ORDER BY ecosystem
        """
    ).fetchall():
        rep.by_ecosystem.append(
            {
                "ecosystem": eco,
                "checkable": checkable,
                "halluc_mentions": halluc,
                "mention_rate": (halluc / checkable) if checkable else None,
            }
        )

    # --- recurrence: the headline ---------------------------------------------------
    for (mid, pid, eco, name, appears, runs, needs_review) in c.execute(
        """
        SELECT me.model_id, me.prompt_id, me.ecosystem, me.name,
               count(DISTINCT me.run_index) AS appearances,
               cr.runs AS runs,
               (me.ecosystem='python' AND bool_or(me.alias_resolved = false)) AS needs_review
        FROM m me JOIN cell_runs cr USING (model_id, prompt_id)
        WHERE me.status='not_found'
        GROUP BY me.model_id, me.prompt_id, me.ecosystem, me.name, cr.runs
        ORDER BY appearances DESC, me.model_id, me.name
        """
    ).fetchall():
        rep.recurrence.append(
            {
                "model_id": mid,
                "prompt_id": pid,
                "ecosystem": eco,
                "name": name,
                "appearances": appears,
                "runs": runs,
                "recurrence_rate": (appears / runs) if runs else None,
                "needs_review": bool(needs_review),
                "status": "needs_review" if needs_review else "confirmed",
            }
        )

    # Distribution of hallucinated (model, prompt, name) cells by how reliably they recur.
    rs = {"cells": 0, "one_off": 0, "recurring_2plus": 0, "majority": 0, "every_run": 0}
    for r in rep.recurrence:
        rs["cells"] += 1
        if r["appearances"] == 1:
            rs["one_off"] += 1
        if r["appearances"] >= 2:
            rs["recurring_2plus"] += 1
        if r["recurrence_rate"] is not None and r["recurrence_rate"] >= 0.5:
            rs["majority"] += 1
        if r["recurrence_rate"] == 1.0 and r["runs"] > 1:
            rs["every_run"] += 1
    rep.recurrence_summary = rs

    # --- needs-review candidates (distinct names) -----------------------------------
    seen: set[tuple[str, str]] = set()
    for r in rep.recurrence:
        if r["needs_review"] and (r["ecosystem"], r["name"]) not in seen:
            seen.add((r["ecosystem"], r["name"]))
            rep.needs_review.append({"ecosystem": r["ecosystem"], "name": r["name"]})

    # --- caveats --------------------------------------------------------------------
    if failed:
        rep.caveats.append(
            f"{failed} model call(s) failed and are excluded from all rates."
        )
    if truncated:
        rep.caveats.append(
            f"{truncated} response(s) were truncated at max_tokens; their package lists "
            "may be incomplete, biasing rates downward."
        )
    if undetermined:
        rep.caveats.append(
            f"{undetermined} mention(s) have an undetermined registry status (network "
            "error or unchecked) and are excluded from rates — re-run the existence pass "
            "to resolve them."
        )
    if rep.needs_review:
        rep.caveats.append(
            f"{len(rep.needs_review)} hallucination candidate(s) need manual review "
            "(unmapped Python import name that 404s). Confirm each before citing."
        )
    c.close()
    return rep


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def render_markdown(rep: Report) -> str:
    t = rep.totals
    lines: list[str] = []
    lines.append("# Slopsquat results\n")
    lines.append(
        f"Source: `{rep.runs_path.name}` × `{rep.registry_path.name}`  \n"
        f"{t['records']} records — {t['ok']} ok, {t['failed']} failed, "
        f"{t['truncated']} truncated.\n"
    )

    lines.append("## Definitions\n")
    lines.append(
        "- **hallucination** — an extracted package name that returned a *definitive* "
        "`not_found` from its registry.\n"
        "- Undetermined statuses (network errors) are excluded from every rate.\n"
        "- **recurrence** — how many of a prompt's repeated runs re-invented the same "
        "name. Recurring names are the registerable risk; one-offs are noise.\n"
    )

    lines.append("## Per model\n")
    lines.append(
        "| model | ok responses | mentions | checkable | halluc | mention rate | "
        "distinct halluc | responses w/ halluc | response rate |\n"
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    for r in rep.per_model:
        lines.append(
            f"| {r['model_id']} | {r['ok_responses']} | {r['mentions']} | "
            f"{r['checkable']} | {r['halluc_mentions']} | {_pct(r['mention_rate'])} | "
            f"{r['distinct_halluc']} | {r['responses_with_halluc']} | "
            f"{_pct(r['response_rate'])} |\n"
        )

    lines.append("\n## By specificity\n")
    lines.append("*Niche prompts target obscure libraries and inflate the rate by design.*\n\n")
    lines.append(
        "| specificity | checkable | halluc | mention rate | distinct halluc |\n"
        "|---|--:|--:|--:|--:|\n"
    )
    for r in rep.by_specificity:
        lines.append(
            f"| {r['specificity']} | {r['checkable']} | {r['halluc_mentions']} | "
            f"{_pct(r['mention_rate'])} | {r['distinct_halluc']} |\n"
        )

    lines.append("\n## By ecosystem\n")
    lines.append("| ecosystem | checkable | halluc | mention rate |\n|---|--:|--:|--:|\n")
    for r in rep.by_ecosystem:
        lines.append(
            f"| {r['ecosystem']} | {r['checkable']} | {r['halluc_mentions']} | "
            f"{_pct(r['mention_rate'])} |\n"
        )

    rs = rep.recurrence_summary
    lines.append("\n## Recurrence\n")
    if rs.get("cells"):
        lines.append(
            f"{rs['cells']} distinct hallucinated (model, prompt, name) cells. "
            f"{rs['recurring_2plus']} recurred in ≥2 runs, {rs['majority']} in a majority "
            f"of runs, {rs['every_run']} in *every* run of their prompt.\n\n"
        )
    lines.append("Top recurring hallucinations:\n\n")
    lines.append("| model | prompt | name | appears | of runs | rate | status |\n|---|---|---|--:|--:|--:|---|\n")
    for r in rep.recurrence[:40]:
        lines.append(
            f"| {r['model_id']} | {r['prompt_id']} | `{r['name']}` | {r['appearances']} | "
            f"{r['runs']} | {_pct(r['recurrence_rate'])} | {r['status']} |\n"
        )

    if rep.needs_review:
        lines.append("\n## Needs manual review\n")
        lines.append(
            "Unmapped Python import names that 404 — ambiguous between an invented name "
            "and a real project with a differing distribution name:\n\n"
        )
        for r in rep.needs_review:
            lines.append(f"- `{r['name']}`\n")

    if rep.caveats:
        lines.append("\n## Caveats\n")
        for cav in rep.caveats:
            lines.append(f"- {cav}\n")

    return "".join(lines)


def write_csv(rep: Report, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["model_id", "prompt_id", "ecosystem", "name", "appearances", "runs",
             "recurrence_rate", "status"]
        )
        for r in rep.recurrence:
            w.writerow(
                [r["model_id"], r["prompt_id"], r["ecosystem"], r["name"],
                 r["appearances"], r["runs"],
                 "" if r["recurrence_rate"] is None else f"{r['recurrence_rate']:.4f}",
                 r["status"]]
            )


def write_charts(rep: Report, out_dir: Path) -> list[Path]:
    """Bar charts of hallucination rate by model and by specificity. Optional — returns
    an empty list (and adds a caveat) if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        rep.caveats.append("matplotlib not installed — charts skipped (`pip install matplotlib`).")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    model_rows = [r for r in rep.per_model if r["mention_rate"] is not None]
    if model_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([r["model_id"] for r in model_rows],
               [100 * r["mention_rate"] for r in model_rows], color="#c0392b")
        ax.set_ylabel("hallucination rate (%)")
        ax.set_title("Hallucination rate by model")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p = out_dir / "rate_by_model.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    spec_rows = [r for r in rep.by_specificity if r["mention_rate"] is not None]
    if spec_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([r["specificity"] for r in spec_rows],
               [100 * r["mention_rate"] for r in spec_rows], color="#e67e22")
        ax.set_ylabel("hallucination rate (%)")
        ax.set_title("Hallucination rate by prompt specificity")
        fig.tight_layout()
        p = out_dir / "rate_by_specificity.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    return written
