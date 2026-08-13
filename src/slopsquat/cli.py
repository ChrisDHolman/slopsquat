"""Command line entry point.

Each subcommand is independently runnable so the pipeline can be inspected stage by
stage. In particular, `prompts` and `extract` make no API calls at all — extraction
accuracy bounds every headline number, so it must be verifiable before spending tokens.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Force UTF-8 on the output streams before rich wraps them. On Windows the default
# console codepage is cp1252, which crashes on characters like ≥, ×, and • — both in a
# legacy console and whenever stdout is piped. This keeps output portable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from slopsquat.config import (
    ConfigError,
    load_env,
    load_aliases,
    load_models,
    load_prompts,
    load_run_config,
    load_stdlib,
)

# Load .env before any command inspects the environment for API keys.
load_env()

app = typer.Typer(
    add_completion=False,
    help="Measure how often, and how reproducibly, LLMs hallucinate package names.",
)
console = Console()


@app.command()
def prompts(
    ecosystem: str = typer.Option(None, help="Filter to one ecosystem."),
    show_text: bool = typer.Option(False, "--text", help="Print the full prompt text."),
) -> None:
    """Validate and summarise the prompt corpus. Makes no API calls."""
    try:
        corpus = load_prompts([ecosystem] if ecosystem else None)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if not corpus:
        console.print("[yellow]no prompts matched[/yellow]")
        raise typer.Exit(1)

    by_eco = collections.Counter(p.ecosystem for p in corpus)
    by_spec = collections.Counter(p.specificity for p in corpus)
    by_domain = collections.Counter(p.domain for p in corpus)

    console.print(f"\n[bold]{len(corpus)} prompts[/bold] validated\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("breakdown")
    table.add_column("counts")
    table.add_row("ecosystem", "  ".join(f"{k}={v}" for k, v in sorted(by_eco.items())))
    table.add_row("specificity", "  ".join(f"{k}={v}" for k, v in sorted(by_spec.items())))
    table.add_row("domains", str(len(by_domain)))
    console.print(table)

    # Niche prompts deliberately push toward obscure libraries and inflate hallucination
    # rates. Surfacing the split here is a standing reminder to slice results by it.
    niche = by_spec.get("niche", 0)
    if niche:
        pct = 100 * niche / len(corpus)
        console.print(
            f"\n[yellow]note:[/yellow] {niche}/{len(corpus)} ({pct:.0f}%) prompts are "
            "'niche' — report their rates separately from everyday prompts."
        )

    if show_text:
        console.print()
        for p in corpus:
            console.print(f"[dim]{p.id}[/dim] [{p.ecosystem}/{p.specificity}] {p.text}")


@app.command()
def config() -> None:
    """Validate configuration and report what a full sweep would do. No API calls."""
    try:
        models, system_prompt = load_models()
        run = load_run_config()
        corpus = load_prompts(run.ecosystems)
        aliases = load_aliases()
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    enabled = [m for m in models if m.enabled]

    table = Table(show_header=True, header_style="bold", title="models")
    table.add_column("id")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("enabled")
    table.add_column("api key")
    for m in models:
        key = "[green]set[/green]" if m.key_present() else "[red]missing[/red]"
        state = "[green]yes[/green]" if m.enabled else "[dim]no[/dim]"
        table.add_row(m.id, m.provider, m.model, state, key)
    console.print(table)

    calls = len(corpus) * run.runs_per_prompt * len(enabled)
    console.print(
        f"\n[bold]sweep size:[/bold] {len(corpus)} prompts × {run.runs_per_prompt} runs "
        f"× {len(enabled)} enabled model(s) = [bold]{calls}[/bold] model calls"
    )

    for lang in ("python", "javascript"):
        console.print(f"  stdlib/{lang}: {len(load_stdlib(lang))} names")
    console.print(f"  python import aliases: {len(aliases)}")

    if not enabled:
        console.print("\n[yellow]no models enabled — a sweep would do nothing[/yellow]")
    missing = [m.id for m in enabled if not m.key_present()]
    if missing:
        console.print(
            f"\n[red]enabled models with no API key:[/red] {', '.join(missing)}"
            "\n  copy .env.example to .env and fill it in"
        )
    placeholder = [m.id for m in models if m.model == "CHANGEME"]
    if placeholder:
        console.print(
            f"\n[yellow]placeholder model id still set for:[/yellow] {', '.join(placeholder)}"
        )
    if not system_prompt.strip():
        console.print("\n[yellow]warning:[/yellow] system_prompt is empty")


@app.command()
def extract(
    file: Path = typer.Option(..., exists=True, help="File containing a model response."),
    ecosystem: str = typer.Option(
        None, help="python | javascript. Both are scanned if omitted."
    ),
    show_filtered: bool = typer.Option(
        True, "--filtered/--no-filtered", help="Show what was excluded, and why."
    ),
) -> None:
    """Extract package names from a saved response. Makes no API calls.

    Run this against real responses before committing to a sweep: extraction accuracy
    bounds every number the study reports, so it should be inspected, not trusted.
    """
    from slopsquat.extract import extract as run_extraction

    if ecosystem and ecosystem not in {"python", "javascript"}:
        console.print(f"[red]unknown ecosystem[/red] {ecosystem!r}")
        raise typer.Exit(1)

    try:
        stdlib = {lang: load_stdlib(lang) for lang in ("python", "javascript")}
        aliases = load_aliases()
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    text = file.read_text(encoding="utf-8", errors="replace")
    result = run_extraction(text, ecosystem=ecosystem, stdlib=stdlib, aliases=aliases)

    if not result.packages:
        console.print("[yellow]no packages extracted[/yellow]")
    else:
        table = Table(show_header=True, header_style="bold", title="extracted")
        table.add_column("name")
        table.add_column("raw")
        table.add_column("eco")
        table.add_column("source")
        table.add_column("origin")
        table.add_column("alias")
        for p in result.packages:
            alias = {True: "resolved", False: "[yellow]unmapped[/yellow]", None: ""}[
                p.alias_resolved
            ]
            origin = p.origin if p.origin == "code_block" else f"[yellow]{p.origin}[/yellow]"
            table.add_row(p.name, p.raw, p.ecosystem, p.source, origin, alias)
        console.print(table)

        unique = len({p.name for p in result.packages})
        console.print(f"\n[bold]{unique}[/bold] unique package name(s) to check")

    if show_filtered and result.filtered:
        console.print("\n[dim]filtered (not sent to any registry):[/dim]")
        for reason, items in sorted(result.filtered.items()):
            console.print(f"  [dim]{reason}:[/dim] {', '.join(sorted(items))}")

    for note in result.notes:
        console.print(f"\n[yellow]note:[/yellow] {note}")

    # An unmapped import name is checked against PyPI as-is, which is correct but is
    # also how a legitimate package can look like a hallucination. Surface it so the
    # alias table can be extended deliberately.
    unmapped = sorted({p.raw for p in result.packages if p.alias_resolved is False})
    if unmapped:
        console.print(
            f"\n[dim]unmapped python import names ({len(unmapped)}): "
            f"{', '.join(unmapped)}[/dim]"
            "\n[dim]  Expected for most packages — import name and distribution name are"
            "\n  usually identical. Only an unmapped name that ALSO 404s at the registry"
            "\n  needs review, since that is ambiguous between a hallucination and a"
            "\n  missing alias. Stage 3 flags that combination.[/dim]"
        )


@app.command()
def check(
    names: list[str] = typer.Argument(..., help="Package names to look up."),
    ecosystem: str = typer.Option("python", help="python | javascript"),
    refresh: bool = typer.Option(False, help="Ignore cached results and re-query."),
) -> None:
    """Look up package names against PyPI or npm. Makes no LLM calls.

    Read-only. A 404 means the registry does not know the name; a timeout or 5xx means
    we do not know, which is reported as an error and never counted as a negative.
    """
    from slopsquat.registry import Status, build_cache, check_names_sync

    if ecosystem not in {"python", "javascript"}:
        console.print(f"[red]unknown ecosystem[/red] {ecosystem!r}")
        raise typer.Exit(1)

    try:
        run = load_run_config()
        aliases = load_aliases()
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    key = "pypi" if ecosystem == "python" else "npm"
    cache_root = run.paths.get("cache")
    cache = build_cache(cache_root) if cache_root else None

    results = check_names_sync(
        list(names),
        ecosystem,
        url_template=run.registries[key],
        user_agent=run.user_agent,
        cache=cache,
        concurrency=run.concurrency_registry,
        delay=run.registry_delay_seconds,
        retries=run.retries_registry,
        refresh=refresh,
    )

    style = {
        Status.EXISTS: "[green]exists[/green]",
        Status.NOT_FOUND: "[red]NOT FOUND[/red]",
        Status.ERROR: "[yellow]error[/yellow]",
    }

    table = Table(show_header=True, header_style="bold", title=f"{key} lookups")
    table.add_column("name")
    table.add_column("status")
    table.add_column("http")
    table.add_column("source")
    table.add_column("detail")
    for r in results:
        table.add_row(
            r.name,
            style.get(r.status, r.status),
            str(r.http_status or ""),
            "cache" if r.from_cache else "live",
            (r.detail or "")[:60],
        )
    console.print(table)

    missing = [r for r in results if r.status == Status.NOT_FOUND]
    errors = [r for r in results if r.status == Status.ERROR]

    console.print(
        f"\n[bold]{len(missing)}[/bold] not found · "
        f"{len(results) - len(missing) - len(errors)} exist · "
        f"{len(errors)} undetermined"
    )

    if errors:
        console.print(
            "\n[yellow]undetermined results are NOT hallucinations.[/yellow]\n"
            "  They were not cached and will be retried on the next run."
        )

    # The genuinely ambiguous case, promised at stage 2: a Python import name that is
    # absent from the alias table AND 404s. It is either a real hallucination or a
    # legitimate package whose distribution name simply is not mapped yet. Only a human
    # can tell, so it is surfaced rather than silently counted.
    if ecosystem == "python" and missing:
        ambiguous = [r.name for r in missing if r.name not in aliases.values()]
        if ambiguous:
            console.print(
                f"\n[yellow]needs review ({len(ambiguous)}):[/yellow] "
                f"{', '.join(ambiguous)}"
                "\n  Absent from PyPI and not in the alias table. Confirm each is genuinely"
                "\n  invented rather than a real project with a differing distribution name"
                "\n  before counting it as a hallucination."
            )


@app.command()
def probe(
    model_id: str = typer.Option(..., "--model", help="Model id from config/models.yaml."),
    prompt: str = typer.Option(None, help="Prompt text. Overrides --prompt-id."),
    prompt_id: str = typer.Option(None, help="Use a prompt from the corpus by id."),
    ecosystem: str = typer.Option(
        None, help="Ecosystem for extraction. Defaults to the corpus prompt's own."
    ),
    show_text: bool = typer.Option(False, "--text", help="Print the full response body."),
) -> None:
    """Send ONE prompt to ONE model and show the response, metadata, and extracted names.

    This is the manual verification tool for the model adapters, and it makes exactly one
    paid API call. Use it to sanity-check a model end to end before committing to a sweep.
    """
    from slopsquat.extract import extract as run_extraction
    from slopsquat.models import complete

    try:
        models, system_prompt = load_models()
        aliases = load_aliases()
        stdlib = {lang: load_stdlib(lang) for lang in ("python", "javascript")}
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    model = next((m for m in models if m.id == model_id), None)
    if model is None:
        console.print(
            f"[red]unknown model id[/red] {model_id!r}\n"
            f"  known: {', '.join(m.id for m in models)}"
        )
        raise typer.Exit(1)
    if not model.key_present():
        console.print(
            f"[red]no API key[/red] for {model.provider} "
            f"(expected {model.env_var} in the environment or .env)"
        )
        raise typer.Exit(1)

    # Resolve the prompt: explicit text wins, else a corpus prompt by id.
    text = prompt
    eco = ecosystem
    if text is None:
        if prompt_id is None:
            console.print("[red]provide --prompt or --prompt-id[/red]")
            raise typer.Exit(1)
        corpus = {p.id: p for p in load_prompts()}
        chosen = corpus.get(prompt_id)
        if chosen is None:
            console.print(f"[red]unknown prompt id[/red] {prompt_id!r}")
            raise typer.Exit(1)
        text = chosen.text
        eco = eco or chosen.ecosystem

    console.print(f"[dim]calling {model.id} ({model.model})…[/dim]")
    resp = complete(model, system_prompt, text)

    meta = Table(show_header=False, title="model call")
    meta.add_column("field", style="bold")
    meta.add_column("value")
    meta.add_row("ok", "[green]yes[/green]" if resp.ok else "[red]no[/red]")
    meta.add_row("resolved model", resp.resolved_model or "—")
    meta.add_row("latency", f"{resp.latency_s:.1f}s")
    meta.add_row("tokens", f"in={resp.input_tokens}  out={resp.output_tokens}")
    meta.add_row("stop reason", str(resp.stop_reason))
    if resp.truncated:
        meta.add_row("truncated", "[red]YES — response hit max_tokens[/red]")
    if resp.error:
        meta.add_row("error", f"[red]{resp.error}[/red]")
    console.print(meta)

    if not resp.ok:
        raise typer.Exit(1)

    if resp.truncated:
        console.print(
            "\n[yellow]warning:[/yellow] response was truncated — extracted names may be"
            " incomplete. Raise max_tokens for this model before trusting a sweep."
        )

    if show_text:
        console.print("\n[dim]--- response ---[/dim]")
        console.print(resp.text)

    result = run_extraction(resp.text, ecosystem=eco, stdlib=stdlib, aliases=aliases)
    if not result.packages:
        console.print("\n[yellow]no packages extracted[/yellow]")
        return

    table = Table(show_header=True, header_style="bold", title="extracted names")
    table.add_column("name")
    table.add_column("eco")
    table.add_column("source")
    for p in result.packages:
        table.add_row(p.name, p.ecosystem, p.source)
    console.print(table)
    console.print(
        f"\n[dim]{len({p.name for p in result.packages})} unique name(s). "
        "Run `slopsquat check …` to see which exist.[/dim]"
    )


@app.command()
def run(
    limit: int = typer.Option(None, help="Only use the first N prompts (smoke test)."),
    runs: int = typer.Option(None, help="Override runs_per_prompt (e.g. a cheap first pass)."),
    model_id: str = typer.Option(
        None, "--model", help="Restrict the sweep to one model id."
    ),
    dry_run: bool = typer.Option(False, help="Show what would run without calling any API."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the spend confirmation."),
    skip_existence: bool = typer.Option(
        False, help="Run the model sweep only; don't resolve registry existence."
    ),
) -> None:
    """Execute the sweep: every enabled model × prompt × run, recorded to runs.jsonl.

    Resumable — re-running continues where an interrupted sweep left off, skipping calls
    already recorded as successful and retrying failed ones. This command spends money.
    """
    from slopsquat.pipeline import build_tasks, load_completed, now_run_id, run_sweep

    try:
        models, system_prompt = load_models()
        run_cfg = load_run_config()
        prompts = load_prompts(run_cfg.ecosystems)
        aliases = load_aliases()
        stdlib = {lang: load_stdlib(lang) for lang in ("python", "javascript")}
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if model_id:
        models = [m for m in models if m.id == model_id]
        if not models:
            console.print(f"[red]unknown model id[/red] {model_id!r}")
            raise typer.Exit(1)

    enabled = [m for m in models if m.enabled]
    missing_key = [m.id for m in enabled if not m.key_present()]
    if missing_key:
        console.print(f"[red]enabled models with no API key:[/red] {', '.join(missing_key)}")
        raise typer.Exit(1)
    if not enabled:
        console.print("[yellow]no enabled models — nothing to do[/yellow]")
        raise typer.Exit(1)

    if limit:
        prompts = prompts[:limit]
    runs_per_prompt = runs if runs is not None else run_cfg.runs_per_prompt

    tasks = build_tasks(enabled, prompts, runs_per_prompt)
    out_path = run_cfg.paths["raw"]
    completed = load_completed(out_path)
    pending = [t for t in tasks if t.key not in completed]

    console.print(
        f"\n[bold]sweep plan[/bold]\n"
        f"  models:   {', '.join(m.id for m in enabled)}\n"
        f"  prompts:  {len(prompts)}\n"
        f"  runs:     {runs_per_prompt} per (prompt, model)\n"
        f"  total:    {len(tasks)} calls\n"
        f"  already done: {len(completed & {t.key for t in tasks})}\n"
        f"  [bold]to run now: {len(pending)}[/bold]\n"
        f"  output:   {out_path}"
    )

    if dry_run:
        console.print("\n[dim]dry run — no API calls made[/dim]")
        return
    if not pending:
        console.print("\n[green]nothing pending — sweep already complete[/green]")
        if not skip_existence:
            _resolve_existence(run_cfg, out_path, console)
        return

    if not yes:
        console.print(
            "\n[yellow]this will make paid API calls.[/yellow] "
            "Re-run with [bold]--yes[/bold] to proceed, or --dry-run to inspect."
        )
        raise typer.Exit(1)

    run_id = now_run_id()
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    counters = {"ok": 0, "failed": 0, "trunc": 0}
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("sweeping", total=len(pending))

        def on_record(rec: dict) -> None:
            if rec.get("ok"):
                counters["ok"] += 1
            else:
                counters["failed"] += 1
            if rec.get("truncated"):
                counters["trunc"] += 1
            progress.update(
                bar,
                advance=1,
                description=f"sweeping (ok={counters['ok']} fail={counters['failed']})",
            )

        summary = run_sweep(
            tasks,
            system_prompt,
            stdlib,
            aliases,
            out_path=out_path,
            run_id=run_id,
            concurrency=run_cfg.concurrency_models,
            retries=run_cfg.retries_model,
            completed=completed,
            progress=on_record,
        )

    console.print(
        f"\n[bold]sweep complete[/bold] — attempted {summary.attempted}, "
        f"[green]{summary.ok} ok[/green], "
        f"{'[red]' if summary.failed else ''}{summary.failed} failed"
        f"{'[/red]' if summary.failed else ''}, "
        f"{summary.truncated} truncated"
    )
    if summary.failed:
        console.print(
            "[dim]failed calls are recorded but excluded from results. Re-run this "
            "command to retry them.[/dim]"
        )
    if summary.truncated:
        console.print(
            "[yellow]some responses were truncated[/yellow] — their package lists may be "
            "incomplete. Consider raising max_tokens and re-running those cells."
        )

    if not skip_existence:
        _resolve_existence(run_cfg, out_path, console)


def _resolve_existence(run_cfg, out_path, console) -> None:
    """Resolve registry existence for every extracted name into a timestamped snapshot."""
    from slopsquat.pipeline import collect_unique_names, write_existence_snapshot
    from slopsquat.registry import build_cache

    names_by_eco = collect_unique_names(out_path)
    total = sum(len(v) for v in names_by_eco.values())
    if not total:
        console.print("\n[dim]no names extracted — skipping existence check[/dim]")
        return

    snapshot_path = out_path.parent / "registry.jsonl"
    cache_root = run_cfg.paths.get("cache")
    cache = build_cache(cache_root) if cache_root else None

    console.print(f"\n[dim]resolving existence for {total} unique name(s)…[/dim]")
    records = write_existence_snapshot(
        names_by_eco,
        snapshot_path,
        registries=run_cfg.registries,
        user_agent=run_cfg.user_agent,
        cache=cache,
        concurrency=run_cfg.concurrency_registry,
        delay=run_cfg.registry_delay_seconds,
        retries=run_cfg.retries_registry,
    )

    from collections import Counter

    by_status = Counter(r["status"] for r in records)
    console.print(
        f"[bold]existence snapshot[/bold] → {snapshot_path}\n"
        f"  exists={by_status.get('exists', 0)}  "
        f"not_found={by_status.get('not_found', 0)}  "
        f"error={by_status.get('error', 0)}"
    )
    if by_status.get("error"):
        console.print(
            "[yellow]some names are undetermined (network error).[/yellow] They are NOT "
            "counted as hallucinations. Re-run to resolve them."
        )


@app.command()
def report(
    charts: bool = typer.Option(False, help="Also render PNG charts (needs matplotlib)."),
    top: int = typer.Option(20, help="How many recurring hallucinations to print."),
) -> None:
    """Analyse the sweep: hallucination rates and recurrence. Makes no API calls.

    Joins runs.jsonl (what models said) with registry.jsonl (what exists) in DuckDB and
    writes a Markdown report plus a recurrence CSV.
    """
    from slopsquat.report import (
        ReportError,
        analyse,
        render_markdown,
        write_charts,
        write_csv,
    )

    try:
        run_cfg = load_run_config()
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    runs_path = run_cfg.paths["raw"]
    registry_path = runs_path.parent / "registry.jsonl"

    try:
        rep = analyse(runs_path, registry_path)
    except ReportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    t = rep.totals
    console.print(
        f"\n[bold]{t['records']} records[/bold] — {t['ok']} ok, {t['failed']} failed, "
        f"{t['truncated']} truncated\n"
    )

    tbl = Table(show_header=True, header_style="bold", title="hallucination rate by model")
    tbl.add_column("model")
    tbl.add_column("ok resp", justify="right")
    tbl.add_column("checkable", justify="right")
    tbl.add_column("halluc", justify="right")
    tbl.add_column("mention rate", justify="right")
    tbl.add_column("distinct", justify="right")
    tbl.add_column("resp rate", justify="right")
    for r in rep.per_model:
        mr = "—" if r["mention_rate"] is None else f"{100 * r['mention_rate']:.1f}%"
        rr = "—" if r["response_rate"] is None else f"{100 * r['response_rate']:.1f}%"
        tbl.add_row(
            r["model_id"], str(r["ok_responses"]), str(r["checkable"]),
            str(r["halluc_mentions"]), mr, str(r["distinct_halluc"]), rr,
        )
    console.print(tbl)

    spec = Table(show_header=True, header_style="bold", title="by specificity (niche inflates)")
    spec.add_column("specificity")
    spec.add_column("checkable", justify="right")
    spec.add_column("halluc", justify="right")
    spec.add_column("rate", justify="right")
    for r in rep.by_specificity:
        mr = "—" if r["mention_rate"] is None else f"{100 * r['mention_rate']:.1f}%"
        spec.add_row(r["specificity"], str(r["checkable"]), str(r["halluc_mentions"]), mr)
    console.print(spec)

    rs = rep.recurrence_summary
    if rs.get("cells"):
        console.print(
            f"\n[bold]recurrence:[/bold] {rs['cells']} hallucinated (model, prompt, name) "
            f"cells — {rs['recurring_2plus']} recurred in ≥2 runs, {rs['every_run']} in "
            f"[bold]every[/bold] run of their prompt."
        )
        rec = Table(show_header=True, header_style="bold", title=f"top {top} recurring")
        rec.add_column("model")
        rec.add_column("prompt")
        rec.add_column("name")
        rec.add_column("appears", justify="right")
        rec.add_column("rate", justify="right")
        rec.add_column("status")
        for r in rep.recurrence[:top]:
            rr = "—" if r["recurrence_rate"] is None else f"{100 * r['recurrence_rate']:.0f}%"
            status = (
                f"[yellow]{r['status']}[/yellow]" if r["needs_review"] else r["status"]
            )
            rec.add_row(
                r["model_id"], r["prompt_id"], r["name"],
                f"{r['appearances']}/{r['runs']}", rr, status,
            )
        console.print(rec)
    else:
        console.print("\n[green]no hallucinations found in checkable mentions.[/green]")

    # Written artefacts.
    report_path = run_cfg.paths.get("report", runs_path.parent / "report.md")
    csv_path = run_cfg.paths.get("csv", runs_path.parent / "recurrence.csv")
    written_charts: list = []
    if charts:
        written_charts = write_charts(rep, runs_path.parent / "charts")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(rep), encoding="utf-8")
    write_csv(rep, csv_path)

    console.print(f"\n[dim]written:[/dim] {report_path}")
    console.print(f"[dim]written:[/dim] {csv_path}")
    for p in written_charts:
        console.print(f"[dim]written:[/dim] {p}")

    if rep.caveats:
        console.print("\n[yellow]caveats:[/yellow]")
        for cav in rep.caveats:
            console.print(f"  • {cav}")


if __name__ == "__main__":
    app()
