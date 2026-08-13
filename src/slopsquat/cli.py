"""Command line entry point.

Each subcommand is independently runnable so the pipeline can be inspected stage by
stage. In particular, `prompts` and `extract` make no API calls at all — extraction
accuracy bounds every headline number, so it must be verifiable before spending tokens.
"""

from __future__ import annotations

import collections
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from slopsquat.config import (
    ConfigError,
    load_aliases,
    load_models,
    load_prompts,
    load_run_config,
    load_stdlib,
)

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
def check(names: list[str]) -> None:
    """Look up package names against PyPI/npm. No LLM calls. [stage 3]"""
    console.print("[yellow]not implemented yet — stage 3[/yellow]")
    raise typer.Exit(1)


@app.command()
def run(
    limit: int = typer.Option(None, help="Only run the first N prompts (smoke test)."),
    dry_run: bool = typer.Option(False, help="Show what would run without calling any API."),
) -> None:
    """Execute the full pipeline. [stage 5]"""
    console.print("[yellow]not implemented yet — stage 5[/yellow]")
    raise typer.Exit(1)


@app.command()
def report() -> None:
    """Recurrence analysis and exports. [stage 6]"""
    console.print("[yellow]not implemented yet — stage 6[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
