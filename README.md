# slopsquat

A measurement harness for **slopsquatting** — the supply-chain attack surface created when
LLMs hallucinate package names that do not exist, which an attacker can then register.

This repository answers one question with data rather than anecdote:

> When an LLM writes code, how often does it invent a package that does not exist —
> and does it invent the **same one** repeatedly?

**Recurrence is the finding.** A model inventing a fake package once is noise. A model
inventing the same fake package across runs, across prompts, and across vendors is a
*predictable* name an attacker can register in advance. Classic typosquatting exploits
scattered human error; slopsquatting exploits reproducible machine error.

## Scope

**Phase 1 (this repository) is detection and measurement only.** It reads public package
registries and never writes to them. No package is registered, reserved, or published.
See [RESEARCH_ETHICS.md](RESEARCH_ETHICS.md).

Phase 1 measures the *precondition* for the attack. It does not measure how often
hallucinated packages are actually installed or executed in the real world — that is a
separate, later question.

## How it works

1. Prompt a configurable set of models with realistic developer coding tasks.
2. Extract every package the model tries to import or install (Python + JavaScript).
3. Check each name against the real PyPI and npm registries.
4. Flag names that do not exist as hallucination candidates.
5. Count recurrence — the same name across prompts, runs, and models.
6. Emit an analysis-ready dataset plus a summary report.

## Install

Requires Python 3.11+. Uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# or:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

uv sync --extra dev
```

## Configure

```bash
cp .env.example .env
# add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

Models, parameters, and run settings live in `config/`. Prompts live in `data/prompts/`.
Nothing about the experiment is hard-coded.

## Run

Each stage runs independently, so you can inspect extraction output **before** spending
any tokens:

```bash
uv run slopsquat prompts                 # validate + list the corpus (no API calls)
uv run slopsquat extract --file <path>   # test extraction on a fixture (no API calls)
uv run slopsquat check requests flask    # registry lookup only (no LLM calls)
uv run slopsquat run --limit 1           # smoke test: one prompt, one run
uv run slopsquat run                     # full sweep
uv run slopsquat report                  # recurrence analysis + exports
```

## Output

- `out/runs.jsonl` — one record per model call: prompt, model, resolved model version,
  raw response, extracted packages, per-package existence status.
- `out/registry-cache/` — cached registry lookups, so repeat runs stay cheap and polite.
- `out/report.md`, `out/recurrence.csv` — analysis outputs.

The JSONL **is** the dataset. DuckDB queries it directly, so there is no hidden ingest
transformation between the raw records and the published numbers.

## Reproducibility notes

- Every record stores the **resolved** model version returned by the API, not just the
  alias requested. `claude-opus-5` and `gpt-…` aliases move; the recorded version does not.
- Model parameters (thinking, effort, temperature where applicable) are part of the config
  and are stored per-record — they are an experimental axis, not a hidden constant.
- Registry lookups distinguish "404, does not exist" from "network error". A network error
  is never recorded as a hallucination.

## Limitations

Stated plainly, because they bound the headline numbers:

- **Extraction accuracy bounds everything.** A package the extractor misses is a false
  negative; a string it wrongly treats as a package is a false positive. Extraction has
  tests and a fixture corpus for this reason, and known gaps are documented.
- **Results are a snapshot.** Model versions change. Claims are scoped to the recorded
  version and date.
- **Prompt design influences rates.** Prompts that nudge toward niche libraries produce
  higher hallucination rates than everyday ones. Results are sliced by prompt category so
  the two are never conflated.
- **Non-existent is not the same as exploitable.** A name only becomes an attack once
  someone registers it.

## Licence

MIT.
