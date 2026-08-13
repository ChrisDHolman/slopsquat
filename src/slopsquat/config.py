"""Configuration loading and validation.

Everything about the experiment lives in config/ and data/ — nothing here hard-codes a
model, a prompt, or a registry URL. Validation is strict and fails loudly: a typo in a
config file must not silently change what the experiment measures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"

VALID_ECOSYSTEMS = {"python", "javascript"}
VALID_SPECIFICITY = {"vague", "detailed", "niche"}
VALID_PROVIDERS = {"anthropic", "openai"}


def load_env(path: Path | None = None) -> bool:
    """Load API keys from .env into the environment.

    Called from the CLI rather than at import time: importing a library should not
    silently mutate the environment, and tests must not pick up a developer's real keys.

    Returns True if a .env file was found and read.
    """
    from dotenv import load_dotenv

    env_path = path or REPO_ROOT / ".env"
    if not env_path.exists():
        return False
    # override=False so an explicitly exported variable always wins over the file —
    # otherwise a stale .env silently shadows the key you just set for one run.
    load_dotenv(env_path, override=False)
    return True


class ConfigError(Exception):
    """Raised when configuration or corpus data is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    model: str
    enabled: bool
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def env_var(self) -> str:
        return {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[self.provider]

    def key_present(self) -> bool:
        return bool(os.environ.get(self.env_var))


@dataclass(frozen=True)
class Prompt:
    id: str
    ecosystem: str
    domain: str
    specificity: str
    text: str


@dataclass(frozen=True)
class RunConfig:
    runs_per_prompt: int
    concurrency_models: int
    concurrency_registry: int
    registry_delay_seconds: float
    retries_model: int
    retries_registry: int
    ecosystems: list[str]
    registries: dict[str, str]
    user_agent: str
    paths: dict[str, Path]


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc


def load_models(path: Path | None = None) -> tuple[list[ModelConfig], str]:
    """Return (models, system_prompt)."""
    path = path or CONFIG_DIR / "models.yaml"
    raw = _load_yaml(path)

    if not isinstance(raw, dict) or "models" not in raw:
        raise ConfigError(f"{path}: expected a top-level 'models' list")

    system_prompt = raw.get("system_prompt", "")
    models: list[ModelConfig] = []
    seen_ids: set[str] = set()

    for entry in raw["models"]:
        for required in ("id", "provider", "model"):
            if required not in entry:
                raise ConfigError(f"{path}: model entry missing '{required}': {entry}")

        if entry["id"] in seen_ids:
            raise ConfigError(f"{path}: duplicate model id {entry['id']!r}")
        seen_ids.add(entry["id"])

        if entry["provider"] not in VALID_PROVIDERS:
            raise ConfigError(
                f"{path}: unknown provider {entry['provider']!r} "
                f"(expected one of {sorted(VALID_PROVIDERS)})"
            )

        models.append(
            ModelConfig(
                id=entry["id"],
                provider=entry["provider"],
                model=entry["model"],
                enabled=bool(entry.get("enabled", True)),
                params=entry.get("params") or {},
            )
        )

    return models, system_prompt


def load_run_config(path: Path | None = None) -> RunConfig:
    path = path or CONFIG_DIR / "run.yaml"
    raw = _load_yaml(path)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping")

    runs = raw.get("runs_per_prompt")
    if not isinstance(runs, int) or runs < 1:
        raise ConfigError(f"{path}: runs_per_prompt must be a positive integer, got {runs!r}")

    ecosystems = raw.get("ecosystems") or []
    unknown = set(ecosystems) - VALID_ECOSYSTEMS
    if unknown:
        raise ConfigError(f"{path}: unknown ecosystem(s) {sorted(unknown)}")

    registries = raw.get("registries") or {}
    for name in ("pypi", "npm"):
        if name not in registries:
            raise ConfigError(f"{path}: registries.{name} is required")
        if "{name}" not in registries[name]:
            raise ConfigError(f"{path}: registries.{name} must contain a '{{name}}' placeholder")

    concurrency = raw.get("concurrency") or {}
    retries = raw.get("retries") or {}
    paths_raw = raw.get("paths") or {}

    return RunConfig(
        runs_per_prompt=runs,
        concurrency_models=int(concurrency.get("models", 4)),
        concurrency_registry=int(concurrency.get("registry", 8)),
        registry_delay_seconds=float(raw.get("registry_delay_seconds", 0.1)),
        retries_model=int(retries.get("model_calls", 2)),
        retries_registry=int(retries.get("registry_calls", 3)),
        ecosystems=list(ecosystems),
        registries=dict(registries),
        user_agent=str(raw.get("user_agent", "slopsquat-research/0.1")),
        paths={k: REPO_ROOT / v for k, v in paths_raw.items()},
    )


def load_prompts(
    ecosystems: list[str] | None = None, prompts_dir: Path | None = None
) -> list[Prompt]:
    """Load and validate the prompt corpus.

    Prompt ids are the stable reference used by every result record, so duplicates are a
    hard error rather than a warning — a duplicate id would silently merge two different
    prompts' results.
    """
    prompts_dir = prompts_dir or DATA_DIR / "prompts"
    if not prompts_dir.is_dir():
        raise ConfigError(f"missing prompts directory: {prompts_dir}")

    prompts: list[Prompt] = []
    seen: dict[str, Path] = {}

    for file in sorted(prompts_dir.glob("*.yaml")):
        entries = _load_yaml(file)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ConfigError(f"{file}: expected a list of prompts")

        for entry in entries:
            for required in ("id", "ecosystem", "domain", "specificity", "text"):
                if required not in entry:
                    raise ConfigError(f"{file}: prompt missing '{required}': {entry}")

            pid = entry["id"]
            if pid in seen:
                raise ConfigError(f"duplicate prompt id {pid!r} in {file} and {seen[pid]}")
            seen[pid] = file

            if entry["ecosystem"] not in VALID_ECOSYSTEMS:
                raise ConfigError(
                    f"{file}: prompt {pid} has unknown ecosystem {entry['ecosystem']!r}"
                )
            if entry["specificity"] not in VALID_SPECIFICITY:
                raise ConfigError(
                    f"{file}: prompt {pid} has unknown specificity "
                    f"{entry['specificity']!r} (expected one of {sorted(VALID_SPECIFICITY)})"
                )
            if not str(entry["text"]).strip():
                raise ConfigError(f"{file}: prompt {pid} has empty text")

            prompts.append(
                Prompt(
                    id=pid,
                    ecosystem=entry["ecosystem"],
                    domain=entry["domain"],
                    specificity=entry["specificity"],
                    text=str(entry["text"]).strip(),
                )
            )

    if ecosystems:
        prompts = [p for p in prompts if p.ecosystem in ecosystems]

    return prompts


def load_stdlib(language: str, stdlib_dir: Path | None = None) -> set[str]:
    """Load the curated stdlib/builtin module list for a language."""
    stdlib_dir = stdlib_dir or DATA_DIR / "stdlib"
    filename = {"python": "python.txt", "javascript": "node.txt"}.get(language)
    if filename is None:
        raise ConfigError(f"no stdlib list for language {language!r}")

    path = stdlib_dir / filename
    if not path.exists():
        raise ConfigError(f"missing stdlib list: {path}")

    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def load_aliases(aliases_dir: Path | None = None) -> dict[str, str]:
    """Load the Python import-name -> PyPI distribution-name table."""
    aliases_dir = aliases_dir or DATA_DIR / "aliases"
    path = aliases_dir / "python_import_to_dist.yaml"
    raw = _load_yaml(path)
    if not isinstance(raw, dict) or "aliases" not in raw:
        raise ConfigError(f"{path}: expected a top-level 'aliases' mapping")
    return dict(raw["aliases"])
