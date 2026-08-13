"""Package-name extraction from model responses.

Extraction accuracy directly bounds the headline numbers: a string wrongly treated as a
package becomes a reported hallucination that isn't one, and a package we miss
understates the rate. Every decision here is therefore explicit and tested rather than
incidental.

Design rules:

1.  **Fenced code blocks are the primary source.** Prose is parsed only as a fallback,
    and anything found that way is tagged `origin="prose"` so it can be excluded from
    a stricter analysis.
2.  **Never guess a distribution name.** Python import names that are not in the alias
    table are checked as-is and marked `alias_resolved=False`, so an unmapped name is
    visible in the data rather than silently transformed.
3.  **Filtering is conservative.** Standard library, relative imports, and modules the
    response itself defines are excluded. Anything else is kept, even if it looks odd —
    deciding a name "obviously isn't real" is the registry's job, not the extractor's.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------- code blocks

# The tag class includes ':' and '/' so a `python:src/utils.py` fence keeps its
# filename — that filename is how we detect modules the response defines itself.
FENCE_RE = re.compile(r"^```([\w+.:/-]*)[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)

# Fence language tag -> the kind of content we should parse it as.
LANG_ALIASES: dict[str, str] = {
    "py": "python",
    "python": "python",
    "python3": "python",
    "requirements": "requirements",
    "txt": "text",
    "js": "javascript",
    "jsx": "javascript",
    "javascript": "javascript",
    "ts": "javascript",
    "tsx": "javascript",
    "typescript": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "node": "javascript",
    "json": "json",
    "sh": "shell",
    "bash": "shell",
    "shell": "shell",
    "zsh": "shell",
    "console": "shell",
    "terminal": "shell",
    "": "untagged",
}


@dataclass(frozen=True)
class ExtractedPackage:
    """One package reference found in a response."""

    raw: str
    """Exactly as it appeared, before normalisation. Kept for auditability."""

    name: str
    """Normalised, registry-ready name."""

    ecosystem: str
    """python | javascript"""

    source: str
    """import | install | manifest — how the model referred to it."""

    origin: str
    """code_block | prose — prose findings are lower confidence by construction."""

    alias_resolved: bool | None = None
    """Python only. True if an import name was mapped to a distribution name via the
    alias table, False if it was checked as-is, None if not applicable."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    packages: list[ExtractedPackage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Human-readable caveats about this extraction, e.g. prose fallback used."""

    filtered: dict[str, list[str]] = field(default_factory=dict)
    """What was excluded and why — stdlib, relative, local. Recorded so a reviewer can
    audit the filtering rather than take it on trust."""

    def add_filtered(self, reason: str, name: str) -> None:
        self.filtered.setdefault(reason, [])
        if name not in self.filtered[reason]:
            self.filtered[reason].append(name)

    def unique_names(self, ecosystem: str | None = None) -> list[str]:
        seen: list[str] = []
        for p in self.packages:
            if ecosystem and p.ecosystem != ecosystem:
                continue
            if p.name not in seen:
                seen.append(p.name)
        return seen

    def to_dict(self) -> dict:
        return {
            "packages": [p.to_dict() for p in self.packages],
            "notes": list(self.notes),
            "filtered": {k: list(v) for k, v in self.filtered.items()},
        }


@dataclass(frozen=True)
class CodeBlock:
    lang: str
    """Normalised content kind: python | javascript | shell | json | requirements |
    untagged | text | other."""

    body: str
    tag: str
    """The raw fence tag, preserved for debugging odd fences."""


def find_code_blocks(text: str) -> list[CodeBlock]:
    """Return fenced code blocks with their language normalised."""
    blocks: list[CodeBlock] = []
    for match in FENCE_RE.finditer(text):
        tag = (match.group(1) or "").strip().lower()
        # A tag may carry a filename (`python:utils.py`); the language is the part
        # before the colon, the whole tag is kept for module-declaration detection.
        lang_key = tag.split(":", 1)[0]
        lang = LANG_ALIASES.get(lang_key, "other")
        blocks.append(CodeBlock(lang=lang, body=match.group(2), tag=tag))
    return blocks


def strip_code_blocks(text: str) -> str:
    """Everything outside fenced blocks — the prose fallback surface."""
    return FENCE_RE.sub("\n", text)


def infer_untagged_language(body: str) -> str | None:
    """Best-effort language guess for an untagged fence.

    Untagged blocks are common in model output and dropping them would understate the
    hallucination rate. Guessing wrong, however, sends a name to the wrong registry, so
    this only commits when the evidence is one-sided.
    """
    py_signals = len(
        re.findall(r"^\s*(?:from\s+[\w.]+\s+import\b|import\s+[\w.]+)", body, re.MULTILINE)
    ) + len(re.findall(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+", body, re.MULTILINE))

    js_signals = (
        len(re.findall(r"\brequire\s*\(", body))
        + len(re.findall(r"^\s*import\s+.*?\bfrom\s+['\"]", body, re.MULTILINE))
        + len(re.findall(r"\b(?:const|let|var)\s+\w+\s*=", body))
        + len(re.findall(r"=>|\bfunction\s*\(", body))
    )

    shell_signals = len(
        re.findall(
            r"^\s*(?:\$\s*)?(?:pip3?|uv|poetry|npm|yarn|pnpm|conda)\b", body, re.MULTILINE
        )
    )

    if shell_signals and shell_signals >= max(py_signals, js_signals):
        return "shell"
    if py_signals > js_signals:
        return "python"
    if js_signals > py_signals:
        return "javascript"
    return None


# ---------------------------------------------------------------- shared helpers

# Version specifiers, extras, and quoting that appear in install commands:
#   requests>=2.0  ·  "requests[socks]"  ·  requests==2.31.0  ·  pkg@1.2.3
SPEC_SPLIT_RE = re.compile(r"[<>=!~\[@;]")

INSTALL_FLAG_RE = re.compile(r"^-")


# Words that end an install command when it appears mid-sentence in prose.
#
# Without this, "just run pip install rich to get started" yields four packages —
# `rich`, `to`, `get`, `started` — three of which are false hallucinations. In a code
# block the command's boundary is unambiguous, so this only applies to prose.
PROSE_STOPWORDS = frozenset(
    """
    a an and as at be but by can could do does first for from get go goes had has have
    here how if in into is it its just like make makes may might need needs next now of
    on once only or should so some start started that the then there these they this
    those to too use used uses using want was we well were what when where which who
    will with would you your run adds add also
    """.split()
)


def clean_install_token(token: str) -> str | None:
    """Reduce one install-command argument to a bare package name, or None to skip."""
    token = token.strip().strip("'\"`,").rstrip(".)!?:")
    if not token or INSTALL_FLAG_RE.match(token):
        return None
    # Local paths, URLs, and VCS installs are not registry packages.
    if token.startswith((".", "/", "~")) or "://" in token or token.startswith("git+"):
        return None
    # A scoped npm package legitimately starts with @; only strip @version elsewhere.
    if token.startswith("@"):
        scope, _, rest = token.partition("/")
        if not rest:
            return None
        rest = SPEC_SPLIT_RE.split(rest)[0]
        return f"{scope}/{rest}" if rest else None
    name = SPEC_SPLIT_RE.split(token)[0]
    return name or None


def install_tokens(arg_string: str, *, prose: bool) -> list[str]:
    """Split an install command's arguments into candidate package names.

    In a fenced code block the command ends at the newline, so every argument is taken.
    In prose the command runs into the surrounding sentence, so consumption stops at the
    first ordinary English word — preferring a missed package over invented ones, since
    a false positive here is indistinguishable from a real hallucination downstream.
    """
    out: list[str] = []
    for token in arg_string.split():
        cleaned = clean_install_token(token)
        if cleaned is None:
            continue
        if prose and cleaned.lower() in PROSE_STOPWORDS:
            break
        out.append(cleaned)
    return out


def looks_like_package_name(name: str) -> bool:
    """Reject obvious non-packages before they reach a registry lookup.

    Deliberately permissive: the registry is the authority on existence, and being too
    clever here would filter out genuine hallucinations (which often look plausible but
    slightly off — exactly the thing being measured).
    """
    if not name or len(name) > 214:  # npm's documented maximum
        return False
    if name in {"-", "_", "."}:
        return False
    # Must contain at least one alphanumeric character.
    return bool(re.search(r"[a-zA-Z0-9]", name))


def extract(
    text: str,
    ecosystem: str | None = None,
    stdlib: dict[str, set[str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> ExtractionResult:
    """Extract package references from a model response.

    `ecosystem` scopes extraction to one language when the prompt targeted one. When
    omitted, both are attempted — a Python prompt can still elicit a JS snippet.
    """
    from slopsquat.extract import javascript as js_mod
    from slopsquat.extract import python as py_mod

    stdlib = stdlib or {}
    aliases = aliases or {}
    result = ExtractionResult()

    want_py = ecosystem in (None, "python")
    want_js = ecosystem in (None, "javascript")

    if want_py:
        py_mod.extract_into(result, text, stdlib.get("python", set()), aliases)
    if want_js:
        js_mod.extract_into(result, text, stdlib.get("javascript", set()))

    if any(p.origin == "prose" for p in result.packages):
        result.notes.append(
            "Some packages were found in prose rather than fenced code blocks; "
            "these are lower confidence and tagged origin='prose'."
        )

    return result
