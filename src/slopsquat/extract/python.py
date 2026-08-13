"""Python package extraction.

Two things make Python harder than it looks:

1.  The **import name and the PyPI distribution name often differ** (`import cv2` needs
    `pip install opencv-python`). Checking the import name against PyPI would 404 and be
    recorded as a hallucination — a false positive. The alias table handles the known
    cases; unmapped names are checked as-is and flagged, never guessed at.

2.  A response often **defines its own modules** (`# file: utils.py` then
    `from utils import helper`). Those are local, not packages. Module names the
    response appears to define are filtered, and the limits of that detection are
    documented below.
"""

from __future__ import annotations

import re

from slopsquat.extract import (
    ExtractedPackage,
    ExtractionResult,
    find_code_blocks,
    install_tokens,
    infer_untagged_language,
    looks_like_package_name,
    strip_code_blocks,
)

# `import a`, `import a.b`, `import a as x`, `import a, b`
IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+([^\n#;]+)", re.MULTILINE)

# `from a.b import c` — a leading dot means relative, handled explicitly.
FROM_RE = re.compile(r"^[ \t]*from[ \t]+(\.*[\w.]+)[ \t]+import\b", re.MULTILINE)

INSTALL_RE = re.compile(
    r"""(?:^|[\s`]|\$\s*)
    (?:
        (?:python3?\s+-m\s+)?pip3?\s+install
      | uv\s+pip\s+install
      | uv\s+add
      | poetry\s+add
      | pipenv\s+install
      | conda\s+install
    )
    \s+([^\n\r&|;`]+)""",
    re.VERBOSE | re.IGNORECASE,
)

# `# file: utils.py`, `# utils.py`, `## src/helpers.py` — a response declaring its own
# module layout. Also matches a fence tag like ```python:utils.py handled separately.
FILE_DECL_RE = re.compile(
    r"^[ \t]*#+[ \t]*(?:file[:\s]+)?([\w./-]+)\.py[ \t]*$", re.MULTILINE | re.IGNORECASE
)

# Requirements-file lines: bare names, optionally with specifiers.
REQ_LINE_RE = re.compile(r"^[ \t]*([A-Za-z0-9][\w.\-]*(?:\[[\w,\- ]+\])?)", re.MULTILINE)


def normalise(name: str) -> str:
    """PEP 503 normalisation — how PyPI itself compares names.

    `Flask_SQLAlchemy`, `flask-sqlalchemy`, and `flask.sqlalchemy` are the same project.
    Normalising here means recurrence counting doesn't split one hallucination across
    three spellings.
    """
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def top_level(module: str) -> str:
    """`os.path` -> `os`. Only the top-level name maps to a distribution."""
    return module.split(".", 1)[0]


def _declared_modules(text: str) -> set[str]:
    """Module names the response appears to define itself.

    Known limitation: only catches modules declared via a filename comment or a
    `lang:filename` fence tag. A response that writes `utils.py` implicitly — with no
    annotation at all — will still have `import utils` treated as a package. This is a
    documented false-positive source rather than a silent one.
    """
    declared: set[str] = set()
    for match in FILE_DECL_RE.finditer(text):
        declared.add(top_level(match.group(1).split("/")[-1]))
    for block in find_code_blocks(text):
        if ":" in block.tag:
            _, _, fname = block.tag.partition(":")
            if fname.endswith(".py"):
                declared.add(top_level(fname.split("/")[-1][: -len(".py")]))
    return declared


def _split_import_targets(clause: str) -> list[str]:
    """`a, b as c, d` -> ['a', 'b', 'd']"""
    names: list[str] = []
    for part in clause.split(","):
        part = part.strip()
        if not part:
            continue
        part = re.split(r"\s+as\s+", part)[0].strip()
        if part:
            names.append(part)
    return names


def _record(
    result: ExtractionResult,
    raw: str,
    source: str,
    origin: str,
    stdlib: set[str],
    aliases: dict[str, str],
    declared: set[str],
    *,
    is_import: bool,
) -> None:
    """Filter, resolve, and record one candidate."""
    if is_import:
        if raw.startswith("."):
            result.add_filtered("relative", raw)
            return
        raw = top_level(raw)

    if not looks_like_package_name(raw):
        result.add_filtered("malformed", raw)
        return

    # Standard-library names are filtered regardless of source: `pip install sqlite3`
    # or a `sqlite3  # built-in` line in requirements is a real module that simply isn't
    # a PyPI package — it 404s but is not a hallucination.
    if raw in stdlib or raw == "__future__":
        result.add_filtered("stdlib", raw)
        return

    if is_import:
        if raw in declared:
            result.add_filtered("declared_in_response", raw)
            return

    alias_resolved: bool | None = None
    name = raw

    if is_import:
        # Import name -> distribution name. Never transform an unmapped name; an
        # invented mapping would convert a real hallucination into a resolved package.
        if raw in aliases:
            name = aliases[raw]
            alias_resolved = True
        else:
            alias_resolved = False

    result.packages.append(
        ExtractedPackage(
            raw=raw,
            name=normalise(name),
            ecosystem="python",
            source=source,
            origin=origin,
            alias_resolved=alias_resolved,
        )
    )


def _scan(
    result: ExtractionResult,
    body: str,
    origin: str,
    stdlib: set[str],
    aliases: dict[str, str],
    declared: set[str],
    *,
    imports: bool = True,
    installs: bool = True,
    requirements: bool = False,
) -> None:
    if imports:
        for match in IMPORT_RE.finditer(body):
            for target in _split_import_targets(match.group(1)):
                _record(
                    result, target, "import", origin, stdlib, aliases, declared, is_import=True
                )
        for match in FROM_RE.finditer(body):
            _record(
                result, match.group(1), "import", origin, stdlib, aliases, declared,
                is_import=True,
            )

    if installs:
        for match in INSTALL_RE.finditer(body):
            for cleaned in install_tokens(match.group(1), prose=origin == "prose"):
                _record(
                    result, cleaned, "install", origin, stdlib, aliases, declared,
                    is_import=False,
                )

    if requirements:
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "-")):
                continue
            match = REQ_LINE_RE.match(line)
            if match:
                for cleaned in install_tokens(match.group(1), prose=False):
                    _record(
                        result, cleaned, "manifest", origin, stdlib, aliases, declared,
                        is_import=False,
                    )


def extract_into(
    result: ExtractionResult, text: str, stdlib: set[str], aliases: dict[str, str]
) -> None:
    """Extract Python packages from `text` into `result`."""
    declared = _declared_modules(text)

    for block in find_code_blocks(text):
        lang = block.lang
        if lang == "untagged":
            inferred = infer_untagged_language(block.body)
            if inferred is None:
                continue
            lang = inferred

        if lang == "python":
            _scan(result, block.body, "code_block", stdlib, aliases, declared)
        elif lang == "shell":
            _scan(
                result, block.body, "code_block", stdlib, aliases, declared, imports=False
            )
        elif lang == "requirements":
            _scan(
                result, block.body, "code_block", stdlib, aliases, declared,
                imports=False, installs=False, requirements=True,
            )
        elif lang == "text":
            # An untagged/txt block naming a requirements file is common enough to be
            # worth catching, but only when it actually looks like one.
            if re.search(r"^[A-Za-z0-9][\w.\-]*\s*[=><~]{1,2}\s*\d", block.body, re.MULTILINE):
                _scan(
                    result, block.body, "code_block", stdlib, aliases, declared,
                    imports=False, installs=False, requirements=True,
                )

    # Prose fallback: install commands written inline, e.g. "just run pip install foo".
    # Imports are NOT scanned in prose — "you should import requests" is discussion, not
    # a dependency declaration, and treating it as one inflates the count.
    prose = strip_code_blocks(text)
    _scan(result, prose, "prose", stdlib, aliases, declared, imports=False)
