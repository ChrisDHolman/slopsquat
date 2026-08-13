"""JavaScript / Node package extraction.

Specifier handling is the fiddly part:

*   **Scoped packages** (`@scope/name`) must keep the scope — `@types/node` and `node`
    are different packages.
*   **Subpaths** must be stripped: `lodash/merge` is a deep import of `lodash`, and
    `@scope/pkg/sub` is a deep import of `@scope/pkg`. Checking the full path against
    the registry would 404 and read as a hallucination.
*   **`node:` prefixes** (`node:fs`) are builtins in modern form and must be recognised
    as such, not looked up.
"""

from __future__ import annotations

import json
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

REQUIRE_RE = re.compile(r"""\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)""")

# `import x from 'y'`, `import {a} from 'y'`, `import * as z from 'y'`
IMPORT_FROM_RE = re.compile(
    r"""^[ \t]*import\b[^;\n]*?\bfrom\s*['"]([^'"]+)['"]""", re.MULTILINE
)

# Side-effect import: `import 'y'`
IMPORT_BARE_RE = re.compile(r"""^[ \t]*import\s*['"]([^'"]+)['"]""", re.MULTILINE)

# Dynamic import: `await import('y')`
IMPORT_DYNAMIC_RE = re.compile(r"""\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)""")

# `export ... from 'y'` — a re-export still declares a dependency.
EXPORT_FROM_RE = re.compile(
    r"""^[ \t]*export\b[^;\n]*?\bfrom\s*['"]([^'"]+)['"]""", re.MULTILINE
)

INSTALL_RE = re.compile(
    r"""(?:^|[\s`]|\$\s*)
    (?:
        npm\s+(?:install|i|add)
      | yarn\s+add
      | pnpm\s+(?:add|install|i)
      | bun\s+(?:add|install)
    )
    \s+([^\n\r&|;`]+)""",
    re.VERBOSE | re.IGNORECASE,
)

DEP_KEYS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")


def normalise(name: str) -> str:
    """npm names are lowercase; scope is part of the identity and is preserved."""
    return name.strip().lower()


def package_root(specifier: str) -> str | None:
    """Reduce a module specifier to the package it belongs to.

    `lodash/merge`        -> `lodash`
    `@scope/pkg/sub/path` -> `@scope/pkg`
    `./local`             -> None (relative)
    `node:fs`             -> `node:fs` (caller filters builtins)
    """
    spec = specifier.strip()
    if not spec:
        return None
    # Relative and absolute paths are local files, not packages.
    if spec.startswith((".", "/")):
        return None
    # URL imports (deno-style, CDN) are not registry packages.
    if "://" in spec:
        return None
    if spec.startswith("@"):
        parts = spec.split("/")
        if len(parts) < 2:
            return None  # bare `@foo` is not a valid scoped package
        # `@/public/hero.jpg` and friends are build-tool path aliases (Vite/webpack
        # `@/` -> project root), not packages: the scope segment is empty.
        if parts[0] == "@" or not parts[1]:
            return None
        return "/".join(parts[:2])
    return spec.split("/", 1)[0]


def _record(
    result: ExtractionResult, raw: str, source: str, origin: str, builtins: set[str]
) -> None:
    root = package_root(raw)
    if root is None:
        result.add_filtered("relative", raw)
        return

    # `node:fs` and bare `fs` are both builtins.
    bare = root[5:] if root.startswith("node:") else root
    if bare in builtins:
        result.add_filtered("builtin", root)
        return

    if not looks_like_package_name(bare):
        result.add_filtered("malformed", raw)
        return

    result.packages.append(
        ExtractedPackage(
            raw=raw,
            name=normalise(root),
            ecosystem="javascript",
            source=source,
            origin=origin,
        )
    )


def _scan_code(result: ExtractionResult, body: str, origin: str, builtins: set[str]) -> None:
    for pattern in (
        REQUIRE_RE,
        IMPORT_FROM_RE,
        IMPORT_BARE_RE,
        IMPORT_DYNAMIC_RE,
        EXPORT_FROM_RE,
    ):
        for match in pattern.finditer(body):
            _record(result, match.group(1), "import", origin, builtins)


def _scan_installs(
    result: ExtractionResult, body: str, origin: str, builtins: set[str]
) -> None:
    for match in INSTALL_RE.finditer(body):
        for cleaned in install_tokens(match.group(1), prose=origin == "prose"):
            _record(result, cleaned, "install", origin, builtins)


def _scan_manifest(
    result: ExtractionResult, body: str, origin: str, builtins: set[str]
) -> None:
    """Parse a package.json-shaped JSON block for declared dependencies."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return
    if not isinstance(data, dict):
        return
    for key in DEP_KEYS:
        deps = data.get(key)
        if isinstance(deps, dict):
            for name in deps:
                _record(result, str(name), "manifest", origin, builtins)


def extract_into(result: ExtractionResult, text: str, builtins: set[str]) -> None:
    """Extract JavaScript packages from `text` into `result`."""
    for block in find_code_blocks(text):
        lang = block.lang
        if lang == "untagged":
            inferred = infer_untagged_language(block.body)
            if inferred is None:
                continue
            lang = inferred

        if lang == "javascript":
            _scan_code(result, block.body, "code_block", builtins)
            _scan_installs(result, block.body, "code_block", builtins)
        elif lang == "shell":
            _scan_installs(result, block.body, "code_block", builtins)
        elif lang == "json":
            _scan_manifest(result, block.body, "code_block", builtins)

    # Prose fallback: install commands only. As with Python, prose *imports* are
    # discussion rather than a dependency declaration and are not counted.
    prose = strip_code_blocks(text)
    _scan_installs(result, prose, "prose", builtins)
