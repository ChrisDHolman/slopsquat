"""JavaScript / Node extraction tests.

Specifier handling is where this goes wrong quietly: a deep import left unstripped
(`lodash/merge`) 404s at the registry and reads as a hallucination, while a scope
stripped off (`@types/node` -> `node`) silently reports the wrong package.
"""

from __future__ import annotations

import pytest

from slopsquat.extract import extract
from slopsquat.extract.javascript import package_root

BUILTINS = {"fs", "path", "http", "https", "crypto", "os", "url", "events", "stream"}


def run(text: str):
    return extract(text, ecosystem="javascript", stdlib={"javascript": BUILTINS})


def names(result) -> set[str]:
    return {p.name for p in result.packages}


# ---------------------------------------------------------------- specifiers


@pytest.mark.parametrize(
    ("specifier", "expected"),
    [
        ("express", "express"),
        ("lodash/merge", "lodash"),
        ("lodash/fp/curry", "lodash"),
        ("@scope/pkg", "@scope/pkg"),
        ("@scope/pkg/sub/path", "@scope/pkg"),
        ("@types/node", "@types/node"),
        ("node:fs", "node:fs"),
        ("./local", None),
        ("../parent/mod", None),
        ("/absolute/path", None),
        ("https://esm.sh/react", None),
        ("@bare", None),
    ],
)
def test_package_root(specifier: str, expected: str | None) -> None:
    assert package_root(specifier) == expected


# ---------------------------------------------------------------- import forms


def test_require() -> None:
    result = run("```js\nconst express = require('express');\n```")
    assert names(result) == {"express"}


def test_import_from() -> None:
    result = run("```js\nimport express from 'express';\nimport { z } from 'zod';\n```")
    assert names(result) == {"express", "zod"}


def test_import_namespace() -> None:
    result = run("```js\nimport * as lodash from 'lodash';\n```")
    assert names(result) == {"lodash"}


def test_bare_side_effect_import() -> None:
    result = run("```js\nimport 'dotenv/config';\n```")
    assert names(result) == {"dotenv"}


def test_dynamic_import() -> None:
    result = run("```js\nconst mod = await import('sharp');\n```")
    assert names(result) == {"sharp"}


def test_export_from_counts_as_dependency() -> None:
    result = run("```js\nexport { default } from 'chalk';\n```")
    assert names(result) == {"chalk"}


def test_typescript_block_is_treated_as_javascript() -> None:
    result = run("```typescript\nimport { Client } from 'pg';\n```")
    assert names(result) == {"pg"}


# ---------------------------------------------------------------- builtins


def test_bare_builtin_filtered() -> None:
    result = run("```js\nconst fs = require('fs');\nconst express = require('express');\n```")
    assert names(result) == {"express"}
    assert "fs" in result.filtered.get("builtin", [])


def test_node_prefixed_builtin_filtered() -> None:
    result = run("```js\nimport fs from 'node:fs/promises';\nimport ky from 'ky';\n```")
    assert names(result) == {"ky"}


def test_relative_import_filtered() -> None:
    result = run("```js\nimport { helper } from './utils.js';\nimport ky from 'ky';\n```")
    assert names(result) == {"ky"}
    assert result.filtered.get("relative")


# ---------------------------------------------------------------- install commands


@pytest.mark.parametrize(
    "command",
    [
        "npm install express",
        "npm i express",
        "npm add express",
        "yarn add express",
        "pnpm add express",
        "pnpm install express",
        "bun add express",
        "$ npm install express",
    ],
)
def test_install_command_forms(command: str) -> None:
    result = run(f"```bash\n{command}\n```")
    assert "express" in names(result)


def test_install_strips_versions_keeps_scope() -> None:
    result = run("```bash\nnpm install express@4.18.2 @types/node@20 zod\n```")
    assert names(result) == {"express", "@types/node", "zod"}


def test_install_ignores_flags() -> None:
    result = run("```bash\nnpm install --save-dev -D typescript\n```")
    assert names(result) == {"typescript"}


# ---------------------------------------------------------------- manifests


def test_package_json_dependencies() -> None:
    result = run(
        "```json\n"
        "{\n"
        '  "name": "my-app",\n'
        '  "dependencies": { "express": "^4.18.0", "zod": "^3.22.0" },\n'
        '  "devDependencies": { "typescript": "^5.4.0" }\n'
        "}\n"
        "```"
    )
    assert names(result) == {"express", "zod", "typescript"}
    assert all(p.source == "manifest" for p in result.packages)


def test_malformed_json_block_is_ignored_not_crashed() -> None:
    result = run('```json\n{ "dependencies": { broken\n```')
    assert names(result) == set()


# ---------------------------------------------------------------- prose


def test_prose_install_captured_and_tagged() -> None:
    result = run("Just run npm install got and you're away.")
    pkg = next(p for p in result.packages if p.name == "got")
    assert pkg.origin == "prose"


def test_prose_require_not_counted() -> None:
    result = run("You'd typically require('express') here, but there are alternatives.")
    assert names(result) == set()


# ---------------------------------------------------------------- realistic


def test_realistic_mixed_response() -> None:
    text = """
Install the dependencies first:

```bash
npm install playwright cheerio p-limit
```

Then:

```javascript
import { chromium } from 'playwright';
import * as cheerio from 'cheerio';
import pLimit from 'p-limit';
import fs from 'node:fs/promises';
import path from 'path';
import { parseRow } from './parsers/row.js';

const limit = pLimit(5);
```

Your package.json will look like:

```json
{ "dependencies": { "playwright": "^1.44.0", "cheerio": "^1.0.0" } }
```
"""
    result = run(text)
    assert names(result) == {"playwright", "cheerio", "p-limit"}
    assert set(result.filtered.get("builtin", [])) >= {"node:fs", "path"}
    assert result.filtered.get("relative")


def test_real_builtins_file_loads() -> None:
    """Guards against drift between the committed node.txt and the extractor."""
    from slopsquat.config import load_stdlib

    builtins = load_stdlib("javascript")
    assert {"fs", "path", "crypto"} <= builtins

    result = extract(
        "```js\nrequire('fs');\nrequire('definitelynotarealpkg999');\n```",
        ecosystem="javascript",
        stdlib={"javascript": builtins},
    )
    assert names(result) == {"definitelynotarealpkg999"}
