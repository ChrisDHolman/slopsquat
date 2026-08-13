"""Python extraction tests.

These bound the headline numbers. A false positive here is a reported hallucination
that isn't one; a miss understates the rate. Both directions are tested.
"""

from __future__ import annotations

import pytest

from slopsquat.extract import extract
from slopsquat.extract.python import normalise, top_level

STDLIB = {"os", "sys", "json", "pathlib", "asyncio", "typing", "re", "csv", "logging"}
ALIASES = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "bs4": "beautifulsoup4"}


def run(text: str):
    return extract(text, ecosystem="python", stdlib={"python": STDLIB}, aliases=ALIASES)


def names(result) -> set[str]:
    return {p.name for p in result.packages}


# ---------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Flask_SQLAlchemy", "flask-sqlalchemy"),
        ("flask.sqlalchemy", "flask-sqlalchemy"),
        ("Requests", "requests"),
        ("zope--interface", "zope-interface"),
        ("a_b.c-d", "a-b-c-d"),
    ],
)
def test_pep503_normalisation(raw: str, expected: str) -> None:
    """One project must not split into several names during recurrence counting."""
    assert normalise(raw) == expected


def test_top_level_only() -> None:
    assert top_level("os.path") == "os"
    assert top_level("concurrent.futures.thread") == "concurrent"


# ---------------------------------------------------------------- imports


def test_basic_import_and_from() -> None:
    result = run("```python\nimport requests\nfrom flask import Flask\n```")
    assert names(result) == {"requests", "flask"}


def test_multi_target_import_with_alias() -> None:
    result = run("```python\nimport numpy as np, pandas as pd, httpx\n```")
    assert names(result) == {"numpy", "pandas", "httpx"}


def test_dotted_import_reduces_to_top_level() -> None:
    result = run("```python\nfrom sqlalchemy.orm import Session\n```")
    assert names(result) == {"sqlalchemy"}


def test_stdlib_is_filtered() -> None:
    result = run("```python\nimport os, sys, requests\nfrom pathlib import Path\n```")
    assert names(result) == {"requests"}
    assert set(result.filtered.get("stdlib", [])) == {"os", "sys", "pathlib"}


def test_relative_imports_filtered() -> None:
    result = run("```python\nfrom .helpers import thing\nfrom ..models import User\n```")
    assert names(result) == set()
    assert result.filtered.get("relative")


def test_future_import_filtered() -> None:
    result = run("```python\nfrom __future__ import annotations\nimport requests\n```")
    assert names(result) == {"requests"}


# ---------------------------------------------------------------- aliases


def test_alias_resolves_import_to_distribution() -> None:
    """`import cv2` needs `pip install opencv-python`. Checking cv2 against PyPI would
    404 and be recorded as a hallucination — the exact false positive to avoid."""
    result = run("```python\nimport cv2\n```")
    pkg = next(p for p in result.packages if p.raw == "cv2")
    assert pkg.name == "opencv-python"
    assert pkg.alias_resolved is True


def test_unmapped_import_is_flagged_not_guessed() -> None:
    """An unmapped name is checked as-is and marked, never transformed."""
    result = run("```python\nimport superfastjson\n```")
    pkg = next(p for p in result.packages if p.raw == "superfastjson")
    assert pkg.name == "superfastjson"
    assert pkg.alias_resolved is False


def test_install_names_are_not_alias_resolved() -> None:
    """`pip install X` already names a distribution — aliasing it would be wrong."""
    result = run("```bash\npip install pillow\n```")
    pkg = next(p for p in result.packages if p.name == "pillow")
    assert pkg.alias_resolved is None
    assert pkg.source == "install"


# ---------------------------------------------------------------- install commands


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "pip3 install requests",
        "python -m pip install requests",
        "python3 -m pip install requests",
        "uv pip install requests",
        "uv add requests",
        "poetry add requests",
        "pipenv install requests",
        "$ pip install requests",
    ],
)
def test_install_command_forms(command: str) -> None:
    result = run(f"```bash\n{command}\n```")
    assert "requests" in names(result)


def test_install_strips_version_specifiers_and_extras() -> None:
    result = run("```bash\npip install 'requests>=2.31' httpx==0.27.0 celery[redis]\n```")
    assert names(result) == {"requests", "httpx", "celery"}


def test_install_ignores_flags_paths_and_urls() -> None:
    result = run(
        "```bash\n"
        "pip install -U --no-cache-dir requests\n"
        "pip install ./local-package\n"
        "pip install git+https://github.com/x/y.git\n"
        "```"
    )
    assert names(result) == {"requests"}


def test_multiple_packages_one_command() -> None:
    result = run("```bash\npip install fastapi uvicorn pydantic\n```")
    assert names(result) == {"fastapi", "uvicorn", "pydantic"}


# ---------------------------------------------------------------- manifests


def test_requirements_block() -> None:
    result = run(
        "```requirements\n"
        "# comment line\n"
        "requests>=2.31.0\n"
        "flask==3.0.0\n"
        "celery[redis]~=5.3\n"
        "-r other.txt\n"
        "```"
    )
    assert names(result) == {"requests", "flask", "celery"}


# ---------------------------------------------------------------- local modules


def test_module_declared_in_response_is_filtered() -> None:
    """A response that writes its own utils.py then imports it is not naming a package."""
    text = (
        "Create these two files.\n\n"
        "```python\n# file: utils.py\ndef helper():\n    return 1\n```\n\n"
        "```python\nfrom utils import helper\nimport requests\n```"
    )
    result = run(text)
    assert names(result) == {"requests"}
    assert "utils" in result.filtered.get("declared_in_response", [])


def test_fence_tag_filename_declares_module() -> None:
    text = (
        "```python:helpers.py\ndef x():\n    pass\n```\n"
        "```python\nfrom helpers import x\nimport httpx\n```"
    )
    result = run(text)
    assert names(result) == {"httpx"}


# ---------------------------------------------------------------- prose handling


def test_prose_install_is_captured_and_tagged() -> None:
    result = run("First, just run pip install rich to get started.")
    pkg = next(p for p in result.packages if p.name == "rich")
    assert pkg.origin == "prose"
    assert result.notes  # extraction records that prose fallback was used


def test_prose_import_is_not_counted() -> None:
    """'you should import requests' is discussion, not a dependency declaration.
    Counting it would inflate the rate."""
    result = run("You should import requests for this, it's the usual choice.")
    assert names(result) == set()


def test_code_block_origin_is_recorded() -> None:
    result = run("```python\nimport requests\n```")
    assert all(p.origin == "code_block" for p in result.packages)


# ---------------------------------------------------------------- untagged fences


def test_untagged_python_block_is_inferred() -> None:
    result = run("```\nimport requests\nfrom flask import Flask\n\ndef go():\n    pass\n```")
    assert names(result) == {"requests", "flask"}


def test_untagged_ambiguous_block_is_skipped() -> None:
    """Guessing wrong sends a name to the wrong registry, so ambiguity means skip."""
    result = run("```\nsome plain text with no code signals at all\n```")
    assert names(result) == set()


# ---------------------------------------------------------------- realistic response


def test_realistic_mixed_response() -> None:
    text = """
You'll want a few libraries for this. Install them with:

```bash
pip install requests beautifulsoup4 pandas
```

Then the scraper itself:

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import csv

from .config import SETTINGS


def scrape(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    return soup
```

If you need JavaScript rendering, `pip install playwright` as well.
"""
    result = run(text)
    assert names(result) == {
        "requests",
        "beautifulsoup4",
        "pandas",
        "playwright",
    }
    assert set(result.filtered.get("stdlib", [])) == {"pathlib", "csv"}
    assert result.filtered.get("relative")


def test_real_data_files_load_and_filter() -> None:
    """Guards against drift between the committed data files and the extractor."""
    from slopsquat.config import load_aliases, load_stdlib

    stdlib = load_stdlib("python")
    aliases = load_aliases()
    assert "os" in stdlib and "asyncio" in stdlib
    assert aliases.get("cv2") == "opencv-python"

    result = extract(
        "```python\nimport os\nimport cv2\nimport notarealpackage123\n```",
        ecosystem="python",
        stdlib={"python": stdlib},
        aliases=aliases,
    )
    assert names(result) == {"opencv-python", "notarealpackage123"}
