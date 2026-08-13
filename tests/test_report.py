"""Report/analysis tests on hand-built JSONL with known-correct expected numbers.

The point is to pin the definitions: an ``error`` status must never count as a
hallucination, recurrence must count distinct runs against the cell's run total, and an
unmapped Python 404 must land in needs-review rather than the confirmed rate.
"""

from __future__ import annotations

import json

import pytest

from slopsquat.report import ReportError, analyse, render_markdown, write_csv


def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _mention(name, source="install", alias_resolved=None):
    return {"name": name, "source": source, "origin": "code_block", "alias_resolved": alias_resolved}


def _rec(run_index, packages, *, ok=True, truncated=False, error=None):
    return {
        "ok": ok,
        "run_id": "r1",
        "prompt_id": "p1",
        "ecosystem": "python",
        "domain": "web",
        "specificity": "detailed",
        "model_id": "m1",
        "run_index": run_index,
        "truncated": truncated,
        "error": error,
        "packages": packages,
    }


@pytest.fixture
def dataset(tmp_path):
    runs = tmp_path / "runs.jsonl"
    reg = tmp_path / "registry.jsonl"
    _write(
        runs,
        [
            _rec(0, [_mention("requests"), _mention("ghostpkg"), _mention("flakey")]),
            _rec(1, [_mention("ghostpkg")]),
            _rec(2, [_mention("mysteryimport", alias_resolved=False)], truncated=True),
            _rec(3, [], ok=False, error="Boom: 500"),  # failed call, excluded
        ],
    )
    _write(
        reg,
        [
            {"ecosystem": "python", "name": "requests", "status": "exists"},
            {"ecosystem": "python", "name": "ghostpkg", "status": "not_found"},
            {"ecosystem": "python", "name": "flakey", "status": "error"},
            {"ecosystem": "python", "name": "mysteryimport", "status": "not_found"},
        ],
    )
    return runs, reg


def test_totals(dataset) -> None:
    rep = analyse(*dataset)
    assert rep.totals == {
        "records": 4,
        "ok": 3,
        "failed": 1,
        "truncated": 1,
        "undetermined_mentions": 1,  # flakey (error)
    }


def test_per_model_rates_exclude_errors(dataset) -> None:
    rep = analyse(*dataset)
    (m,) = rep.per_model
    assert m["ok_responses"] == 3
    assert m["mentions"] == 5
    assert m["checkable"] == 4  # flakey (error) excluded
    assert m["halluc_mentions"] == 3  # ghostpkg x2 + mysteryimport
    assert m["distinct_halluc"] == 2
    assert m["mention_rate"] == pytest.approx(3 / 4)
    assert m["responses_with_halluc"] == 3
    assert m["response_rate"] == pytest.approx(1.0)


def test_error_status_is_never_a_hallucination(dataset) -> None:
    rep = analyse(*dataset)
    names = {r["name"] for r in rep.recurrence}
    assert "flakey" not in names  # an error is undetermined, not invented


def test_recurrence_counts_distinct_runs(dataset) -> None:
    rep = analyse(*dataset)
    ghost = next(r for r in rep.recurrence if r["name"] == "ghostpkg")
    assert ghost["appearances"] == 2
    assert ghost["runs"] == 3
    assert ghost["recurrence_rate"] == pytest.approx(2 / 3)
    assert ghost["needs_review"] is False


def test_unmapped_python_404_is_needs_review(dataset) -> None:
    rep = analyse(*dataset)
    mystery = next(r for r in rep.recurrence if r["name"] == "mysteryimport")
    assert mystery["needs_review"] is True
    assert mystery["status"] == "needs_review"
    assert {n["name"] for n in rep.needs_review} == {"mysteryimport"}


def test_recurrence_summary_buckets(dataset) -> None:
    rep = analyse(*dataset)
    rs = rep.recurrence_summary
    assert rs["cells"] == 2
    assert rs["recurring_2plus"] == 1  # ghostpkg
    assert rs["one_off"] == 1  # mysteryimport
    assert rs["majority"] == 1  # ghostpkg at 2/3
    assert rs["every_run"] == 0


def test_caveats_present(dataset) -> None:
    rep = analyse(*dataset)
    joined = " ".join(rep.caveats)
    assert "failed" in joined
    assert "truncated" in joined
    assert "undetermined" in joined
    assert "manual review" in joined


def test_markdown_and_csv_render(dataset, tmp_path) -> None:
    rep = analyse(*dataset)
    md = render_markdown(rep)
    assert "# Slopsquat results" in md
    assert "ghostpkg" in md
    csv_path = tmp_path / "rec.csv"
    write_csv(rep, csv_path)
    body = csv_path.read_text(encoding="utf-8")
    assert "ghostpkg" in body and "mysteryimport" in body


def test_missing_files_raise(tmp_path) -> None:
    with pytest.raises(ReportError):
        analyse(tmp_path / "nope.jsonl", tmp_path / "reg.jsonl")
