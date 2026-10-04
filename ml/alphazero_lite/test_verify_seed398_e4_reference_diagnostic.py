"""Regression tests for read-only verification of published E4 evidence."""

from __future__ import annotations

import json

import pytest

from ml.alphazero_lite import verify_seed398_e4_reference_diagnostic as verifier
from ml.alphazero_lite.seed398_e4_report import calculate_report


def _published_bytes() -> dict[str, bytes]:
    return {
        name: (verifier.DATA / name).read_bytes()
        for name in (
            "registration.json",
            "outcomes.jsonl",
            "analysis.json",
            "publication-binding.json",
        )
    }


def test_untouched_publication_verifies_without_writes() -> None:
    before = _published_bytes()
    assert verifier.verify()["status"] == "verified"
    assert _published_bytes() == before


@pytest.mark.parametrize("altered", ["decision", "paired_matrix"])
def test_altered_analysis_is_rejected_and_untouched(altered: str) -> None:
    path = verifier.DATA / "analysis.json"
    before = _published_bytes()
    original = path.read_bytes()
    report = json.loads(original)
    if altered == "decision":
        report["decision"] = "altered"
    else:
        report["paired_matrix"][0]["1536"]["delta"] += 1
    path.write_text(json.dumps(report))
    changed = path.read_bytes()
    try:
        with pytest.raises(ValueError, match="analysis_reconciliation"):
            verifier.verify()
        assert path.read_bytes() == changed
        assert _published_bytes() == {**before, "analysis.json": changed}
    finally:
        path.write_bytes(original)


def test_missing_analysis_fails_without_creating_it() -> None:
    path = verifier.DATA / "analysis.json"
    before = _published_bytes()
    original = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises(ValueError, match="analysis_missing"):
            verifier.verify()
        assert not path.exists()
        assert {
            name: (verifier.DATA / name).read_bytes()
            for name in before
            if name != "analysis.json"
        } == {name: value for name, value in before.items() if name != "analysis.json"}
    finally:
        path.write_bytes(original)


def test_altered_ledger_bytes_fail() -> None:
    path = verifier.DATA / "outcomes.jsonl"
    before = _published_bytes()
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError, match="publication_outcomes_binding"):
            verifier.verify()
        assert _published_bytes()["registration.json"] == before["registration.json"]
        assert _published_bytes()["analysis.json"] == before["analysis.json"]
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("mode", ["partial", "duplicate"])
def test_partial_or_duplicate_outcomes_fail(mode: str) -> None:
    registration = json.loads((verifier.DATA / "registration.json").read_bytes())
    baseline = json.loads(
        (
            verifier.ROOT / "docs/data/seed398-paired-first-action/analysis.json"
        ).read_bytes()
    )
    ledger = (verifier.DATA / "outcomes.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in ledger]
    if mode == "partial":
        rows.pop()
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        calculate_report(registration, rows, baseline)
