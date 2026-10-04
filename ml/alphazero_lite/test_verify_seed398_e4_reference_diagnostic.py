"""Regression tests for read-only verification of published E4 evidence."""

from __future__ import annotations

import json
import shutil

import pytest

from ml.alphazero_lite import verify_seed398_e4_reference_diagnostic as verifier
from ml.alphazero_lite.seed398_e4_report import calculate_report


@pytest.fixture
def isolated_publication(tmp_path, monkeypatch):
    data = tmp_path / "publication"
    shutil.copytree(verifier.DATA, data)
    monkeypatch.setattr(verifier, "DATA", data)
    return data


def _all_bytes(data):
    return {path.name: path.read_bytes() for path in data.iterdir() if path.is_file()}


def test_receipt_chain_required_and_read_only(isolated_publication) -> None:
    before = _all_bytes(isolated_publication)
    assert verifier.verify()["status"] == "verified"
    assert _all_bytes(isolated_publication) == before


@pytest.mark.parametrize(
    ("receipt_name", "error"),
    [
        ("verifier-correction-receipt.json", "correction_receipt_missing"),
        ("verifier-correction-receipt-v2.json", "correction_successor_receipt_missing"),
    ],
)
def test_missing_receipt_fails_read_only(
    isolated_publication, receipt_name, error
) -> None:
    (isolated_publication / receipt_name).unlink()
    before = _all_bytes(isolated_publication)
    with pytest.raises(ValueError, match=error):
        verifier.verify()
    assert _all_bytes(isolated_publication) == before


@pytest.mark.parametrize(
    ("receipt_name", "field", "value", "error"),
    [
        (
            "verifier-correction-receipt.json",
            "new_verifier_sha256",
            "0" * 64,
            "correction_predecessor_hash",
        ),
        (
            "verifier-correction-receipt.json",
            "helper_sha256",
            "0" * 64,
            "correction_helper_identity",
        ),
        (
            "verifier-correction-receipt-v2.json",
            "previous_verifier_sha256",
            "0" * 64,
            "correction_previous_verifier_identity",
        ),
        (
            "verifier-correction-receipt-v2.json",
            "current_verifier_sha256",
            "0" * 64,
            "correction_current_verifier_identity",
        ),
        (
            "verifier-correction-receipt-v2.json",
            "helper_sha256",
            "0" * 64,
            "correction_successor_helper_identity",
        ),
        (
            "verifier-correction-receipt-v2.json",
            "predecessor_receipt_sha256",
            "0" * 64,
            "correction_predecessor_hash",
        ),
    ],
)
def test_altered_receipt_identity_fails_read_only(
    isolated_publication, receipt_name, field, value, error
) -> None:
    path = isolated_publication / receipt_name
    record = json.loads(path.read_bytes())
    record[field] = value
    path.write_text(json.dumps(record))
    before = _all_bytes(isolated_publication)
    with pytest.raises(ValueError, match=error):
        verifier.verify()
    assert _all_bytes(isolated_publication) == before


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
