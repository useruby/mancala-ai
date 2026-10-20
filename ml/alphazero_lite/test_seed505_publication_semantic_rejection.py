"""Corrected semantic rejection coverage for the seed505 publication."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed505_publication_supplement as verifier


ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed505-strength-screen-v3"


@pytest.fixture
def copied_publication(tmp_path: Path) -> Path:
    target = tmp_path / "publication"
    shutil.copytree(PUBLICATION, target)
    return target


def _verify(publication: Path) -> dict:
    return verifier.verify(ROOT, out=publication)


def _json(path: Path, update) -> None:
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rebind_receipt(publication: Path) -> None:
    receipt_path = publication / "publication-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in receipt["files"]:
        receipt["files"][relative] = verifier.sha(publication / relative)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_unmodified_copy_passes_real_root_and_explicit_out(
    copied_publication: Path,
) -> None:
    result = _verify(copied_publication)
    assert result["status"] == "verified_complete"
    assert result["games"] == 1024
    assert result["pass"] is False


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("score", "published_result_mismatch"),
        ("bounds", "published_result_mismatch"),
        ("pass", "published_result_mismatch"),
        ("runtime", "report_runtime:0"),
        ("suite", "suite_association:0"),
        ("ledger", "ledger_replay:0:0"),
    ],
)
def test_rebound_receipt_reaches_expected_semantic_rejection(
    copied_publication: Path, corruption: str, expected_code: str
) -> None:
    # Establish that all source identities and runtime dependencies are valid,
    # and that this precise invocation accepts the untouched copied publication.
    assert _verify(copied_publication)["status"] == "verified_complete"

    if corruption in {"score", "bounds", "pass"}:

        def alter(result: dict) -> None:
            if corruption == "score":
                result["score"] += 0.01
            elif corruption == "bounds":
                result["bootstrap"]["one_sided_lower_95"] += 0.01
            else:
                result["pass"] = True

        _json(copied_publication / "results.json", alter)
    elif corruption == "runtime":
        report_path = copied_publication / "batches/report-0000.json"
        _json(
            report_path,
            lambda report: report["notes"].update(challenger_simulations=256),
        )
        ledger_path = copied_publication / "outcomes.jsonl"
        lines = ledger_path.read_text().splitlines()
        for index, line in enumerate(lines[:2]):
            row = json.loads(line)
            row["batch_report_sha256"] = verifier.sha(report_path)
            lines[index] = json.dumps(row, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")
    elif corruption == "suite":
        report_path = copied_publication / "batches/report-0000.json"
        _json(
            report_path,
            lambda report: report["notes"].update(suite_sha256="0" * 64),
        )
        ledger_path = copied_publication / "outcomes.jsonl"
        lines = ledger_path.read_text().splitlines()
        for index, line in enumerate(lines[:2]):
            row = json.loads(line)
            row["batch_report_sha256"] = verifier.sha(report_path)
            lines[index] = json.dumps(row, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")
    else:
        ledger_path = copied_publication / "outcomes.jsonl"
        lines = ledger_path.read_text().splitlines()
        row = json.loads(lines[0])
        row["winner"] = "challenger"
        lines[0] = json.dumps(row, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")

    _rebind_receipt(copied_publication)
    with pytest.raises(ValueError, match=expected_code):
        _verify(copied_publication)
