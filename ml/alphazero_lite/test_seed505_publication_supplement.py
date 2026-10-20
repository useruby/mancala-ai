"""Semantic corruption tests for the read-only seed505 supplement."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed505_publication_supplement as verifier
from ml.alphazero_lite import seed505_strength_screen as frozen_runner


@pytest.fixture
def publication(tmp_path: Path) -> Path:
    source = (
        Path(__file__).resolve().parents[2] / "docs/data/seed505-strength-screen-v3"
    )
    target = tmp_path / "publication"
    shutil.copytree(source, target)
    return target


def _json(path: Path, update) -> None:
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rebind_receipt(path: Path) -> None:
    receipt_path = path / "publication-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in receipt["files"]:
        receipt["files"][relative] = verifier.sha(path / relative)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "corruption", ["score", "bounds", "pass", "runtime", "suite", "ledger"]
)
def test_rebound_ordinary_receipt_does_not_hide_semantic_corruption(
    publication: Path, corruption: str
) -> None:
    if corruption in {"score", "bounds", "pass"}:

        def alter(result):
            if corruption == "score":
                result["score"] += 0.01
            elif corruption == "bounds":
                result["bootstrap"]["one_sided_lower_95"] += 0.01
            else:
                result["pass"] = True

        _json(publication / "results.json", alter)
    elif corruption == "runtime":
        path = publication / "batches/report-0000.json"
        _json(path, lambda report: report["notes"].update(challenger_simulations=256))
        ledger_path = publication / "outcomes.jsonl"
        lines = ledger_path.read_text().splitlines()
        for i, line in enumerate(lines[:2]):
            row = json.loads(line)
            row["batch_report_sha256"] = verifier.sha(path)
            lines[i] = json.dumps(row, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")
    elif corruption == "suite":
        path = publication / "batches/report-0000.json"
        _json(path, lambda report: report["notes"].update(suite_sha256="0" * 64))
        ledger_path = publication / "outcomes.jsonl"
        lines = ledger_path.read_text().splitlines()
        for i, line in enumerate(lines[:2]):
            row = json.loads(line)
            row["batch_report_sha256"] = verifier.sha(path)
            lines[i] = json.dumps(row, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")
    else:
        ledger_path = publication / "outcomes.jsonl"
        lines = ledger_path.read_text().splitlines()
        row = json.loads(lines[0])
        row["winner"] = "challenger"
        lines[0] = json.dumps(row, sort_keys=True)
        ledger_path.write_text("\n".join(lines) + "\n")
    _rebind_receipt(publication)
    with pytest.raises(ValueError):
        verifier.verify(publication.parent, publication)


def test_frozen_resume_guard_rejects_valid_pair_before_skip_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose the historical guard defect; intended resume validates then skips."""
    out = tmp_path / "seed505"
    out.mkdir()
    monkeypatch.setattr(frozen_runner, "OUT", out)
    monkeypatch.setattr(frozen_runner, "SEED504", Path("checkpoint"))
    monkeypatch.setattr(frozen_runner, "E4_SHA256", "bound")
    monkeypatch.setattr(frozen_runner, "INITIALIZER_SHA256", "bound")
    monkeypatch.setattr(frozen_runner, "sha", lambda _path: "bound")
    registration = {"source_sha256": {}, "runtime": {"runtime_identities_sha256": {}}}
    (out / "registration.json").write_text(json.dumps(registration))
    (out / "execution-freeze.json").write_text(
        json.dumps(
            {"status": "frozen_before_execution", "registration_sha256": "bound"}
        )
    )
    (out / "openings.jsonl").write_text('{"opening_index":0}\n')
    (out / "outcomes.jsonl").write_text(
        '{"opening_index":0,"F_player":0}\n{"opening_index":0,"F_player":1}\n'
    )
    (out / "initializer.npz").write_bytes(b"synthetic")
    monkeypatch.setattr(
        frozen_runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("arena must not be invoked"),
    )

    # Observed defect: 1 distinct opening versus 2 seat-pair keys is called partial.
    # Intended: validate this complete pair, continue into the skip branch, and run
    # no arena process for opening 0.
    with pytest.raises(ValueError, match="partial_or_duplicate_committed_pair"):
        frozen_runner.run(tmp_path, execute=True)
