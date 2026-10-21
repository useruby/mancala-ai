"""Semantic rejection controls for the append-only seed512 verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite.verify_seed512_supplement import (
    audit_assets,
    audit_ledger,
    require_result_match,
)


def full_ledger() -> list[dict[str, int]]:
    return [
        {"opening_index": i, "challenger_player": seat, "batch_game_position": seat}
        for i in range(1024)
        for seat in (0, 1)
    ]


def test_copied_publication_audit_passes() -> None:
    audit_ledger(full_ledger())


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("seat", "seed512_supplement_duplicate_seat:0:0"),
        ("position", "seed512_supplement_duplicate_position:0:0"),
    ],
)
def test_rebound_ledger_receipt_does_not_hide_duplicate(
    change: str, error: str
) -> None:
    ledger = full_ledger()
    if change == "seat":
        ledger[1]["challenger_player"] = 0
    else:
        ledger[1]["batch_game_position"] = 0
    with pytest.raises(ValueError, match=error):
        audit_ledger(ledger)


def test_artifact_mutation_rejected_after_receipt_rebind(tmp_path: Path) -> None:
    out = tmp_path / "docs/data/seed512-fixed-composition-evaluation"
    art = tmp_path / "artifact"
    art.mkdir()
    component_files = {"model.npz": "a" * 64}
    (art / "model.npz").write_bytes(b"authentic")
    registration = {
        "components": {
            "E4": {
                "artifact": "artifact",
                "checkpoint_source": "absent.npz",
                "checkpoint_sha256": "b" * 64,
                "files": component_files,
            },
            "initializer": {
                "artifact": "artifact",
                "checkpoint_source": "absent.npz",
                "checkpoint_sha256": "b" * 64,
                "files": component_files,
            },
        }
    }
    out.mkdir(parents=True)
    (out / "registration.json").write_text(json.dumps(registration))
    prior = tmp_path / "docs/data/seed511-fixed-composition-screen-v4"
    prior.mkdir(parents=True)
    (prior / "registration.json").write_text(json.dumps({"runtime_assets": {}}))
    # A newly computed receipt hash cannot authorize bytes differing from the recorded identity.
    with pytest.raises(
        ValueError, match="seed512_supplement_artifact_hash:E4:model.npz"
    ):
        audit_assets(tmp_path)


def test_changed_result_rejected_after_receipt_rebind() -> None:
    with pytest.raises(ValueError, match="seed512_supplement_result_mismatch"):
        require_result_match({"score": 0.5}, {"score": 0.6})
