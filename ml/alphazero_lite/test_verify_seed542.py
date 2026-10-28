"""Tamper tests for the read-only seed542 verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed542 as verifier


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    monkeypatch.setattr(verifier, "CORPUS", tmp_path)
    inventory = []
    for seed in range(542000, 542064):
        path = tmp_path / f"shard-{seed}.jsonl"
        data = f"shard {seed}\n".encode()
        path.write_bytes(data)
        inventory.append(
            {
                "event": "finished",
                "seed": seed,
                "exit_status": 0,
                "output_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (tmp_path / "execution-inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inventory)
    )
    artifacts = {
        "accounting.json": {"status": "complete", "rows": 12},
        "development-membership.json": {"included": ["d:1"]},
        "training-protocol.json": {"schedule_sha256": "abc"},
    }
    for name, payload in artifacts.items():
        (tmp_path / name).write_text(json.dumps(payload))
    monkeypatch.setattr(verifier.rebuild, "build_artifacts", lambda: artifacts)
    return artifacts


def test_verifier_is_read_only_and_rebuilds_expected_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _fixture(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = verifier.verify()
    assert set(result) == set(artifacts)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize(
    ("filename", "bad_payload"),
    [
        ("accounting.json", {"rows": 11}),
        ("development-membership.json", {"included": []}),
        ("training-protocol.json", {"schedule_sha256": "corrupt"}),
    ],
)
def test_verifier_rejects_corrupted_archived_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    bad_payload: dict,
) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / filename).write_text(json.dumps(bad_payload))
    with pytest.raises(ValueError, match=f"archived_evidence_mismatch:{filename}"):
        verifier.verify()


def test_verifier_rejects_corrupt_inventory_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    path = tmp_path / "shard-542000.jsonl"
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="inventory_shard_hash_mismatch:542000"):
        verifier.verify()
