"""Focused semantic tests for the portable seed449 reconstruction."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import train
from ml.alphazero_lite import verify_seed450_seed449_correction as verifier

ROOT = Path(__file__).resolve().parents[2]


def _row24() -> dict:
    with gzip.open(ROOT / verifier.COMPRESSED_BASE / "fresh.jsonl.gz", "rt") as stream:
        return next(
            json.loads(line) for number, line in enumerate(stream, 1) if number == 24
        )


def _validate(row: dict, requested: str = "sharpened") -> None:
    train.validate_policy_target(
        np.asarray(row["policy"], dtype=np.float32),
        state=row["state"],
        path=Path("fixture"),
        row_number=1,
        policy_target_mode=requested,
        declared_mode=train.declared_policy_target_mode_for_row(row),
        row=row,
    )


def test_fresh_row_24_uses_actual_mode_precedence_and_preserved_target():
    row = _row24()
    assert row["policy_target_actual_mode"] == "exact_root_one_hot"
    assert row["policy_target_mode"] == "sharpened"
    assert train.declared_policy_target_mode_for_row(row) == "exact_root_one_hot"
    _validate(row)
    row["policy"] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    with pytest.raises(ValueError, match="exact_root_one_hot"):
        _validate(row)


def test_ordinary_declared_mode_matches_requested_and_conflict_rejected():
    row = _row24()
    row.pop("policy_target_actual_mode")
    row.pop("exact_selected_action")
    row.pop("exact_optimal_actions")
    row.pop("exact_action_margins")
    row["teacher_source"] = "ordinary"
    row["policy_target_mode"] = "sharpened"
    _validate(row)
    row["policy_target_actual_mode"] = "default"
    with pytest.raises(ValueError, match="does not match requested"):
        _validate(row, "sharpened")


def test_exact_root_optimal_set_uniform_validates_metadata_and_target():
    row = _row24()
    row["policy_target_actual_mode"] = "exact_root_optimal_set_uniform"
    row["policy"] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    _validate(row)
    row["exact_action_margins"]["5"] = -23
    with pytest.raises(ValueError, match="equal margins"):
        _validate(row)


def test_exact_root_one_hot_validates_shape_and_missing_mode_behavior():
    row = _row24()
    row["policy_target_actual_mode"] = "exact_root_one_hot"
    _validate(row)
    row.pop("policy_target_actual_mode")
    row.pop("policy_target_mode")
    with pytest.raises(ValueError, match="must declare"):
        _validate(row, "sharpened")


def test_lane_a_transform_and_derivative_hash_are_checked(tmp_path):
    registration = json.loads((ROOT / verifier.SEED416).read_text())
    paths = verifier._reconstruct_derivatives(ROOT, registration, tmp_path)
    assert set(paths) == set(verifier.SOURCES)
    assert all(path.is_file() for path in paths.values())


def test_source_order_rejects_mutation(tmp_path):
    registration = json.loads((ROOT / verifier.SEED416).read_text())
    registration["replays"][0], registration["replays"][1] = (
        registration["replays"][1],
        registration["replays"][0],
    )
    with pytest.raises(ValueError, match="source_order"):
        verifier._reconstruct_derivatives(ROOT, registration, tmp_path)


def test_compressed_hash_failure_is_distinct_from_semantic_failure(tmp_path):
    source = ROOT / verifier.COMPRESSED_BASE / "fresh.jsonl.gz"
    changed = tmp_path / "fresh.jsonl.gz"
    changed.write_bytes(source.read_bytes() + b"x")
    assert verifier.sha(changed) != verifier.COMPRESSED_SHA256["fresh"]


def _single_source_fixture(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    compressed_dir = source_root / verifier.COMPRESSED_BASE
    compressed_dir.mkdir(parents=True)
    return source_root, compressed_dir


def test_tampered_compressed_snapshot_fails_hash_check(tmp_path, monkeypatch):
    source_root, compressed_dir = _single_source_fixture(tmp_path, monkeypatch)
    original = ROOT / verifier.COMPRESSED_BASE / "fresh.jsonl.gz"
    (compressed_dir / "fresh.jsonl.gz").write_bytes(original.read_bytes() + b"x")
    monkeypatch.setattr(verifier, "SOURCES", ("fresh",))
    registration = json.loads((ROOT / verifier.SEED416).read_text())
    registration["replays"] = registration["replays"][:1]
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="compressed_snapshot_hash_invalid:fresh"):
        verifier._reconstruct_derivatives(source_root, registration, output)


def test_content_tampering_is_rejected_after_compressed_hash_is_rebound(
    tmp_path, monkeypatch
):
    source_root, compressed_dir = _single_source_fixture(tmp_path, monkeypatch)
    original = ROOT / verifier.COMPRESSED_BASE / "fresh.jsonl.gz"
    changed_path = compressed_dir / "fresh.jsonl.gz"
    with (
        gzip.open(original, "rt", encoding="utf-8") as source,
        gzip.open(changed_path, "wt", encoding="utf-8") as changed,
    ):
        first = json.loads(next(source))
        first["value"] = -first["value"]
        changed.write(json.dumps(first, separators=(",", ":")) + "\n")
        for line in source:
            changed.write(line)
    monkeypatch.setattr(verifier, "SOURCES", ("fresh",))
    monkeypatch.setattr(
        verifier,
        "COMPRESSED_SHA256",
        {"fresh": hashlib.sha256(changed_path.read_bytes()).hexdigest()},
    )
    registration = json.loads((ROOT / verifier.SEED416).read_text())
    registration["replays"] = registration["replays"][:1]
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="decompressed_source_identity_invalid:fresh"):
        verifier._reconstruct_derivatives(source_root, registration, output)
