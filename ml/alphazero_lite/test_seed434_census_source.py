"""Semantic and portable-publication tests for seed434."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ml.alphazero_lite.seed433_census_correction import calculate
from ml.alphazero_lite.verify_seed434_census_source import (
    _positions,
    compare_groups,
    compare_rows,
    reconstruct_sources,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed432-policy-target-compatibility"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_registration(root: Path) -> dict:
    source_dir = root / "docs/data/seed426-canonical-overlap/sources"
    source_dir.mkdir(parents=True)
    specs = []
    for name, weight, value_mode in zip(
        (
            "fresh",
            "generic_bootstrap",
            "random_teacher",
            "opening_disagreement",
            "stability",
        ),
        (1, 4, 1, 8, 4),
        ("default", "sharpened", "sharpened", "sharpened", "sharpened"),
        strict=True,
    ):
        row = {
            "state": [4 / 48] * 12 + [0, 0, 0] + [0] * 12,
            "policy": [1, 0, 0, 0, 0, 0],
            "value": 0,
            "policy_target_mode": "sharpened",
            "value_target_mode": value_mode,
        }
        raw = (json.dumps(row) + "\n").encode()
        path = source_dir / f"{name}.jsonl.gz"
        with gzip.open(path, "wb") as stream:
            stream.write(raw)
        specs.append(
            {
                "name": name,
                "weight": weight,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "value_target_mode": value_mode,
            }
        )
    return {"replays": specs}


def test_altered_source_bytes_rejected(tmp_path):
    registration = _synthetic_registration(tmp_path)
    path = tmp_path / "docs/data/seed426-canonical-overlap/sources/fresh.jsonl.gz"
    row = {
        "state": [4 / 48] * 12 + [0, 0, 0] + [0] * 12,
        "policy": [0, 1, 0, 0, 0, 0],
        "value": 0,
        "policy_target_mode": "sharpened",
        "value_target_mode": "default",
    }
    with gzip.open(path, "wb") as stream:
        stream.write((json.dumps(row) + "\n").encode())
    with pytest.raises(ValueError, match="decompressed_source_hash_mismatch:fresh"):
        list(reconstruct_sources(tmp_path, registration))


def test_forged_split_position_identity_rejected(tmp_path):
    relative = "split.json.gz"
    path = tmp_path / relative
    train, validation = [0], [1]
    split = {"train_positions": train, "validation_positions": validation}
    with gzip.open(path, "wt") as stream:
        json.dump(split, stream)
    registration = {
        "training": {
            "source_row_split": {
                "path": relative,
                "sha256": _hash(path),
                "train_count": 1,
                "validation_count": 1,
                "train_positions_sha256": hashlib.sha256(
                    (0).to_bytes(8, "little", signed=True)
                ).hexdigest(),
                "validation_positions_sha256": hashlib.sha256(
                    (1).to_bytes(8, "little", signed=True)
                ).hexdigest(),
            }
        }
    }
    registration["training"]["source_row_split"]["train_positions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="train_position_identity_mismatch"):
        _positions(tmp_path, registration, 2, [(2, 1)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("target", [0, 1, 0, 0, 0, 0]),
        ("policy_coefficient", 0.5),
        ("source", "other"),
        ("line", 99),
        ("replay_multiplicity", 3),
        ("partition", "validation"),
        ("input_hex", "00"),
        ("input_sha256", "0" * 64),
        ("canonical_state", "forged"),
    ],
)
def test_forged_reconstructed_row_fields_rejected(field, value):
    expected = [
        {
            "compact_row": 0,
            "target": [1, 0, 0, 0, 0, 0],
            "policy_coefficient": 1.0,
            "source": "fresh",
            "line": 1,
            "replay_multiplicity": 1,
            "partition": "train",
            "input_hex": "abcd",
            "input_sha256": "x",
            "canonical_state": "state",
        }
    ]
    forged = [dict(expected[0], **{field: value})]
    with pytest.raises(ValueError, match="seed432_row_source_mismatch:0"):
        compare_rows(expected, forged)


@pytest.mark.parametrize(
    "field",
    [
        "canonical_states",
        "compact_rows",
        "source_lines",
        "replay_multiplicities",
        "input_sha256",
    ],
)
def test_forged_group_fields_rejected(field):
    expected = [
        {
            "partition": "train",
            "input_sha256": "id",
            "canonical_states": ["state"],
            "compact_rows": [0],
            "source_lines": [["fresh", 1]],
            "replay_multiplicities": [1],
        }
    ]
    forged = [dict(expected[0], **{field: ["forged"]})]
    with pytest.raises(ValueError, match="seed432_group_source_mismatch"):
        compare_groups(expected, forged)


def test_corrected_production_calculation_uses_control_gate(tmp_path):
    source = tmp_path / "rows"
    source.mkdir()
    rows = []
    for target in ([1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]):
        rows.append(
            {
                "partition": "train",
                "active_stones": 40,
                "input_sha256": "same",
                "target": target,
                "source": "fresh",
                "exposure_weight": 1.0,
                "equal_input_weight": 1.0,
                "treatment_exposure_weight": 0.0,
                "treatment_equal_input_weight": 0.0,
            }
        )
    with gzip.open(source / "row-accounting.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (source / "results.json").write_text("{}\n")
    result = calculate(source, None)
    assert result["classification"] == "material_policy_target_disagreement"
    primary = result["populations"]["train/gt32"]
    assert primary["exposure"]["mean_js_nats"] >= 0.05
    assert primary["seed429_treatment_exposure"]["mean_js_nats"] == 0


def test_actual_publication_and_relocated_cli_are_read_only(tmp_path):
    expected = verify(ROOT)
    assert expected["compact_rows"] == 87625
    assert expected["expanded_positions"] == 149448
    evidence = [
        DATA / name
        for name in (
            "manifest.json",
            "results.json",
            "row-accounting.jsonl.gz",
            "group-accounting.jsonl.gz",
        )
    ]
    before = {str(p.relative_to(ROOT)): _hash(p) for p in evidence}
    relocated = tmp_path / "checkout"
    shutil.copytree(
        ROOT / "ml",
        relocated / "ml",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in (
        "docs/data/seed426-canonical-overlap/sources",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed432-policy-target-compatibility",
        "docs/data/seed433-seed432-census-correction",
    ):
        source = ROOT / relative
        target = relocated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed434_census_source",
            "--root",
            str(relocated),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(relocated)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"all_seed432_rows_reproduce": true' in result.stdout

    # Refresh the archive digest after an evidence-only forgery: source reconstruction
    # must still reject the target mismatch before accepting downstream bindings.
    relocated_data = relocated / "docs/data/seed432-policy-target-compatibility"
    rows_path = relocated_data / "row-accounting.jsonl.gz"
    with gzip.open(rows_path, "rt") as stream:
        forged_rows = [json.loads(line) for line in stream]
    forged_rows[0]["target"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    with gzip.open(rows_path, "wt") as stream:
        for row in forged_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest_path = relocated_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_accounting_sha256"] = _hash(rows_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="seed432_row_source_mismatch:0"):
        verify(relocated)

    # Similarly refresh the group-archive hash after forging a group field.
    shutil.copy2(DATA / "row-accounting.jsonl.gz", rows_path)
    manifest["row_accounting_sha256"] = _hash(rows_path)
    groups_path = relocated_data / "group-accounting.jsonl.gz"
    with gzip.open(groups_path, "rt") as stream:
        groups = [json.loads(line) for line in stream]
    groups[0]["source_lines"][0][1] += 1
    with gzip.open(groups_path, "wt") as stream:
        for group in groups:
            stream.write(json.dumps(group, separators=(",", ":")) + "\n")
    manifest["group_accounting_sha256"] = _hash(groups_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="seed432_group_source_mismatch"):
        verify(relocated)

    after = {key: _hash(ROOT / key) for key in before}
    assert after == before
