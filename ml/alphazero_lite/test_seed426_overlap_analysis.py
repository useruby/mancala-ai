"""Unit and portable-verifier regression tests for the seed426 publication."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

from ml.alphazero_lite.seed426_overlap_analysis import (
    ACCOUNTING_DEFINITIONS,
    DECISION_RULES,
    IDENTITY_DEFINITIONS,
    audit,
    float32_identity,
    state_identity,
)
from ml.alphazero_lite.verify_seed426_overlap_audit import verify

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
WEIGHTS = (1, 4, 1, 8, 4)
VALUE_MODES = ("default", "sharpened", "sharpened", "sharpened", "sharpened")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(state: dict[str, object]) -> str:
    identity = (
        tuple(state["player_pits"]),
        tuple(state["opponent_pits"]),
        state["player_store"],
        state["opponent_store"],
        state["current_player"],
    )
    return json.dumps(identity, separators=(",", ":"))


def _f32(values: list[object]) -> bytes:
    return struct.pack("<" + "f" * len(values), *(float(value) for value in values))


def _state(seed: int) -> list[float]:
    pits = [4 + seed] * 6 + [3] * 6
    return [value / 48 for value in pits] + [0.0, 0.0, 0.0] + [0.0] * 12


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _fixture(root: Path) -> Path:
    data = root / "docs/data/seed426-canonical-overlap"
    split_dir = root / "docs/data/seed416-policy-target-softening/training-freeze-v3"
    ml = root / "ml/alphazero_lite"
    ml.mkdir(parents=True, exist_ok=True)
    for name in (
        "seed426_overlap_analysis.py",
        "verify_seed426_overlap_audit.py",
        "run_seed426_overlap_audit.py",
        "train.py",
        "self_play.py",
        "fresh_p1_adapter_teacher_audit.py",
    ):
        shutil.copyfile(ROOT / "ml/alphazero_lite" / name, ml / name)

    states = [_state(0), _state(0), _state(1), _state(2), _state(3)]
    compact = []
    source_raw: dict[str, bytes] = {}
    target_checks = {}
    for source_index, name in enumerate(SOURCE_NAMES):
        row = {
            "state": states[source_index],
            "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "value": 0.25,
            "policy_target_mode": "sharpened",
            "value_target_mode": VALUE_MODES[source_index],
        }
        raw = (json.dumps(row, separators=(",", ":")) + "\n").encode()
        source_raw[name] = raw
        (data / "sources" / f"{name}.jsonl.gz").parent.mkdir(
            parents=True, exist_ok=True
        )
        (data / "sources" / f"{name}.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
        input_id = float32_identity(states[source_index]).hex()
        state = {
            "player_pits": [round(value * 48) for value in states[source_index][:6]],
            "opponent_pits": [
                round(value * 48) for value in states[source_index][6:12]
            ],
            "player_store": round(states[source_index][12] * 48),
            "opponent_store": round(states[source_index][13] * 48),
            "current_player": round(states[source_index][14]),
        }
        compact.append(
            {
                "source": name,
                "raw_line": 1,
                "compact_row": source_index,
                "active_stones": sum(state["player_pits"])
                + sum(state["opponent_pits"]),
                "canonical_identity": _canonical(state),
                "input_identity": input_id,
            }
        )
        target_checks[name] = {
            "eligible_rows": 1,
            "policy_float32_sha256": _digest(_f32(row["policy"])),
            "value_float32_sha256": _digest(_f32([row["value"]])),
        }

    train_positions = [0, 5]
    validation_positions = [
        position for position in range(sum(WEIGHTS)) if position not in train_positions
    ]
    train_source_rows, validation_source_rows = [0, 2], [1, 3, 4]
    split = {
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "train_source_rows": train_source_rows,
        "validation_source_rows": validation_source_rows,
    }
    split_raw = gzip.compress(
        (json.dumps(split, sort_keys=True) + "\n").encode(), mtime=0
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "source-row-split.json.gz").write_bytes(split_raw)
    expected_rows = []
    weighted_position = 0
    for index, (name, weight) in enumerate(zip(SOURCE_NAMES, WEIGHTS, strict=True)):
        for copy in range(weight):
            expected_rows.append(
                {
                    **compact[index],
                    "copy": copy,
                    "weighted_position": weighted_position,
                    "partition": "train"
                    if weighted_position in train_positions
                    else "validation",
                    "weight": 1,
                }
            )
            weighted_position += 1
    (data / "row-accounting.jsonl.gz").parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(data / "row-accounting.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in expected_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    source_specs = [
        {
            "name": name,
            "path": f"/not-used/{name}.jsonl",
            "sha256": _digest(source_raw[name]),
            "weight": weight,
            "value_target_mode": mode,
        }
        for name, weight, mode in zip(SOURCE_NAMES, WEIGHTS, VALUE_MODES, strict=True)
    ]
    registration = {
        "replays": source_specs,
        "source_hashes": {
            "ml/alphazero_lite/train.py": _digest(
                (ROOT / "ml/alphazero_lite/train.py").read_bytes()
            ),
            "ml/alphazero_lite/self_play.py": _digest(
                (ROOT / "ml/alphazero_lite/self_play.py").read_bytes()
            ),
        },
        "training": {
            "source_row_split": {
                "path": "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
                "sha256": _digest(split_raw),
                "train_positions_sha256": _digest(
                    b"".join(struct.pack("<q", n) for n in train_positions)
                ),
                "validation_positions_sha256": _digest(
                    b"".join(struct.pack("<q", n) for n in validation_positions)
                ),
            }
        },
    }
    registration_raw = _write_json(
        root / "docs/data/seed416-policy-target-softening/registration-v3.json",
        registration,
    )
    census = audit(expected_rows)
    source_accounting = {}
    for name, weight in zip(SOURCE_NAMES, WEIGHTS, strict=True):
        positions = [row for row in expected_rows if row["source"] == name]
        source_accounting[name] = {
            "weight": weight,
            "eligible_source_rows": 1,
            "weighted_positions": weight,
            "train_source_rows": int(
                any(row["partition"] == "train" for row in positions)
            ),
            "validation_source_rows": int(
                any(row["partition"] == "validation" for row in positions)
            ),
            "train_weighted_positions": sum(
                row["partition"] == "train" for row in positions
            ),
            "validation_weighted_positions": sum(
                row["partition"] == "validation" for row in positions
            ),
        }
    results = {
        "schema": "seed426-canonical-overlap-results-v1",
        "identity_census": census,
        "weighted_positions": weighted_position,
        "eligible_source_rows": len(compact),
        "source_accounting": source_accounting,
        "split_positions": {
            "train": len(train_positions),
            "validation": len(validation_positions),
        },
        "eligibility": {
            "frozen_call_exclude_buckets": None,
            "raw_jsonl_lines": 5,
            "eligible_compact_rows": 5,
            "skipped_raw_lines": [],
            "rule": "the frozen load_jsonl_replay call supplies no exclusion buckets; every parsed raw row must validate and load",
        },
        "loader_parity": {
            "loaded_x_shape": [len(states), 27],
            "loaded_x_float32_sha256": _digest(
                b"".join(float32_identity(state) for state in states)
            ),
            "loaded_policy_float32_sha256": _digest(
                b"".join(_f32([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for _ in states)
            ),
            "loaded_value_float32_sha256": _digest(
                b"".join(_f32([0.25]) for _ in states)
            ),
            "source_targets": target_checks,
        },
        "decision": "canonical_state_overlap_detected",
        "recommendation": "separately_authorized_validation_metric_correction_using_identity_disjoint_holdout",
    }
    result_raw = _write_json(data / "results.json", results)
    del result_raw
    manifest_sources = [
        {
            "name": name,
            "sha256": _digest(source_raw[name]),
            "weight": weight,
            "policy_target_mode": "sharpened",
            "value_target_mode": mode,
            "eligibility": "no exclusion buckets",
        }
        for name, weight, mode in zip(SOURCE_NAMES, WEIGHTS, VALUE_MODES, strict=True)
    ]
    execution = {
        "runner": "run_seed426_overlap_audit.py",
        "analysis": "seed426_overlap_analysis.py",
        "frozen_train": "train.py",
        "kalah_v3_encoder": "self_play.py",
        "state_decoder": "fresh_p1_adapter_teacher_audit.py",
        "verifier": "verify_seed426_overlap_audit.py",
    }
    snapshots = data / "execution-source-snapshots"
    snapshots.mkdir(exist_ok=True)
    execution_hashes = {}
    for name, filename in execution.items():
        raw = (ml / filename).read_bytes()
        (snapshots / filename).write_bytes(raw)
        execution_hashes[f"execution-source-snapshots/{filename}"] = _digest(raw)
    manifest = {
        "schema": "seed426-canonical-overlap-manifest-v1",
        "registrations": {
            "seed416-v3": _digest(registration_raw),
            "source_split": _digest(split_raw),
            "frozen_train": registration["source_hashes"]["ml/alphazero_lite/train.py"],
            "kalah_v3_encoder": registration["source_hashes"][
                "ml/alphazero_lite/self_play.py"
            ],
        },
        "sources": manifest_sources,
        "identity_definitions": IDENTITY_DEFINITIONS,
        "accounting": ACCOUNTING_DEFINITIONS,
        "decision_rules": DECISION_RULES,
        "chronology": {"amendment": "synthetic portable verifier fixture"},
        "execution_sources": {
            key: _digest((ml / filename).read_bytes())
            for key, filename in execution.items()
        },
        "execution_source_snapshots": execution_hashes,
    }
    _write_json(data / "manifest.json", manifest)
    return root


def _fixture_dir(tmp_path: Path) -> Path:
    root = tmp_path / "relocated"
    return _fixture(root)


def _rebind_registration(
    root: Path,
    registration: dict[str, object],
    split: dict[str, object],
    split_raw: bytes,
) -> None:
    split_path = root / registration["training"]["source_row_split"]["path"]
    split_path.write_bytes(split_raw)
    registration["training"]["source_row_split"]["sha256"] = _digest(split_raw)
    for partition in ("train", "validation"):
        registration["training"]["source_row_split"][
            f"{partition}_positions_sha256"
        ] = _digest(
            b"".join(struct.pack("<q", int(n)) for n in split[f"{partition}_positions"])
        )
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration_raw = _write_json(registration_path, registration)
    manifest_path = root / "docs/data/seed426-canonical-overlap/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["registrations"]["seed416-v3"] = _digest(registration_raw)
    manifest["registrations"]["source_split"] = _digest(split_raw)
    _write_json(manifest_path, manifest)


def test_canonical_identity_includes_stores_and_player() -> None:
    base = {
        "player_pits": [1] * 6,
        "opponent_pits": [2] * 6,
        "player_store": 3,
        "opponent_store": 4,
        "current_player": 0,
    }
    assert state_identity(base) != state_identity({**base, "player_store": 5})
    assert state_identity(base) != state_identity({**base, "opponent_store": 6})
    assert state_identity(base) != state_identity({**base, "current_player": 1})


def test_exact_input_identity_weighted_rows_and_additive_attribution() -> None:
    vector = [0.0, 0.5, 1.0]
    assert float32_identity(vector) == b"\x00\x00\x00\x00\x00\x00\x00?\x00\x00\x80?"
    assert float32_identity([1.0]) != float32_identity([1.0 + 2**-23])
    row = {
        "canonical_identity": "s",
        "input_identity": "s",
        "partition": "train",
        "source": "x",
        "compact_row": 0,
        "weight": 1,
        "active_stones": 42,
    }
    validation = {**row, "partition": "validation", "source": "y", "compact_row": 1}
    report = audit([row, validation])
    summary = report["canonical"]
    assert summary["intersection_unique"] == 1
    assert summary["validation_unseen_unique"] == 0
    assert (
        summary["validation_overlap_denominators"]["weighted_positions"]["numerator"]
        == 1
    )
    assert (
        sum(
            pattern["shared_validation_positions"]
            for pattern in summary["additive_membership_patterns"]
        )
        == 1
    )
    assert audit([row])["canonical"]["buckets"]["<=16"]["validation_unique"] == 0


def test_within_source_duplicate_groups_zero_overlap_and_weighted_copies() -> None:
    train_copy_a = {
        "canonical_identity": "duplicate",
        "input_identity": "encoded-a",
        "partition": "train",
        "source": "source-a",
        "compact_row": 2,
        "weight": 1,
        "active_stones": 30,
    }
    train_copy_b = {**train_copy_a, "compact_row": 3}
    validation_copy = {**train_copy_a, "partition": "validation", "compact_row": 4}
    report = audit([train_copy_a, train_copy_b, validation_copy])
    canonical = report["canonical"]
    assert canonical["intersection_unique"] == 1
    assert canonical["validation_overlap_denominators"]["eligible_source_rows"] == {
        "numerator": 1,
        "denominator": 1,
        "fraction": 1.0,
    }
    assert canonical["source_pairwise_validation_exposure"]["within_source"] == 1

    no_overlap = audit(
        [train_copy_a, {**validation_copy, "canonical_identity": "unseen"}]
    )["canonical"]
    assert no_overlap["intersection_unique"] == 0
    assert no_overlap["validation_unseen_unique"] == 1
    assert (
        no_overlap["validation_overlap_denominators"]["weighted_positions"]["fraction"]
        == 0.0
    )

    # Repeated positions from the same compact row are weighted copies; source-
    # row and identity denominators deduplicate them while position counts do not.
    weighted = audit(
        [train_copy_a, train_copy_b, validation_copy, {**validation_copy, "copy": 1}]
    )["canonical"]
    assert (
        weighted["validation_overlap_denominators"]["eligible_source_rows"][
            "denominator"
        ]
        == 1
    )
    assert (
        weighted["validation_overlap_denominators"]["weighted_positions"]["denominator"]
        == 2
    )


def test_actual_verifier_entrypoint_is_portable_and_read_only(tmp_path: Path) -> None:
    root = _fixture_dir(tmp_path)
    before = {
        path.relative_to(root): _digest(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    verifier = root / "ml/alphazero_lite/verify_seed426_overlap_audit.py"
    completed = subprocess.run(
        [sys.executable, str(verifier), "--root", str(root)],
        cwd=elsewhere,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    after = {
        path.relative_to(root): _digest(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after
    evidence = [
        json.loads(line)
        for line in gzip.open(
            root / "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz",
            "rt",
        )
    ]
    assert evidence[0]["raw_line"] == 1 and evidence[0]["compact_row"] == 0
    assert evidence[1]["raw_line"] == 1 and evidence[1]["compact_row"] == 1
    assert not (root / "model-artifact").exists()
    assert not (root / ".tmp").exists()


def test_verifier_rejects_missing_or_altered_snapshot(tmp_path: Path) -> None:
    root = _fixture_dir(tmp_path)
    snapshot = root / "docs/data/seed426-canonical-overlap/sources/fresh.jsonl.gz"
    original = snapshot.read_bytes()
    snapshot.unlink()
    with pytest.raises(ValueError, match="missing_replay_snapshot:fresh"):
        verify(root)
    snapshot.write_bytes(gzip.compress(b"changed\n", mtime=0))
    with pytest.raises(ValueError, match="replay_snapshot_hash_mismatch:fresh"):
        verify(root)
    snapshot.write_bytes(original)


def test_verifier_rejects_altered_split_bytes_and_partition_membership(
    tmp_path: Path,
) -> None:
    root = _fixture_dir(tmp_path)
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    split_path = root / registration["training"]["source_row_split"]["path"]
    original = split_path.read_bytes()
    split_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="registered_split_hash_mismatch"):
        verify(root)
    split = json.loads(gzip.decompress(original))
    split["train_positions"][0], split["validation_positions"][0] = (
        split["validation_positions"][0],
        split["train_positions"][0],
    )
    split_raw = gzip.compress(
        (json.dumps(split, sort_keys=True) + "\n").encode(), mtime=0
    )
    _rebind_registration(root, registration, split, split_raw)
    with pytest.raises(ValueError, match="source_row_membership_mismatch"):
        verify(root)


def test_verifier_rejects_row_mapping_and_weight_multiplicity(tmp_path: Path) -> None:
    root = _fixture_dir(tmp_path)
    data = root / "docs/data/seed426-canonical-overlap"
    evidence = data / "row-accounting.jsonl.gz"
    original = evidence.read_bytes()
    rows = [json.loads(line) for line in gzip.decompress(original).splitlines()]
    rows[0]["compact_row"] = 99
    evidence.write_bytes(
        gzip.compress(
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            ),
            mtime=0,
        )
    )
    with pytest.raises(ValueError, match="row_mapping_mismatch"):
        verify(root)
    evidence.write_bytes(original)
    rows = [json.loads(line) for line in gzip.decompress(original).splitlines()]
    rows[1]["copy"] = 9
    evidence.write_bytes(
        gzip.compress(
            b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            ),
            mtime=0,
        )
    )
    with pytest.raises(ValueError, match="row_mapping_mismatch"):
        verify(root)
    evidence.write_bytes(original)
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    registration["replays"][1]["weight"] = 3
    raw = _write_json(registration_path, registration)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["registrations"]["seed416-v3"] = _digest(raw)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="registered_weights_mismatch"):
        verify(root)


def test_verifier_rejects_target_modes_encoder_and_published_accounting(
    tmp_path: Path,
) -> None:
    root = _fixture_dir(tmp_path)
    data = root / "docs/data/seed426-canonical-overlap"
    source_path = data / "sources/fresh.jsonl.gz"
    original_source = source_path.read_bytes()
    row = json.loads(gzip.decompress(original_source))
    row["policy_target_mode"] = "invalid-mode"
    raw = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    source_path.write_bytes(gzip.compress(raw, mtime=0))
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][0]["eligibility"] = "unexpected filtering"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest_loader_contract_mismatch"):
        verify(root)

    _fixture(root)
    source_path.write_bytes(gzip.compress(raw, mtime=0))
    registration = json.loads(registration_path.read_text())
    new_hash = _digest(raw)
    registration["replays"][0]["sha256"] = new_hash
    registration_raw = _write_json(registration_path, registration)
    manifest = json.loads(manifest_path.read_text())
    manifest["registrations"]["seed416-v3"] = _digest(registration_raw)
    manifest["sources"][0]["sha256"] = new_hash
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="target_mode_declaration_mismatch:fresh:1"):
        verify(root)

    _fixture(root)
    state = json.loads(gzip.decompress(source_path.read_bytes()))
    state["state"][15] = 1.0
    altered = (json.dumps(state, separators=(",", ":")) + "\n").encode()
    source_path.write_bytes(gzip.compress(altered, mtime=0))
    registration = json.loads(registration_path.read_text())
    registration["replays"][0]["sha256"] = _digest(altered)
    registration_raw = _write_json(registration_path, registration)
    manifest = json.loads(manifest_path.read_text())
    manifest["registrations"]["seed416-v3"] = _digest(registration_raw)
    manifest["sources"][0]["sha256"] = _digest(altered)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="row_mapping_mismatch"):
        verify(root)

    _fixture(root)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["accounting"]["weighted_position_order"] = "adjacent duplicates"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest_accounting_definition_mismatch"):
        verify(root)

    _fixture(root)
    results_path = data / "results.json"
    results = json.loads(results_path.read_text())
    results["identity_census"]["canonical"]["intersection_unique"] += 1
    _write_json(results_path, results)
    with pytest.raises(ValueError, match="published_accounting_mismatch"):
        verify(root)
