from __future__ import annotations

import json
import hashlib
import gzip
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import train
from ml.alphazero_lite.seed416_policy_target_softening import resume_action
from ml.alphazero_lite.seed416_policy_target_softening import (
    bootstrap_paired,
    entropy,
    softened_policy,
    transform_jsonl,
    transform_row,
)
from ml.alphazero_lite.verify_seed416_policy_target_softening import verify


def _row() -> dict:
    state = [4 / 48] * 12 + [0.0, 0.0, 0.0]
    return {
        "state": state,
        "policy": [0.0, 0.1, 0.2, 0.3, 0.4, 0.0],
        "stored_policy_target": [0.0, 0.1, 0.2, 0.3, 0.4, 0.0],
        "value": 0.25,
        "policy_target_mode": "sharpened",
        "policy_target_actual_mode": "sharpened",
        "teacher_source": "puct",
    }


def test_softening_preserves_support_legality_and_immutable_fields() -> None:
    original = _row()
    transformed, changed = transform_row(original, "B")
    assert changed
    assert transformed["state"] == original["state"]
    assert transformed["value"] == original["value"]
    assert transformed["policy"] == transformed["stored_policy_target"]
    assert [p == 0 for p in transformed["policy"]] == [
        p == 0 for p in original["policy"]
    ]
    assert sum(transformed["policy"]) == pytest.approx(1.0)
    assert entropy(transformed["policy"]) > entropy(original["policy"])
    assert transformed["policy_transform_provenance"]["source_semantics"].endswith(
        "not recovered visit counts"
    )


def test_exact_root_target_is_unchanged() -> None:
    row = _row()
    row.update(
        teacher_source="exact_root_tablebase",
        policy_target_actual_mode="exact_root_one_hot",
        policy=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        stored_policy_target=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    result, changed = transform_row(row, "B")
    assert not changed
    assert result == row


def test_derivatives_are_immutable_and_keep_row_multiplicity(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(_row()) for _ in range(3)) + "\n")
    output = tmp_path / "derivative.jsonl"
    metadata = transform_jsonl(source, output, "B")
    assert metadata["rows"] == metadata["changed_rows"] == 3
    assert len(output.read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        transform_jsonl(source, output, "B")


def test_bootstrap_requires_same_complete_opening_clusters() -> None:
    rows = []
    for lane, score in (("A", 0.5), ("B", 0.6)):
        for opening in range(512):
            rows.extend(
                {"lane": lane, "opening_id": str(opening), "opponent_score": score}
                for _ in range(2)
            )
    result = bootstrap_paired(rows, samples=100, seed=416)
    assert result["primary_mean_B_minus_A"] == pytest.approx(0.1)
    assert result["primary_95_percentile_interval"] == pytest.approx([0.1, 0.1])
    with pytest.raises(ValueError, match="opening_seat_pair_incomplete"):
        bootstrap_paired(rows[:-1], samples=10)


def test_zero_policy_rejected_as_invalid_normalization() -> None:
    with pytest.raises(ValueError, match="invalid_policy_target"):
        softened_policy([0.0] * 6)


def test_illegal_policy_mass_is_rejected() -> None:
    row = _row()
    row["state"] = [0.0] * 6 + [4 / 48] * 6 + [0.0, 0.0, 0.0]
    row["policy"] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="illegal_mass"):
        transform_row(row, "B")


def test_source_row_split_and_weighted_multiplicity_parity(tmp_path) -> None:
    row = _row()
    row["policy"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    row["stored_policy_target"] = list(row["policy"])
    row["state"] = [4 / 48] * 12 + [0.0, 0.0, 0.0]
    row["legal_moves"] = [0, 1, 2, 3, 4, 5]
    row["policy_target_actual_mode"] = "sharpened"
    source_a, source_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    payload = "\n".join(json.dumps(row) for _ in range(3)) + "\n"
    source_a.write_text(payload)
    source_b.write_text(payload)
    derivative = tmp_path / "softened.jsonl"
    transform_jsonl(source_b, derivative, "B")
    paths = [source_a, source_b]
    train.set_seed(416)
    xa, pa, va, ra = train.load_jsonl_replay(
        paths, [1, 4], policy_target_mode="sharpened", value_target_mode="default"
    )
    train.set_seed(416)
    xb, pb, vb, rb = train.load_jsonl_replay(
        [source_a, derivative],
        [1, 4],
        policy_target_mode="sharpened",
        value_target_mode="default",
    )
    assert np.array_equal(xa, xb)
    assert np.array_equal(va, vb)
    assert np.array_equal(ra, rb)
    assert np.bincount(ra).tolist() == [1, 1, 1, 4, 4, 4]
    train.set_seed(416)
    tr_a, val_a = train.split_replay_positions_by_source_row(ra, val_split=0.1)
    train.set_seed(416)
    tr_b, val_b = train.split_replay_positions_by_source_row(rb, val_split=0.1)
    assert np.array_equal(tr_a, tr_b)
    assert np.array_equal(val_a, val_b)
    assert set(ra[tr_a]).isdisjoint(set(ra[val_a]))


def test_interrupted_lane_resume_skips_only_unchanged_completed_lane() -> None:
    completed = {"epochs": {"1": "a" * 64, "2": "b" * 64}}
    assert resume_action(completed, completed.copy()) == "skip"
    assert resume_action(None, {"epochs": {}}) == "run"
    with pytest.raises(ValueError, match="resumed_lane_evidence_mismatch"):
        resume_action(completed, {"epochs": {"1": "f" * 64}})


def _synthetic_publication(root: Path) -> None:
    data = root / "docs/data/seed416-policy-target-softening"
    data.mkdir(parents=True)
    suite = [
        {
            "state_hash": f"{index:064x}",
            "pit_sum": 48,
            "state": {"player_pits": [4] * 6, "opponent_pits": [4] * 6},
        }
        for index in range(512)
    ]
    suite_path = data / "openings-v3.jsonl"
    suite_path.write_text("".join(json.dumps(row) + "\n" for row in suite))
    proof_path = data / "opening-exclusion-proof-v3.json"
    proof_path.write_text(
        json.dumps(
            {"excluded_identities": [], "combined_union_count": 0, "suite_overlap": 0}
        )
    )
    freeze_dir = data / "training-freeze-v3"
    freeze_dir.mkdir()
    split_path = freeze_dir / "source-row-split.json.gz"
    split_path.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "train_positions": [],
                    "validation_positions": [],
                    "train_source_rows": [],
                    "validation_source_rows": [],
                }
            ).encode(),
            mtime=0,
        )
    )
    permutation_path = freeze_dir / "epoch-permutations.json.gz"
    empty_permutations = [[], [], [], []]
    permutation_path.write_bytes(
        gzip.compress(json.dumps(empty_permutations).encode(), mtime=0)
    )
    registration_path = data / "registration-v3.json"
    epoch_hashes = {
        str(epoch): hashlib.sha256(f"E{epoch}".encode()).hexdigest()
        for epoch in range(1, 5)
    }
    permutations = {
        str(epoch): hashlib.sha256(json.dumps([]).encode()).hexdigest()
        for epoch in range(1, 5)
    }
    registration_path.write_text(
        json.dumps(
            {
                "schema": "seed416-policy-target-softening-registration-v1",
                "status": "registered_before_training_and_model_probing",
                "suite": {
                    "sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest()
                },
                "replays": [{"weight": weight} for weight in [1, 4, 1, 8, 4]],
                "exclusion_proof_sha256": hashlib.sha256(
                    proof_path.read_bytes()
                ).hexdigest(),
                "training": {
                    "source_row_split": {
                        "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                        "train_count": 0,
                        "validation_count": 0,
                    },
                    "epoch_permutations": {
                        "sha256": hashlib.sha256(
                            permutation_path.read_bytes()
                        ).hexdigest(),
                        "epoch_sha256": permutations,
                    },
                },
                "source_hashes": {},
            }
        )
    )
    supersession_path = data / "registration-supersession-v3.json"
    supersession_path.write_text(
        json.dumps(
            {
                "replacement_registration_sha256": hashlib.sha256(
                    registration_path.read_bytes()
                ).hexdigest()
            }
        )
    )
    training_lanes = {
        lane: {"epochs": epoch_hashes, "permutations": permutations}
        for lane in ("A", "B")
    }
    training_path = data / "training-results.json"
    training_path.write_text(
        json.dumps(
            {
                "registration_sha256": hashlib.sha256(
                    registration_path.read_bytes()
                ).hexdigest(),
                "lanes": training_lanes,
            }
        )
    )
    binding_path = data / "evaluation-binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "registration_sha256": hashlib.sha256(
                    registration_path.read_bytes()
                ).hexdigest(),
                "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "native_probe_sha256": "a" * 64,
                "tablebase_sha256": "b" * 64,
                "runtime_policy_sha256": "c" * 64,
                "training_sha256": hashlib.sha256(
                    training_path.read_bytes()
                ).hexdigest(),
                "candidates": {
                    lane: {"checkpoint_sha256": epoch_hashes["4"]}
                    for lane in ("A", "B")
                },
            }
        )
    )
    outcome_path = data / "outcome-binding.json"
    outcome_path.write_text(
        json.dumps(
            {
                "evaluation_binding_sha256": hashlib.sha256(
                    binding_path.read_bytes()
                ).hexdigest(),
                "status": "completed_2048_games",
                "matched_initial_seed_contexts": 1024,
            }
        )
    )
    rows = []
    for lane in ("A", "B"):
        for opening in range(512):
            for seat in (0, 1):
                game = {
                    "opening_contract": "arena_player_relative_v2",
                    "winner": "challenger",
                    "challenger_player": seat,
                }
                rows.append(
                    {
                        "lane": lane,
                        "opening_id": str(opening),
                        "opponent_score": 1.0,
                        "game": game,
                    }
                )
    ledger_path = data / "outcome-ledger.jsonl"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    from ml.alphazero_lite.seed416_policy_target_softening import bootstrap_paired

    analysis = bootstrap_paired(rows, samples=10000, seed=416)
    analysis.update(
        {
            "outcome_ledger_sha256": hashlib.sha256(
                ledger_path.read_bytes()
            ).hexdigest(),
            "outcome_binding_sha256": hashlib.sha256(
                outcome_path.read_bytes()
            ).hexdigest(),
        }
    )
    matrix_path = data / "per-opening-matrix.json"
    matrix_path.write_text(json.dumps(analysis["per_opening"]))
    (data / "analysis.json").write_text(json.dumps(analysis))


def test_public_verifier_is_read_only_and_rejects_altered_ledger(tmp_path) -> None:
    _synthetic_publication(tmp_path)
    paths = sorted(
        path
        for path in (tmp_path / "docs/data/seed416-policy-target-softening").rglob("*")
        if path.is_file()
    )
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert verify(tmp_path)["valid"]
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after
    ledger = tmp_path / "docs/data/seed416-policy-target-softening/outcome-ledger.jsonl"
    lines = ledger.read_text().splitlines()
    altered = json.loads(lines[0])
    altered["opponent_score"] = 0.0
    lines[0] = json.dumps(altered)
    ledger.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        verify(tmp_path)
