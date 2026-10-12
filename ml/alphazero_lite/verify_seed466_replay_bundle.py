"""Independent, read-only reconstruction verifier for seed466."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_MANIFEST_ID = "seed466-five-source-lane-A-v1"
PACKET = Path("docs/data/seed464-dependency-closure-audit/packet")
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
WEIGHTS = (1, 4, 1, 8, 4)
PACKET_MANIFEST_SHA256 = (
    "d4ea022be45e6e6e259d163b8ef2deec1ac06761dec09f6f6ce4463fa507ec03"
)
PINNED_SOURCE_HASHES = {
    "ml/alphazero_lite/train.py": "2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32",
    "ml/alphazero_lite/verify_seed434_census_source.py": "4b632366f7b1ad920b77c692c0b9993de5420b723deba71bc9d081522789ec51",
    "ml/alphazero_lite/exact_root_policy_targets.py": "2aca4c1e2e9e9e199fb1d951b143c506785c23121531e26628a59c9afaf79898",
    "ml/alphazero_lite/seed429_policy_normalization.py": "b2743afc7bc4acdadff9e25fc456f8dc3757dfd0c98c9bc214aee805d70011d6",
    "ml/alphazero_lite/seed432_policy_target_census.py": "7b8dc48bdc1cf53d8dcfc53446e37edb2f84fdd583009d0693cc43a8e97204eb",
    "ml/alphazero_lite/seed433_census_correction.py": "3b4b7cf7387ade4fc5598f1360059a2b19418bbecd46e7020a1f5808746567eb",
    "ml/alphazero_lite/build_opening_suite.py": "5c15d541574905ff9d7c75bc8daf8813311e22f22f220563e18cb9b85fb8bd87",
    "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
}
EXPECTED_SOURCE_FILES = {
    *PINNED_SOURCE_HASHES,
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/verify_seed466_replay_bundle.py",
}
EXPECTED_INPUT_FILES = {
    "packet-manifest.json",
    *(f"derivatives/A/{source}.jsonl.gz" for source in SOURCES),
    "docs/data/seed416-policy-target-softening/registration-v3.json",
    "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
    "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
    "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
    *(
        f"docs/data/seed426-canonical-overlap/sources/{source}.jsonl.gz"
        for source in SOURCES
    ),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(bundle: Path, root: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity"] != EXPECTED_MANIFEST_ID:
        raise ValueError("manifest_identity_mismatch")
    if set(manifest["source_closure"]) != EXPECTED_SOURCE_FILES:
        raise ValueError("source_closure_inventory_mismatch")
    for source, expected in manifest["source_closure"].items():
        if digest((root / source).read_bytes()) != expected:
            raise ValueError(f"source_identity_mismatch:{source}")
        if source in PINNED_SOURCE_HASHES and expected != PINNED_SOURCE_HASHES[source]:
            raise ValueError(f"pinned_source_identity_mismatch:{source}")
    if set(manifest["source_inputs"]) != EXPECTED_INPUT_FILES:
        raise ValueError("source_input_inventory_mismatch")
    packet = root / PACKET
    if digest((packet / "packet-manifest.json").read_bytes()) != PACKET_MANIFEST_SHA256:
        raise ValueError("seed464_packet_manifest_identity_mismatch")
    for relative, record in manifest["source_inputs"].items():
        raw = (packet / relative).read_bytes()
        if digest(raw) != record["compressed_or_stored_sha256"]:
            raise ValueError(f"source_input_identity:{relative}")
        if (
            "decompressed_sha256" in record
            and digest(gzip.decompress(raw)) != record["decompressed_sha256"]
        ):
            raise ValueError(f"source_input_decompressed_identity:{relative}")
    registration_path = (
        packet / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    if [item["name"] for item in registration["replays"]] != list(SOURCES):
        raise ValueError("registered_source_order_mismatch")
    if [int(item["weight"]) for item in registration["replays"]] != list(WEIGHTS):
        raise ValueError("registered_source_weight_mismatch")
    compact: list[tuple[np.ndarray, np.ndarray, float, int, int, int, int]] = []
    source_lengths: list[int] = []
    for source_id, source_name in enumerate(SOURCES):
        path = packet / f"derivatives/A/{source_name}.jsonl.gz"
        packed = path.read_bytes()
        record = next(
            item
            for item in json.loads((packet / "packet-manifest.json").read_text())[
                "members"
            ]
            if item["portable_path"] == f"derivatives/A/{source_name}.jsonl.gz"
        )
        if digest(packed) != record["archived_sha256"]:
            raise ValueError(f"compressed_source_identity:{source_name}")
        raw = gzip.decompress(packed)
        if digest(raw) != record["expected_sha256"]:
            raise ValueError(f"decompressed_source_identity:{source_name}")
        rows = raw.splitlines()
        source_lengths.append(len(rows))
        declared_mode = registration["replays"][source_id]["value_target_mode"]
        for row_no, line in enumerate(rows, 1):
            row = json.loads(line)
            effective_mode = row.get("value_target_mode", declared_mode)
            if effective_mode != declared_mode:
                raise ValueError(f"effective_mode:{source_name}:{row_no}")
            state = np.asarray(row["state"], dtype=np.float32)
            policy = np.asarray(row["policy"], dtype=np.float32)
            value = np.float32(row["value"])
            policy_mode = str(
                row.get(
                    "policy_target_actual_mode",
                    row.get("policy_target_mode", "sharpened"),
                )
            )
            registered_value_mode = str(
                registration["replays"][source_id]["value_target_mode"]
            )
            declared_value_mode = row.get("value_target_mode")
            effective_value_mode = (
                str(declared_value_mode)
                if declared_value_mode is not None
                else registered_value_mode
            )
            if effective_value_mode != registered_value_mode:
                raise ValueError(f"effective_value_mode:{source_name}:{row_no}")
            if state.shape != (27,) or policy.shape != (6,):
                raise ValueError(f"input_shape:{source_name}:{row_no}")
            if (
                not np.isfinite(state).all()
                or not np.isfinite(policy).all()
                or not np.isfinite(value)
                or not -1 <= float(value) <= 1
                or np.any(policy < 0)
                or not np.isclose(
                    float(policy.astype(np.float64).sum()), 1.0, atol=1e-6
                )
            ):
                raise ValueError(f"target_or_state_invalid:{source_name}:{row_no}")
            player = int(round(float(state[14])))
            if player not in (0, 1) or abs(float(state[14]) - player) > 1e-6:
                raise ValueError(f"state_player:{source_name}:{row_no}")
            legal_pits = state[6:12] if player else state[:6]
            legal_moves = np.flatnonzero(np.rint(legal_pits * 48).astype(np.int64) > 0)
            if not len(legal_moves) or np.any(
                policy[np.setdiff1d(np.arange(6), legal_moves)] > 1e-6
            ):
                raise ValueError(f"illegal_policy:{source_name}:{row_no}")
            if policy_mode == "exact_root_one_hot" and (
                not np.isclose(float(policy.max()), 1.0, atol=1e-6)
                or np.count_nonzero(policy > 1e-6) != 1
            ):
                raise ValueError(f"exact_root_policy:{source_name}:{row_no}")
            if row.get("teacher_source") == "exact_root_tablebase":
                active = int(row["active_pit_stones"])
                simulations = int(row["puct_simulations_executed"])
                margins = {
                    int(move): int(margin)
                    for move, margin in row["exact_action_margins"].items()
                }
                optimal = {int(move) for move in row["exact_optimal_actions"]}
                if (
                    not 0 <= active <= 16
                    or simulations != 0
                    or not optimal
                    or optimal
                    != {
                        move
                        for move, margin in margins.items()
                        if margin == max(margins.values())
                    }
                    or any(move not in legal_moves for move in optimal)
                ):
                    raise ValueError(f"exact_root_provenance:{source_name}:{row_no}")
            compact.append(
                (
                    state,
                    policy,
                    float(value),
                    source_id,
                    row_no,
                    {
                        "sharpened": 0,
                        "exact_root_one_hot": 1,
                        "exact_root_optimal_set_uniform": 2,
                    }[policy_mode],
                    {"default": 0, "sharpened": 1}[effective_value_mode],
                )
            )
    expected_x = np.stack([row[0] for row in compact])
    expected_p = np.stack([row[1] for row in compact])
    expected_v = np.asarray([[row[2]] for row in compact], dtype=np.float32)
    expected_coeff = np.ones(len(compact), dtype=np.float32)
    expanded = np.concatenate(
        [
            np.tile(
                np.arange(
                    sum(source_lengths[:i]),
                    sum(source_lengths[: i + 1]),
                    dtype=np.int64,
                ),
                weight,
            )
            for i, weight in enumerate(WEIGHTS)
        ]
    )
    split_spec = registration["training"]["source_row_split"]
    split_file = packet / split_spec["path"]
    if digest(split_file.read_bytes()) != split_spec["sha256"]:
        raise ValueError("split_source_hash")
    frozen = json.loads(gzip.decompress(split_file.read_bytes()))
    train_expected = np.asarray(frozen["train_positions"], dtype=np.int64)
    validation_expected = np.asarray(frozen["validation_positions"], dtype=np.int64)
    if (
        digest(train_expected.astype("<i8", copy=False).tobytes())
        != split_spec["train_positions_sha256"]
    ):
        raise ValueError("train_positions_decoded_identity")
    if (
        digest(validation_expected.astype("<i8", copy=False).tobytes())
        != split_spec["validation_positions_sha256"]
    ):
        raise ValueError("validation_positions_decoded_identity")
    permutation_spec = registration["training"]["epoch_permutations"]
    permutation_file = packet / permutation_spec["path"]
    if digest(permutation_file.read_bytes()) != permutation_spec["sha256"]:
        raise ValueError("permutation_source_hash")
    all_permutations = json.loads(gzip.decompress(permutation_file.read_bytes()))
    permutation_full = np.asarray(all_permutations[0], dtype=np.int64)
    if (
        digest(json.dumps(all_permutations[0]).encode())
        != permutation_spec["epoch_sha256"]["1"]
    ):
        raise ValueError("epoch_permutation_decoded_identity")
    permutation = permutation_full[:8192]
    expected = {
        "inputs": expected_x,
        "policy_targets": expected_p,
        "value_targets": expected_v,
        "policy_coefficients": expected_coeff,
        "expanded_replay_indices": expanded,
        "train_positions": train_expected,
        "validation_positions": validation_expected,
        "first16_minibatch_positions": np.asarray(permutation, dtype=np.int64),
        "source_ids": np.asarray([row[3] for row in compact], dtype=np.uint8),
        "raw_lines": np.asarray([row[4] for row in compact], dtype=np.int32),
        "fresh_source_mask": np.asarray(
            [row[3] == 0 for row in compact], dtype=np.bool_
        ),
        "multiplicities": np.asarray(
            [WEIGHTS[row[3]] for row in compact], dtype=np.uint8
        ),
        "compact_rows": np.arange(len(compact), dtype=np.int32),
        "eligible_mask": np.ones(len(compact), dtype=np.bool_),
        "effective_value_mode_ids": np.asarray(
            [row[6] for row in compact], dtype=np.uint8
        ),
        "policy_target_mode_ids": np.asarray(
            [row[5] for row in compact], dtype=np.uint8
        ),
    }
    # Masks are independently derived from the encoded pit counts and player bit.
    masks = np.zeros((len(compact), 6), dtype=np.uint8)
    for index, state in enumerate(expected_x):
        pits = state[6:12] if round(float(state[14])) else state[:6]
        masks[index, np.flatnonzero(np.rint(pits * 48).astype(np.int64) > 0)] = 1
    expected["legal_masks"] = masks
    membership_path = (
        packet
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    membership = [
        json.loads(line) for line in gzip.open(membership_path, "rt", encoding="utf-8")
    ]
    selected = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    canonical: dict[str, list[dict[str, object]]] = {}
    evaluation_identity: list[str] = []
    evaluation_sources: list[int] = []
    evaluation_lines: list[int] = []
    evaluation_compact: list[int] = []
    evaluation_policy: list[np.ndarray] = []
    evaluation_value: list[float] = []
    identity_groups: dict[str, int] = {}
    group_ids: list[int] = []
    for occurrence in selected:
        source = str(occurrence["source"])
        if source not in canonical:
            source_path = (
                packet
                / f"docs/data/seed426-canonical-overlap/sources/{source}.jsonl.gz"
            )
            source_spec = next(
                item for item in registration["replays"] if item["name"] == source
            )
            if (
                digest(gzip.decompress(source_path.read_bytes()))
                != source_spec["sha256"]
            ):
                raise ValueError(f"canonical_source_decoded_identity:{source}")
            canonical[source] = [
                json.loads(line)
                for line in gzip.open(source_path, "rt", encoding="utf-8")
            ]
        raw_line = int(occurrence["raw_line"])
        record = canonical[source][raw_line - 1]
        identity = str(occurrence["input_identity"])
        identity_groups.setdefault(identity, len(identity_groups))
        evaluation_identity.append(identity)
        evaluation_sources.append(SOURCES.index(source))
        evaluation_lines.append(raw_line)
        evaluation_compact.append(int(occurrence["compact_row"]))
        evaluation_policy.append(np.asarray(record["policy"], dtype=np.float32))
        evaluation_value.append(float(str(record["value"])))
        group_ids.append(identity_groups[identity])
    expected_evaluation = {
        "input_identity": np.asarray(evaluation_identity, dtype="U256"),
        "source_ids": np.asarray(evaluation_sources, dtype=np.uint8),
        "raw_lines": np.asarray(evaluation_lines, dtype=np.int32),
        "compact_rows": np.asarray(evaluation_compact, dtype=np.int32),
        "policy_targets": np.asarray(evaluation_policy, dtype=np.float32),
        "value_targets": np.asarray(evaluation_value, dtype=np.float32),
        "identity_group_ids": np.asarray(group_ids, dtype=np.int32),
    }
    for filename, key in (
        ("training.npz", "training"),
        ("evaluation.npz", "evaluation"),
    ):
        raw_file = (bundle / filename).read_bytes()
        if digest(raw_file) != manifest["files"][filename]:
            raise ValueError(f"routine_file_hash:{filename}")
        with np.load(bundle / filename, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        records = manifest["arrays"][key]
        for name, array in arrays.items():
            if digest(array.tobytes(order="C")) != records[name]["sha256"]:
                raise ValueError(f"routine_array_hash:{key}:{name}")
            if key == "training":
                if name not in expected or not np.array_equal(array, expected[name]):
                    raise ValueError(f"independent_reconstruction_mismatch:{name}")
            elif name not in expected_evaluation or not np.array_equal(
                array, expected_evaluation[name]
            ):
                raise ValueError(f"independent_evaluation_mismatch:{name}")
        if arrays.keys() != records.keys():
            raise ValueError(f"array_inventory_mismatch:{key}")
        if list(arrays) != manifest["array_order"][key]:
            raise ValueError(f"array_order_mismatch:{key}")
    train = expected["train_positions"]
    fresh = expected["fresh_source_mask"][expanded[train]]
    if len(train) != 134502 or int(fresh.sum()) != 60015:
        raise ValueError("training_count_reconciliation")
    if len(expected["validation_positions"]) != 14946:
        raise ValueError("validation_count_reconciliation")
    count = manifest["counts"]
    if (
        count["compact_rows"] != 87625
        or count["expanded_occurrences_including_validation"] != 149448
        or count["expanded_training"] != 134502
        or count["fresh_expanded"] != 60015
        or count["validation"] != 14946
    ):
        raise ValueError("training_count_reconciliation")
    if count["evaluation_pooled_occurrences"] != len(selected) or count[
        "evaluation_pooled_identities"
    ] != len(identity_groups):
        raise ValueError("evaluation_count_reconciliation")
    if count["evaluation_fresh_occurrences"] != sum(
        source == "fresh" for source in (row["source"] for row in selected)
    ):
        raise ValueError("fresh_evaluation_count_reconciliation")
    masses = manifest["policy_masses"]
    compact_train = expanded[train]
    source_train = expected["source_ids"][compact_train]
    expected_source_counts = {
        source: int(np.count_nonzero(source_train == index))
        for index, source in enumerate(SOURCES)
    }
    expected_source_masses = dict(expected_source_counts)
    if masses != {
        "training_all_sources": 134502.0,
        "training_fresh": 60015.0,
        "training_by_source": expected_source_masses,
    }:
        raise ValueError("policy_mass_reconciliation")
    if manifest["training_occurrences_by_source"] != expected_source_counts:
        raise ValueError("source_occurrence_reconciliation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    verify(args.bundle, args.root)
    print("seed466 independent reconstruction passed")


if __name__ == "__main__":
    main()
