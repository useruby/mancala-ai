"""Build and read the portable seed466 replay-input bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path("docs/data/seed464-dependency-closure-audit/packet")
OUT = Path("docs/data/seed466-replay-input-bundle")
SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
WEIGHTS = (1, 4, 1, 8, 4)
EXPECTED_MANIFEST_ID = "seed466-five-source-lane-A-v1"
PACKET_MANIFEST_SHA256 = (
    "d4ea022be45e6e6e259d163b8ef2deec1ac06761dec09f6f6ce4463fa507ec03"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]


def _verify_pinned_inputs(packet: Path) -> None:
    manifest_bytes = (packet / "packet-manifest.json").read_bytes()
    if sha(manifest_bytes) != PACKET_MANIFEST_SHA256:
        raise ValueError("seed464_packet_manifest_identity_mismatch")
    packet_manifest = json.loads(manifest_bytes)
    paths = [
        *(f"derivatives/A/{name}.jsonl.gz" for name in SOURCES),
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        *(
            f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            for name in SOURCES
        ),
    ]
    for relative in paths:
        member = next(
            (
                item
                for item in packet_manifest["members"]
                if item["portable_path"] == relative
            ),
            None,
        )
        if member is None:
            raise ValueError(f"packet_member_missing:{relative}")
        stored = (packet / relative).read_bytes()
        if sha(stored) != member["archived_sha256"]:
            raise ValueError(f"packet_member_compressed_identity:{relative}")
        decoded = gzip.decompress(stored) if relative.endswith(".gz") else stored
        if (
            relative.startswith("derivatives/")
            and sha(decoded) != member["expected_sha256"]
        ):
            raise ValueError(f"packet_member_decoded_identity:{relative}")


def _read_inputs(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    packet = root / PACKET
    _verify_pinned_inputs(packet)
    registration = json.loads(
        (
            packet / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    if [item["name"] for item in registration["replays"]] != list(SOURCES):
        raise ValueError("source_order_mismatch")
    if [int(item["weight"]) for item in registration["replays"]] != list(WEIGHTS):
        raise ValueError("source_weight_mismatch")
    compact_rows: list[dict[str, Any]] = []
    source_ids: list[int] = []
    multiplicities: list[int] = []
    coefficients: list[float] = []
    raw_lines: list[int] = []
    effective_values: list[str] = []
    effective_policies: list[str] = []
    xs: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    base = 0
    for source_id, (name, weight) in enumerate(zip(SOURCES, WEIGHTS, strict=True)):
        member = packet / f"derivatives/A/{name}.jsonl.gz"
        compressed = member.read_bytes()
        entry = next(
            m
            for m in json.loads((packet / "packet-manifest.json").read_text())[
                "members"
            ]
            if m["portable_path"] == f"derivatives/A/{name}.jsonl.gz"
        )
        if sha(compressed) != entry["archived_sha256"]:
            raise ValueError(f"compressed_identity_mismatch:{name}")
        raw = gzip.decompress(compressed)
        if sha(raw) != entry["expected_sha256"]:
            raise ValueError(f"decompressed_identity_mismatch:{name}")
        rows = [json.loads(line) for line in raw.splitlines()]
        spec = registration["replays"][source_id]
        mode = spec["value_target_mode"]
        for row_number, row in enumerate(rows, 1):
            declared = row.get("value_target_mode")
            effective = declared if declared is not None else mode
            if effective != mode:
                raise ValueError(f"value_mode_mismatch:{name}:{row_number}")
            x = np.asarray(row["state"], dtype=np.float32)
            p = np.asarray(row["policy"], dtype=np.float32)
            v = np.asarray([row["value"]], dtype=np.float32)
            if x.shape != (27,) or p.shape != (6,):
                raise ValueError(f"row_shape:{name}:{row_number}")
            if (
                not np.isfinite(x).all()
                or not np.isfinite(p).all()
                or not np.isfinite(v).all()
            ):
                raise ValueError(f"row_nonfinite:{name}:{row_number}")
            if np.any(p < 0) or not np.isclose(
                float(p.astype(np.float64).sum()), 1.0, atol=1e-6
            ):
                raise ValueError(f"policy_target_invalid:{name}:{row_number}")
            legal = np.zeros(6, dtype=np.uint8)
            player = int(round(float(x[14])))
            if player not in (0, 1) or abs(float(x[14]) - player) > 1e-6:
                raise ValueError(f"state_player_invalid:{name}:{row_number}")
            pits = x[6:12] if player else x[0:6]
            legal[np.flatnonzero(np.rint(pits * 48).astype(np.int64) > 0)] = 1
            if not legal.any() or np.any(p[legal == 0] > 1e-6):
                raise ValueError(f"illegal_policy_target:{name}:{row_number}")
            policy_mode = row.get(
                "policy_target_actual_mode", row.get("policy_target_mode")
            )
            if policy_mode == "exact_root_one_hot" and (
                not np.isclose(float(p.max()), 1.0, atol=1e-6)
                or np.count_nonzero(p > 1e-6) != 1
            ):
                raise ValueError(f"exact_root_policy_invalid:{name}:{row_number}")
            if row.get("teacher_source") == "exact_root_tablebase":
                active = int(row["active_pit_stones"])
                simulations = int(row["puct_simulations_executed"])
                margins = {
                    int(move): int(margin)
                    for move, margin in row["exact_action_margins"].items()
                }
                optimal = {int(move) for move in row["exact_optimal_actions"]}
                if not 0 <= active <= 16 or simulations != 0 or not optimal:
                    raise ValueError(f"exact_root_provenance:{name}:{row_number}")
                if optimal != {
                    move
                    for move, margin in margins.items()
                    if margin == max(margins.values())
                }:
                    raise ValueError(f"exact_root_optimal_set:{name}:{row_number}")
                if any(move not in np.flatnonzero(legal) for move in optimal):
                    raise ValueError(f"exact_root_illegal_action:{name}:{row_number}")
            xs.append(x)
            ps.append(p)
            vs.append(v)
            masks.append(legal)
            coefficient = 1.0
            compact_rows.append(
                {
                    "source": name,
                    "source_id": source_id,
                    "raw_line": row_number,
                    "compact_row": base,
                    "multiplicity": weight,
                    "policy_coefficient": coefficient,
                    "effective_value_mode": effective,
                }
            )
            source_ids.append(source_id)
            multiplicities.append(weight)
            coefficients.append(coefficient)
            raw_lines.append(row_number)
            effective_values.append(effective)
            effective_policies.append(
                str(
                    row.get(
                        "policy_target_actual_mode",
                        row.get("policy_target_mode", "sharpened"),
                    )
                )
            )
            base += 1
    n = len(xs)
    replay = np.concatenate(
        [
            np.tile(
                np.arange(
                    sum(
                        len(_jsonl_gzip(packet / f"derivatives/A/{s}.jsonl.gz"))
                        for s in SOURCES[:i]
                    ),
                    sum(
                        len(_jsonl_gzip(packet / f"derivatives/A/{s}.jsonl.gz"))
                        for s in SOURCES[: i + 1]
                    ),
                    dtype=np.int64,
                ),
                w,
            )
            for i, w in enumerate(WEIGHTS)
        ]
    )
    split_path = packet / registration["training"]["source_row_split"]["path"]
    if (
        sha(split_path.read_bytes())
        != registration["training"]["source_row_split"]["sha256"]
    ):
        raise ValueError("split_compressed_hash_mismatch")
    split = json.loads(gzip.decompress(split_path.read_bytes()))
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    val_positions = np.asarray(split["validation_positions"], dtype=np.int64)
    if (
        sha(train_positions.astype("<i8", copy=False).tobytes())
        != registration["training"]["source_row_split"]["train_positions_sha256"]
    ):
        raise ValueError("train_positions_decompressed_identity_mismatch")
    if (
        sha(val_positions.astype("<i8", copy=False).tobytes())
        != registration["training"]["source_row_split"]["validation_positions_sha256"]
    ):
        raise ValueError("validation_positions_decompressed_identity_mismatch")
    perm_path = packet / registration["training"]["epoch_permutations"]["path"]
    if (
        sha(perm_path.read_bytes())
        != registration["training"]["epoch_permutations"]["sha256"]
    ):
        raise ValueError("permutation_hash_mismatch")
    permutations = json.loads(gzip.decompress(perm_path.read_bytes()))
    if (
        sha(json.dumps(permutations[0]).encode())
        != registration["training"]["epoch_permutations"]["epoch_sha256"]["1"]
    ):
        raise ValueError("epoch_permutation_decompressed_identity_mismatch")
    first16 = np.asarray(permutations[0][:8192], dtype=np.int64)
    arrays = {
        "inputs": np.asarray(xs, dtype=np.float32),
        "legal_masks": np.asarray(masks, dtype=np.uint8),
        "policy_targets": np.asarray(ps, dtype=np.float32),
        "value_targets": np.asarray(vs, dtype=np.float32),
        "policy_coefficients": np.asarray(coefficients, dtype=np.float32),
        "expanded_replay_indices": replay,
        "train_positions": train_positions,
        "validation_positions": val_positions,
        "first16_minibatch_positions": first16,
        "fresh_source_mask": np.asarray([s == 0 for s in source_ids], dtype=np.bool_),
        "source_ids": np.asarray(source_ids, dtype=np.uint8),
        "raw_lines": np.asarray(raw_lines, dtype=np.int32),
        "multiplicities": np.asarray(multiplicities, dtype=np.uint8),
        "compact_rows": np.arange(n, dtype=np.int32),
        "eligible_mask": np.ones(n, dtype=np.bool_),
        "effective_value_mode_ids": np.asarray(
            [0 if mode == "default" else 1 for mode in effective_values], dtype=np.uint8
        ),
        "policy_target_mode_ids": np.asarray(
            [
                {
                    "sharpened": 0,
                    "exact_root_one_hot": 1,
                    "exact_root_optimal_set_uniform": 2,
                }[mode]
                for mode in effective_policies
            ],
            dtype=np.uint8,
        ),
    }
    if n != 87625 or len(replay) != 149448 or len(train_positions) != 134502:
        raise ValueError("frozen_count_mismatch")
    return arrays, {
        "compact_rows": compact_rows,
        "registration": registration,
        "membership_path": packet
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
    }


def _evaluation(root: Path, detail: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = _jsonl_gzip(detail["membership_path"])
    packet = root / PACKET
    cache: dict[str, list[dict[str, Any]]] = {}
    selected = [
        row for row in rows if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    identities: list[str] = []
    sources: list[int] = []
    lines: list[int] = []
    compact: list[int] = []
    policy: list[np.ndarray] = []
    value: list[float] = []
    grouping: list[int] = []
    identity_ids: dict[str, int] = {}
    for row in selected:
        name = row["source"]
        if name not in cache:
            source_path = (
                packet / f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            )
            source_spec = next(
                item
                for item in detail["registration"]["replays"]
                if item["name"] == name
            )
            if sha(gzip.decompress(source_path.read_bytes())) != source_spec["sha256"]:
                raise ValueError(
                    f"canonical_source_decompressed_identity_mismatch:{name}"
                )
            cache[name] = _jsonl_gzip(source_path)
        source_row = cache[name][int(row["raw_line"]) - 1]
        ident = row["input_identity"]
        identity_ids.setdefault(ident, len(identity_ids))
        identities.append(ident)
        sources.append(SOURCES.index(name))
        lines.append(int(row["raw_line"]))
        compact.append(int(row["compact_row"]))
        policy.append(np.asarray(source_row["policy"], dtype=np.float32))
        value.append(float(source_row["value"]))
        grouping.append(identity_ids[ident])
    return {
        "input_identity": np.asarray(identities, dtype="U256"),
        "source_ids": np.asarray(sources, dtype=np.uint8),
        "raw_lines": np.asarray(lines, dtype=np.int32),
        "compact_rows": np.asarray(compact, dtype=np.int32),
        "policy_targets": np.asarray(policy, dtype=np.float32),
        "value_targets": np.asarray(value, dtype=np.float32),
        "identity_group_ids": np.asarray(grouping, dtype=np.int32),
    }


def build(root: Path = ROOT, out: Path | None = None) -> Path:
    destination = out or root / OUT
    if destination.exists():
        raise ValueError("publication_exists")
    arrays, detail = _read_inputs(root)
    evaluation = _evaluation(root, detail)
    destination.mkdir(parents=True)
    np.savez(destination / "training.npz", **dict(arrays))
    np.savez(destination / "evaluation.npz", **dict(evaluation))
    sources = [
        "ml/alphazero_lite/seed466_replay_bundle.py",
        "ml/alphazero_lite/verify_seed466_replay_bundle.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
        "ml/alphazero_lite/exact_root_policy_targets.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/seed432_policy_target_census.py",
        "ml/alphazero_lite/seed433_census_correction.py",
        "ml/alphazero_lite/build_opening_suite.py",
        "ml/alphazero_lite/kalah_rules.py",
    ]
    consumed_paths = [
        "packet-manifest.json",
        *(f"derivatives/A/{name}.jsonl.gz" for name in SOURCES),
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        detail["registration"]["training"]["source_row_split"]["path"],
        detail["registration"]["training"]["epoch_permutations"]["path"],
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        *(
            f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            for name in SOURCES
        ),
    ]
    source_inputs = {}
    for relative in consumed_paths:
        raw = (root / PACKET / relative).read_bytes()
        record: dict[str, str] = {"compressed_or_stored_sha256": sha(raw)}
        if relative.endswith(".gz"):
            decoded = gzip.decompress(raw)
            record["decompressed_sha256"] = sha(decoded)
        source_inputs[relative] = record
    files = {
        name: sha((destination / name).read_bytes())
        for name in ("training.npz", "evaluation.npz")
    }
    training_fresh = arrays["fresh_source_mask"][
        arrays["expanded_replay_indices"][arrays["train_positions"]]
    ]
    compact_training = arrays["expanded_replay_indices"][arrays["train_positions"]]
    training_source_ids = arrays["source_ids"][compact_training]
    per_source_counts = {
        source: int(np.count_nonzero(training_source_ids == index))
        for index, source in enumerate(SOURCES)
    }
    per_source_masses = {
        source: float(
            arrays["policy_coefficients"][
                compact_training[training_source_ids == index]
            ].sum(dtype=np.float64)
        )
        for index, source in enumerate(SOURCES)
    }
    manifest = {
        "identity": EXPECTED_MANIFEST_ID,
        "historical_status_note": "seed464 remains classified invalid_attempt_dependency_closure_incomplete; this separately constructed bundle does not upgrade its execution evidence",
        "files": files,
        "source_inputs": source_inputs,
        "arrays": {
            "training": {
                key: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": sha(value.tobytes(order="C")),
                }
                for key, value in arrays.items()
            },
            "evaluation": {
                key: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": sha(value.tobytes(order="C")),
                }
                for key, value in evaluation.items()
            },
        },
        "source_closure": {name: sha((root / name).read_bytes()) for name in sources},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation": sys.implementation.name,
        },
        "recipe": {
            "sources": list(SOURCES),
            "source_id_codes": {source: index for index, source in enumerate(SOURCES)},
            "weights": list(WEIGHTS),
            "policy_target_mode": "sharpened",
            "policy_target_mode_precedence": "policy_target_actual_mode, then policy_target_mode; checked against the frozen sharpened request",
            "value_target_mode_precedence": "per-source registered mode is passed to the loader; any row declaration must match it; missing declaration uses the registered mode",
            "target_mode_codes": {
                "policy": {
                    "sharpened": 0,
                    "exact_root_one_hot": 1,
                    "exact_root_optimal_set_uniform": 2,
                },
                "value": {"default": 0, "sharpened": 1},
            },
            "multiplicity": "tile compact indices source-by-source; source weight from registration",
            "input_encoding": "kalah_v3; 27 float32 features",
            "eligibility": "all derivative rows; no bucket exclusions",
            "split": "unchanged seed416 lane-A frozen source-row split",
            "permutation": "unchanged seed416 epoch-1 permutation; first 8192 positions",
        },
        "counts": {
            "compact_rows": len(arrays["inputs"]),
            "expanded_occurrences_including_validation": len(
                arrays["expanded_replay_indices"]
            ),
            "expanded_training": len(arrays["train_positions"]),
            "fresh_expanded": int(training_fresh.sum()),
            "validation": len(arrays["validation_positions"]),
            "evaluation_pooled_occurrences": len(evaluation["value_targets"]),
            "evaluation_pooled_identities": len(
                set(evaluation["input_identity"].tolist())
            ),
            "evaluation_fresh_occurrences": int(
                np.count_nonzero(evaluation["source_ids"] == 0)
            ),
            "evaluation_fresh_identities": len(
                set(
                    evaluation["input_identity"][evaluation["source_ids"] == 0].tolist()
                )
            ),
        },
        "policy_masses": {
            "training_all_sources": float(
                arrays["policy_coefficients"][
                    arrays["expanded_replay_indices"][arrays["train_positions"]]
                ].sum(dtype=np.float64)
            ),
            "training_fresh": float(
                arrays["policy_coefficients"][
                    arrays["expanded_replay_indices"][arrays["train_positions"]]
                ][training_fresh].sum(dtype=np.float64)
            ),
            "training_by_source": per_source_masses,
        },
        "training_occurrences_by_source": per_source_counts,
        "array_order": {
            "training": list(arrays),
            "evaluation": list(evaluation),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return destination


def read_bundle(
    path: Path, expected_identity: str = EXPECTED_MANIFEST_ID
) -> dict[str, dict[str, np.ndarray]]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("identity") != expected_identity:
        raise ValueError("manifest_identity_mismatch")
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in ("training", "evaluation"):
        file_name = f"{name}.npz"
        raw = (path / file_name).read_bytes()
        if sha(raw) != manifest["files"][file_name]:
            raise ValueError(f"member_hash_mismatch:{file_name}")
        with np.load(path / file_name, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        expected_arrays = manifest["arrays"][name]
        if arrays.keys() != expected_arrays.keys():
            raise ValueError(f"array_members_mismatch:{name}")
        if list(arrays) != manifest["array_order"][name]:
            raise ValueError(f"array_order_mismatch:{name}")
        for key, value in arrays.items():
            record = expected_arrays[key]
            if (
                str(value.dtype) != record["dtype"]
                or list(value.shape) != record["shape"]
            ):
                raise ValueError(f"array_schema_mismatch:{name}:{key}")
            if sha(value.tobytes(order="C")) != record["sha256"]:
                raise ValueError(f"array_hash_mismatch:{name}:{key}")
        result[name] = arrays
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "read"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        print(build(args.root, args.bundle))
    else:
        print(
            {
                k: {n: list(a.shape) for n, a in v.items()}
                for k, v in read_bundle(args.bundle or args.root / OUT).items()
            }
        )


if __name__ == "__main__":
    main()
