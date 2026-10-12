"""Within-source canonical-multiplicity policy coefficient normalization."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from typing import Any
from pathlib import Path

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.kalah_rules import KalahGame


def normalize_policy_coefficients(
    coefficients: np.ndarray,
    identities: list[str],
    active_stones: np.ndarray,
    sources: list[str],
    training_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Equalize positive policy mass per canonical identity per source/stone bucket.

    Input and output are compact source-row vectors. The returned coefficients are
    float32, matching the production policy-loss input representation. Replay-copy
    multiplicity is intentionally not expanded here.
    """
    q = np.asarray(coefficients)
    stones = np.asarray(active_stones)
    training = np.asarray(training_mask)
    n = len(identities)
    if q.dtype.kind not in "fi" or q.shape != (n,) or len(sources) != n:
        raise ValueError("coefficient_row_shape_mismatch")
    if stones.shape != (n,) or training.shape != (n,):
        raise ValueError("row_metadata_shape_mismatch")
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("malformed_policy_coefficients")
    if not np.isin(training, [False, True]).all():
        raise ValueError("malformed_training_mask")
    if not np.isfinite(stones).all() or (stones < 0).any():
        raise ValueError("malformed_active_stone_counts")

    original = np.asarray(q, dtype=np.float32)
    normalized = original.copy()
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in range(n):
        if not training[row] or stones[row] <= 16:
            continue
        bucket = "17-32" if stones[row] <= 32 else ">32"
        buckets[(sources[row], bucket)].append(row)

    census: dict[str, Any] = {}
    for (source, bucket), rows in sorted(buckets.items()):
        by_identity: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            if original[row] > 0:
                by_identity[identities[row]].append(row)
        group_mass = {
            identity: float(original[indexes].astype(np.float64).sum())
            for identity, indexes in by_identity.items()
        }
        multiplicities = [len(indexes) for indexes in by_identity.values()]
        total = float(original[rows].astype(np.float64).sum())
        positive_groups = len(group_mass)
        if positive_groups:
            for identity, indexes in by_identity.items():
                mass = group_mass[identity]
                scale = total / (positive_groups * mass)
                normalized[indexes] = np.asarray(
                    original[indexes].astype(np.float64) * scale,
                    dtype=np.float32,
                )
        output_mass = float(normalized[rows].astype(np.float64).sum())
        census[f"{source}/{bucket}"] = {
            "training_rows": len(rows),
            "positive_rows": int(np.count_nonzero(original[rows])),
            "positive_identities": positive_groups,
            "positive_identity_rows_repeated_beyond_first": int(
                sum(max(0, count - 1) for count in multiplicities)
            ),
            "identities_with_multiple_compact_rows": int(
                sum(count > 1 for count in multiplicities)
            ),
            "maximum_positive_compact_row_multiplicity": max(multiplicities, default=0),
            "zero_mass_identities": len(
                {identities[i] for i in rows} - set(by_identity)
            ),
            "original_mass": total,
            "normalized_mass": output_mass,
            "original_coefficient_range": [
                float(original[rows].min()),
                float(original[rows].max()),
            ],
            "normalized_coefficient_range": [
                float(normalized[rows].min()),
                float(normalized[rows].max()),
            ],
            "effective_concentration": (
                sum(value * value for value in group_mass.values()) / (total * total)
                if total
                else 0.0
            ),
        }
        reconciliation = _group_reconciliation(original, normalized, rows, identities)
        census[f"{source}/{bucket}"].update(reconciliation)
        census[f"{source}/{bucket}"]["effective_concentration_after"] = reconciliation[
            "normalized_effective_concentration"
        ]
    return normalized, {
        "rows_changed": int(np.count_nonzero(normalized != original)),
        "normalization_is_noop": bool(np.array_equal(normalized, original)),
        "groups": census,
        "coefficient_mass_conservation_scope": (
            "sum of compact-row coefficients per source and active-stone bucket; "
            "not realized minibatch gradient contribution"
        ),
    }


def _group_reconciliation(
    original: np.ndarray,
    normalized: np.ndarray,
    rows: list[int],
    identities: list[str],
) -> dict[str, Any]:
    original_mass: dict[str, float] = defaultdict(float)
    normalized_mass: dict[str, float] = defaultdict(float)
    for row in rows:
        original_mass[identities[row]] += float(original[row])
        normalized_mass[identities[row]] += float(normalized[row])
    positive_ids = [key for key, value in original_mass.items() if value > 0]
    changed_ids = {identities[row] for row in rows if original[row] != normalized[row]}
    values = [normalized_mass[key] for key in positive_ids]
    positive_total = sum(values)
    return {
        "affected_rows": int(sum(original[row] != normalized[row] for row in rows)),
        "affected_identities": len(changed_ids),
        "original_positive_identity_mass_range": [
            min((original_mass[key] for key in positive_ids), default=0.0),
            max((original_mass[key] for key in positive_ids), default=0.0),
        ],
        "normalized_positive_identity_mass_range": [
            min(values, default=0.0),
            max(values, default=0.0),
        ],
        "normalized_effective_concentration": (
            sum(value * value for value in values) / (positive_total * positive_total)
            if positive_total
            else 0.0
        ),
        "mass_delta": float(
            sum(float(normalized[row]) - float(original[row]) for row in rows)
        ),
    }


def reconstruct_frozen_census(root: Path) -> dict[str, Any]:
    """Reconstruct #416 compact rows, frozen membership, and seed429 treatment."""
    from ml.alphazero_lite.train import load_jsonl_replay

    historic = root / "docs/data/seed416-policy-target-softening"
    registration_path = historic / "registration-v3.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    records = registration["replays"]
    paths = [Path(record["path"]) for record in records]
    source_weight = [int(record["weight"]) for record in records]
    for record, path in zip(records, paths, strict=True):
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"registered_replay_hash_mismatch:{record['name']}")
    x, _policy, _value, replay, coefficients = load_jsonl_replay(
        paths,
        source_weight,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[row["value_target_mode"] for row in records],
        include_policy_loss_weights=True,
    )
    split_path = root / registration["training"]["source_row_split"]["path"]
    split_identity = registration["training"]["source_row_split"]
    if sha256_file(split_path) != split_identity["sha256"]:
        raise ValueError("frozen_source_row_split_hash_mismatch")
    with gzip.open(split_path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    train_positions = np.asarray(split["train_positions"], dtype=np.int64)
    validation_positions = np.asarray(split["validation_positions"], dtype=np.int64)
    if (
        len(train_positions) != split_identity["train_count"]
        or len(validation_positions) != split_identity["validation_count"]
        or hashlib.sha256(train_positions.tobytes()).hexdigest()
        != split_identity["train_positions_sha256"]
        or hashlib.sha256(validation_positions.tobytes()).hexdigest()
        != split_identity["validation_positions_sha256"]
    ):
        raise ValueError("frozen_split_membership_identity_mismatch")
    if (
        np.any(train_positions < 0)
        or np.any(validation_positions < 0)
        or np.any(train_positions >= len(replay))
        or np.any(validation_positions >= len(replay))
        or np.intersect1d(train_positions, validation_positions).size
    ):
        raise ValueError("frozen_split_position_invalid")

    n = len(x)
    identities = [canonical_identity_from_encoded_state(row.tolist()) for row in x]
    active_stones = np.rint(x[:, :12].astype(np.float64) * 48).sum(axis=1)
    compact_sources: list[str] = []
    compact_weights: list[int] = []
    source_start = 0
    for record, path in zip(records, paths, strict=True):
        with path.open(encoding="utf-8") as stream:
            count = sum(1 for line in stream if line.strip())
        compact_sources.extend([record["name"]] * count)
        compact_weights.extend([int(record["weight"])] * count)
        source_start += count
    if len(compact_sources) != n:
        raise ValueError("compact_source_row_mapping_mismatch")
    # The v3 split is defined over expanded np.tile positions. All weighted copies
    # of each source row must resolve to a single partition before compacting.
    train_rows = set(int(value) for value in replay[train_positions])
    validation_rows = set(int(value) for value in replay[validation_positions])
    if train_rows & validation_rows or train_rows | validation_rows != set(range(n)):
        raise ValueError("frozen_split_compact_membership_not_partition")
    training_mask = np.zeros(n, dtype=bool)
    training_mask[list(train_rows)] = True
    normalized, summary = normalize_policy_coefficients(
        coefficients, identities, active_stones, compact_sources, training_mask
    )
    for source_index, record in enumerate(records):
        source_rows = [
            i for i, name in enumerate(compact_sources) if name == record["name"]
        ]
        positions = np.isin(replay, source_rows)
        expanded_train = int(
            np.count_nonzero(np.isin(train_positions, np.flatnonzero(positions)))
        )
        expanded_validation = int(
            np.count_nonzero(np.isin(validation_positions, np.flatnonzero(positions)))
        )
        record["compact_rows"] = len(source_rows)
        record["expanded_replay_positions"] = int(np.count_nonzero(positions))
        record["expanded_training_positions"] = expanded_train
        record["expanded_validation_positions"] = expanded_validation
        record["replay_weight"] = source_weight[source_index]
    summary.update(
        {
            "schema": "seed429-compact-coefficient-census-v1",
            "classification": (
                "normalization is a no-op"
                if summary["normalization_is_noop"]
                else "normalization changes float32 coefficients"
            ),
            "registered_seed416_registration_sha256": sha256_file(registration_path),
            "replay_sources": records,
            "source_row_count": n,
            "expanded_replay_position_count": len(replay),
            "training_compact_row_count": len(train_rows),
            "validation_compact_row_count": len(validation_rows),
            "training_expanded_position_count": len(train_positions),
            "validation_expanded_position_count": len(validation_positions),
            "validation_coefficients_unchanged": bool(
                np.array_equal(
                    normalized[list(validation_rows)],
                    coefficients[list(validation_rows)],
                )
            ),
            "low_stone_coefficients_unchanged": bool(
                np.array_equal(
                    normalized[active_stones <= 16], coefficients[active_stones <= 16]
                )
            ),
            "control_coefficients_sha256": hashlib.sha256(
                np.asarray(coefficients, dtype="<f4").tobytes()
            ).hexdigest(),
            "treatment_coefficients_sha256": hashlib.sha256(
                np.asarray(normalized, dtype="<f4").tobytes()
            ).hexdigest(),
            "source_weights_are_copy_multiplicity_not_new_compact_rows": True,
            "expanded_replay_copy_counts_are_separately_recorded": True,
            "target_arrays_preserved_by_construction": True,
        }
    )
    return summary


def sha256_file(path: Path) -> str:
    """Hash a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_identity_from_encoded_state(encoded: list[float]) -> str:
    """Return Kalah's player-relative canonical identity for a loaded row."""
    if len(encoded) < 15 or not np.isfinite(np.asarray(encoded[:15])).all():
        raise ValueError("malformed_encoded_state")
    state = {
        "player_pits": [int(round(float(value) * 48)) for value in encoded[:6]],
        "opponent_pits": [int(round(float(value) * 48)) for value in encoded[6:12]],
        "player_store": int(round(float(encoded[12]) * 48)),
        "opponent_store": int(round(float(encoded[13]) * 48)),
        "current_player": int(round(float(encoded[14]))),
    }
    return suites.canonical_key(KalahGame.from_state(state).to_state())
