"""Portable source-to-publication proof for the seed432/433 census."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ml.alphazero_lite.exact_root_policy_targets import validate_exact_root_metadata
from ml.alphazero_lite.seed429_policy_normalization import (
    canonical_identity_from_encoded_state,
    normalize_policy_coefficients,
)
from ml.alphazero_lite.seed432_policy_target_census import summarize
from ml.alphazero_lite.seed433_census_correction import calculate, sha

SOURCES = (
    "fresh",
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
WEIGHTS = (1, 4, 1, 8, 4)
FEATURES = 27


def _float32_bytes(values: list[object]) -> bytes:
    return struct.pack("<" + "f" * len(values), *(float(value) for value in values))


def _decode_state(
    encoded: list[object], source: str, line: int
) -> tuple[int, str, list[int]]:
    values = np.asarray(encoded, dtype=np.float32)
    if len(encoded) != FEATURES or not np.isfinite(values).all():
        raise ValueError(f"state_shape_or_finiteness:{source}:{line}")

    def stones(start: int, count: int) -> list[int]:
        result = []
        for item in values[start : start + count]:
            scaled = float(item) * 48
            rounded = round(scaled)
            if abs(scaled - rounded) > 1e-6 or rounded < 0:
                raise ValueError(f"state_decode:{source}:{line}")
            result.append(int(rounded))
        return result

    player = round(float(values[14]))
    if player not in (0, 1) or abs(float(values[14]) - player) > 1e-6:
        raise ValueError(f"state_player:{source}:{line}")
    state = {
        "player_pits": stones(0, 6),
        "opponent_pits": stones(6, 6),
        "player_store": stones(12, 1)[0],
        "opponent_store": stones(13, 1)[0],
        "current_player": player,
    }
    acting_pits = state["player_pits"] if player == 0 else state["opponent_pits"]
    legal_moves = [i for i, count in enumerate(acting_pits) if count > 0]
    return (
        sum(state["player_pits"] + state["opponent_pits"]),
        canonical_identity_from_encoded_state(values.tolist()),
        legal_moves,
    )


def reconstruct_sources(
    root: Path, registration: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Decode registered snapshots as the frozen loader does, without sharpening targets."""
    specs = registration["replays"]
    if [s["name"] for s in specs] != list(SOURCES) or [
        s["weight"] for s in specs
    ] != list(WEIGHTS):
        raise ValueError("registered_source_order_or_weights_mismatch")
    compact = 0
    for spec in specs:
        path = (
            root
            / "docs/data/seed426-canonical-overlap/sources"
            / f"{spec['name']}.jsonl.gz"
        )
        digest = hashlib.sha256()
        line_number = 0
        with gzip.open(path, "rb") as stream:
            for raw in stream:
                digest.update(raw)
                line_number += 1
                if not raw.endswith(b"\n"):
                    raise ValueError(
                        f"source_line_terminator:{spec['name']}:{line_number}"
                    )
                row = json.loads(raw)
                declared = row.get(
                    "policy_target_actual_mode", row.get("policy_target_mode")
                )
                if declared not in {
                    "sharpened",
                    "exact_root_one_hot",
                    "exact_root_optimal_set_uniform",
                }:
                    raise ValueError(f"policy_mode:{spec['name']}:{line_number}")
                value_mode = row.get("value_target_mode")
                if value_mode is None:
                    if spec["value_target_mode"] != "default":
                        raise ValueError(
                            f"value_mode_missing:{spec['name']}:{line_number}"
                        )
                elif value_mode != spec["value_target_mode"]:
                    raise ValueError(f"value_mode:{spec['name']}:{line_number}")
                target = np.asarray(row["policy"], dtype=np.float32)
                if (
                    target.shape != (6,)
                    or not np.isfinite(target).all()
                    or (target < 0).any()
                    or not np.isclose(
                        float(target.astype(np.float64).sum()), 1.0, atol=1e-6
                    )
                ):
                    raise ValueError(f"policy_target:{spec['name']}:{line_number}")
                if declared == "exact_root_one_hot" and (
                    not np.isclose(float(target.max()), 1.0, atol=1e-6)
                    or np.count_nonzero(target > 1e-6) != 1
                ):
                    raise ValueError(f"exact_root_one_hot:{spec['name']}:{line_number}")
                if declared == "exact_root_optimal_set_uniform":
                    _selected, optimal = validate_exact_root_metadata(row)
                    expected = np.zeros(6, dtype=np.float32)
                    expected[optimal] = 1.0 / len(optimal)
                    if not np.allclose(target, expected, atol=1e-6):
                        raise ValueError(
                            f"exact_root_uniform_target:{spec['name']}:{line_number}"
                        )
                stones, canonical, legal_moves = _decode_state(
                    row["state"], spec["name"], line_number
                )
                if not legal_moves:
                    raise ValueError(f"no_legal_moves:{spec['name']}:{line_number}")
                illegal = [
                    i
                    for i, probability in enumerate(target)
                    if i not in legal_moves and probability > 1e-6
                ]
                if illegal:
                    raise ValueError(
                        f"illegal_policy_moves:{spec['name']}:{line_number}"
                    )
                if (
                    not math.isfinite(float(row["value"]))
                    or not -1 <= float(row["value"]) <= 1
                ):
                    raise ValueError(f"value_target:{spec['name']}:{line_number}")
                coefficient = 1.0
                if row.get("teacher_source") == "exact_root_tablebase":
                    validate_exact_root_metadata(row)
                    active_pit_stones = int(str(row["active_pit_stones"]))
                    simulations = int(str(row["puct_simulations_executed"]))
                    if (
                        active_pit_stones < 0
                        or active_pit_stones > 16
                        or simulations != 0
                    ):
                        raise ValueError(
                            f"exact_root_provenance:{spec['name']}:{line_number}"
                        )
                input_bytes = _float32_bytes(row["state"])
                yield {
                    "source": spec["name"],
                    "line": line_number,
                    "compact_row": compact,
                    "replay_multiplicity": int(spec["weight"]),
                    "policy_coefficient": coefficient,
                    "active_stones": stones,
                    "canonical_state": canonical,
                    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                    "input_hex": input_bytes.hex(),
                    "target": target.tolist(),
                    "target_sum": float(target.astype(np.float64).sum()),
                    "exposure_weight": int(spec["weight"]) * coefficient,
                    "equal_input_weight": coefficient,
                }
                compact += 1
        if digest.hexdigest() != spec["sha256"]:
            raise ValueError(f"decompressed_source_hash_mismatch:{spec['name']}")


def _positions(
    root: Path, registration: dict[str, Any], n: int, weights: list[tuple[int, int]]
) -> tuple[list[str], int, int]:
    spec = registration["training"]["source_row_split"]
    path = root / spec["path"]
    if sha(path) != spec["sha256"]:
        raise ValueError("frozen_split_hash_mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        split = json.load(stream)
    chunks = []
    offset = 0
    for count, weight in weights:
        chunks.append(
            np.tile(np.arange(offset, offset + count, dtype=np.int64), weight)
        )
        offset += count
    expanded = np.concatenate(chunks)
    membership = [""] * n
    sets: list[set[int]] = []
    for label, key, count_key, hash_key in (
        ("train", "train_positions", "train_count", "train_positions_sha256"),
        (
            "validation",
            "validation_positions",
            "validation_count",
            "validation_positions_sha256",
        ),
    ):
        pos = np.asarray(split[key], dtype=np.int64)
        packed = pos.astype("<i8", copy=False).tobytes()
        if (
            len(pos) != spec[count_key]
            or hashlib.sha256(packed).hexdigest() != spec[hash_key]
        ):
            raise ValueError(f"{label}_position_identity_mismatch")
        if (
            len(set(map(int, pos))) != len(pos)
            or np.any(pos < 0)
            or np.any(pos >= len(expanded))
        ):
            raise ValueError(f"{label}_position_range_or_duplicate")
        sets.append(set(map(int, pos)))
        for row in set(map(int, expanded[pos])):
            if membership[row]:
                raise ValueError("compact_partition_overlap")
            membership[row] = label
    if (
        sets[0] & sets[1]
        or sets[0] | sets[1] != set(range(len(expanded)))
        or any(not p for p in membership)
    ):
        raise ValueError("split_coverage_mismatch")
    return membership, len(sets[0]), len(sets[1])


def compare_rows(
    reconstructed: list[dict[str, Any]], published: list[dict[str, Any]]
) -> None:
    """Require exact equality for every independently reconstructed row field."""
    if reconstructed != published:
        first = next(
            (
                i
                for i, pair in enumerate(zip(reconstructed, published))
                if pair[0] != pair[1]
            ),
            min(len(reconstructed), len(published)),
        )
        raise ValueError(f"seed432_row_source_mismatch:{first}")


def compare_groups(
    reconstructed: list[dict[str, Any]], published: list[dict[str, Any]]
) -> None:
    """Compare complete group records, including all member-order fields."""
    if reconstructed != published:
        raise ValueError("seed432_group_source_mismatch")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "docs/data/seed432-policy-target-compatibility"
    correction = root / "docs/data/seed433-seed432-census-correction"
    manifest = json.loads((data / "manifest.json").read_text())
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    registration = json.loads(registration_path.read_text())
    if sha(registration_path) != manifest["registration_sha256"]:
        raise ValueError("registration_manifest_binding_mismatch")
    for relative, digest in manifest["execution_sources"].items():
        if sha(root / relative) != digest:
            raise ValueError(f"seed432_execution_source_mismatch:{relative}")
    if (
        sha(data / "row-accounting.jsonl.gz") != manifest["row_accounting_sha256"]
        or sha(data / "group-accounting.jsonl.gz")
        != manifest["group_accounting_sha256"]
    ):
        raise ValueError("seed432_archive_hash_mismatch")
    rows = list(reconstruct_sources(root, registration))
    counts = [
        sum(row["source"] == spec["name"] for row in rows)
        for spec in registration["replays"]
    ]
    weights = [
        (count, spec["weight"])
        for count, spec in zip(counts, registration["replays"], strict=True)
    ]
    membership, train_count, validation_count = _positions(
        root, registration, len(rows), weights
    )
    for row in rows:
        row["partition"] = membership[row["compact_row"]]
    coefficients = np.asarray(
        [row["policy_coefficient"] for row in rows], dtype=np.float32
    )
    treatment, _ = normalize_policy_coefficients(
        coefficients,
        [row["canonical_state"] for row in rows],
        np.asarray([row["active_stones"] for row in rows]),
        [row["source"] for row in rows],
        np.asarray([row["partition"] == "train" for row in rows]),
    )
    for row, value in zip(rows, treatment, strict=True):
        row.update(
            treatment_coefficient=float(value),
            treatment_exposure_weight=row["replay_multiplicity"] * float(value),
            treatment_equal_input_weight=float(value),
        )
    with gzip.open(data / "row-accounting.jsonl.gz", "rt", encoding="utf-8") as stream:
        published_rows = [json.loads(line) for line in stream]
    compare_rows(rows, published_rows)
    populations: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = (
            "le16"
            if row["active_stones"] <= 16
            else "17-32"
            if row["active_stones"] <= 32
            else "gt32"
        )
        populations.setdefault(f"{row['partition']}/{bucket}", []).append(row)
    rebuilt_populations = {
        name: {kind: summarize(group, kind) for kind in ("exposure", "equal_input")}
        | {
            f"seed429_treatment_{kind}": summarize(group, kind, treatment=True)
            for kind in ("exposure", "equal_input")
        }
        for name, group in populations.items()
    }
    published_results = json.loads((data / "results.json").read_text())
    if rebuilt_populations != published_results["populations"]:
        raise ValueError("seed432_results_source_mismatch")
    actual_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        actual_groups.setdefault((row["partition"], row["input_sha256"]), []).append(
            row
        )
    reconstructed_groups = [
        {
            "partition": p,
            "input_sha256": identity,
            "canonical_states": sorted({r["canonical_state"] for r in group}),
            "compact_rows": [r["compact_row"] for r in group],
            "source_lines": [[r["source"], r["line"]] for r in group],
            "replay_multiplicities": [r["replay_multiplicity"] for r in group],
        }
        for (p, identity), group in sorted(actual_groups.items())
    ]
    with gzip.open(
        data / "group-accounting.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        compare_groups(reconstructed_groups, [json.loads(line) for line in stream])
    correction_receipt = json.loads(
        (correction / "correction-receipt.json").read_text()
    )
    if (
        sha(data / "manifest.json") != correction_receipt["seed432_manifest_sha256"]
        or sha(data / "results.json") != correction_receipt["seed432_results_sha256"]
    ):
        raise ValueError("seed433_binding_chain_mismatch")
    for relative, key in (
        ("ml/alphazero_lite/seed433_census_correction.py", "calculation_source_sha256"),
        ("ml/alphazero_lite/verify_seed433.py", "verifier_source_sha256"),
    ):
        if sha(root / relative) != correction_receipt[key]:
            raise ValueError(f"seed433_execution_source_mismatch:{relative}")
    corrected = json.loads((correction / "corrected-results.json").read_text())
    if (
        sha(correction / "corrected-results.json")
        != correction_receipt["corrected_results_sha256"]
        or calculate(data, None) != corrected
    ):
        raise ValueError("seed433_corrected_results_mismatch")
    expanded_count = sum(count * weight for count, weight in weights)
    return {
        "status": "valid",
        "compact_rows": len(rows),
        "expanded_positions": expanded_count,
        "training_positions": train_count,
        "validation_positions": validation_count,
        "classification": corrected["classification"],
        "all_seed432_rows_reproduce": True,
        "all_seed433_corrected_results_reproduce": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
