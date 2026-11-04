"""Seed552 preregistration and opening-cluster analysis for C108 versus P."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed542_expanded_policy_strength_screen as suite
from ml.alphazero_lite import seed551_controlled_policy_strength as seed551

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/seed552-c108-serving-p"
SEED511 = "docs/data/seed511-fixed-composition-screen-v4"
OPENINGS_PATH = f"{SEED511}/openings.jsonl"
OPENINGS_SHA256 = "27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d"
OPENINGS = 1024
GAMES = 2048
SEED = 545
SIMULATIONS = 128
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 545
C_CHECKPOINT = "docs/data/seed549-controlled-training/control-update-108.npz"
P_CHECKPOINT = "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/P-E4/model.npz"
INITIALIZER = "docs/data/seed542-selfplay-4096/initializer.npz"
C_SHA256 = "5339fcc06b1284515181352b47414ef82ff326d22b6848aa1cd7f445e47852ad"
P_SHA256 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
INITIALIZER_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
TABLEBASE_POINTER = "model-artifact/runtime/kalah_v1_21.kvtb"
TABLEBASE_OBJECT_SHA256 = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
TABLEBASE_OBJECT_SIZE = 709_635_072
HISTORICAL_TABLEBASE_POINTER_SHA256 = "a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c"
HISTORICAL_TABLEBASE_POINTER_COMMIT = "ff0805ea6572dcf297a2f13db66abab28ad915e8"
NATIVE_PROBE = "model-artifact/runtime/kalah_v1_tablebase"
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
}
ASSIGNMENTS = {
    "challenger": "C update-108 control policy and its own trunk",
    "challenger_policy_artifact": "C108",
    "challenger_policy_trunk": "C108",
    "challenger_value_artifact": "initializer",
    "challenger_value_trunk": "initializer",
    "opponent": "P historical seed504 E4 policy and its own trunk",
    "opponent_policy_artifact": "P-E4",
    "opponent_policy_trunk": "P-E4",
    "opponent_value_artifact": "initializer",
    "opponent_value_trunk": "initializer",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_openings(root: Path) -> tuple[bytes, list[dict[str, Any]]]:
    path = root / OPENINGS_PATH
    raw = path.read_bytes()
    if sha256(path) != OPENINGS_SHA256:
        raise ValueError("seed552_indexed_suite_hash_mismatch")
    if len(raw.splitlines()) != OPENINGS:
        raise ValueError("seed552_indexed_suite_row_count_mismatch")
    helper = suite.import_seed511_suite
    rows, _ = helper(root)
    if len(rows) != OPENINGS:
        raise ValueError("seed552_replayed_suite_row_count_mismatch")
    return raw, rows


def opening_overlaps(root: Path, openings: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind every overlap population and retain a row-level report."""
    root = root.resolve()
    prior_overlap = seed551.overlap_report(root, openings)
    e4_registration_path = root / "docs/data/seed504-policy-only-ablation/registration.json"
    e4_registration = json.loads(e4_registration_path.read_text())
    p_canonical: set[str] = set()
    p_float32: set[str] = set()
    row_count = 0
    shards = e4_registration["bindings"]["training_shards"]
    for shard in shards:
        path = root / shard["path"]
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"seed552_P_training_shard_hash_mismatch:{shard['path']}")
        for line in path.read_text().splitlines():
            if not line:
                continue
            encoded = np.asarray(json.loads(line)["state"], dtype="<f4")
            state = suite.state_from_encoded(encoded)
            p_canonical.add(seed551.canonical_identity(state))
            p_float32.add(hashlib.sha256(encoded.tobytes()).hexdigest())
            row_count += 1
    if row_count != e4_registration["bindings"]["training_rows"]:
        raise ValueError("seed552_P_training_population_count_mismatch")
    rows = []
    counts = {"C_training": {"canonical": 0, "production_float32": 0},
              "C_development": {"canonical": 0, "production_float32": 0},
              "P_policy_training": {"canonical": 0, "production_float32": 0}}
    for index, (opening, old) in enumerate(zip(openings, prior_overlap["per_opening"], strict=True)):
        canonical = seed551.canonical_identity(opening["state"])
        f32 = seed551.production_float32_identity(opening["state"])
        row = {"opening_index": index, "canonical_identity": canonical, "production_float32_identity": f32}
        row["C_training_canonical_overlap"] = old["control_canonical_overlap"]
        row["C_training_production_float32_overlap"] = old["control_production_float32_overlap"]
        row["C_development_canonical_overlap"] = old["shared_development_canonical_overlap"]
        row["C_development_production_float32_overlap"] = old["shared_development_production_float32_overlap"]
        row["P_policy_training_canonical_overlap"] = canonical in p_canonical
        row["P_policy_training_production_float32_overlap"] = f32 in p_float32
        for population, fields in (
            ("C_training", ("C_training_canonical_overlap", "C_training_production_float32_overlap")),
            ("C_development", ("C_development_canonical_overlap", "C_development_production_float32_overlap")),
            ("P_policy_training", ("P_policy_training_canonical_overlap", "P_policy_training_production_float32_overlap")),
        ):
            counts[population]["canonical"] += row[fields[0]]
            counts[population]["production_float32"] += row[fields[1]]
        rows.append(row)
    return {
        "schema": "seed552_training_population_overlap_v1",
        "opening_count": len(openings),
        "population_rows": {"C_training": prior_overlap["training_rows"]["control"],
                            "C_development": prior_overlap["shared_development_members"],
                            "P_policy_training": row_count},
        "overlap_opening_counts": counts,
        "P_policy_training_registration_sha256": sha256(e4_registration_path),
        "P_policy_training_shards": shards,
        "seed461_continuation_overlap_unresolved": True,
        "per_opening": rows,
    }


def analyze(ledger: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score C wins as one and bootstrap whole opening clusters with replacement."""
    if len(ledger) != GAMES:
        raise ValueError("seed552_analysis_requires_all_2048_games")
    points = {
        "C": 1.0, "draw": 0.5, "P": 0.0,
        "capped_C": 1.0, "capped_draw": 0.5, "capped_P": 0.0,
    }
    grouped: dict[int, dict[int, dict[str, Any]]] = {i: {} for i in range(OPENINGS)}
    terminal = {"C": 0, "draw": 0, "P": 0}
    capped_count = 0
    for row in ledger:
        opening, seat = int(row["opening_index"]), int(row["game_position"])
        if opening not in grouped or seat not in (0, 1) or seat in grouped[opening]:
            raise ValueError("seed552_duplicate_opening_or_seat")
        outcome = row["outcome"]
        if outcome not in points:
            raise ValueError("seed552_invalid_outcome")
        grouped[opening][seat] = row
        if row.get("capped_nonterminal"):
            capped_count += 1
        else:
            terminal[outcome] += 1
    if any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError("seed552_missing_seat_pair")
    clusters = np.asarray(
        [np.mean([points[grouped[i][s]["outcome"]] for s in (0, 1)]) for i in range(OPENINGS)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(0, OPENINGS, size=(BOOTSTRAP_RESAMPLES, OPENINGS))
    means = clusters[sample_indices].mean(axis=1)
    lower = float(np.quantile(means, 0.05, method="linear"))
    interval = [float(np.quantile(means, q, method="linear")) for q in (0.025, 0.975)]
    valid_decision = capped_count == 0
    advance = valid_decision and lower > 0.50
    result = {
        "status": "complete",
        "games": len(ledger),
        "opening_pairs": OPENINGS,
        "terminal_wins": terminal["C"],
        "terminal_draws": terminal["draw"],
        "terminal_losses": terminal["P"],
        "capped_nonterminal_games": capped_count,
        "capped_nonterminal_prevents_valid_decision": capped_count > 0,
        "score": float(clusters.mean()),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sampling": "opening clusters with replacement; duplicate selections retained",
        "quantile_method": "NumPy linear",
        "one_sided_95_lower": lower,
        "descriptive_95_interval": interval,
        "decision": "advance_to_separate_1200_simulation_screen" if advance else "no_valid_strength_decision" if capped_count else "do_not_advance_C108",
        "strength_decision_valid": valid_decision,
        "promotion_or_deployment_authorized": False,
    }
    opening_rows = [
        {
            "opening_index": i,
            "seat0_C_score": points[grouped[i][0]["outcome"]],
            "seat1_C_score": points[grouped[i][1]["outcome"]],
            "opening_cluster_score": float(clusters[i]),
            "capped_nonterminal_games": sum(bool(grouped[i][s].get("capped_nonterminal")) for s in (0, 1)),
        }
        for i in range(OPENINGS)
    ]
    return result, opening_rows
