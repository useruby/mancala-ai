"""Preregistered T512 versus C128 policy-strength screen protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed542_expanded_policy_strength_screen as suite
from ml.alphazero_lite import seed551_controlled_policy_strength as seed551

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/seed555-policy-strength-screen"
SEED554 = "docs/data/seed554-fixed-state-teacher-budget"
SEED511 = "docs/data/seed511-fixed-composition-screen-v4"
OPENINGS_PATH = f"{SEED511}/openings.jsonl"
OPENINGS_SHA256 = "27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d"
OPENINGS = 1024
GAMES = 2048
SEED = 545
SIMULATIONS = 128
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 545
T512_CHECKPOINT = f"{SEED554}/treatment-update-108.npz"
C128_CHECKPOINT = f"{SEED554}/control-update-108.npz"
INITIALIZER = "docs/data/seed542-selfplay-4096/initializer.npz"
T512_SHA256 = "8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e"
C128_SHA256 = "6957ac4a9db529f51ec184a04c01304c2a6453de300c69f7c53de16634626205"
INITIALIZER_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
TABLEBASE_PATH = "model-artifact/runtime/kalah_v1_21.kvtb"
TABLEBASE_SHA256 = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
TABLEBASE_SIZE = 709_635_072
TABLEBASE_POINTER_SHA256 = (
    "a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c"
)
TABLEBASE_POINTER_COMMIT = "ff0805ea6572dcf297a2f13db66abab28ad915e8"
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
    "challenger": "T512 update-108 policy and its own trunk",
    "challenger_policy_artifact": "T512",
    "challenger_policy_trunk": "T512",
    "challenger_value_artifact": "initializer",
    "challenger_value_trunk": "initializer",
    "opponent": "C128 update-108 policy and its own trunk",
    "opponent_policy_artifact": "C128",
    "opponent_policy_trunk": "C128",
    "opponent_value_artifact": "initializer",
    "opponent_value_trunk": "initializer",
}
ARENA_RULES = {
    "games_per_opening": 2,
    "simulations_per_side": SIMULATIONS,
    "c_puct": 1.25,
    "workers": 1,
    "search_options": SEARCH_OPTIONS,
    "exact_root_threshold": 16,
    "exact_root_objective": "final_score_margin",
    "tie_rule": "highest legal network prior then lowest move index",
    "exact_leaf_solving": False,
    "random_opening_plies": 0,
    "max_plies": 200,
    "seed_contract": "azlite_eval_seed_v2",
    "seed": "545 + opening_index",
    "deterministic_root_policy": True,
    "temperature": 0.0,
    "root_noise": False,
    "fpu": 0.0,
    "subtree_reuse": False,
    "value_normalization": False,
    "tactical_root_bias": 0.0,
}
ANALYSIS_RULES = {
    "T512_win": 1.0,
    "terminal_draw": 0.5,
    "loss": 0.0,
    "seat_average_per_opening": True,
    "bootstrap_clusters": "opening",
    "resamples": BOOTSTRAP_RESAMPLES,
    "seed": BOOTSTRAP_SEED,
    "replacement": True,
    "quantiles": [0.025, 0.05, 0.975],
    "numpy_quantile_method": "linear",
    "capped_nonterminal_blocks_valid_decision": True,
    "score_and_bootstrap_with_caps": None,
    "advance_rule": "all 2048 games valid and terminal and one-sided 95% lower bound strictly greater than 0.50: propose_T512_vs_serving_P_screen",
    "complete_otherwise": "do_not_advance_teacher_budget_branch",
    "incomplete_invalid_or_capped": "no_valid_advancement_decision",
    "positive_result_does_not_establish_superiority_to_serving_P": True,
    "positive_result_does_not_authorize_deployment": True,
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
    if sha256(path) != OPENINGS_SHA256 or len(raw.splitlines()) != OPENINGS:
        raise ValueError("seed555_indexed_suite_identity_mismatch")
    openings, _ = suite.import_seed511_suite(root)
    if len(openings) != OPENINGS:
        raise ValueError("seed555_replayed_suite_count_mismatch")
    return raw, openings


def overlap_report(root: Path, openings: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind opening overlap to the actual #554 derivatives and development set."""
    root = root.resolve()
    seed554 = root / SEED554
    binding_path = seed554 / "training-input-binding.json"
    binding = json.loads(binding_path.read_text())
    prior = seed551.overlap_report(root, openings)
    training_sets: dict[str, tuple[set[str], set[str], int]] = {}
    for label, directory in (("T512", "treatment"), ("C128", "control")):
        canonical: set[str] = set()
        float32: set[str] = set()
        count = 0
        for shard in range(6):
            relative = f"{directory}/shard-{shard:03d}.jsonl"
            path = seed554 / relative
            if sha256(path) != binding["derivative_sha256"][f"{directory}/{shard:03d}"]:
                raise ValueError(f"seed555_derivative_hash_mismatch:{relative}")
            for line in path.read_text().splitlines():
                row = json.loads(line)
                encoded = np.asarray(row["state"], dtype="<f4")
                state = suite.state_from_encoded(encoded)
                canonical.add(seed551.canonical_identity(state))
                float32.add(hashlib.sha256(encoded.tobytes()).hexdigest())
                count += 1
        if count != binding["rows"]:
            raise ValueError(f"seed555_derivative_row_count_mismatch:{label}")
        training_sets[label] = canonical, float32, count

    per_opening: list[dict[str, Any]] = []
    counts = {
        name: {"canonical": 0, "production_float32": 0}
        for name in ("T512_training", "C128_training", "unchanged_development")
    }
    for index, (opening, dev) in enumerate(
        zip(openings, prior["per_opening"], strict=True)
    ):
        state = opening["state"]
        canonical = seed551.canonical_identity(state)
        float32 = seed551.production_float32_identity(state)
        row: dict[str, Any] = {
            "opening_index": index,
            "canonical_identity": canonical,
            "production_float32_identity": float32,
        }
        for label, population in training_sets.items():
            name = f"{label}_training"
            row[f"{name}_canonical_overlap"] = canonical in population[0]
            row[f"{name}_production_float32_overlap"] = float32 in population[1]
            counts[name]["canonical"] += row[f"{name}_canonical_overlap"]
            counts[name]["production_float32"] += row[
                f"{name}_production_float32_overlap"
            ]
        row["unchanged_development_canonical_overlap"] = dev[
            "shared_development_canonical_overlap"
        ]
        row["unchanged_development_production_float32_overlap"] = dev[
            "shared_development_production_float32_overlap"
        ]
        counts["unchanged_development"]["canonical"] += row[
            "unchanged_development_canonical_overlap"
        ]
        counts["unchanged_development"]["production_float32"] += row[
            "unchanged_development_production_float32_overlap"
        ]
        per_opening.append(row)
    return {
        "schema": "seed555_training_population_overlap_v1",
        "opening_count": OPENINGS,
        "training_rows": {label: data[2] for label, data in training_sets.items()},
        "unchanged_development_members": prior["shared_development_members"],
        "overlap_opening_counts": counts,
        "training_input_binding_sha256": sha256(binding_path),
        "seed461_continuation_overlap_unresolved": True,
        "historical_games_or_outcomes_imported": False,
        "per_opening": per_opening,
    }


def analyze(
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score T512 and bootstrap opening clusters with replacement."""
    if len(ledger) != GAMES:
        raise ValueError("seed555_analysis_requires_all_2048_games")
    points = {"T512": 1.0, "draw": 0.5, "C128": 0.0}
    grouped: dict[int, dict[int, dict[str, Any]]] = {i: {} for i in range(OPENINGS)}
    terminals = {"T512": 0, "draw": 0, "C128": 0}
    capped = 0
    capped_outcomes = {"T512": 0, "draw": 0, "C128": 0}
    for row in ledger:
        opening, seat = int(row["opening_index"]), int(row["game_position"])
        if opening not in grouped or seat not in (0, 1) or seat in grouped[opening]:
            raise ValueError("seed555_duplicate_opening_or_seat")
        outcome = row["outcome"]
        if outcome not in points:
            raise ValueError("seed555_invalid_outcome")
        grouped[opening][seat] = row
        if row.get("capped_nonterminal"):
            capped += 1
            capped_outcomes[outcome] += 1
        else:
            terminals[outcome] += 1
    if any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError("seed555_missing_seat_pair")
    clusters = np.asarray(
        [
            np.mean([points[grouped[i][seat]["outcome"]] for seat in (0, 1)])
            for i in range(OPENINGS)
        ],
        dtype=np.float64,
    )
    valid = capped == 0
    pair_capped = {
        index: sum(
            bool(grouped[index][seat].get("capped_nonterminal")) for seat in (0, 1)
        )
        for index in range(OPENINGS)
    }
    if valid:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = clusters[
            rng.integers(0, OPENINGS, size=(BOOTSTRAP_RESAMPLES, OPENINGS))
        ]
        means = samples.mean(axis=1)
        lower = float(np.quantile(means, 0.05, method="linear"))
        interval = [
            float(np.quantile(means, q, method="linear")) for q in (0.025, 0.975)
        ]
        score = float(clusters.mean())
    else:
        lower = None
        interval = None
        score = None
    result = {
        "status": "complete",
        "games": GAMES,
        "opening_pairs": OPENINGS,
        "terminal_wins": terminals["T512"],
        "terminal_draws": terminals["draw"],
        "terminal_losses": terminals["C128"],
        "capped_nonterminal_games": capped,
        "capped_outcomes_by_winner": capped_outcomes,
        "capped_nonterminal_prevents_valid_decision": bool(capped),
        "score": score,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sampling": "opening clusters with replacement; duplicate selections retained",
        "quantile_method": "NumPy linear",
        "one_sided_95_lower": lower,
        "descriptive_95_interval": interval,
        "decision": "propose_T512_vs_serving_P_screen"
        if valid and lower > 0.5
        else "no_valid_advancement_decision"
        if not valid
        else "do_not_advance_teacher_budget_branch",
        "strength_decision_valid": valid,
        "promotion_or_deployment_authorized": False,
    }
    scores = [
        {
            "opening_index": i,
            "seat0_T512_score": points[grouped[i][0]["outcome"]],
            "seat1_T512_score": points[grouped[i][1]["outcome"]],
            "opening_cluster_score": None if pair_capped[i] else float(clusters[i]),
            "capped_nonterminal_games": pair_capped[i],
        }
        for i in range(OPENINGS)
    ]
    return result, scores
