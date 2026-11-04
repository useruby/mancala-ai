"""Seed551 policy-only composition screen protocol and analysis helpers.

The screen compares the update-108 control and treatment policies while each
uses its own trunk and both use the same frozen initializer value component.
It is an exploratory reused-suite strength screen, not a promotion gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed542_expanded_policy_strength_screen as suite

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/seed551-controlled-policy-strength-screen"
CORPUS = "docs/data/seed549-matched-selfplay"
TRAINING = "docs/data/seed549-controlled-training"
INITIALIZER = "docs/data/seed542-selfplay-4096/initializer.npz"
CONTROL_CHECKPOINT = f"{TRAINING}/control-update-108.npz"
TREATMENT_CHECKPOINT = f"{TRAINING}/treatment-update-108.npz"
CONTROL_SHA256 = "5339fcc06b1284515181352b47414ef82ff326d22b6848aa1cd7f445e47852ad"
TREATMENT_SHA256 = "5006ec04a09c3b286753c23bd34d7216387f5cca65e8d5705b0432b0ad2efe00"
OPENINGS = 1024
GAMES = OPENINGS * 2
SIMULATIONS = 128
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 545
SEED_CONTRACT = "azlite_eval_seed_v2"
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
ASSIGNMENTS = {
    "C_policy": "C update-108 policy",
    "C_policy_trunk": "C update-108 own trunk",
    "C_value": "frozen initializer value network",
    "C_value_trunk": "frozen initializer own trunk",
    "T_policy": "T update-108 policy",
    "T_policy_trunk": "T update-108 own trunk",
    "T_value": "frozen initializer value network",
    "T_value_trunk": "frozen initializer own trunk",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_identity(state: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(state)).hexdigest()


def production_float32_identity(state: dict[str, Any]) -> str:
    vector = np.asarray(
        suite.encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
    )
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _training_identities(root: Path, arm: str) -> tuple[set[str], set[str], int]:
    """Read state identities only after checking their registered source bytes."""
    manifest = json.loads((root / CORPUS / f"{arm}-source-manifest.json").read_text())
    registration = json.loads((root / TRAINING / "registration.json").read_text())
    expected = registration.get("input_sha256", {})
    manifest_relative = f"{CORPUS}/{arm}-source-manifest.json"
    post_generation = json.loads(
        (root / CORPUS / "post-generation-receipt.json").read_text()
    )
    manifest_name = f"{arm}-source-manifest.json"
    if sha256(root / manifest_relative) != post_generation.get("products", {}).get(
        manifest_name
    ):
        raise ValueError(f"seed551_source_manifest_hash_mismatch:{arm}")
    canonical: set[str] = set()
    float32: set[str] = set()
    count = 0
    if manifest.get("arm") != arm:
        raise ValueError(f"seed551_source_manifest_arm_mismatch:{arm}")
    rows = [row for row in manifest.get("shards", []) if row["partition"] == "training"]
    if post_generation.get("generation_registration_sha256") != sha256(
        root / CORPUS / "registration.json"
    ):
        raise ValueError("seed551_corpus_registration_receipt_mismatch")
    for row in rows:
        relative = row["path"]
        digest = row["sha256"]
        if (
            expected.get(relative) not in (None, digest)
            or sha256(root / relative) != digest
        ):
            raise ValueError(f"seed551_training_source_identity_mismatch:{relative}")
        for example in _jsonl(root / relative):
            encoded = np.asarray(example["state"], dtype="<f4")
            state = suite.state_from_encoded(encoded)
            canonical.add(canonical_identity(state))
            float32.add(hashlib.sha256(encoded.tobytes()).hexdigest())
            count += 1
    if not count:
        raise ValueError(f"seed551_empty_training_population:{arm}")
    return canonical, float32, count


def overlap_report(root: Path, openings: list[dict[str, Any]]) -> dict[str, Any]:
    """Count suite overlap without filtering, substituting, or reordering rows."""
    root = root.resolve()
    populations = {
        arm: _training_identities(root, arm) for arm in ("control", "treatment")
    }
    training_registration = json.loads(
        (root / TRAINING / "registration.json").read_text()
    )

    membership_path = root / CORPUS / "shared-development-membership.jsonl"
    post_generation = json.loads(
        (root / CORPUS / "post-generation-receipt.json").read_text()
    )
    membership_relative = f"{CORPUS}/shared-development-membership.jsonl"
    if sha256(membership_path) != post_generation.get("products", {}).get(
        "shared-development-membership.jsonl"
    ) or sha256(membership_path) != training_registration.get("input_sha256", {}).get(
        membership_relative
    ):
        raise ValueError("seed551_development_membership_hash_mismatch")
    membership = [row for row in _jsonl(membership_path) if row.get("included")]
    development_canonical: set[str] = set()
    development_float32: set[str] = set()
    cached_shards: dict[str, list[dict[str, Any]]] = {}
    shard_hashes: dict[str, str] = {}
    for arm in ("control", "treatment"):
        source_manifest = json.loads(
            (root / CORPUS / f"{arm}-source-manifest.json").read_text()
        )
        shard_hashes.update(
            {record["path"]: record["sha256"] for record in source_manifest["shards"]}
        )
    for member in membership:
        source = member["source"]
        relative = source["path"]
        if relative not in cached_shards:
            shard_path = root / relative
            if sha256(shard_path) != shard_hashes.get(relative) or sha256(
                shard_path
            ) != training_registration.get("input_sha256", {}).get(relative):
                raise ValueError(f"seed551_development_source_hash_mismatch:{relative}")
            cached_shards[relative] = _jsonl(shard_path)
        record = cached_shards[relative][int(source["row_number"])]
        encoded = np.asarray(record["state"], dtype="<f4")
        state = suite.state_from_encoded(encoded)
        identity = hashlib.sha256(encoded.tobytes()).hexdigest()
        if identity != member.get("production_float32_sha256"):
            raise ValueError("seed551_development_member_identity_mismatch")
        development_canonical.add(canonical_identity(state))
        development_float32.add(identity)

    per_opening: list[dict[str, Any]] = []
    counts = {
        name: {"canonical": 0, "production_float32": 0}
        for name in ("control", "treatment", "shared_development")
    }
    for index, opening in enumerate(openings):
        if opening.get("opening_index") != index:
            raise ValueError(f"seed551_opening_order_mismatch:{index}")
        state = opening["state"]
        can = canonical_identity(state)
        f32 = production_float32_identity(state)
        row: dict[str, Any] = {
            "opening_index": index,
            "canonical_identity": can,
            "production_float32_identity": f32,
        }
        for name, population in populations.items():
            row[f"{name}_canonical_overlap"] = can in population[0]
            row[f"{name}_production_float32_overlap"] = f32 in population[1]
            counts[name]["canonical"] += row[f"{name}_canonical_overlap"]
            counts[name]["production_float32"] += row[
                f"{name}_production_float32_overlap"
            ]
        row["shared_development_canonical_overlap"] = can in development_canonical
        row["shared_development_production_float32_overlap"] = (
            f32 in development_float32
        )
        counts["shared_development"]["canonical"] += row[
            "shared_development_canonical_overlap"
        ]
        counts["shared_development"]["production_float32"] += row[
            "shared_development_production_float32_overlap"
        ]
        per_opening.append(row)
    return {
        "schema": "seed551_training_population_overlap_v1",
        "exploratory_reused_suite": True,
        "suite_openings_retained": len(openings),
        "training_rows": {name: result[2] for name, result in populations.items()},
        "overlap_opening_counts": counts,
        "shared_development_members": len(membership),
        "per_opening": per_opening,
        "seed461_continuation_overlap_unresolved": True,
        "historical_suite_outcomes_imported": False,
    }


def analyze(
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the preregistered opening-cluster bootstrap from validated records."""
    if len(ledger) != GAMES:
        raise ValueError("seed551_analysis_requires_all_2048_validated_games")
    grouped: dict[int, dict[int, dict[str, Any]]] = {i: {} for i in range(OPENINGS)}
    points = {"T": 1.0, "draw": 0.5, "C": 0.0}
    wins = draws = losses = capped = 0
    capped_results = {"T": 0, "draw": 0, "C": 0}
    for row in ledger:
        opening = int(row["opening_index"])
        seat = int(row["game_position"])
        if opening not in grouped or seat not in (0, 1) or seat in grouped[opening]:
            raise ValueError("seed551_ledger_opening_or_seat_duplicate")
        outcome = row["outcome"]
        if outcome not in points:
            raise ValueError("seed551_ledger_outcome_invalid")
        capped += bool(row.get("capped_nonterminal", False))
        if row.get("capped_nonterminal", False):
            capped_results[outcome] += 1
        else:
            wins += outcome == "T"
            draws += outcome == "draw"
            losses += outcome == "C"
        grouped[opening][seat] = row
    if any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError("seed551_ledger_missing_seat_pair")
    clusters = np.asarray(
        [
            np.mean([points[grouped[i][seat]["outcome"]] for seat in (0, 1)])
            for i in range(OPENINGS)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = clusters[rng.integers(0, OPENINGS, size=(BOOTSTRAP_RESAMPLES, OPENINGS))]
    means = sampled.mean(axis=1)
    score = float(clusters.mean())
    lower = float(np.quantile(means, 0.05, method="linear"))
    complete = len(ledger) == GAMES
    advance = complete and lower > 0.50
    result = {
        "status": "complete",
        "games": GAMES,
        "opening_pairs": OPENINGS,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "capped_nonterminal_results_by_current_store_margin": capped_results,
        "capped_points_policy": "use the arena's current store-margin winner at ply 200, ties 0.5, and report capped results separately from terminal outcomes",
        "capped_nonterminal_games": int(capped),
        "score": score,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "replacement": True,
        "preserved_cluster_multiplicity": True,
        "quantile_method": "linear",
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(means, 0.025, method="linear")),
            float(np.quantile(means, 0.975, method="linear")),
        ],
        "decision": "advance_to_separate_serving_P_screen"
        if advance
        else "do_not_advance_source_evaluator_branch",
        "strength_decision_valid": complete,
        "non_advancement_does_not_establish_universal_source_evaluator_inferiority": True,
        "promotion_or_deployment_authorized": False,
        "fresh_holdout": False,
        "seed461_continuation_overlap_unresolved": True,
    }
    opening_rows = [
        {
            "opening_index": index,
            "seat0_T_score": points[grouped[index][0]["outcome"]],
            "seat1_T_score": points[grouped[index][1]["outcome"]],
            "opening_cluster_score": float(clusters[index]),
            "seat0_capped_nonterminal": bool(grouped[index][0]["capped_nonterminal"]),
            "seat1_capped_nonterminal": bool(grouped[index][1]["capped_nonterminal"]),
            "capped_nonterminal_games": sum(
                bool(grouped[index][seat]["capped_nonterminal"]) for seat in (0, 1)
            ),
        }
        for index in range(OPENINGS)
    ]
    return result, opening_rows
