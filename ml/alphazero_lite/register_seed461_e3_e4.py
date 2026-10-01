"""Freeze the seed461 constant-LR E3 versus E4 arena before evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
BASE_REG = DATA / "seed461-cosine-lr-ablation-registration.json"
TRAINING = DATA / "seed461-cosine-lr-ablation-training.json"
TRAINING_BIND = DATA / "seed461-cosine-lr-ablation-training-binding.json"
PR387 = DATA / "seed461-e2-e4-average-registration.json"
PR388_BIND = DATA / "seed461-e2-e4-average-evaluation-binding.json"
CANONICAL = ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl"
EXPLORATORY = DATA / "seed461-batch-order-sensitivity-exploratory-openings.jsonl"
REG = DATA / "seed461-e3-e4-registration.json"
SUITE = DATA / "seed461-e3-e4-openings.jsonl"
ORDERS = [38611, 38612, 38613, 38614, 38615]
CONSUMED = [
    DATA / name
    for name in (
        "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
        "seed461-lr-sensitivity-openings-v2.jsonl",
        "seed461-cosine-lr-ablation-openings.jsonl",
        "seed461-e2-e4-average-openings.jsonl",
    )
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def register() -> None:
    base = json.loads(BASE_REG.read_text())
    training = json.loads(TRAINING.read_text())
    training_bind = json.loads(TRAINING_BIND.read_text())
    old_registration = json.loads(PR387.read_text())
    old_eval_binding = json.loads(PR388_BIND.read_text())
    if training_bind["registration_sha256"] != sha(BASE_REG):
        raise ValueError("training_registration_hash_mismatch")
    if training_bind["training_sha256"] != sha(TRAINING):
        raise ValueError("training_record_hash_mismatch")

    replay_paths = [Path(row["path"]) for row in base["training"]["replays"]]
    for replay, record in zip(replay_paths, base["training"]["replays"], strict=True):
        if sha(replay) != record["sha256"]:
            raise ValueError(f"frozen_replay_hash_mismatch:{replay}")
    proof, excluded = population.build_proof(CANONICAL, EXPLORATORY, replay_paths)
    consumed = {}
    for path in CONSUMED:
        rows = suites.load_suite_jsonl(str(path))
        identities = population.keys(rows)
        consumed[path.name] = {
            "path": str(path),
            "sha256": sha(path),
            "unique_state_identities": len(identities),
        }
        excluded |= identities
    openings = population.select_holdout(excluded, seed=389)
    suite_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in openings
    ).encode()
    suite_identities = population.keys(openings)
    if (
        len(openings) != 256
        or len(suite_identities) != 256
        or suite_identities & excluded
    ):
        raise ValueError("suite_overlap_or_count_failure")
    if any(
        row["pit_sum"] <= 32 or KalahGame.from_state(row["state"]).over()
        for row in openings
    ):
        raise ValueError("suite_opening_ineligible")
    if SUITE.exists() and SUITE.read_bytes() != suite_bytes:
        raise ValueError("immutable_suite_conflict")
    if not SUITE.exists():
        SUITE.write_bytes(suite_bytes)

    proof["consumed_suites_through_388"] = consumed
    proof["exclusion_counts"]["consumed_suites_through_388"] = len(
        set().union(
            *(population.keys(suites.load_suite_jsonl(str(path))) for path in CONSUMED)
        )
    )
    proof["exclusion_counts"]["union"] = len(excluded)
    proof["suite_identity_overlap"] = {
        "suite_states": len(suite_identities),
        "excluded_states": len(excluded),
        "overlap": len(suite_identities & excluded),
        "proof": "canonical state SHA-256 identities; zero intersection",
    }

    source_hashes, source_paths, counts = {}, {}, {}
    for order in ORDERS:
        key = str(order)
        run = f"order_{order}_A"
        record = training["trajectories"][run]
        source_hashes[key], source_paths[key], counts[key] = {}, {}, {}
        for epoch in ("E3", "E4"):
            checkpoint = (
                ROOT / ".tmp/seed461-cosine-lr-ablation/training" / run / f"{epoch}.npz"
            )
            expected = record["epochs"][epoch]
            if sha(checkpoint) != expected:
                raise ValueError(f"source_checkpoint_hash_mismatch:{run}:{epoch}")
            source_hashes[key][epoch] = expected
            source_paths[key][epoch] = str(checkpoint)
            row = next(
                item for item in record["history"] if item["epoch"] == int(epoch[1:])
            )
            counts[key][epoch] = {
                "epoch_updates": row["optimizer_updates"],
                "cumulative_updates": sum(
                    item["optimizer_updates"]
                    for item in record["history"][: int(epoch[1:])]
                ),
                "epoch_examples_exposed": row["examples_sampled"],
                "cumulative_examples_exposed": sum(
                    item["examples_sampled"]
                    for item in record["history"][: int(epoch[1:])]
                ),
            }

    runtime = old_registration["evaluation"]["runtime_contract"]
    opponent = old_registration["evaluation"]["opponent_binding"]
    # #388 independently binds the same exact frozen opponent/runtime; chain it.
    if old_eval_binding["opponent"] != opponent:
        raise ValueError("pr388_opponent_binding_mismatch")
    value = {
        "schema": "seed461-e3-e4-registration-v1",
        "status": "prospectively_registered_before_candidate_binding_and_games",
        "arms": {
            "A": "original constant-LR E4 checkpoint",
            "B": "original constant-LR E3 checkpoint",
        },
        "treatment": "epoch count only; LR, scheduler, data, initialization, split, architecture, and losses are inherited unchanged from PR #386 constant-LR A trajectories",
        "training": {
            "performed": False,
            "source_pr": 386,
            "orders": ORDERS,
            "source_registration_sha256": sha(BASE_REG),
            "training_record_sha256": sha(TRAINING),
            "training_binding_sha256": sha(TRAINING_BIND),
            "source_checkpoint_sha256": source_hashes,
            "source_paths": source_paths,
            "update_and_exposure_counts": counts,
        },
        "evaluation": {
            "opponent": "exact frozen seed455 opponent from PR #388",
            "opponent_binding": opponent,
            "runtime_contract": runtime,
            "suite": {
                "path": str(SUITE),
                "sha256": sha(SUITE),
                "seed": 389,
                "openings": 256,
                "nonterminal": True,
                "active_pit_stones_gt": 32,
                "identity_overlap_proof": proof,
            },
            "games": 5120,
            "games_per_candidate": 512,
            "games_per_opening": 2,
            "simulations_per_side": 384,
            "c_puct": 1.25,
            "arena_seed": 389,
            "seed_contract": "azlite_eval_seed_v2",
            "outcome_dependent_extensions": False,
        },
        "analysis": {
            "bootstrap": {
                "resamples": 10000,
                "seed": 389,
                "cluster": "shared opening",
                "interval": "95% percentile",
            },
            "decision_thresholds": {
                "mean_effect_at_least": 0.03,
                "lower_interval_strictly_above": 0.0,
                "nonnegative_paired_effects_at_least": 4,
                "worst_paired_effect_at_least": -0.05,
                "B_score_range_strictly_smaller": True,
                "B_minimum_score_at_least_A": True,
            },
            "inference_scope": "conditional on these five orders and this dataset",
            "promotion": "never; do not change production checkpoint selection",
            "preserve_rejections": [
                "PR #386 cosine-LR rejection",
                "PR #388 checkpoint-averaging rejection",
            ],
        },
    }
    if REG.exists() and json.loads(REG.read_text()) != value:
        raise ValueError("immutable_registration_conflict")
    save_json(REG, value)
    print(f"registration_sha256={sha(REG)}")
    print(f"suite_sha256={sha(SUITE)}")
    print(f"zero_overlap={proof['suite_identity_overlap']['overlap']}")


if __name__ == "__main__":
    register()
