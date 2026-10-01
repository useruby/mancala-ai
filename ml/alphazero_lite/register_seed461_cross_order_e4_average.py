"""Preregister cross-order E4 averaging before exporting candidates or games."""

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
TRAINING_BINDING = DATA / "seed461-cosine-lr-ablation-training-binding.json"
PR389_REG = DATA / "seed461-e3-e4-registration.json"
PR389_BINDING = DATA / "seed461-e3-e4-evaluation-binding.json"
CANONICAL = ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl"
EXPLORATORY = DATA / "seed461-batch-order-sensitivity-exploratory-openings.jsonl"
REPLAY_ROOT = (
    ROOT
    / ".tmp/seed455-nextgen-s461-default-value-root16/runs/seed455-nextgen-s461-default-value-root16-iter1"
)
REG = DATA / "seed461-cross-order-e4-average-registration.json"
SUITE = DATA / "seed461-cross-order-e4-average-openings.jsonl"
ORDERS = (38611, 38612, 38613, 38614, 38615)
CONSUMED_THROUGH_389 = (
    "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
    "seed461-lr-sensitivity-openings-v2.jsonl",
    "seed461-cosine-lr-ablation-openings.jsonl",
    "seed461-e2-e4-average-openings.jsonl",
    "seed461-e3-e4-openings.jsonl",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def register() -> None:
    base = json.loads(BASE_REG.read_text())
    training = json.loads(TRAINING.read_text())
    training_binding = json.loads(TRAINING_BINDING.read_text())
    prior = json.loads(PR389_REG.read_text())
    prior_binding = json.loads(PR389_BINDING.read_text())
    if training_binding["registration_sha256"] != sha(BASE_REG):
        raise ValueError("source_registration_binding_mismatch")
    if training_binding["training_sha256"] != sha(TRAINING):
        raise ValueError("source_training_binding_mismatch")

    replay_paths = [Path(item["path"]) for item in base["training"]["replays"]]
    for replay_path, item in zip(
        replay_paths, base["training"]["replays"], strict=True
    ):
        if sha(replay_path) != item["sha256"]:
            raise ValueError(f"frozen_replay_hash_mismatch:{replay_path}")
    proof, excluded = population.build_proof(CANONICAL, EXPLORATORY, replay_paths)
    consumed = {}
    consumed_identities: set[str] = set()
    for name in CONSUMED_THROUGH_389:
        path = DATA / name
        expected_hash = (
            prior["evaluation"]["suite"]["sha256"]
            if name == "seed461-e3-e4-openings.jsonl"
            else None
        )
        if expected_hash is not None and sha(path) != expected_hash:
            raise ValueError("pr389_suite_hash_mismatch")
        rows = suites.load_suite_jsonl(str(path))
        identities = population.keys(rows)
        consumed_identities |= identities
        excluded |= identities
        consumed[name] = {
            "path": str(path),
            "sha256": sha(path),
            "unique_state_identities": len(identities),
        }

    suite_rows = population.select_holdout(excluded, seed=390)
    suite_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in suite_rows
    ).encode()
    suite_ids = population.keys(suite_rows)
    if len(suite_rows) != 256 or len(suite_ids) != 256 or suite_ids & excluded:
        raise ValueError("suite_overlap_or_count_failure")
    if any(
        row["pit_sum"] <= 32 or KalahGame.from_state(row["state"]).over()
        for row in suite_rows
    ):
        raise ValueError("suite_opening_ineligible")
    if SUITE.exists() and SUITE.read_bytes() != suite_bytes:
        raise ValueError("immutable_suite_conflict")
    if not SUITE.exists():
        SUITE.write_bytes(suite_bytes)

    proof["consumed_suites_through_389"] = consumed
    proof["exclusion_counts"]["consumed_suites_through_389"] = len(consumed_identities)
    proof["exclusion_counts"]["union"] = len(excluded)
    proof["suite_identity_overlap"] = {
        "suite_states": len(suite_ids),
        "excluded_states": len(excluded),
        "overlap": len(suite_ids & excluded),
        "proof": "canonical state SHA-256 identities; zero intersection",
    }

    source = {}
    for order in ORDERS:
        run = f"order_{order}_A"
        record = training["trajectories"][run]
        e4_path = (
            REPLAY_ROOT.parent.parent.parent
            / "seed461-cosine-lr-ablation"
            / "training"
            / run
            / "E4.npz"
        )
        expected = record["epochs"]["E4"]
        if sha(e4_path) != expected:
            raise ValueError(f"source_e4_identity_mismatch:{run}")
        history = record["history"]
        if len(history) != 4 or any(item["learning_rate"] != 0.001 for item in history):
            raise ValueError(f"source_constant_lr_e4_record_mismatch:{run}")
        source[str(order)] = {
            "run": run,
            "checkpoint": str(e4_path),
            "checkpoint_sha256": expected,
            "source_record_selected_sha256": record["selected_sha256"],
            "full_epochs_completed": record["metrics"]["full_epochs_completed"],
            "optimizer_updates": record["metrics"]["optimizer_updates"],
            "examples_sampled": record["metrics"]["examples_sampled"],
            "initialization_sha256": record["initialization_sha256"],
        }

    opponent = prior["evaluation"]["opponent_binding"]
    if prior_binding["opponent"] != opponent:
        raise ValueError("pr389_opponent_binding_mismatch")
    evaluation = prior["evaluation"]
    value = {
        "schema": "seed461-cross-order-e4-average-registration-v1",
        "status": "prospectively_registered_before_candidate_binding_and_games",
        "hypothesis": "The uniform arithmetic mean of independently ordered constant-LR A-arm E4 trajectories from one initializer improves strength while retaining four epochs.",
        "distinction_from_pr387_388": "Cross-order E4 mean, not within-trajectory E2-E4 averaging; #386, #388, and #389 rejections remain preserved.",
        "training": {
            "performed": False,
            "source_pr": 386,
            "source_registration_sha256": sha(BASE_REG),
            "training_record_sha256": sha(TRAINING),
            "training_binding_sha256": sha(TRAINING_BINDING),
            "initializer_sha256": source["38611"]["initialization_sha256"],
            "orders": list(ORDERS),
            "source_e4": source,
            "treatment": "P = 0.2 * sum(checkpoint[order_38611_A..order_38615_A, E4]); includes trunk, policy head, and value head; float64 accumulation then source dtype cast",
            "inference": "single network; no ensemble",
        },
        "candidates": {
            "P": "uniform mean of all five validated E4 checkpoints, coefficient 0.2 each",
            **{
                f"A{i}": f"individual order_{order}_A E4 checkpoint"
                for i, order in enumerate(ORDERS, 1)
            },
        },
        "evaluation": {
            "opponent": "exact frozen seed455 native runtime from PR #389",
            "opponent_binding": opponent,
            "runtime_contract": evaluation["runtime_contract"],
            "suite": {
                "path": str(SUITE),
                "sha256": sha(SUITE),
                "seed": 390,
                "openings": 256,
                "nonterminal": True,
                "active_pit_stones_gt": 32,
                "identity_overlap_proof": proof,
            },
            "games": 3072,
            "games_per_model": 512,
            "games_per_opening": 2,
            "simulations_per_side": 384,
            "c_puct": 1.25,
            "arena_seed": 390,
            "seed_contract": "azlite_eval_seed_v2",
            "outcome_dependent_extensions": False,
        },
        "analysis": {
            "primary_effect": "mean opening score of P minus mean opening score across A1-A5",
            "bootstrap": {
                "resamples": 10000,
                "seed": 390,
                "cluster": "shared opening",
                "interval": "95% percentile",
            },
            "decision_thresholds": {
                "primary_mean_improvement_at_least": 0.03,
                "primary_interval_lower_strictly_above": 0.0,
                "P_absolute_score_interval_lower_strictly_above": 0.5,
                "P_score_at_least_baselines": 4,
                "worst_P_minus_Ai_at_least": -0.05,
            },
            "inference_scope": "conditional on these five source trajectories and this dataset; no inference to order stability",
            "advance": "independent confirmation only when every registered criterion passes",
            "preserve_rejections": [
                "PR #386 cosine-LR rejection",
                "PR #388 within-trajectory averaging rejection",
                "PR #389 E3 versus E4 rejection",
            ],
            "promotion": "never; no production-selection changes",
        },
        "costs_and_limitations": {
            "additional_training_runs": 0,
            "training_runs_required_for_future_construction": 5,
            "arena_games": 3072,
            "scope": "one constructed model, not five independent treatment models",
        },
    }
    if REG.exists() and json.loads(REG.read_text()) != value:
        raise ValueError("immutable_registration_conflict")
    write_json(REG, value)
    print(f"registration_sha256={sha(REG)}")
    print(f"suite_sha256={sha(SUITE)}")
    print(f"zero_overlap={proof['suite_identity_overlap']['overlap']}")


if __name__ == "__main__":
    register()
