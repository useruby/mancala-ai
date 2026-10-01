#!/usr/bin/env python3
"""Register the preregistered paired seed461 Adam learning-rate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_seed461_order_exploratory as frozen
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / ".tmp/seed461-lr-sensitivity"
MANIFEST = ROOT / "docs/data/seed461-lr-sensitivity-registration-v2.json"
SUITE = ROOT / "docs/data/seed461-lr-sensitivity-openings-v2.jsonl"
ORDER_SEEDS = tuple(range(38411, 38416))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def register() -> None:
    if MANIFEST.exists() or SUITE.exists():
        raise RuntimeError("registration_already_exists")
    replay_paths, replay_records, parent = frozen.inputs()
    for path, record in zip(replay_paths, replay_records, strict=True):
        if not path.is_file() or sha(path) != record["sha256"]:
            raise RuntimeError(f"frozen_replay_mismatch:{path}")
    proof, excluded = population.build_proof(
        ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl",
        ROOT / "docs/data/seed461-batch-order-sensitivity-exploratory-openings.jsonl",
        replay_paths,
    )
    pr383_suite = (
        ROOT / "docs/data/seed461-batch-order-sensitivity-confirmation-openings.jsonl"
    )
    pr383_keys = population.keys(suites.load_suite_jsonl(str(pr383_suite)))
    excluded |= pr383_keys
    proof["pr383_confirmation"] = {
        "path": str(pr383_suite),
        "sha256": sha(pr383_suite),
        "unique_state_identities": len(pr383_keys),
    }
    proof["exclusion_counts"]["pr383_confirmation"] = len(pr383_keys)
    proof["exclusion_counts"]["union"] = len(excluded)
    selected = population.select_holdout(excluded, seed=384)
    suites.write_suite_jsonl(selected, str(SUITE))
    if len(population.keys(selected)) != 256 or population.keys(selected) & excluded:
        raise RuntimeError("holdout_exclusion_failure")
    proof["holdout"] = {
        "path": str(SUITE),
        "sha256": sha(SUITE),
        "seed": 384,
        "openings": 256,
        "unique_identities": 256,
        "nonterminal": True,
        "active_pit_stones_gt": 32,
        "overlap_with_exclusion_union": 0,
    }
    runtime = resolve_strength_comparison_runtime_contract(
        current_artifact=ROOT / ".tmp/seed461-order-confirmation/opponent-artifact",
        challenger_artifact=ROOT / ".tmp/seed461-order-confirmation/artifacts/T1-E4",
    )
    if runtime is None or runtime.get("exact_root_solve_threshold") != 16:
        raise RuntimeError("frozen_runtime_preflight_failed")
    write_json(
        MANIFEST,
        {
            "schema": "seed461-lr-sensitivity-registration-v1",
            "status": "registered_before_training",
            "created_before_training": True,
            "population_coverage": proof,
            "training": {
                "seed": 461,
                "parent": str(parent),
                "parent_sha256": sha(parent),
                "replays": replay_records,
                "weights": [1, 4, 1, 8, 4],
                "value_target_modes": [
                    "default",
                    "sharpened",
                    "sharpened",
                    "sharpened",
                    "sharpened",
                ],
                "architecture": {"input": [96, 3], "model": "residual_v3"},
                "epochs": 4,
                "batch_size": 512,
                "grad_clip": 1.0,
                "value_loss": "huber",
                "huber_delta": 1.0,
                "value_loss_weight": 0.3,
                "policy_target_mode": "sharpened",
                "validation_split": 0.1,
                "save_top_k": 3,
                "scheduler": "none",
                "selection": "best_validation",
                "split_seed": 461,
                "orders": {
                    f"order_{seed}": {"order_seed": seed, "A_lr": 0.001, "B_lr": 0.0005}
                    for seed in ORDER_SEEDS
                },
                "pair_invariants": [
                    "initialization_hash",
                    "split_membership_hash",
                    "replay_multiplicity_hash",
                    "epoch_permutation_hashes",
                ],
            },
            "evaluation": {
                "opponent": "seed455_native_runtime_frozen",
                "opponent_artifact_binding": "PR #383 frozen seed455 native-runtime artifact",
                "runtime_contract": runtime,
                "simulations_per_side": 384,
                "suite_sha256": sha(SUITE),
                "openings": 256,
                "seats": [0, 1],
                "games": 5120,
                "outcome_dependent_extensions": False,
                "checkpoint": "production-selected best_validation independently per run",
            },
            "analysis": {
                "primary": "mean of five paired B-minus-A opening-score differences",
                "bootstrap": {
                    "samples": 10000,
                    "seed": 384,
                    "cluster": "opening; same resampled openings across all orders",
                    "interval": "95% percentile interval",
                },
                "decision": {
                    "advance": "mean>=0.03 and lower_95_bound>0 and >=4 pairs nonnegative and minimum pair>=-0.05",
                    "otherwise": "retain LR 0.001; classify rejection or inconclusive",
                    "stability_claim": "only if between-order spread decreases",
                },
                "inference_scope": "conditional on these five order seeds and frozen dataset",
            },
            "holdout_exclusions": [
                "conservative historical population",
                "frozen replay states",
                "PR #382 evaluation starts",
                "PR #383 evaluation starts",
            ],
            "limitations": [
                "no production adoption without another dataset/generation replication"
            ],
        },
    )


def train_all() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "registered_before_training" or not manifest.get(
        "created_before_training"
    ):
        raise RuntimeError("registration_status_invalid")
    if sha(SUITE) != manifest["evaluation"]["suite_sha256"]:
        raise RuntimeError("registered_suite_hash_mismatch")
    spec = manifest["training"]
    for row in spec["replays"]:
        if sha(Path(row["path"])) != row["sha256"]:
            raise RuntimeError("frozen_replay_mismatch")
    if sha(Path(spec["parent"])) != spec["parent_sha256"]:
        raise RuntimeError("frozen_parent_mismatch")
    frozen.WORKDIR = WORKDIR
    results: dict[str, Any] = {
        "registration_sha256": sha(MANIFEST),
        "trajectories": {},
        "treatments": {"A": {"lr": 0.001}, "B": {"lr": 0.0005}},
    }
    started = time.monotonic()
    for seed in ORDER_SEEDS:
        for arm, lr in (("A", 0.001), ("B", 0.0005)):
            label = f"order_{seed}_{arm}"
            run_manifest = {"training": {**spec, "orders": {label: seed}}}
            result = frozen.train_one(run_manifest, label, lr=lr)
            results["trajectories"][label] = result
        a = results["trajectories"][f"order_{seed}_A"]
        b = results["trajectories"][f"order_{seed}_B"]
        for key in ("permutation_sha256",):
            if a[key] != b[key]:
                raise RuntimeError(f"paired_order_invariant_failure:{seed}:{key}")
        if a["metrics"]["train_split_count"] != b["metrics"]["train_split_count"]:
            raise RuntimeError(f"paired_split_count_mismatch:{seed}")
    results["wall_clock_seconds"] = time.monotonic() - started
    record = WORKDIR / "training.json"
    write_json(record, results)
    write_json(
        ROOT / "docs/data/seed461-lr-sensitivity-training-amendment.json",
        {
            "schema": "seed461-lr-sensitivity-training-amendment-v1",
            "timing": "after_training_before_evaluation",
            "registration_sha256": sha(MANIFEST),
            "training_record_sha256": sha(record),
            "training_record": str(record),
            "checkpoints": {
                name: {
                    "epochs": item["epochs"],
                    "selected_sha256": item["selected_sha256"],
                }
                for name, item in results["trajectories"].items()
            },
        },
    )


def validate_runtime(manifest: dict[str, Any]) -> None:
    """Revalidate the entire frozen evaluation runtime before costly work."""
    evaluation = manifest["evaluation"]
    opponent = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    challenger = ROOT / ".tmp/seed461-order-confirmation/artifacts/T1-E4"
    registered = evaluation["runtime_contract"]
    runtime = resolve_strength_comparison_runtime_contract(
        current_artifact=opponent, challenger_artifact=challenger
    )
    if runtime != registered:
        raise RuntimeError("registered_runtime_contract_mismatch")
    binding_path = ROOT / "docs/data/seed461-lr-sensitivity-evaluation-binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        sha(binding_path)
        != "d1dcbbc861d26c121f2bb40d06d4a86b553e0fc26abcb16c4beb31aad180d91d"
    ):
        raise RuntimeError("frozen_evaluation_binding_hash_mismatch")
    for filename, key in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
    ):
        expected_hash = binding["opponent"][key]
        if sha(opponent / filename) != expected_hash:
            raise RuntimeError(f"frozen_opponent_identity_mismatch:{filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "validate-runtime", "train"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "train":
        train_all()
    else:
        if not MANIFEST.is_file() or not SUITE.is_file():
            raise RuntimeError("registration_missing")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if sha(SUITE) != manifest["evaluation"]["suite_sha256"]:
            raise RuntimeError("registered_suite_hash_mismatch")
        validate_runtime(manifest)
        print("runtime_and_suite_preflight_valid")


if __name__ == "__main__":
    main()
