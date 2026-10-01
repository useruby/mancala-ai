#!/usr/bin/env python3
"""Preregister and execute the paired fixed-E4 cosine-LR ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_seed461_lr_sensitivity as prior
from ml.alphazero_lite import run_seed461_order_exploratory as frozen
from ml.alphazero_lite import seed461_order_population as population

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
WORK = ROOT / ".tmp/seed461-cosine-lr-ablation"
REG = DATA / "seed461-cosine-lr-ablation-registration.json"
SUITE = DATA / "seed461-cosine-lr-ablation-openings.jsonl"
ORDERS = tuple(range(38611, 38616))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def register() -> None:
    if REG.exists() or SUITE.exists():
        raise RuntimeError("registration_already_exists")
    replay_paths, replay_records, parent = frozen.inputs()
    for path, record in zip(replay_paths, replay_records, strict=True):
        if not path.is_file() or sha(path) != record["sha256"]:
            raise RuntimeError(f"frozen_replay_mismatch:{path}")
    proof, excluded = population.build_proof(
        ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl",
        DATA / "seed461-batch-order-sensitivity-exploratory-openings.jsonl",
        replay_paths,
    )
    consumed = {
        "pr383": DATA / "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
        "pr384_385": DATA / "seed461-lr-sensitivity-openings-v2.jsonl",
    }
    for name, path in consumed.items():
        keys = population.keys(suites.load_suite_jsonl(str(path)))
        excluded |= keys
        proof.setdefault("consumed_suites", {})[name] = {
            "path": str(path),
            "sha256": sha(path),
            "unique_state_identities": len(keys),
        }
    proof["exclusion_counts"] = {
        **proof["exclusion_counts"],
        "consumed_pr383_pr384_pr385": sum(
            row["unique_state_identities"] for row in proof["consumed_suites"].values()
        ),
        "union": len(excluded),
    }
    selected = population.select_holdout(excluded, seed=386)
    suites.write_suite_jsonl(selected, str(SUITE))
    identities = population.keys(selected)
    if len(identities) != 256 or identities & excluded:
        raise RuntimeError("holdout_exclusion_failure")
    runtime = prior.resolve_strength_comparison_runtime_contract(
        current_artifact=ROOT / ".tmp/seed461-order-confirmation/opponent-artifact",
        challenger_artifact=ROOT / ".tmp/seed461-order-confirmation/artifacts/T1-E4",
    )
    if runtime is None:
        raise RuntimeError("frozen_runtime_preflight_failed")
    write_json(
        REG,
        {
            "schema": "seed461-cosine-lr-ablation-v1",
            "status": "registered_before_training",
            "created_before_training": True,
            "population_coverage": proof,
            "holdout": {
                "path": str(SUITE),
                "sha256": sha(SUITE),
                "seed": 386,
                "openings": 256,
                "unique_identities": 256,
                "nonterminal": True,
                "active_pit_stones_gt": 32,
                "excluded_identity_overlap": 0,
            },
            "training": {
                "seed": 461,
                "parent": str(parent),
                "parent_sha256": sha(parent),
                "replays": replay_records,
                "orders": list(ORDERS),
                "architecture": {"input": [96, 3], "model": "residual_v3"},
                "epochs": 4,
                "batch_size": 512,
                "grad_clip": 1.0,
                "value_loss": "huber",
                "huber_delta": 1.0,
                "value_loss_weight": 0.3,
                "policy_target_mode": "sharpened",
                "validation_split": 0.1,
                "split_seed": 461,
                "arms": {
                    "A": {"lr": 0.001, "lr_scheduler": "none"},
                    "B": {
                        "lr": 0.001,
                        "lr_scheduler": "cosine",
                        "T_max": 4,
                        "eta_min": 0.0,
                    },
                },
                "selection": "best_validation; descriptive only",
            },
            "evaluation": {
                "primary_checkpoint": "E4",
                "opponent": "frozen seed455 native runtime from PR #384",
                "runtime_contract": runtime,
                "simulations_per_side": 384,
                "c_puct": 1.25,
                "arena_seed": 386,
                "seed_contract": "azlite_eval_seed_v2",
                "games": 5120,
                "outcome_dependent_extensions": False,
            },
            "analysis": {
                "bootstrap_samples": 10000,
                "bootstrap_seed": 386,
                "cluster": "shared opening",
                "confidence": 0.95,
                "inference": "conditional on these five orders and frozen dataset",
            },
        },
    )


def train_all() -> None:
    registration = json.loads(REG.read_text())
    if (
        registration["status"] != "registered_before_training"
        or sha(SUITE) != registration["holdout"]["sha256"]
    ):
        raise RuntimeError("registration_or_suite_identity_failure")
    frozen.WORKDIR = WORK
    trajectories: dict[str, Any] = {}
    for seed in ORDERS:
        for arm, scheduler in (("A", "none"), ("B", "cosine")):
            run = f"order_{seed}_{arm}"
            manifest = {
                "training": {
                    "seed": 461,
                    "parent": registration["training"]["parent"],
                    "replays": registration["training"]["replays"],
                    "orders": {run: seed},
                }
            }
            trajectories[run] = frozen.train_one(
                manifest, run, lr=0.001, lr_scheduler=scheduler
            )
        a, b = trajectories[f"order_{seed}_A"], trajectories[f"order_{seed}_B"]
        if a["permutation_sha256"] != b["permutation_sha256"]:
            raise RuntimeError(f"paired_permutation_mismatch:{seed}")
        for key in (
            "optimizer_updates",
            "train_split_count",
            "validation_count",
            "train_split_sha256",
            "validation_split_sha256",
        ):
            if a["metrics"][key] != b["metrics"][key]:
                raise RuntimeError(f"paired_invariant_mismatch:{seed}:{key}")
        for key in (
            "initialization_sha256",
            "replay_multiplicity_sha256",
            "permutation_sha256",
        ):
            if a[key] != b[key]:
                raise RuntimeError(f"paired_invariant_mismatch:{seed}:{key}")
        with (
            np.load(WORK / "training" / f"order_{seed}_A" / "E1.npz") as a_e1,
            np.load(WORK / "training" / f"order_{seed}_B" / "E1.npz") as b_e1,
        ):
            if a_e1.files != b_e1.files or any(
                not np.array_equal(a_e1[key], b_e1[key]) for key in a_e1.files
            ):
                raise RuntimeError(f"paired_e1_tensor_mismatch:{seed}")
        expected_lrs = [0.001, 0.0008535533905932737, 0.0005, 0.00014644660940672628]
        actual = [row["learning_rate"] for row in b["history"]]
        if not np.allclose(actual, expected_lrs, rtol=1e-8, atol=1e-12):
            raise RuntimeError(f"cosine_learning_rate_mismatch:{seed}:{actual}")
    record = {"registration_sha256": sha(REG), "trajectories": trajectories}
    write_json(WORK / "training.json", record)
    write_json(
        DATA / "seed461-cosine-lr-ablation-training-binding.json",
        {
            "registration_sha256": sha(REG),
            "training_sha256": sha(WORK / "training.json"),
            "trajectories": {
                key: {
                    "epochs": value["epochs"],
                    "selected_sha256": value["selected_sha256"],
                }
                for key, value in trajectories.items()
            },
        },
    )


def publish_training_record() -> None:
    record_path = WORK / "training.json"
    record = json.loads(record_path.read_text())
    binding = json.loads(
        (DATA / "seed461-cosine-lr-ablation-training-binding.json").read_text()
    )
    if (
        sha(REG) != binding["registration_sha256"]
        or sha(record_path) != binding["training_sha256"]
    ):
        raise RuntimeError("training_record_binding_mismatch")
    published = DATA / "seed461-cosine-lr-ablation-training.json"
    write_json(published, record)
    binding["published_training_record"] = {
        "path": str(published.relative_to(ROOT)),
        "sha256": sha(published),
    }
    write_json(DATA / "seed461-cosine-lr-ablation-training-binding.json", binding)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "train", "publish-training"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "train":
        train_all()
    else:
        publish_training_record()


if __name__ == "__main__":
    main()
