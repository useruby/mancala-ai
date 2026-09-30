#!/usr/bin/env python3
"""Register and train the sealed seed461 primary-minibatch-order confirmation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_seed461_order_exploratory as exploratory
from ml.alphazero_lite import seed461_order_population as population

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / ".tmp/seed461-order-confirmation"
CANONICAL = ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl"
EXPLORATORY = (
    ROOT / "docs/data/seed461-batch-order-sensitivity-exploratory-openings.jsonl"
)
ORDERS = {"O0": None, **{f"T{i}": 38310 + i for i in range(1, 6)}}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def register(manifest_path: Path, suite_path: Path) -> None:
    if manifest_path.exists() or suite_path.exists():
        raise RuntimeError("confirmation_registration_already_exists")
    replay_paths, replays, parent = exploratory.inputs()
    for path, record in zip(replay_paths, replays, strict=True):
        if not path.is_file() or exploratory.sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen_replay_mismatch:{path}")
    if not parent.is_file():
        raise RuntimeError("frozen_parent_missing")
    proof, excluded = population.build_proof(CANONICAL, EXPLORATORY, replay_paths)
    selected = population.select_holdout(excluded)
    suites.write_suite_jsonl(selected, str(suite_path))
    identities = population.keys(selected)
    if identities & excluded:
        raise RuntimeError("registered_suite_overlaps_exclusion_union")
    proof["holdout"] = {
        "seed": 383,
        "path": str(suite_path),
        "sha256": exploratory.sha256(suite_path),
        "openings": len(selected),
        "unique_identities": len(identities),
        "nonterminal": True,
        "active_pit_stones_gt": 32,
        "overlap_with_exclusion_union": 0,
    }
    write_json(
        manifest_path,
        {
            "schema": "seed461-batch-order-sensitivity-confirmation-v1",
            "status": "registered_before_training",
            "population_coverage": proof,
            "training": {
                "seed": 461,
                "parent": str(parent),
                "parent_sha256": exploratory.sha256(parent),
                "replays": replays,
                "orders": ORDERS,
                "expected_o0": exploratory.EXPECTED_O0,
                "epochs": 4,
                "batch_size": 512,
                "lr": 0.001,
                "validation_split": 0.1,
                "selection": "best_validation",
            },
            "evaluation": {
                "opponent": "frozen_seed455",
                "simulations_per_side": 384,
                "both_seats_per_opening": True,
                "games": 6144,
                "extensions": "forbidden",
            },
            "analysis": {
                "primary": "five paired E4-minus-O0 contrasts",
                "bootstrap_samples": 10000,
                "bootstrap_seed": 383,
                "interval": "Bonferroni-adjusted 99%",
            },
        },
    )


def train_all(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "registered_before_training":
        raise RuntimeError("confirmation_registration_status_invalid")
    spec = manifest["training"]
    if exploratory.sha256(Path(spec["parent"])) != spec["parent_sha256"]:
        raise RuntimeError("frozen_parent_mismatch")
    for record in spec["replays"]:
        path = Path(record["path"])
        if not path.is_file() or exploratory.sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen_replay_mismatch:{path}")
    if (
        exploratory.sha256(Path(manifest["population_coverage"]["holdout"]["path"]))
        != manifest["population_coverage"]["holdout"]["sha256"]
    ):
        raise RuntimeError("confirmation_suite_mismatch")
    exploratory.WORKDIR = WORKDIR
    started = time.monotonic()
    results: dict[str, Any] = {
        "manifest_sha256": exploratory.sha256(manifest_path),
        "trajectories": {},
    }
    reference = None
    for label in ORDERS:
        result = exploratory.train_one(manifest, label)
        if label == "O0" and result["epochs"] != spec["expected_o0"]:
            raise RuntimeError("o0_reproduction_drift")
        if reference is not None:
            for key in ("optimizer_updates", "train_split_count", "validation_count"):
                if result["metrics"][key] != reference["metrics"][key]:
                    raise RuntimeError(f"trajectory_contract_drift:{label}:{key}")
        reference = result if reference is None else reference
        results["trajectories"][label] = result
    results["wall_clock_seconds"] = time.monotonic() - started
    write_json(WORKDIR / "training.json", results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "train"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args()
    if args.stage == "register":
        if args.suite is None:
            parser.error("register requires --suite")
        register(args.manifest, args.suite)
    else:
        train_all(args.manifest)


if __name__ == "__main__":
    main()
