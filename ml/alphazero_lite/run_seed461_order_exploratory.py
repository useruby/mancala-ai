#!/usr/bin/env python3
"""Register and train the non-confirmatory seed461 order experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import checkpoint_trajectory_diagnostic as trajectory
from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / ".tmp/seed461-order-exploratory"
GENERATION = (
    ROOT
    / "docs/data/alphazero-lite-generations/seed455-nextgen-s461-default-value-root16/generation.json"
)
REPLACEMENTS = ROOT / ".tmp/consumed-suite-recovery/replacement-registry.json"
ORDERS = {"O0": None, **{f"T{i}": 38210 + i for i in range(1, 6)}}
EXPECTED_O0 = {
    "E1": "643407f0b070acc1603e90e14e4ad8fd4579e4e34e3c109d9b0691f08267a1dc",
    "E2": "73914b801e61be7eada0366619fbc2defb23999fb163ef0da49134594ec44488",
    "E3": "9d3db177d22f12b0c8952233f3fc6c5dad5b0529f3990932699e8839afa0d2b8",
    "E4": "4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def inputs() -> tuple[list[Path], list[dict[str, Any]], Path]:
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    source_records = generation["replay"]["sources"]
    paths = [ROOT / row["artifact"]["path"] for row in source_records]
    records = [
        {
            "name": row["name"],
            "path": str(path),
            "sha256": row["artifact"]["sha256"],
            "weight": row["weight"],
            "value_target_mode": "default" if row["name"] == "fresh" else "sharpened",
        }
        for row, path in zip(source_records, paths)
    ]
    return paths, records, ROOT / generation["training"]["config"]["init_checkpoint"]


def register(manifest_path: Path, suite_path: Path) -> None:
    if manifest_path.exists() or suite_path.exists():
        raise RuntimeError("exploratory_registration_already_exists")
    if not REPLACEMENTS.is_file():
        raise RuntimeError("replacement_registry_missing")
    replacements = json.loads(REPLACEMENTS.read_text(encoding="utf-8"))
    if replacements["status"] != "exploratory_only_not_historical_registry":
        raise RuntimeError("replacement_registry_status_invalid")
    paths, records, parent = inputs()
    for path, record in zip(paths, records):
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen_replay_mismatch:{path}")
    replacement_keys = set()
    for entry in replacements["entries"].values():
        path = Path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"replacement_suite_mismatch:{path}")
        replacement_keys |= {
            suites.canonical_key(row["state"])
            for row in suites.load_suite_jsonl(str(path))
        }
    training_keys = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    training_keys.add(
                        suites.canonical_key(
                            trajectory.state_from_row(json.loads(line))
                        )
                    )
    prefixes = suites.enumerate_legal_prefixes(8)
    population, _, _ = suites.deduplicate_openings(prefixes)
    selected = suites.select_diverse(
        [
            row
            for row in suites.stratify_openings(population)
            if suites.canonical_key(row["state"]) not in replacement_keys
            and suites.canonical_key(row["state"]) not in training_keys
        ],
        256,
        382,
    )
    suites.write_suite_jsonl(selected, str(suite_path))
    identities = [suites.canonical_key(row["state"]) for row in selected]
    if len(set(identities)) != 256:
        raise RuntimeError("exploratory_suite_not_unique")
    write_json(
        manifest_path,
        {
            "schema": "seed461-batch-order-sensitivity-exploratory-v1",
            "status": "exploratory_not_sealed_v2_confirmation",
            "limitation": "Historical consumed suites A-AP were replaced after their original replay exclusions were lost.",
            "replacement_registry": {
                "path": str(REPLACEMENTS),
                "sha256": sha256(REPLACEMENTS),
            },
            "suite": {
                "path": str(suite_path),
                "sha256": sha256(suite_path),
                "identities": identities,
            },
            "training": {
                "seed": 461,
                "parent": str(parent),
                "parent_sha256": sha256(parent),
                "replays": records,
                "orders": ORDERS,
                "expected_o0": EXPECTED_O0,
            },
        },
    )


def load_registered(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("exploratory_registration_missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "exploratory_not_sealed_v2_confirmation":
        raise RuntimeError("exploratory_registration_status_invalid")
    if sha256(Path(manifest["suite"]["path"])) != manifest["suite"]["sha256"]:
        raise RuntimeError("exploratory_suite_drift")
    return manifest


def train_one(
    manifest: dict[str, Any],
    label: str,
    lr: float = 0.001,
    lr_scheduler: str = "none",
) -> dict[str, Any]:
    spec = manifest["training"]
    paths = [Path(row["path"]) for row in spec["replays"]]
    train.set_seed(461)
    x, policy, value, replay, weights = train.load_jsonl_replay(
        paths,
        [row["weight"] for row in spec["replays"]],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[row["value_target_mode"] for row in spec["replays"]],
        include_policy_loss_weights=True,
    )
    model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(model, Path(spec["parent"]))
    initialization_hash = hashlib.sha256(
        json.dumps(
            {
                key: value.tolist()
                for key, value in train.checkpoint_from_model(model).items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    output = WORKDIR / "training" / label
    output.mkdir(parents=True, exist_ok=True)
    epochs: dict[str, str] = {}
    permutations: dict[str, str] = {}
    history: list[dict[str, Any]] = []

    def on_epoch(
        epoch: int, _optimizer: torch.optim.Optimizer, current: torch.nn.Module
    ) -> None:
        checkpoint = output / f"E{epoch}.npz"
        np.savez(checkpoint, **train.checkpoint_from_model(current))
        epochs[f"E{epoch}"] = sha256(checkpoint)

    train.train(
        model,
        x,
        policy,
        value,
        replay,
        policy_loss_weights=weights,
        epochs=4,
        batch_size=512,
        lr=lr,
        device=torch.device("cpu"),
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=1.0,
        save_top_k=3,
        lr_scheduler=lr_scheduler,
        final_checkpoint="best_validation",
        primary_order_seed=spec["orders"][label],
        epoch_history=history,
        epoch_callback=on_epoch,
        permutation_callback=lambda epoch, values: permutations.__setitem__(
            str(epoch), hashlib.sha256(json.dumps(values).encode()).hexdigest()
        ),
    )
    selected = output / "selected.npz"
    np.savez(selected, **train.checkpoint_from_model(model))
    return {
        "epochs": epochs,
        "selected_sha256": sha256(selected),
        "history": history,
        "permutation_sha256": permutations,
        "initialization_sha256": initialization_hash,
        "replay_multiplicity_sha256": hashlib.sha256(replay.tobytes()).hexdigest(),
        "metrics": model.last_train_metrics,
    }


def train_all(manifest_path: Path) -> None:
    manifest = load_registered(manifest_path)
    started = time.monotonic()
    results = {"manifest_sha256": sha256(manifest_path), "trajectories": {}}
    reference: dict[str, Any] | None = None
    for label in ORDERS:
        result = train_one(manifest, label)
        if label == "O0" and result["epochs"] != EXPECTED_O0:
            raise RuntimeError(f"o0_reproduction_drift:{result['epochs']}")
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
