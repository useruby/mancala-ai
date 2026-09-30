"""Export production-selected seed461 LR checkpoints and run the fixed arena."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".tmp/seed461-lr-sensitivity"
SUITE = ROOT / "docs/data/seed461-lr-sensitivity-openings-v2.jsonl"
REGISTRATION = ROOT / "docs/data/seed461-lr-sensitivity-registration-v2.json"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
RUNTIME_POLICY = ROOT / "model-artifact/current/search_policy.json"
ORDERS = range(38411, 38416)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def selected_epoch(result: dict[str, Any]) -> str:
    epoch = min(result["history"], key=lambda item: item["validation_total_loss"])[
        "epoch"
    ]
    name = f"E{epoch}"
    if result["selected_sha256"] != result["epochs"][name]:
        raise RuntimeError("best_validation_checkpoint_binding_failure")
    return name


def main() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    training_path = WORK / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    amendment_path = ROOT / "docs/data/seed461-lr-sensitivity-training-amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment["registration_sha256"] != sha(REGISTRATION):
        raise RuntimeError("training_amendment_registration_hash_mismatch")
    if amendment["training_record_sha256"] != sha(training_path):
        raise RuntimeError("training_amendment_record_hash_mismatch")
    if sha(SUITE) != registration["evaluation"]["suite_sha256"]:
        raise RuntimeError("registered_suite_hash_mismatch")
    if (
        sha(Path(registration["training"]["parent"]))
        != registration["training"]["parent_sha256"]
    ):
        raise RuntimeError("parent_hash_mismatch")
    runtime_contract = registration["evaluation"]["runtime_contract"]
    binding: dict[str, Any] = {
        "schema": "seed461-lr-sensitivity-evaluation-binding-v1",
        "registration_sha256": sha(REGISTRATION),
        "training_sha256": sha(training_path),
        "training_amendment_sha256": sha(amendment_path),
        "suite_sha256": sha(SUITE),
        "candidates": {},
    }
    for order in ORDERS:
        a = training["trajectories"][f"order_{order}_A"]
        b = training["trajectories"][f"order_{order}_B"]
        if a["permutation_sha256"] != b["permutation_sha256"]:
            raise RuntimeError(f"paired_permutation_mismatch:{order}")
        for arm, result in (("A", a), ("B", b)):
            run = f"order_{order}_{arm}"
            epoch = selected_epoch(result)
            checkpoint = WORK / "training" / run / "selected.npz"
            if sha(checkpoint) != result["selected_sha256"]:
                raise RuntimeError(f"selected_file_hash_mismatch:{run}")
            artifact = WORK / "artifacts" / run
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--out-dir",
                    str(artifact),
                    "--version",
                    f"seed461-lr-{run}",
                    "--model-type",
                    "residual_v3",
                    "--rules-version",
                    "kalah_v1",
                    "--input-encoding",
                    "kalah_v3",
                ],
                check=True,
                cwd=ROOT,
            )
            shutil.copy2(RUNTIME_POLICY, artifact / "search_policy.json")
            contract = resolve_strength_comparison_runtime_contract(
                current_artifact=OPPONENT, challenger_artifact=artifact
            )
            if contract != runtime_contract:
                raise RuntimeError(f"runtime_contract_mismatch:{run}")
            binding["candidates"][run] = {
                "arm": arm,
                "order_seed": order,
                "selected_epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
                "artifact": str(artifact),
                "model_sha256": sha(artifact / "model.npz"),
                "weights_sha256": sha(artifact / "weights.json"),
                "metadata_sha256": sha(artifact / "metadata.json"),
                "search_policy_sha256": sha(artifact / "search_policy.json"),
                "runtime_contract": contract,
            }
    binding["opponent"] = {
        "artifact": str(OPPONENT),
        "weights_sha256": sha(OPPONENT / "weights.json"),
        "metadata_sha256": sha(OPPONENT / "metadata.json"),
        "search_policy_sha256": sha(OPPONENT / "search_policy.json"),
    }
    write_json(
        ROOT / "docs/data/seed461-lr-sensitivity-evaluation-binding.json", binding
    )

    for run, candidate in binding["candidates"].items():
        report = WORK / "arena" / f"{run}.json"
        games = WORK / "arena" / f"{run}-games.jsonl"
        report.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                candidate["artifact"],
                "--current",
                str(OPPONENT),
                "--games",
                "512",
                "--games-per-opening",
                "2",
                "--opening-prefixes-jsonl",
                str(SUITE),
                "--suite-sha256",
                sha(SUITE),
                "--challenger-simulations",
                "384",
                "--current-simulations",
                "384",
                "--seed",
                "384",
                "--workers",
                "24",
                "--c-puct",
                "1.25",
                "--seed-contract",
                "azlite_eval_seed_v2",
                "--game-jsonl",
                str(games),
                "--out",
                str(report),
            ],
            check=True,
            cwd=ROOT,
        )
    binding["reports"] = {
        run: {
            "report": str(WORK / "arena" / f"{run}.json"),
            "report_sha256": sha(WORK / "arena" / f"{run}.json"),
            "games": str(WORK / "arena" / f"{run}-games.jsonl"),
            "games_sha256": sha(WORK / "arena" / f"{run}-games.jsonl"),
        }
        for run in binding["candidates"]
    }
    binding["status"] = "completed_fixed_5120_games"
    write_json(
        ROOT / "docs/data/seed461-lr-sensitivity-evaluation-binding.json", binding
    )


if __name__ == "__main__":
    main()
