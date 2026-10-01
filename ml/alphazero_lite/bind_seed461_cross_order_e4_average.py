"""Build and immutably bind the averaged model and reused E4 baselines."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.checkpoint_average import average_checkpoints, load_npz
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-cross-order-e4-average-registration.json"
SUITE = DATA / "seed461-cross-order-e4-average-openings.jsonl"
SOURCE = DATA / "seed461-e2-e4-average-candidate-binding.json"
DEST = DATA / "seed461-cross-order-e4-average-candidate-binding.json"
WORK = ROOT / ".tmp/seed461-cross-order-e4-average"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
ORDERS = (38611, 38612, 38613, 38614, 38615)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def construct() -> None:
    registration = json.loads(REG.read_text())
    source_binding = json.loads(SOURCE.read_text())
    old_reg = DATA / "seed461-e2-e4-average-registration.json"
    old_binding = DATA / "seed461-e2-e4-average-evaluation-binding.json"
    previous = json.loads(DEST.read_text()) if DEST.exists() else None
    if source_binding.get("registration_sha256") != sha(old_reg):
        raise ValueError("reused_candidate_registration_mismatch")
    if (
        json.loads(old_binding.read_text())["opponent"]
        != registration["evaluation"]["opponent_binding"]
    ):
        raise ValueError("reused_candidate_opponent_mismatch")
    opponent = registration["evaluation"]["opponent_binding"]
    contract = registration["evaluation"]["runtime_contract"]
    validate_frozen_opponent_identity(OPPONENT, opponent, contract)

    checkpoint_paths = [
        Path(registration["training"]["source_e4"][str(order)]["checkpoint"])
        for order in ORDERS
    ]
    expected_hashes = [
        registration["training"]["source_e4"][str(order)]["checkpoint_sha256"]
        for order in ORDERS
    ]
    if [sha(path) for path in checkpoint_paths] != expected_hashes:
        raise ValueError("source_e4_checkpoint_hash_mismatch")

    averaged_checkpoint = WORK / "P" / "checkpoint.npz"
    artifact_p = WORK / "P" / "artifact"
    candidates: dict[str, Any] = {}
    if previous is not None:
        treatment = previous["candidates"]["P"]
        if sha(averaged_checkpoint) != treatment["checkpoint_sha256"]:
            raise ValueError("bound_treatment_checkpoint_changed")
        if any(
            sha(Path(treatment["artifact"]) / name) != digest
            for name, digest in treatment["artifact_sha256"].items()
        ):
            raise ValueError("bound_treatment_artifact_changed")
    else:
        averaged_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        averaged = average_checkpoints(
            [load_npz(str(path)) for path in checkpoint_paths]
        )
        np.savez(averaged_checkpoint, **averaged)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                "--checkpoint",
                str(averaged_checkpoint),
                "--out-dir",
                str(artifact_p),
                "--version",
                "seed461-cross-order-e4-average-P",
                "--model-type",
                "residual_v3",
                "--rules-version",
                "kalah_v1",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=ROOT,
            check=True,
        )
        sidecar = (
            Path(source_binding["candidates"][f"order_{ORDERS[0]}_A"]["artifact"])
            / "search_policy.json"
        )
        shutil.copy2(sidecar, artifact_p / "search_policy.json")
        if (artifact_p / "search_policy.json").read_bytes() != sidecar.read_bytes():
            raise ValueError("treatment_runtime_sidecar_copy_mismatch")

    # Reuse each already-bound A-arm E4 artifact from the prior evaluation.
    for index, order in enumerate(ORDERS, 1):
        old = source_binding["candidates"][f"order_{order}_A"]
        checkpoint = checkpoint_paths[index - 1]
        if (
            old["checkpoint_sha256"] != expected_hashes[index - 1]
            or sha(Path(old["checkpoint"])) != expected_hashes[index - 1]
        ):
            raise ValueError(f"baseline_checkpoint_binding_mismatch:A{index}")
        for name, digest in old["artifact_sha256"].items():
            if sha(Path(old["artifact"]) / name) != digest:
                raise ValueError(f"baseline_artifact_changed:A{index}:{name}")
        candidates[f"A{index}"] = {
            "artifact": old["artifact"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected_hashes[index - 1],
            "artifact_sha256": old["artifact_sha256"],
            "runtime_contract": contract,
            "source_run": f"order_{order}_A",
            "epoch": "E4",
        }

    if previous is None:
        hashes = {
            name: sha(artifact_p / name)
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
        if hashes["model.npz"] != sha(averaged_checkpoint):
            raise ValueError("treatment_model_checkpoint_mismatch")
        resolved = resolve_strength_comparison_runtime_contract(
            current_artifact=OPPONENT, challenger_artifact=artifact_p
        )
        if resolved != contract:
            raise ValueError("treatment_runtime_contract_mismatch")
        candidates["P"] = {
            "artifact": str(artifact_p),
            "checkpoint": str(averaged_checkpoint),
            "checkpoint_sha256": sha(averaged_checkpoint),
            "artifact_sha256": hashes,
            "runtime_contract": resolved,
            "coefficients": {str(order): 0.2 for order in ORDERS},
            "source_checkpoints": [str(path) for path in checkpoint_paths],
        }
    else:
        candidates["P"] = previous["candidates"]["P"]

    value = {
        "schema": "seed461-cross-order-e4-average-candidate-binding-v1",
        "registration_sha256": sha(REG),
        "suite_sha256": sha(SUITE),
        "source_training_sha256": registration["training"]["training_record_sha256"],
        "source_candidate_binding_sha256": sha(SOURCE),
        "opponent": opponent,
        "runtime_contract": contract,
        "candidates": candidates,
    }
    if previous is not None and previous != value:
        raise ValueError("immutable_candidate_binding_conflict")
    if previous is None:
        save(DEST, value)
    print(f"candidate_binding_sha256={sha(DEST)}")
    print(f"P_checkpoint_sha256={value['candidates']['P']['checkpoint_sha256']}")


if __name__ == "__main__":
    construct()
