"""Export E3 candidates and freeze all E3/E4 candidate identities."""

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
DATA = ROOT / "docs/data"
REG = DATA / "seed461-e3-e4-registration.json"
SUITE = DATA / "seed461-e3-e4-openings.jsonl"
SOURCE_BIND = DATA / "seed461-e2-e4-average-candidate-binding.json"
DEST = DATA / "seed461-e3-e4-candidate-binding.json"
WORK = ROOT / ".tmp/seed461-e3-e4"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
ORDERS = (38611, 38612, 38613, 38614, 38615)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def construct() -> None:
    registration = json.loads(REG.read_text())
    source_binding = json.loads(SOURCE_BIND.read_text())
    old_reg = DATA / "seed461-e2-e4-average-registration.json"
    old_eval = json.loads(
        (DATA / "seed461-e2-e4-average-evaluation-binding.json").read_text()
    )
    if source_binding["registration_sha256"] != sha(old_reg):
        raise ValueError("source_candidate_registration_mismatch")
    if sha(SUITE) != registration["evaluation"]["suite"]["sha256"]:
        raise ValueError("suite_hash_mismatch")
    frozen_runtime = registration["evaluation"]["runtime_contract"]
    frozen_opponent = registration["evaluation"]["opponent_binding"]
    if old_eval["opponent"] != frozen_opponent:
        raise ValueError("seed455_opponent_binding_mismatch")
    for filename, key in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "sidecar_sha256"),
    ):
        if sha(OPPONENT / filename) != frozen_opponent[key]:
            raise ValueError(f"opponent_artifact_hash_mismatch:{filename}")
    for path_key, hash_key in (
        ("exact_root_native_probe", "exact_root_native_probe_sha256"),
        ("exact_root_tablebase", "exact_root_tablebase_sha256"),
    ):
        path = Path(frozen_runtime[path_key])
        if (
            sha(path) != frozen_runtime[hash_key]
            or frozen_runtime[hash_key]
            != frozen_opponent[
                "native_probe_sha256" if "probe" in path_key else "tablebase_sha256"
            ]
        ):
            raise ValueError(f"runtime_identity_mismatch:{path_key}")

    binding: dict[str, Any] = {
        "schema": "seed461-e3-e4-candidate-binding-v1",
        "registration_sha256": sha(REG),
        "suite_sha256": sha(SUITE),
        "source_training_sha256": registration["training"]["training_record_sha256"],
        "opponent": frozen_opponent,
        "runtime_contract": frozen_runtime,
        "candidates": {},
    }
    prior = json.loads(DEST.read_text()) if DEST.exists() else None
    if prior and any(
        prior.get(key) != binding[key] for key in binding if key != "candidates"
    ):
        raise ValueError("immutable_candidate_binding_header_mismatch")
    for order in ORDERS:
        old_name = f"order_{order}_A"
        old_a = source_binding["candidates"][old_name]
        for epoch, arm in (("E4", "A"), ("E3", "B")):
            run = f"order_{order}_{arm}"
            checkpoint = Path(
                registration["training"]["source_paths"][str(order)][epoch]
            )
            checkpoint_hash = registration["training"]["source_checkpoint_sha256"][
                str(order)
            ][epoch]
            if sha(checkpoint) != checkpoint_hash:
                raise ValueError(f"source_checkpoint_hash_mismatch:{run}")
            if epoch == "E4":
                artifact = Path(old_a["artifact"])
                artifacts = old_a["artifact_sha256"]
                if (
                    sha(Path(old_a["checkpoint"])) != old_a["checkpoint_sha256"]
                    or old_a["checkpoint_sha256"] != checkpoint_hash
                ):
                    raise ValueError(f"reused_e4_checkpoint_identity_mismatch:{run}")
                for filename, expected in artifacts.items():
                    if sha(artifact / filename) != expected:
                        raise ValueError(
                            f"reused_e4_artifact_hash_mismatch:{run}:{filename}"
                        )
                contract = old_a["runtime_contract"]
            else:
                artifact = WORK / "artifacts" / run
                artifact.mkdir(parents=True, exist_ok=True)
                if not (artifact / "model.npz").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--out-dir",
                            str(artifact),
                            "--version",
                            f"seed461-e3-e4-{run}",
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
                    shutil.copy2(
                        Path(old_a["artifact"]) / "search_policy.json",
                        artifact / "search_policy.json",
                    )
                artifacts = {
                    name: sha(artifact / name)
                    for name in (
                        "model.npz",
                        "weights.json",
                        "metadata.json",
                        "search_policy.json",
                    )
                }
                if artifacts["model.npz"] != checkpoint_hash:
                    raise ValueError(f"candidate_model_checkpoint_mismatch:{run}")
                contract = resolve_strength_comparison_runtime_contract(
                    current_artifact=OPPONENT, challenger_artifact=artifact
                )
            if contract != frozen_runtime:
                raise ValueError(f"candidate_runtime_contract_mismatch:{run}")
            row = {
                "arm": arm,
                "epoch": epoch,
                "artifact": str(artifact),
                "artifact_sha256": artifacts,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "runtime_contract": contract,
                "source_checkpoint_sha256": checkpoint_hash,
            }
            if prior and prior["candidates"].get(run) != row:
                raise ValueError(f"immutable_candidate_identity_mismatch:{run}")
            binding["candidates"][run] = row
    if prior is not None and prior != binding:
        raise ValueError("immutable_candidate_binding_mismatch")
    if prior is None:
        save(DEST, binding)
    print(f"candidate_binding_sha256={sha(DEST)}")


if __name__ == "__main__":
    construct()
