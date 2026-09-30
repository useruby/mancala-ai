"""Bind hash-verified seed461 evaluation artifacts to the frozen runtime policy."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / ".tmp/seed461-order-confirmation"
SOURCE = (
    ROOT
    / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1"
)
CURRENT = ROOT / "model-artifact/current"
EXPECTED = {
    "checkpoint.npz": "c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1",
    "weights.json": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
    "metadata.json": "8b9f4d02395271accd5accb5cde20ec6ddc4ecd63ded192b0b84d9ecf6c649ac",
    "search_policy.json": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_copy(source: Path, target: Path, expected: str) -> None:
    if not source.is_file() or sha(source) != expected:
        raise RuntimeError(f"artifact_identity_mismatch:{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if sha(target) != expected:
        raise RuntimeError(f"artifact_copy_mismatch:{target}")


def main() -> None:
    opponent = WORKDIR / "opponent-artifact"
    checked_copy(
        SOURCE / "weights.json", opponent / "weights.json", EXPECTED["weights.json"]
    )
    checked_copy(
        SOURCE / "metadata.json", opponent / "metadata.json", EXPECTED["metadata.json"]
    )
    checked_copy(
        CURRENT / "search_policy.json",
        opponent / "search_policy.json",
        EXPECTED["search_policy.json"],
    )
    training = WORKDIR / "training.json"
    if not training.is_file():
        raise RuntimeError("completed_training_missing")
    rows = []
    for trajectory in ("O0", "T1", "T2", "T3", "T4", "T5"):
        for epoch in ("E1", "E4"):
            checkpoint = WORKDIR / "training" / trajectory / f"{epoch}.npz"
            artifact = WORKDIR / "artifacts" / f"{trajectory}-{epoch}"
            if not checkpoint.is_file() or sha(checkpoint) != sha(
                artifact / "model.npz"
            ):
                raise RuntimeError(f"exported_model_mismatch:{trajectory}:{epoch}")
            checked_copy(
                CURRENT / "search_policy.json",
                artifact / "search_policy.json",
                EXPECTED["search_policy.json"],
            )
            contract = resolve_strength_comparison_runtime_contract(
                current_artifact=opponent, challenger_artifact=artifact
            )
            if contract is None or contract["exact_root_solve_threshold"] != 16:
                raise RuntimeError(f"unbound_runtime_contract:{trajectory}:{epoch}")
            rows.append(
                {
                    "trajectory": trajectory,
                    "epoch": epoch,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha(checkpoint),
                    "artifact": str(artifact),
                    "model_sha256": sha(artifact / "model.npz"),
                    "weights_sha256": sha(artifact / "weights.json"),
                    "metadata_sha256": sha(artifact / "metadata.json"),
                    "runtime_contract": contract,
                }
            )
    result = {
        "schema": "seed461-order-candidate-artifact-binding-v1",
        "timing": "after_training_before_arena_games",
        "training_sha256": sha(training),
        "amendment_sha256": sha(
            ROOT
            / "docs/data/seed461-batch-order-sensitivity-evaluation-identity-amendment.json"
        ),
        "opponent_artifact": str(opponent),
        "candidates": rows,
    }
    out = (
        ROOT
        / "docs/data/seed461-batch-order-sensitivity-candidate-artifact-binding.json"
    )
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
