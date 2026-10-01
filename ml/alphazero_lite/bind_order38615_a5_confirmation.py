"""Bind the registered frozen candidate and exact evaluation identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "order38615-a5-confirmation-registration.json"
CANDIDATE_BINDING = DATA / "order38615-a5-confirmation-candidate-binding.json"
EVALUATION_BINDING = DATA / "order38615-a5-confirmation-evaluation-binding.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"immutable_binding_conflict:{path.name}")
    if not path.exists():
        path.write_text(text)


def bind() -> None:
    reg = json.loads(REG.read_text())
    ev = reg["evaluation"]
    source = json.loads(Path(ev["candidate_binding_source"]).read_text())
    source_row = source["candidates"][ev["candidate"]]
    artifact = Path(source_row["artifact"])
    checkpoint = Path(source_row["checkpoint"])
    if sha(checkpoint) != ev["candidate_checkpoint_sha256"]:
        raise ValueError("checkpoint_hash_mismatch")
    for filename, expected in source_row["artifact_sha256"].items():
        if sha(artifact / filename) != expected:
            raise ValueError(f"candidate_artifact_hash_mismatch:{filename}")
    if source_row["artifact_sha256"]["model.npz"] != sha(checkpoint):
        raise ValueError("candidate_model_checkpoint_mismatch")
    opponent = ev["opponent_binding"]
    validate_frozen_opponent_identity(
        Path(opponent["artifact"]), opponent, ev["runtime_contract"]
    )
    candidate = {
        "schema": "order38615-a5-confirmation-candidate-binding-v1",
        "registration_sha256": sha(REG),
        "source_binding_sha256": ev["candidate_binding_source_sha256"],
        "source_training_sha256": reg["training"]["training_record_sha256"],
        "candidate": source_row,
        "opponent": opponent,
        "runtime_contract": ev["runtime_contract"],
    }
    save(CANDIDATE_BINDING, candidate)
    evaluation = {
        "schema": "order38615-a5-confirmation-evaluation-binding-v1",
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": sha(CANDIDATE_BINDING),
        "opponent": opponent,
        "reports": {},
    }
    save(EVALUATION_BINDING, evaluation)
    print(f"candidate_binding_sha256={sha(CANDIDATE_BINDING)}")
    print(f"evaluation_binding_sha256={sha(EVALUATION_BINDING)}")


if __name__ == "__main__":
    bind()
