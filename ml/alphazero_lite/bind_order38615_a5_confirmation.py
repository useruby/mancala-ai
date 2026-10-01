"""Bind the registered frozen candidate and exact evaluation identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.order38615_confirmation_validation import (
    sha256_file,
    validate_candidate_source,
    validate_confirmation_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "order38615-a5-confirmation-registration.json"
CANDIDATE_BINDING = DATA / "order38615-a5-confirmation-candidate-binding.json"
EVALUATION_BINDING = DATA / "order38615-a5-confirmation-evaluation-binding.json"


def sha(path: Path) -> str:
    return sha256_file(path)


def save(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"immutable_binding_conflict:{path.name}")
    if not path.exists():
        path.write_text(text)


def bind() -> None:
    reg = json.loads(REG.read_text())
    ev = reg["evaluation"]
    validate_candidate_source(REG, reg)
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
    if EVALUATION_BINDING.exists():
        evaluation = json.loads(EVALUATION_BINDING.read_text())
    else:
        save(EVALUATION_BINDING, evaluation)
    validate_confirmation_inputs(
        REG,
        CANDIDATE_BINDING,
        EVALUATION_BINDING,
        reg,
        json.loads(CANDIDATE_BINDING.read_text()),
        json.loads(EVALUATION_BINDING.read_text()),
    )
    print(f"candidate_binding_sha256={sha(CANDIDATE_BINDING)}")
    print(f"evaluation_binding_sha256={sha(EVALUATION_BINDING)}")


if __name__ == "__main__":
    bind()
