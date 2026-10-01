"""Execute the preregistered constant-LR E3 versus E4 arena."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.alphazero_lite import run_seed461_e2_e4_average_arena as arena

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"


def validate_inputs(reg: dict[str, Any], candidates: dict[str, Any]) -> None:
    if arena.sha(arena.SUITE) != reg["evaluation"]["suite"]["sha256"]:
        raise ValueError("registered_suite_hash_mismatch")
    if candidates.get("schema") != "seed461-e3-e4-candidate-binding-v1":
        raise ValueError("candidate_binding_schema_mismatch")
    for key, expected in (
        ("registration_sha256", arena.sha(arena.REG)),
        ("suite_sha256", arena.sha(arena.SUITE)),
        ("source_training_sha256", reg["training"]["training_record_sha256"]),
        ("opponent", reg["evaluation"]["opponent_binding"]),
    ):
        if candidates.get(key) != expected:
            raise ValueError(f"candidate_binding_mismatch:{key}")
    if candidates.get("runtime_contract") != reg["evaluation"]["runtime_contract"]:
        raise ValueError("candidate_runtime_binding_mismatch")
    expected = {
        f"order_{order}_{arm}"
        for order in reg["training"]["orders"]
        for arm in ("A", "B")
    }
    if set(candidates.get("candidates", {})) != expected:
        raise ValueError("candidate_set_mismatch")
    for run, row in candidates["candidates"].items():
        if row["runtime_contract"] != candidates["runtime_contract"]:
            raise ValueError(f"candidate_runtime_contract_mismatch:{run}")
        if arena.sha(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError(f"candidate_checkpoint_hash_mismatch:{run}")
        for name, expected_hash in row["artifact_sha256"].items():
            if arena.sha(Path(row["artifact"]) / name) != expected_hash:
                raise ValueError(f"candidate_artifact_hash_mismatch:{run}:{name}")
        if row["artifact_sha256"]["model.npz"] != row["checkpoint_sha256"]:
            raise ValueError(f"candidate_model_checkpoint_mismatch:{run}")
        actual = arena.resolve_strength_comparison_runtime_contract(
            current_artifact=arena.OPPONENT, challenger_artifact=Path(row["artifact"])
        )
        if actual != reg["evaluation"]["runtime_contract"]:
            raise ValueError(f"resolved_runtime_contract_mismatch:{run}")


def main() -> None:
    arena.REG = DATA / "seed461-e3-e4-registration.json"
    arena.SUITE = DATA / "seed461-e3-e4-openings.jsonl"
    arena.CANDIDATES = DATA / "seed461-e3-e4-candidate-binding.json"
    arena.BINDING = DATA / "seed461-e3-e4-evaluation-binding.json"
    arena.WORK = ROOT / ".tmp/seed461-e3-e4"
    arena.OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
    arena.validate_inputs = validate_inputs
    arena.main()


if __name__ == "__main__":
    main()
