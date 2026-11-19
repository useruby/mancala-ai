"""Independent Seed 615 freeze verifier and create-only publisher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed599_replay_coverage as seed599
from ml.alphazero_lite import seed615_opening_diversity as seed615
from ml.alphazero_lite import seed615_training_executor as executor
from ml.alphazero_lite.verify_seed604_runtime_certification import (
    verify as verify_seed604_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed615-source-opening-diversity")
FREEZE = DATA / "training-execution-freeze-v3.json"
READINESS = DATA / "training-readiness-evidence-v3.json"
LOCK = Path(
    "docs/data/seed604-replay-optimization/runtime-certification-v3/requirements-lock.txt"
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read(root: Path, path: Path) -> bytes:
    result = root / path
    if not result.is_file():
        raise ValueError(f"required_artifact_missing:{path}")
    return result.read_bytes()


def independently_expected_freeze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    seed599.read_only_preflight(root)
    preparation = seed615.verify(root)
    if not preparation["passed"] or preparation["training_or_evaluation_started"]:
        raise ValueError("seed615_preparation_not_verified")
    registration = json.loads(read(root, DATA / "registration.json"))
    config = json.loads(
        read(root, Path("docs/data/seed599-replay-coverage/training-B.json"))
    )
    schedule = json.loads(
        read(root, Path("docs/data/seed599-replay-coverage/batch-schedule-B.json"))
    )
    membership = [
        json.loads(line)
        for line in read(root, DATA / "candidate-membership.jsonl").splitlines()
        if line
    ]
    if (
        sha(canonical(membership)) != registration["treatment"]["membership_sha256"]
        or sha(canonical(schedule))
        != registration["treatment"]["batch_index_schedule_sha256"]
        or len(schedule) != registration["treatment"]["updates"]
        or config["rng_seed"] != 599
    ):
        raise ValueError("registered_training_inputs_mismatch")
    parent = Path(registration["treatment"]["initial_checkpoint"])
    control = Path(registration["control"]["checkpoint"])
    if (
        sha(read(root, parent))
        != registration["treatment"]["initial_checkpoint_sha256"]
    ):
        raise ValueError("parent_checkpoint_identity_mismatch")
    if sha(read(root, control)) != registration["control"]["checkpoint_sha256"]:
        raise ValueError("control_checkpoint_identity_mismatch")
    source_hashes = {
        relative: sha(read(root, Path(relative)))
        for relative in sorted(executor.SOURCE_PATHS)
    }
    return {
        "schema": "seed615-training-execution-freeze-v1",
        "status": "frozen-awaiting-owner-authorization",
        "training_authorized": False,
        "bindings": {
            "registration_sha256": sha(read(root, DATA / "registration.json")),
            "parent_checkpoint_sha256": registration["treatment"][
                "initial_checkpoint_sha256"
            ],
            "b512_checkpoint_sha256": registration["control"]["checkpoint_sha256"],
            "membership_sha256": registration["treatment"]["membership_sha256"],
            "batch_schedule_sha256": registration["treatment"][
                "batch_index_schedule_sha256"
            ],
            "heldout_provenance_sha256": seed615.HELDOUT_SHA256,
            "seed599_training_configuration_sha256": sha(
                read(root, Path("docs/data/seed599-replay-coverage/training-B.json"))
            ),
            "runtime_certification_sha256": sha(
                read(root, DATA / "training-runtime-certification-v3.json")
            ),
            "superseded_freeze_sha256": sha(
                read(root, DATA / "training-execution-freeze.json")
            ),
            "superseded_readiness_sha256": sha(
                read(root, DATA / "training-readiness-evidence.json")
            ),
            "superseded_v2_freeze_sha256": sha(
                read(root, DATA / "training-execution-freeze-v2.json")
            ),
            "superseded_v2_readiness_sha256": sha(
                read(root, DATA / "training-readiness-evidence-v2.json")
            ),
            "superseded_v2_runtime_certification_sha256": sha(
                read(root, DATA / "training-runtime-certification-v2.json")
            ),
            "schedule_rows": len(schedule),
            "membership_rows": len(membership),
        },
        "architecture": {
            "model_type": "residual_v3",
            "input_size": 27,
            "hidden_sizes": [96, 3],
        },
        "parameter_groups": {
            "trainable": list(executor.TRAINABLE),
            "frozen": list(executor.FROZEN),
        },
        "optimizer": {
            "name": "Adam",
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "scheduler": None,
        },
        "training": {
            "updates": 512,
            "batch_size": 32,
            "gradient_clip_norm": 1.0,
            "loss": "masked_policy_cross_entropy",
            "device": "cpu",
            "torch_threads": 1,
        },
        "rng": {
            "training_seed": 599,
            "semantics": "Seed 599 train.set_seed(599) before architecture construction; then load exact P-E4 state and use fresh Adam; frozen Seed 599 B schedule indexes sorted Seed 615 membership rows; no per-update randomness.",
        },
        "runtime": dict(executor.RUNTIME),
        "hardware_requirements": {
            "architecture": "x86_64",
            "device": "CPU only",
            "threads": 1,
            "cuda_execution": False,
        },
        "source_sha256": source_hashes,
        "dependency_lock": str(LOCK),
        "dependency_lock_sha256": sha(read(root, LOCK)),
        "output_contract": {
            "directory": str(executor.OUTPUT),
            "create_only": True,
            "updates": 512,
            "result_schema": "seed615-training-result-v1",
            "artifacts": [
                "training-trace.jsonl",
                "seed615-S615-update-512.npz",
                "training-receipt.json",
                "training-integrity-report.json",
            ],
            "final_checkpoint_only": True,
            "intermediate_checkpoint_selection": False,
        },
        "execution_policy": "Validate all inputs and exact owner authorization before creating output; one attempt only; no automatic retry or recovery; preserve partial output for forensic review.",
        "diagnostics": {
            "compare_final_tensors_with_b512": True,
            "report_equality_without_selection": True,
            "report_trainable_movement_from_parent_without_selection": True,
        },
    }


def expected_freeze(root: Path = ROOT) -> dict[str, Any]:
    return independently_expected_freeze(root)


def expected_readiness(root: Path, freeze_raw: bytes) -> dict[str, Any]:
    freeze = json.loads(freeze_raw)
    lock_raw = read(root, LOCK)
    runtime_result = verify_seed604_runtime()
    if not runtime_result.get("passed"):
        raise ValueError("reused_runtime_certification_failed")
    return {
        "schema": "seed615-training-readiness-evidence-v1",
        "execution_freeze_sha256": sha(freeze_raw),
        "runtime_certification": {
            "runtime_contract": freeze["runtime"],
            "dependency_lock_sha256": sha(lock_raw),
            "certificate_sha256": freeze["bindings"]["runtime_certification_sha256"],
            "seed604_runtime_verification": runtime_result,
            "reused_seed604_certification": "runtime identity, dependency wheel hashes, and deterministic CPU capability only; no Seed 615 code claims inherited",
            "runtime_identity_checked_in_fresh_process": True,
            "training_authorized": False,
        },
        "membership_verification": {
            "passed": True,
            "states": 2048,
            "sha256": freeze["bindings"]["membership_sha256"],
        },
        "schedule_verification": {
            "passed": True,
            "updates": 512,
            "sha256": freeze["bindings"]["batch_schedule_sha256"],
        },
        "owner_authorization_present": False,
        "training_authorized": False,
        "production_training_output_present": False,
        "evaluation_started": False,
        "source_sha256": freeze["source_sha256"],
    }


def verify(
    root: Path = ROOT, *, allow_execution_artifacts: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    raw = read(root, FREEZE)
    freeze = json.loads(raw)
    if raw != canonical(freeze) + b"\n" or freeze != expected_freeze(root):
        raise ValueError("execution_freeze_contract_mismatch")
    certificate = json.loads(
        read(root, DATA / "training-runtime-certification-v3.json")
    )
    manifest_path = Path(
        certificate.get("dependency_evidence", {}).get("seed604_runtime_manifest", "")
    )
    if (
        certificate.get("schema") != "seed615-training-runtime-certification-v1"
        or certificate.get("status") != "verified-for-deterministic-cpu-execution"
        or certificate.get("runtime") != freeze.get("runtime")
        or certificate.get("hardware_requirements")
        != freeze.get("hardware_requirements")
        or certificate.get("training_authorized") is not False
        or certificate.get("seed615_specific_checks", {}).get("source_sha256")
        != freeze.get("source_sha256")
        or certificate.get("dependency_evidence", {}).get("requirements_lock_sha256")
        != freeze.get("dependency_lock_sha256")
        or certificate.get("dependency_evidence", {})
        .get("verification", {})
        .get("passed")
        is not True
        or sha(read(root, manifest_path))
        != certificate.get("dependency_evidence", {}).get(
            "seed604_runtime_manifest_sha256"
        )
        or certificate.get("seed615_specific_checks", {})
        .get("deterministic_cpu_smoke_test", {})
        .get("bitwise_equal")
        is not True
    ):
        raise ValueError("seed615_runtime_certification_invalid")
    # The independent derivation and guarded executor must agree exactly.
    loaded, executor_raw = executor.load_freeze(root)
    if loaded != freeze or executor_raw != raw:
        raise ValueError("executor_freeze_verification_disagreement")
    readiness_raw = read(root, READINESS)
    readiness = json.loads(readiness_raw)
    if (
        readiness != expected_readiness(root, raw)
        or readiness_raw != canonical(readiness) + b"\n"
    ):
        raise ValueError("training_readiness_evidence_mismatch")
    authorization_exists = (root / executor.AUTHORIZATION).exists()
    output_exists = (root / executor.OUTPUT).exists()
    if allow_execution_artifacts:
        if not authorization_exists or not output_exists:
            raise ValueError("authorized_result_artifacts_missing")
    elif authorization_exists or output_exists:
        raise ValueError("unexpected_authorization_or_production_output")
    return {
        "passed": True,
        "freeze_sha256": sha(raw),
        "training_authorized": authorization_exists,
        "production_training_output_present": output_exists,
        "independent_contract_agrees_with_executor": True,
    }


def publish(root: Path = ROOT) -> None:
    root = root.resolve()
    freeze_path, readiness_path = root / FREEZE, root / READINESS
    if freeze_path.exists() or readiness_path.exists():
        raise FileExistsError("freeze_or_readiness_destination_exists")
    value = expected_freeze(root)
    raw = canonical(value) + b"\n"
    if value.get("training_authorized") is not False:
        raise ValueError("training_must_remain_unauthorized")
    readiness = expected_readiness(root, raw)
    with freeze_path.open("xb") as handle:
        handle.write(raw)
    with readiness_path.open("xb") as handle:
        handle.write(canonical(readiness) + b"\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish(args.root)
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
