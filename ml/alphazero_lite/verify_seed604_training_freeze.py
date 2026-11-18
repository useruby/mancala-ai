"""Independent, read-only verification and create-only publication of Seed 604 freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed604_training_executor as executor
from ml.alphazero_lite import seed604_replay_optimization as registered
from ml.alphazero_lite.verify_seed604_runtime_certification import (
    ARTIFACTS as CERTIFICATION_DIR,
    verify as verify_runtime_certification,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed604-replay-optimization")
FREEZE_PATH = DATA / "training-execution-freeze.json"
READINESS_PATH = DATA / "execution-readiness-evidence.json"
CERTIFICATION_FILES = (
    "README.md",
    "environment.json",
    "requirements-lock.txt",
    "runtime-identity.json",
    "synthetic-test-results.json",
    "sha256-manifest.json",
)
FREEZE_KEYS = {
    "schema",
    "status",
    "execution_authorized",
    "bindings",
    "runtime",
    "source_sha256",
    "output_contract",
    "b512_reproduction_gate",
}
GATE = {
    "stop_after_update": 512,
    "compare_every_model_tensor": True,
    "exact_names_shapes_dtypes_values": True,
    "require_complete_adam_state": True,
    "require_adam_step": 512,
    "continue_same_optimizer": True,
    "resume_at_update": 513,
    "fail_closed": True,
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _bytes(root: Path, path: Path) -> bytes:
    target = root / path
    if not target.is_file():
        raise ValueError(f"required_artifact_missing:{path}")
    return target.read_bytes()


def _sha(root: Path, path: Path) -> str:
    return digest(_bytes(root, path))


def independently_derived_contract(root: Path = ROOT) -> dict[str, Any]:
    """Derive the executor contract from registration and authenticated bytes."""
    root = root.resolve()
    preflight = registered.read_only_preflight(root)
    reg = json.loads(_bytes(root, DATA / "registration.json"))
    cohort = Path("docs/data/seed604-evaluation-opening-cohort-v5")
    schedule = registered.batch_schedule()
    bindings = {
        "seed604_registration_sha256": _sha(root, DATA / "registration.json"),
        "parent_checkpoint_sha256": preflight["authenticated_seed599_inputs"][
            "parent_checkpoint_sha256"
        ],
        "b512_checkpoint_sha256": preflight["authenticated_seed599_inputs"][
            "b512_checkpoint_sha256"
        ],
        "membership_sha256": preflight["authenticated_seed599_inputs"][
            "membership_sha256"
        ],
        "training_configuration_sha256": preflight["authenticated_seed599_inputs"][
            "training_configuration_sha256"
        ],
        "b512_schedule_sha256": preflight["authenticated_seed599_inputs"][
            "published_schedule_sha256"
        ],
        "b2048_schedule_sha256": digest(canonical_bytes(schedule)),
        "cohort_registration_sha256": _sha(root, cohort / "registration.json"),
        "cohort_manifest_sha256": _sha(root, cohort / "opening-manifest.json"),
        "cohort_verification_report_sha256": _sha(
            root, cohort / "verification-report.json"
        ),
        "cohort_runtime_provenance_sha256": _sha(
            root, cohort / "runtime-provenance.json"
        ),
    }
    expected_registration = reg["treatment"]["batch_schedule_sha256"]
    if bindings["b2048_schedule_sha256"] != expected_registration:
        raise ValueError("registered_B2048_schedule_mismatch")
    source_paths = sorted(executor.FREEZE_SOURCE_PATHS)
    source_sha256 = {path: _sha(root, Path(path)) for path in source_paths}
    derived = {
        "bindings": bindings,
        "runtime": dict(executor.RUNTIME),
        "source_sha256": source_sha256,
        "output_contract": {
            "directory": str(executor.OUTPUT),
            "create_only": True,
            "updates": registered.UPDATES,
            "result_schema": "seed604-b2048-training-result-v1",
        },
        "b512_reproduction_gate": dict(GATE),
    }
    # Cross-check against the certified executor contract without using it as the
    # derivation. This detects accidental divergence from the guarded loader.
    if executor.expected_freeze_contract(root) != derived:
        raise ValueError("independent_contract_disagrees_with_executor")
    return derived


def expected_freeze(root: Path = ROOT) -> dict[str, Any]:
    return {
        "schema": "seed604-training-execution-freeze-v1",
        "status": "frozen-awaiting-owner-authorization",
        "execution_authorized": False,
        **independently_derived_contract(root),
    }


def certification_evidence(root: Path = ROOT) -> dict[str, Any]:
    runtime_result = verify_runtime_certification()
    artifact_hashes = {
        name: _sha(root, CERTIFICATION_DIR / name) for name in CERTIFICATION_FILES
    }
    identity = json.loads(_bytes(root, CERTIFICATION_DIR / "runtime-identity.json"))
    source_inventory = identity["source_sha256"]
    return {
        "artifact_sha256": artifact_hashes,
        "verification_result": runtime_result,
        "certified_source_sha256": source_inventory,
    }


def expected_readiness(
    root: Path = ROOT, freeze_raw: bytes | None = None
) -> dict[str, Any]:
    if freeze_raw is None:
        freeze_raw = _bytes(root, FREEZE_PATH)
    freeze = json.loads(freeze_raw)
    certification = certification_evidence(root)
    return {
        "schema": "seed604-execution-readiness-evidence-v1",
        "purpose": "supplemental-to-frozen-executor-contract",
        "execution_freeze_sha256": digest(freeze_raw),
        "runtime_certification_v3": certification,
        "source_sha256": {
            "execution_freeze_source_sha256": freeze["source_sha256"],
            "runtime_certified_source_sha256": certification["certified_source_sha256"],
        },
        "execution_freeze_validation": {
            "passed": True,
            "freeze_sha256": digest(freeze_raw),
            "validator": "ml.alphazero_lite.verify_seed604_training_freeze",
        },
        "owner_authorization_present": False,
        "owner_authorization_statement": "Owner authorization remains absent; training is not authorized.",
        "executor_enforces_certification_link": False,
        "future_authorization_requirement": "Validate both this certification link and the execution freeze before owner authorization.",
    }


def _validate_freeze(raw: bytes, expected: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution_freeze_malformed") from error
    if not isinstance(value, dict) or set(value) != FREEZE_KEYS:
        raise ValueError("execution_freeze_fields_mismatch")
    if value.get("schema") != "seed604-training-execution-freeze-v1":
        raise ValueError("execution_freeze_schema_mismatch")
    if (
        value.get("status") != "frozen-awaiting-owner-authorization"
        or value.get("execution_authorized") is not False
    ):
        raise ValueError("execution_freeze_status_or_authorization_invalid")
    if raw != canonical_bytes(value) + b"\n":
        raise ValueError("execution_freeze_serialization_not_canonical")
    if value != expected:
        raise ValueError("execution_freeze_contract_mismatch")
    return value


def _validate_readiness(value: Any, expected: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value != expected:
        raise ValueError("execution_readiness_binding_mismatch")


def _require_preparation_only(root: Path) -> None:
    if (root / executor.AUTHORIZATION).exists():
        raise ValueError("training_authorization_artifact_exists")
    if (root / executor.OUTPUT).exists():
        raise ValueError("production_training_output_exists")


def publish(root: Path = ROOT) -> None:
    """Publish freeze and supplemental link with exclusive, non-overwriting writes."""
    root = root.resolve()
    freeze_path, readiness_path = root / FREEZE_PATH, root / READINESS_PATH
    if freeze_path.exists() or readiness_path.exists():
        raise FileExistsError("execution_freeze_or_readiness_destination_exists")
    raw = canonical_bytes(expected_freeze(root)) + b"\n"
    _validate_freeze(raw, expected_freeze(root))
    readiness = expected_readiness(root, raw)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("xb") as file:
        file.write(raw)
    # If the second exclusive create fails, preserve the first artifact and fail
    # closed. Never delete or regenerate a published freeze automatically.
    with readiness_path.open("xb") as file:
        file.write(canonical_bytes(readiness) + b"\n")


def verify(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    raw = _bytes(root, FREEZE_PATH)
    freeze = _validate_freeze(raw, expected_freeze(root))
    loaded, loader_raw = executor.load_freeze(root)
    if loaded != freeze or loader_raw != raw:
        raise ValueError("certified_executor_load_freeze_mismatch")
    readiness_raw = _bytes(root, READINESS_PATH)
    try:
        readiness = json.loads(readiness_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution_readiness_malformed") from error
    _validate_readiness(readiness, expected_readiness(root, raw))
    runtime_result = verify_runtime_certification()
    if not runtime_result.get("passed"):
        raise ValueError("runtime_certification_not_verified")
    for relative, expected_hash in freeze["source_sha256"].items():
        if _sha(root, Path(relative)) != expected_hash:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    _require_preparation_only(root)
    return {
        "freeze_integrity_verified": True,
        "runtime_certification_bound": True,
        "training_authorized": False,
        "b512_reproduction_completed": False,
        "runtime_certification_verification": runtime_result,
    }


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
