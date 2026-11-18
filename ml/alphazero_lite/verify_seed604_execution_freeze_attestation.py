"""Read-only verifier and append-only publisher for the Seed 604 load receipt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed604_training_executor as executor
from ml.alphazero_lite import verify_seed604_training_freeze as freeze_verifier
from ml.alphazero_lite.verify_seed604_runtime_certification import (
    verify as verify_runtime_certification,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed604-replay-optimization")
FREEZE = DATA / "training-execution-freeze.json"
READINESS = DATA / "execution-readiness-evidence.json"
ATTESTATION = DATA / "execution-freeze-validation-attestation.json"
SOURCE_COMMIT = "cc5e72cdbbcbd7f25695335f313b4c060c8f93dd"
EXECUTOR_SHA256 = "6dedc41def9823fe79258ae3706b5bdf189db351aad299453c0dc89293a42ded"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read(root: Path, relative: Path) -> bytes:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"required_attestation_input_missing:{relative}")
    return path.read_bytes()


def _require_no_execution(root: Path) -> None:
    if (root / executor.AUTHORIZATION).exists():
        raise ValueError("training_authorization_artifact_exists")
    if (root / executor.OUTPUT).exists():
        raise ValueError("production_B2048_output_exists")


def validate_executor_source(raw: bytes) -> str:
    observed = sha256(raw)
    if observed != EXECUTOR_SHA256:
        raise ValueError("certified_executor_source_hash_mismatch")
    return observed


def _check_source_commit(root: Path) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def collect(root: Path = ROOT) -> dict[str, Any]:
    """Run all direct validators and return a deterministic receipt payload."""
    root = root.resolve()
    freeze_raw = _read(root, FREEZE)
    readiness_raw = _read(root, READINESS)
    _check_source_commit(root)
    executor_path = root / "ml/alphazero_lite/seed604_training_executor.py"
    executor_sha = validate_executor_source(executor_path.read_bytes())

    # This direct call to the pre-existing executor is the standalone load receipt.
    freeze, returned_bytes = executor.load_freeze(root)
    if returned_bytes != freeze_raw:
        raise ValueError("load_freeze_returned_different_bytes")
    freeze_result = {
        "passed": True,
        "freeze_sha256": sha256(returned_bytes),
        "schema": freeze["schema"],
        "status": freeze["status"],
        "execution_authorized": freeze["execution_authorized"],
        "registered_source_hash_count": len(freeze["source_sha256"]),
        "complete_registered_source_hash_set_passed": (
            set(freeze["source_sha256"]) == executor.FREEZE_SOURCE_PATHS
        ),
    }
    if not freeze_result["complete_registered_source_hash_set_passed"]:
        raise ValueError("registered_source_hash_set_incomplete")

    independent = freeze_verifier.verify(root)
    if (
        independent.get("freeze_integrity_verified") is not True
        or independent.get("runtime_certification_bound") is not True
        or independent.get("training_authorized") is not False
        or independent.get("b512_reproduction_completed") is not False
    ):
        raise ValueError("independent_freeze_verification_incomplete")
    runtime = verify_runtime_certification()
    if runtime.get("passed") is not True:
        raise ValueError("runtime_certification_verification_failed")
    _require_no_execution(root)

    return {
        "schema": "seed604-execution-freeze-validation-attestation-v1",
        "purpose": "append-only-validation-receipt; does not replace either bound evidence file",
        "execution_freeze_sha256": sha256(freeze_raw),
        "execution_readiness_evidence_sha256": sha256(readiness_raw),
        "source_commit": SOURCE_COMMIT,
        "certified_executor_source_sha256": executor_sha,
        "direct_load_freeze_validation": freeze_result,
        "independent_freeze_verifier": independent,
        "runtime_certification_independently_verified": runtime,
        "training_authorization_artifact_absent": True,
        "production_B2048_output_directory_absent": True,
        "b512_reproduction_occurred": False,
        "execution_freeze_integrity_is_not_execution_authorization": True,
    }


def validate_record(record: Any, expected: dict[str, Any]) -> None:
    if not isinstance(record, dict) or record != expected:
        raise ValueError("validation_attestation_mismatch")


def publish(root: Path = ROOT) -> None:
    """Create the attestation exactly once; existing artifacts are never replaced."""
    root = root.resolve()
    path = root / ATTESTATION
    if path.exists():
        raise FileExistsError("validation_attestation_destination_exists")
    payload = collect(root)
    raw = canonical_bytes(payload) + b"\n"
    with path.open("xb") as destination:
        destination.write(raw)


def verify(root: Path = ROOT) -> dict[str, Any]:
    """Recompute source/evidence identities and all validation results read-only."""
    root = root.resolve()
    raw = _read(root, ATTESTATION)
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("validation_attestation_malformed") from error
    if raw != canonical_bytes(record) + b"\n":
        raise ValueError("validation_attestation_serialization_not_canonical")
    expected = collect(root)
    validate_record(record, expected)
    return {
        "passed": True,
        "freeze_sha256": expected["execution_freeze_sha256"],
        "readiness_sha256": expected["execution_readiness_evidence_sha256"],
        "load_freeze_passed": True,
        "training_authorized": False,
        "b512_reproduction_occurred": False,
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
