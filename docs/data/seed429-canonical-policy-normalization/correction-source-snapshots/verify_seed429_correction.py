"""Supplemental verifier for the pre-arena seed429 launcher correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.verify_seed429_publication import verify as verify_publication


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_protocol_identity(
    receipt: dict[str, Any], registration: dict[str, Any]
) -> None:
    if receipt.get("protocol_identity") != registration.get("evaluation"):
        raise ValueError("correction_protocol_seed_or_search_change")
    if receipt.get("protocol_unchanged") is not True:
        raise ValueError("correction_protocol_not_declared_unchanged")


def verify_correction(root: Path, *, require_complete: bool = False) -> dict[str, Any]:
    root = root.resolve()
    data = root / "docs/data/seed429-canonical-policy-normalization"
    registration_path = data / "registration.json"
    binding_path = data / "runtime-binding.json"
    receipt_path = data / "arena-correction-receipt.json"
    for path in (registration_path, binding_path, receipt_path):
        if not path.is_file():
            raise ValueError(f"correction_chain_file_missing:{path.name}")
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    binding_sha = _sha(binding_path)
    if receipt.get("registration_sha256") != registration_sha:
        raise ValueError("correction_original_registration_mismatch")
    if receipt.get("runtime_binding_sha256") != binding_sha:
        raise ValueError("correction_runtime_binding_mismatch")
    suite_path = root / registration["suite_path"]
    if (
        receipt.get("suite_sha256") != registration["suite_sha256"]
        or _sha(suite_path) != registration["suite_sha256"]
    ):
        raise ValueError("correction_suite_identity_mismatch")
    if receipt.get("original_execution_source_snapshots") != registration.get(
        "execution_source_snapshots"
    ):
        raise ValueError("correction_original_snapshot_set_mismatch")
    for relative, expected in receipt["original_execution_source_snapshots"].items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"correction_original_snapshot_mismatch:{relative}")
    correction_hashes = {}
    for relative, expected in receipt["corrected_source_hashes"].items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"correction_source_mismatch:{relative}")
        snapshot = root / receipt["corrected_source_snapshots"][relative]
        if not snapshot.is_file() or _sha(snapshot) != expected:
            raise ValueError(f"correction_source_snapshot_mismatch:{relative}")
        correction_hashes[relative] = expected
    if receipt.get("games_completed_before_correction") != 0:
        raise ValueError("correction_not_prospective_to_arena")
    verify_protocol_identity(receipt, registration)
    if receipt.get("training_results_sha256") != _sha(data / "training-results.json"):
        raise ValueError("correction_training_binding_mismatch")
    training = json.loads((data / "training-results.json").read_text(encoding="utf-8"))
    for lane in ("A", "B"):
        for epoch, expected in receipt["training_lanes"][lane]["epoch_sha256"].items():
            checkpoint = (
                root / training["lanes"][lane]["checkpoint_dir"] / f"E{epoch}.npz"
            )
            if _sha(checkpoint) != expected:
                raise ValueError(f"correction_checkpoint_mismatch:{lane}:E{epoch}")
    for lane in ("A", "B"):
        candidate = receipt["candidate_artifacts"][lane]
        artifact = root / candidate["artifact"]
        for filename, expected in candidate["artifact_files"].items():
            if _sha(artifact / filename) != expected:
                raise ValueError(
                    f"correction_runtime_artifact_mismatch:{lane}:{filename}"
                )
    if (
        receipt.get("A_E4_sha256")
        != registration["training"]["expected_A_epoch_hashes"]["4"]
    ):
        raise ValueError("correction_A_E4_identity_mismatch")
    if receipt.get("protocol_identity") != registration["evaluation"]:
        raise ValueError("correction_protocol_seed_or_search_change")
    if binding.get("registration_sha256") != registration_sha:
        raise ValueError("correction_runtime_binding_registration_mismatch")
    if not require_complete:
        return {
            "valid": True,
            "complete": False,
            "registration_sha256": registration_sha,
            "correction_sources": correction_hashes,
        }
    result = verify_publication(root, require_complete=True)
    outcomes = json.loads((data / "outcome-binding.json").read_text(encoding="utf-8"))
    if outcomes.get("correction_receipt_sha256") != _sha(receipt_path):
        raise ValueError("correction_outcome_binding_mismatch")
    return {**result, "correction_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_correction(args.root, require_complete=args.require_complete),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
