"""Independent, copy-friendly verifier for Seed 629 supervisor result records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LIMITS = {
    "wall_seconds",
    "cpu_seconds",
    "process_rss_bytes",
    "execution_concurrency",
    "output_and_temporary_storage_bytes",
}
RESOURCE_CATEGORIES = {
    "cpu_exhaustion",
    "memory_oom",
    "memory_limit",
    "wall_time_expiration",
    "storage_exhaustion",
    "concurrency_rejection",
}
NON_RESOURCE_CATEGORIES = {"residual_descendants", "ordinary_worker_failure"}
TRUSTED_OS_EVIDENCE_FUTURE = (
    "independently collected raw cgroupfs control/stat/event files plus kernel identity, "
    "and filesystem-native quota records, captured by a separately privileged "
    "root-owned collector and authenticated outside the publisher"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(
    result_path: Path, freeze_path: Path, receipt_path: Path | None = None
) -> dict[str, Any]:
    """Verify exact freeze binding, measured ceilings, receipts, and artifacts."""
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    if result.get("schema") != "seed629-supervisor-execution-result-v1":
        raise ValueError("supervisor_result_schema_mismatch")
    if result.get("execution_freeze_sha256") != _sha(freeze_raw):
        raise ValueError("supervisor_freeze_digest_mismatch")
    if result.get("authorized_run_identity") != freeze.get("authorized_run_identity"):
        raise ValueError("supervisor_run_identity_mismatch")
    limits = freeze.get("production_resource_limits")
    if not isinstance(limits, dict) or set(LIMITS) - set(limits):
        raise ValueError("frozen_resource_limits_incomplete")
    if result.get("limits") != limits:
        raise ValueError("supervisor_limits_mismatch")
    evidence_digest = result.pop("evidence_sha256", None)
    if evidence_digest != _sha(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ):
        raise ValueError("supervisor_evidence_hash_mismatch")
    measurements = result.get("measurements", {})
    if (
        measurements.get("cpu_usage_usec", 0) > limits["cpu_seconds"] * 1_000_000
        and result.get("breach_category") != "cpu_exhaustion"
    ):
        raise ValueError("cpu_limit_misclassified")
    if (
        measurements.get("wall_seconds", 0) > limits["wall_seconds"]
        and result.get("breach_category") != "wall_time_expiration"
    ):
        raise ValueError("wall_limit_misclassified")
    if measurements.get("memory_peak_bytes", 0) > result.get("cgroup", {}).get(
        "memory_max", 0
    ):
        raise ValueError("cgroup_memory_ceiling_exceeded")
    events = measurements.get("memory_events", {})
    if events.get("oom_kill", 0) and result.get("breach_category") != "memory_oom":
        raise ValueError("oom_evidence_misclassified")
    category = result.get("breach_category")
    if category not in RESOURCE_CATEGORIES | NON_RESOURCE_CATEGORIES | {None} and not (
        isinstance(category, str)
        and category.startswith("external_termination_signal_")
    ):
        raise ValueError("unknown_resource_category")
    if result.get("breach_category") and result.get("status") != "failure":
        raise ValueError("resource_failure_claimed_as_success")
    resource_breach = result.get("breach_category") in RESOURCE_CATEGORIES
    if resource_breach and receipt_path is None:
        raise ValueError("frozen_failure_receipt_required_for_resource_termination")
    if receipt_path is not None:
        frozen_receipt_path = freeze.get("failure_receipt_contract", {}).get(
            "create_only_path"
        )
        if str(receipt_path) != frozen_receipt_path:
            raise ValueError("failure_receipt_path_not_frozen")
        receipt = json.loads(receipt_path.read_bytes())
        if resource_breach and (
            receipt.get("schema") != "seed624-authenticated-execution-failure-v1"
            or receipt.get("failure_category") != "external_resource_limit_termination"
            or receipt.get("execution_freeze_sha256") != _sha(freeze_raw)
            or receipt.get("authorized_run_identity")
            != result.get("authorized_run_identity")
        ):
            raise ValueError("failure_receipt_binding_mismatch")
        if _sha(receipt_path.read_bytes()) != result.get("receipt_sha256"):
            raise ValueError("failure_receipt_hash_mismatch")
    for artifact in result.get("partial_artifacts", []):
        path = Path(artifact)
        if path.exists() and path.is_symlink():
            raise ValueError("artifact_path_is_symlink")
    return {
        "status": "record_consistent_not_os_enforcement_verified",
        "verification_scope": "record-format-and-internal-consistency-only",
        "os_enforcement_verified": False,
        "trusted_os_evidence_required": TRUSTED_OS_EVIDENCE_FUTURE,
        "execution_freeze_sha256": _sha(freeze_raw),
        "authorized_run_identity": result["authorized_run_identity"],
        "breach_category": result.get("breach_category"),
        "evidence_sha256": evidence_digest,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.result, args.freeze, args.receipt), sort_keys=True, indent=2
        )
    )


if __name__ == "__main__":
    main()
