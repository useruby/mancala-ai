"""Read-only verifier for the retrospective seed414 coverage correction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ml.alphazero_lite import seed414_post_execution_audit as audit


def verify() -> dict[str, Any]:
    inputs = (
        audit.ORIGINAL_REGISTRATION,
        audit.ORIGINAL_PROOF,
        audit.LEDGER,
        audit.LEDGER_BINDING,
        audit.LEDGER_ANALYSIS,
        audit.SUITE,
    )
    before = {path: path.read_bytes() for path in inputs}
    proof, receipt = audit.build_audit()
    corrected = json.loads(audit.CORRECTED.read_text())
    published_receipt = json.loads(audit.RECEIPT.read_text())
    if corrected != proof or published_receipt != receipt:
        raise ValueError("post_execution_audit_publication_mismatch")
    if (
        receipt["audit_timing"] != "after_execution"
        or receipt["preregistration_exclusion_proof_was_complete"]
    ):
        raise ValueError("post_execution_caveat_missing")
    expected = hashlib.sha256(
        (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if receipt["bindings"]["corrected_proof_sha256"] != expected:
        raise ValueError("corrected_proof_receipt_binding_mismatch")
    if {path: path.read_bytes() for path in inputs} != before:
        raise ValueError("audit_verifier_mutated_source_evidence")
    return {
        "status": "verified",
        "games": receipt["counts"]["games"],
        "additional_identities": receipt["counts"]["additional_identities"],
        "corrected_union": receipt["counts"]["corrected_union"],
        "frozen_suite_collisions": receipt["collision_report"]["collision_count"],
        "decision": receipt["decision"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
