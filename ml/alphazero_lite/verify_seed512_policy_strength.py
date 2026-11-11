"""Read-only verifier for a #512 paired root-evaluation evidence file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed512_policy_strength import (
    ARTIFACT_HASHES,
    ROOT,
    reconstruct_gate,
    verify_freeze,
)
from ml.alphazero_lite.seed557_retrospective import canonical_hash, search_seed


def verify_evidence(root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    frozen = verify_freeze(root)
    if evidence.get("schema") != "seed512-policy-strength-evidence-v1":
        raise ValueError("evidence_schema_mismatch")
    if (
        evidence.get("freeze_sha256")
        != hashlib.sha256(
            (
                root / "docs/data/seed512-policy-strength-evaluation/freeze.json"
            ).read_bytes()
        ).hexdigest()
        or evidence.get("membership_sha256")
        != frozen["freeze"]["cohort"]["membership_sha256"]
        or evidence.get("artifact_hashes") != ARTIFACT_HASHES
    ):
        raise ValueError("evidence_freeze_binding_mismatch")
    paired = evidence.get("pairs")
    if not isinstance(paired, dict) or set(paired) != {"control", "treatment"}:
        raise ValueError("evidence_arm_set_mismatch")
    members = {row["state_hash"]: row for row in frozen["membership"]["roots"]}
    if set(paired["control"]) != set(members) or set(paired["treatment"]) != set(
        members
    ):
        raise ValueError("evidence_membership_or_pair_incomplete")
    oracle_path = root / frozen["freeze"]["cohort"]["oracle_records"]
    oracle_rows = {
        row["state_hash"]: row
        for row in (
            json.loads(line) for line in oracle_path.read_text().splitlines() if line
        )
    }
    for arm in ("control", "treatment"):
        for key, member in members.items():
            row = paired[arm][key]
            margins = {
                int(move): int(value)
                for move, value in oracle_rows[key]["action_margins"].items()
            }
            action = int(row["selected_action"])
            margin = margins[action]
            best = max(margins.values())
            wdl = (margin > 0) - (margin < 0)
            oracle_wdl = (best > 0) - (best < 0)
            if (
                int(row["opening_index"]) != int(member["opening_index"])
                or int(row["seed"]) != int(search_seed(canonical_hash(member["state"])))
                or action not in member["legal_moves"]
                or int(row["exact_final_margin"]) != margin
                or int(row["regret"]) != best - margin
                or int(row["wdl"]) != wdl
                or int(row["oracle_wdl"]) != oracle_wdl
            ):
                raise ValueError(f"invalid_root_evidence:{arm}:{key}")
    observed_gate = reconstruct_gate(paired)
    if evidence.get("gate") != observed_gate:
        raise ValueError("gate_reconstruction_mismatch")
    return {"status": "verified", "roots": len(members), "gate": observed_gate}


def verify_gate_only(evidence: dict[str, Any]) -> dict[str, Any]:
    """Read-only synthetic fixture path used to test gate reconstruction."""
    if evidence.get("schema") != "seed512-policy-strength-evidence-v1":
        raise ValueError("evidence_schema_mismatch")
    observed_gate = reconstruct_gate(evidence["pairs"])
    if evidence.get("gate") != observed_gate:
        raise ValueError("gate_reconstruction_mismatch")
    return observed_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify_evidence(args.root, json.loads(args.evidence.read_bytes()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
