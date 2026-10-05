#!/usr/bin/env python3
"""Read-only Torch/runtime-independent verifier for seed422 publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed422_adam_memory_analysis import analyze, validate_ledger
from ml.alphazero_lite.seed422_exclusions import ROOT, build_union, sha256

DATA = ROOT / "docs/data/seed422-adam-first-moment"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify() -> dict[str, Any]:
    registration_path = DATA / "registration.json"
    registration = read_json(registration_path)
    declared_registration_sha = registration.pop("registration_sha256")
    if canonical_sha(registration) != declared_registration_sha:
        raise ValueError("registration_canonical_hash_mismatch")
    registration["registration_sha256"] = declared_registration_sha

    proof_path = DATA / "opening-exclusion-proof.json"
    proof = read_json(proof_path)
    rebuilt_ids, rebuilt = build_union()
    if (
        proof["identity_count"] != len(rebuilt_ids)
        or proof["identity_set_sha256"] != rebuilt["identity_set_sha256"]
        or proof["excluded_identities"] != sorted(rebuilt_ids)
    ):
        raise ValueError("historical_exclusion_union_rebuild_mismatch")
    if registration["exclusion_proof"]["sha256"] != sha256(proof_path):
        raise ValueError("registration_exclusion_proof_hash_mismatch")

    suite_path = DATA / "openings.jsonl"
    suite = read_jsonl(suite_path)
    if registration["suite"]["sha256"] != sha256(suite_path):
        raise ValueError("registered_opening_suite_hash_mismatch")
    if len(suite) != 512:
        raise ValueError("opening_suite_count_mismatch")
    suite_ids = set()
    from ml.alphazero_lite.seed422_adam_memory_analysis import replay_opening

    for row in suite:
        state = replay_opening(row)
        from ml.alphazero_lite.seed422_exclusions import suites

        suite_ids.add(suites.canonical_key(state))
    if len(suite_ids) != 512 or suite_ids & rebuilt_ids:
        raise ValueError("suite_duplicate_or_exclusion_overlap")

    ledger_path = DATA / "outcome-ledger.jsonl"
    ledger = read_jsonl(ledger_path)
    validate_ledger(ledger, suite)
    if len(ledger) != 2048:
        raise ValueError("arena_game_count_mismatch")
    result = analyze(ledger)
    analysis_path = DATA / "analysis.json"
    published_analysis = read_json(analysis_path)
    for key, value in result.items():
        if published_analysis.get(key) != value:
            raise ValueError(f"published_analysis_mismatch:{key}")
    if published_analysis["outcome_ledger_sha256"] != sha256(ledger_path):
        raise ValueError("analysis_ledger_hash_mismatch")
    if read_json(DATA / "opening-matrix.json") != result["opening_matrix"]:
        raise ValueError("opening_matrix_mismatch")
    if published_analysis["registration_sha256"] != sha256(registration_path):
        raise ValueError("analysis_registration_hash_mismatch")

    training = read_json(DATA / "training-results.json")
    if training["registration_sha256"] != sha256(registration_path):
        raise ValueError("training_registration_hash_mismatch")
    if training["optimizer_updates_per_lane"] != 1052:
        raise ValueError("optimizer_update_count_mismatch")
    if (
        training["lanes"]["A"]["epochs"]["4"]
        != registration["training"]["expected_A_E4_checkpoint_sha256"]
    ):
        raise ValueError("A_checkpoint_did_not_reproduce_published_identity")
    if training["lanes"]["A"]["permutations"] != training["lanes"]["B"]["permutations"]:
        raise ValueError("lane_batch_plan_mismatch")
    if (
        training["lanes"]["A"]["multiplicity_sha256"]
        != training["lanes"]["B"]["multiplicity_sha256"]
    ):
        raise ValueError("lane_exposure_mismatch")
    for lane, betas in (("A", [0.9, 0.999]), ("B", [0.0, 0.999])):
        if training["lanes"][lane]["optimizer_parameter_group"]["betas"] != betas:
            raise ValueError(f"optimizer_betas_mismatch:{lane}")
    return {
        "verified": True,
        "torch_required": False,
        "runtime_artifacts_required": False,
        "excluded_identity_count": len(rebuilt_ids),
        "suite_count": len(suite),
        "game_count": len(ledger),
        "decision": published_analysis["decision"],
    }


def main() -> None:
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
