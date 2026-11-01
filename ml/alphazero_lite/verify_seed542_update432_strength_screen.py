"""Independent verification for the update-432 paired strength screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import (
    run_seed542_expanded_policy_strength_screen as runner,
    seed542_expanded_policy_strength_screen as protocol,
    seed542_update432_strength_screen as experiment,
    verify_seed542_expanded_policy_strength_screen as base_verifier,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate_composition(
    root: Path, output: Path, registration: dict[str, Any]
) -> None:
    experiment.authenticate_composition(root, output, registration)


def advance_eligible(validated_games: int, one_sided_lower: float) -> bool:
    return validated_games == 2048 and one_sided_lower > 0.50


def verify(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    runner.RUNTIME_SOURCES = (*runner.RUNTIME_SOURCES, *experiment.SOURCE_FILES)
    registration_path = output / "registration.json"
    freeze_path = output / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        registration.get("status") != "prospective_frozen_before_arena_games"
        or registration.get("candidate", {}).get("training_update") != 432
        or registration.get("candidate", {}).get("sole_candidate") is not True
        or sha(registration_path) != freeze.get("registration_sha256")
        or sha(output / "openings.jsonl") != freeze.get("suite_sha256")
        or sha(output / "command-preflight.jsonl")
        != freeze.get("command_preflight_sha256")
    ):
        raise ValueError("update432_screen_freeze_binding_mismatch")
    for relative, expected in freeze["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"update432_screen_source_hash_mismatch:{relative}")
    for relative, expected in freeze["authenticated_file_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(
                f"update432_screen_authenticated_input_mismatch:{relative}"
            )
    if (
        sha(root / "docs/data/seed542-training-budget-432/D-update-432.npz")
        != registration["candidate"]["training_checkpoint_sha256"]
    ):
        raise ValueError("update432_screen_candidate_checkpoint_changed")
    if sha(output / "openings.jsonl") != sha(
        root / "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl"
    ):
        raise ValueError("update432_screen_reused_suite_changed")

    openings, import_report = protocol.import_seed511_suite(root)
    if import_report["suite_sha256"] != freeze["suite_sha256"]:
        raise ValueError("update432_screen_suite_import_hash_mismatch")
    overlap = protocol.overlap_report(root, openings)
    published_overlap = json.loads((output / "overlap-summary.json").read_text())
    if published_overlap != {
        key: value for key, value in overlap.items() if key != "per_opening"
    }:
        raise ValueError("update432_screen_overlap_summary_mismatch")
    published_overlap_rows = [
        json.loads(line)
        for line in (output / "overlap-per-opening.jsonl").read_text().splitlines()
    ]
    if published_overlap_rows != overlap["per_opening"]:
        raise ValueError("update432_screen_overlap_rows_mismatch")
    if (
        registration["disclosures"]["historical_seed542_expanded_training_overlap"]
        != overlap["aggregates"]["seed542_expanded_training"]
        or registration["disclosures"][
            "seed461_historical_overlap_limitation_unresolved"
        ]
        is not True
    ):
        raise ValueError("update432_screen_required_disclosure_missing")

    observed = base_verifier._observed_evidence_paths(root, output, registration)
    before = base_verifier._snapshot(observed)
    ledger = [
        json.loads(line)
        for line in (output / "outcomes.jsonl").read_text().splitlines()
        if line
    ]
    if len(ledger) != 2048:
        raise ValueError("update432_screen_ledger_game_count_mismatch")
    committed: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        committed.setdefault(int(row["opening_index"]), []).append(row)
    if set(committed) != set(range(1024)):
        raise ValueError("update432_screen_ledger_opening_coverage_mismatch")
    old_authenticate = base_verifier._authenticate_composition
    base_verifier._authenticate_composition = authenticate_composition
    try:
        replayed = []
        for index in range(1024):
            pair = base_verifier.validate_pair(root, output, index)
            base_verifier.reconcile_pair_ledger(index, pair, committed[index])
            replayed.extend(pair)
    finally:
        base_verifier._authenticate_composition = old_authenticate
    result = base_verifier.independent_analysis(replayed)[0]
    published = json.loads((output / "results.json").read_text())
    per_opening = [
        json.loads(line)
        for line in (output / "per-opening-results.jsonl").read_text().splitlines()
        if line
    ]
    expected_result, expected_rows = base_verifier.independent_analysis(replayed)
    if published != expected_result or per_opening != expected_rows:
        raise ValueError("update432_screen_registered_bootstrap_mismatch")
    if len(per_opening) != 1024 or result["games"] != 2048:
        raise ValueError("update432_screen_analysis_coverage_mismatch")
    if published.get("pass") != advance_eligible(2048, published["one_sided_95_lower"]):
        raise ValueError("update432_screen_advancement_rule_mismatch")
    if before != base_verifier._snapshot(observed):
        raise ValueError("update432_screen_verification_mutated_evidence")
    return {
        "status": "all_1024_pairs_and_2048_games_independently_verified",
        "validated_openings": 1024,
        "validated_games": 2048,
        "published_analysis_matches_registered_bootstrap": True,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.ROOT)
    parser.add_argument("--out", type=Path, default=experiment.OUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.out), indent=2, sort_keys=True))


TRAINING_PATH = experiment.TRAINING / "D-update-432.npz"

if __name__ == "__main__":
    main()
