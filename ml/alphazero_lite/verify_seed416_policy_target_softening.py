#!/usr/bin/env python3
"""Portable, read-only verifier for published seed416 ablation evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed416_public_validation import (
    lane_games_sha256,
    validate_openings,
    validate_records,
)
from ml.alphazero_lite.seed416_validation_analysis import bootstrap_paired


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify(root: Path) -> dict[str, Any]:
    data = root / "docs/data/seed416-policy-target-softening"
    registration_path = data / "registration-v3.json"
    suite_path = data / "openings-v3.jsonl"
    proof_path = data / "opening-exclusion-proof-v3.json"
    training_path = data / "training-results.json"
    supersession_path = data / "registration-supersession-v3.json"
    split_path = data / "training-freeze-v3/source-row-split.json.gz"
    permutation_path = data / "training-freeze-v3/epoch-permutations.json.gz"
    binding_path = data / "evaluation-binding.json"
    outcomes_path = data / "outcome-binding.json"
    amendment_path = data / "post-first-lane-accounting-amendment.json"
    ledger_path = data / "outcome-ledger.jsonl"
    matrix_path = data / "per-opening-matrix.json"
    analysis_path = data / "analysis.json"
    archive_path = (
        data / "verify_seed416_policy_target_softening_pre_correction.archive.json"
    )
    receipt_path = data / "post-execution-correction-receipt.json"
    evidence_paths = [
        registration_path,
        suite_path,
        proof_path,
        binding_path,
        outcomes_path,
        ledger_path,
        matrix_path,
        analysis_path,
        archive_path,
        receipt_path,
        training_path,
        supersession_path,
        split_path,
        permutation_path,
    ]
    if amendment_path.exists():
        evidence_paths.append(amendment_path)
    identities = [digest(path) for path in evidence_paths]
    registration = read_json(registration_path)
    if registration.get("schema") != "seed416-policy-target-softening-registration-v1":
        raise ValueError("registration_schema_invalid")
    if registration.get("status") != "registered_before_training_and_model_probing":
        raise ValueError("registration_timing_invalid")
    if [row["weight"] for row in registration["replays"]] != [1, 4, 1, 8, 4]:
        raise ValueError("registered_replay_weights_invalid")
    if registration.get("exclusion_proof_sha256") != digest(proof_path):
        raise ValueError("exclusion_proof_binding_mismatch")
    supersession = read_json(supersession_path)
    if supersession.get("replacement_registration_sha256") != digest(registration_path):
        raise ValueError("registration_supersession_binding_mismatch")
    for freeze_path, key in (
        (split_path, "source_row_split"),
        (permutation_path, "epoch_permutations"),
    ):
        declared = registration["training"][key]
        if declared["sha256"] != digest(freeze_path):
            raise ValueError(f"training_freeze_hash_mismatch:{key}")
    split = json.loads(gzip.decompress(split_path.read_bytes()))
    train_positions = split["train_positions"]
    validation_positions = split["validation_positions"]
    if (
        len(train_positions)
        != registration["training"]["source_row_split"]["train_count"]
        or len(validation_positions)
        != registration["training"]["source_row_split"]["validation_count"]
        or set(train_positions) & set(validation_positions)
        or len(set(train_positions)) != len(train_positions)
        or len(set(validation_positions)) != len(validation_positions)
        or set(split["train_source_rows"]) & set(split["validation_source_rows"])
    ):
        raise ValueError("source_row_split_integrity_invalid")
    permutations = json.loads(gzip.decompress(permutation_path.read_bytes()))
    if len(permutations) != 4 or any(
        len(permutation) != len(train_positions)
        or sorted(permutation) != list(range(len(train_positions)))
        for permutation in permutations
    ):
        raise ValueError("epoch_permutation_integrity_invalid")
    permutation_hashes = {
        str(index + 1): hashlib.sha256(json.dumps(values).encode()).hexdigest()
        for index, values in enumerate(permutations)
    }
    if (
        permutation_hashes
        != registration["training"]["epoch_permutations"]["epoch_sha256"]
    ):
        raise ValueError("epoch_permutation_registration_mismatch")
    if digest(suite_path) != registration["suite"]["sha256"]:
        raise ValueError("suite_hash_mismatch")
    amendment = read_json(amendment_path) if amendment_path.exists() else None
    if amendment is not None and (
        amendment.get("registration_sha256") != digest(registration_path)
        or amendment.get("evaluation_binding_sha256") != digest(binding_path)
        or amendment.get("suite_sha256") != digest(suite_path)
        or not amendment.get("no_adaptive_extension")
    ):
        raise ValueError("execution_amendment_binding_invalid")
    archive = read_json(archive_path)
    receipt = read_json(receipt_path)
    if (
        archive.get("source_commit") != "9692fc1"
        or archive.get("source_sha256")
        != "ed15ad725d002bc6df99d53c0f0251f31c5af66ab0ba04bcfc4fe4e00ce10788"
        or archive.get("source_path")
        != "ml/alphazero_lite/verify_seed416_policy_target_softening.py"
        or receipt.get("archived_verifier_sha256") != archive.get("source_sha256")
        or receipt.get("archive_record_sha256") != digest(archive_path)
        or receipt.get("registration_sha256") != digest(registration_path)
        or receipt.get("registration_supersession_sha256") != digest(supersession_path)
        or receipt.get("evaluation_binding_sha256") != digest(binding_path)
        or receipt.get("outcome_binding_sha256") != digest(outcomes_path)
        or receipt.get("post_first_lane_accounting_amendment_sha256")
        != digest(amendment_path)
        or receipt.get("corrected_verifier_sha256") != digest(Path(__file__))
        or receipt.get("validation_helper_sha256")
        != digest(root / "ml/alphazero_lite/seed416_public_validation.py")
        or receipt.get("analysis_helper_sha256")
        != digest(root / "ml/alphazero_lite/seed416_validation_analysis.py")
    ):
        raise ValueError("post_execution_correction_receipt_invalid")
    replacements = amendment.get("source_replacements", {}) if amendment else {}
    for name, expected in registration.get("source_hashes", {}).items():
        source = root / name
        actual = digest(source) if source.is_file() else None
        replacement = replacements.get(name)
        if (
            actual != expected
            and not (
                replacement is not None
                and replacement.get("preregistered_sha256") == expected
                and replacement.get("post_run_sha256") == actual
            )
            and not (
                name == "ml/alphazero_lite/verify_seed416_policy_target_softening.py"
                and actual == receipt["corrected_verifier_sha256"]
            )
        ):
            raise ValueError(f"execution_source_hash_mismatch:{name}")
    suite = read_jsonl(suite_path)
    if len(suite) != 512 or len({row["state_hash"] for row in suite}) != 512:
        raise ValueError("suite_count_or_uniqueness_invalid")
    proof = read_json(proof_path)
    validate_openings(suite, proof)
    binding = read_json(binding_path)
    if binding["registration_sha256"] != digest(registration_path) or binding[
        "suite_sha256"
    ] != digest(suite_path):
        raise ValueError("evaluation_binding_mismatch")
    training = read_json(training_path)
    if training.get("registration_sha256") != digest(registration_path):
        raise ValueError("training_result_binding_mismatch")
    if binding.get("training_sha256") != digest(training_path):
        raise ValueError("training_evaluation_binding_mismatch")
    if set(training.get("lanes", {})) != {"A", "B"}:
        raise ValueError("training_lane_accounting_invalid")
    for lane in ("A", "B"):
        record = training["lanes"][lane]
        if set(record["epochs"]) != {"1", "2", "3", "4"}:
            raise ValueError(f"training_epoch_accounting_invalid:{lane}")
        candidate = binding["candidates"][lane]
        if candidate["checkpoint_sha256"] != record["epochs"]["4"]:
            raise ValueError(f"candidate_e4_checkpoint_binding_mismatch:{lane}")
        if record["permutations"] != training["lanes"]["A"]["permutations"]:
            raise ValueError(f"paired_training_order_mismatch:{lane}")
    for value in (
        binding.get("native_probe_sha256"),
        binding.get("tablebase_sha256"),
        binding.get("runtime_policy_sha256"),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("runtime_identity_missing")
    outcome_binding = read_json(outcomes_path)
    if outcome_binding["evaluation_binding_sha256"] != digest(binding_path):
        raise ValueError("outcome_binding_mismatch")
    if (
        outcome_binding.get("status") != "completed_2048_games"
        or outcome_binding.get("matched_initial_seed_contexts") != 1024
    ):
        raise ValueError("outcome_completion_status_invalid")
    if amendment is not None and (
        outcome_binding.get("lanes", {}).get("A", {}).get("report_sha256")
        != amendment["A_lane"]["report_sha256"]
        or outcome_binding.get("lanes", {}).get("A", {}).get("games_sha256")
        != amendment["A_lane"]["games_sha256"]
    ):
        raise ValueError("execution_amendment_A_binding_invalid")
    rows = read_jsonl(ledger_path)
    if len(rows) != 2048:
        raise ValueError("outcome_ledger_count_invalid")
    validate_records(rows, suite)
    for lane in ("A", "B"):
        if (
            lane_games_sha256(rows, lane)
            != outcome_binding["lanes"][lane]["games_sha256"]
        ):
            raise ValueError(f"bound_public_games_hash_mismatch:{lane}")
    counts: Counter[tuple[str, str]] = Counter()
    seats: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        lane = row["lane"]
        opening = str(row["opening_id"])
        game = row["game"]
        if lane not in {"A", "B"} or not 0 <= int(opening) < 512:
            raise ValueError("outcome_identity_invalid")
        if game.get("opening_contract") != "arena_player_relative_v2":
            raise ValueError("outcome_opening_contract_invalid")
        if game.get("winner") not in {"challenger", "current", "draw"}:
            raise ValueError("outcome_winner_invalid")
        score = (
            1.0
            if game["winner"] == "challenger"
            else 0.5
            if game["winner"] == "draw"
            else 0.0
        )
        if score != float(row["opponent_score"]):
            raise ValueError("outcome_score_mismatch")
        key = (lane, opening)
        counts[key] += 1
        seats.setdefault(key, set()).add(int(game["challenger_player"]))
    if len(counts) != 1024 or any(value != 2 for value in counts.values()):
        raise ValueError("outcome_per_opening_accounting_invalid")
    if any(value != {0, 1} for value in seats.values()):
        raise ValueError("seat_pairing_invalid")
    analysis = read_json(analysis_path)
    if analysis.get("outcome_ledger_sha256") != digest(ledger_path):
        raise ValueError("analysis_ledger_binding_mismatch")
    if analysis.get("outcome_binding_sha256") != digest(outcomes_path):
        raise ValueError("analysis_binding_mismatch")
    matrix = read_json(matrix_path)
    if matrix != analysis["per_opening"] or len(matrix) != 512:
        raise ValueError("opening_matrix_mismatch")
    if analysis.get("samples") != 10000 or analysis.get("seed") != 416:
        raise ValueError("bootstrap_registration_mismatch")
    recomputed = bootstrap_paired(rows, samples=10000, seed=416)
    for key in (
        "primary_mean_B_minus_A",
        "primary_95_percentile_interval",
        "B_opponent_score_mean",
        "B_opponent_score_95_percentile_interval",
        "decision",
        "per_opening",
    ):
        if recomputed[key] == analysis[key]:
            continue
        if key in {"decision", "per_opening"} or not np.allclose(
            np.asarray(recomputed[key], dtype=float),
            np.asarray(analysis[key], dtype=float),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(f"analysis_recomputation_mismatch:{key}")
    decision = (
        "advance_to_separate_confirmation"
        if analysis["primary_mean_B_minus_A"] >= 0.03
        and analysis["primary_95_percentile_interval"][0] > 0
        and analysis["B_opponent_score_mean"] >= 0.53
        and analysis["B_opponent_score_95_percentile_interval"][0] > 0.50
        else "retain_baseline"
    )
    if analysis["decision"] != decision:
        raise ValueError("decision_threshold_mismatch")
    final_identities = [digest(path) for path in evidence_paths]
    if identities != final_identities:
        raise ValueError("verifier_modified_evidence")
    return {
        "valid": True,
        "games": len(rows),
        "openings": 512,
        "decision": analysis["decision"],
        "primary_mean": analysis["primary_mean_B_minus_A"],
        "primary_interval": analysis["primary_95_percentile_interval"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
