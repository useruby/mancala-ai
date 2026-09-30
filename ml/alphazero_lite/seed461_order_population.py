"""Conservative state-identity exclusions for the seed461 order confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import checkpoint_trajectory_diagnostic as trajectory
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite.kalah_rules import KalahGame

CANONICAL_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
SOURCE_RUNNERS = {
    "A-C": "run_pr249_fresh_suite_generalization.py",
    "D-F": "run_pr251_cross_seed_strength_residual_transfer.py",
    "G-I": "run_pr252_phase_target_delta_attribution.py",
    "J-L": "run_pr253_semantic_receiver_target_surgery.py",
    "M-R": "run_pr254_third_seed_budget_replay.py",
    "S-U": "run_pr256_fresh768_replay_replication.py",
    "V-X": "run_pr258_two_replay_aggregation.py",
    "Y-AA": "run_pr259_two_replay_second_epoch.py",
    "AB-AD": "run_pr260_value_head_refresh.py",
    "AE-AG": "run_pr261_policy_representation.py",
    "AH-AJ": "run_pr262_policy_hidden_capacity.py",
    "AK-AM": "run_pr264_joint_alphazero_iteration.py",
    "AN-AP": "run_pr265_unique_data_scale.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(rows: list[dict[str, Any]]) -> set[str]:
    return {suites.canonical_key(row["state"]) for row in rows}


def pr249_population() -> list[dict[str, Any]]:
    population = pr249.all_openings()
    if len(population) != 28_961 or len(keys(population)) != 28_961:
        raise RuntimeError("pr249_population_identity_mismatch")
    return population


def replay_keys(paths: list[Path]) -> set[str]:
    result = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                result.add(
                    suites.canonical_key(trajectory.state_from_row(json.loads(line)))
                )
    return result


def build_proof(
    canonical: Path, exploratory: Path, replay_paths: list[Path]
) -> tuple[dict[str, Any], set[str]]:
    if sha256(canonical) != CANONICAL_SHA:
        raise RuntimeError("canonical_reconstruction_sha_mismatch")
    population = pr249_population()
    population_keys = keys(population)
    canonical_keys = keys(suites.load_suite_jsonl(str(canonical)))
    if not canonical_keys <= population_keys:
        raise RuntimeError("canonical_not_covered_by_pr249_population")
    exploratory_keys = keys(suites.load_suite_jsonl(str(exploratory)))
    frozen_replay_keys = replay_keys(replay_paths)
    excluded = population_keys | exploratory_keys | frozen_replay_keys
    return (
        {
            "schema": "seed461-order-population-coverage-v1",
            "claim": "A-AP are covered by the complete PR249 source population; this is a conservative superset proof, not suite reconstruction.",
            "canonical": {
                "path": str(canonical),
                "sha256": CANONICAL_SHA,
                "members": 128,
            },
            "historical_source_population": {
                "function": "run_pr249_fresh_suite_generalization.all_openings",
                "unique_state_identities": len(population_keys),
                "source_runners": SOURCE_RUNNERS,
            },
            "exclusion_counts": {
                "pr249_population": len(population_keys),
                "pr382_exploratory": len(exploratory_keys),
                "frozen_replays": len(frozen_replay_keys),
                "union": len(excluded),
            },
        },
        excluded,
    )


def select_holdout(excluded: set[str], seed: int = 383) -> list[dict[str, Any]]:
    prefixes = suites.enumerate_legal_prefixes(8)
    population, _, _ = suites.deduplicate_openings(prefixes)
    eligible = [
        row
        for row in population
        if suites.canonical_key(row["state"]) not in excluded
        and row["pit_sum"] > 32
        and not KalahGame.from_state(row["state"]).over()
    ]
    selected = suites.select_diverse(suites.stratify_openings(eligible), 256, seed)
    if len(selected) != 256 or len(keys(selected)) != 256 or keys(selected) & excluded:
        raise RuntimeError("holdout_selection_contract_failure")
    return selected
