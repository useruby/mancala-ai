"""Read-only verifier for frozen seed559 feasibility evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import arena
from ml.alphazero_lite.seed557_retrospective import (
    authenticate_runtime,
    canonical_hash,
    expected_oracle_identities,
    load_authenticated_archive,
)
from ml.alphazero_lite.seed559_fixed_position_feasibility import (
    DEFAULT_ANALYSIS_DATA,
    _excluded_hashes,
    EVALUATOR_ARTIFACTS,
    _source_candidates,
    _wdl,
    analyze_published_run,
    build_evaluators,
    evaluator_artifact_hashes,
    select,
    verify_routing_boundaries,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _asset_sha(path: Path) -> str:
    contents = path.read_bytes()
    marker = b"version https://git-lfs.github.com/spec/v1\n"
    if contents.startswith(marker):
        for line in contents.decode("ascii").splitlines()[1:]:
            if line.startswith("oid sha256:"):
                return line.removeprefix("oid sha256:")
        raise ValueError("invalid_git_lfs_pointer")
    return _sha(contents)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify(data: Path = DEFAULT_ANALYSIS_DATA) -> dict[str, Any]:
    cohort_doc = json.loads((data / "cohort-manifest.json").read_text())
    cohort = cohort_doc["states"]
    reconstructed, audit = select()
    if cohort != reconstructed:
        raise ValueError("cohort_reconstruction_mismatch")
    published_audit = json.loads((data / "count-audit.json").read_text())
    if audit != published_audit:
        raise ValueError("selection_count_audit_mismatch")
    openings, records, _ = load_authenticated_archive()
    eligible = _source_candidates(openings, records)
    excluded557, excluded418 = _excluded_hashes()
    eligible_unique = {row["state_hash"] for row in eligible}
    supplement = json.loads((data / "count-audit-supplement.json").read_text())
    expected_supplement = {
        "eligible_unique_states_before_exclusion": len(eligible_unique),
        "unique_states_excluded_by_seed557": len(eligible_unique & excluded557),
        "unique_states_excluded_by_seed418": len(eligible_unique & excluded418),
        "unique_states_excluded_by_both": len(
            eligible_unique & excluded557 & excluded418
        ),
        "eligible_occurrence_matches_seed557": sum(
            row["state_hash"] in excluded557 for row in eligible
        ),
        "eligible_occurrence_matches_seed418": sum(
            row["state_hash"] in excluded418 for row in eligible
        ),
        "unique_remaining_states": audit["unique_remaining_states"],
    }
    if any(supplement.get(key) != value for key, value in expected_supplement.items()):
        raise ValueError("count_audit_supplement_reconstruction_mismatch")
    if len(cohort) != 512 or len({row["state_hash"] for row in cohort}) != 512:
        raise ValueError("cohort_count_or_state_uniqueness_mismatch")
    if len({row["opening_index"] for row in cohort}) != 512:
        raise ValueError("more_than_one_root_per_opening")
    freeze_bytes = (data / "execution-freeze.json").read_bytes()
    freeze = json.loads(freeze_bytes)
    if freeze["cohort_sha256"] != _sha((data / "cohort-manifest.json").read_bytes()):
        raise ValueError("cohort_freeze_hash_mismatch")
    if freeze["selection_audit_sha256"] != _sha(
        (data / "count-audit.json").read_bytes()
    ):
        raise ValueError("count_audit_freeze_hash_mismatch")
    if freeze["authenticated_policies"] != authenticate_runtime():
        raise ValueError("authenticated_policy_identity_mismatch")
    expected_runtime = freeze["authenticated_policies"]
    expected_evaluators = {name: expected_runtime[name] for name in EVALUATOR_ARTIFACTS}
    observed_evaluators = evaluator_artifact_hashes(
        EVALUATOR_ARTIFACTS, expected_evaluators
    )
    recorded_evaluators = freeze.get("evaluator_artifacts")
    if recorded_evaluators is not None:
        expected_binding = {
            name: {
                "path": str(path.relative_to(Path(__file__).resolve().parents[2])),
                "files": observed_evaluators[name],
            }
            for name, path in EVALUATOR_ARTIFACTS.items()
        }
        if recorded_evaluators != expected_binding:
            raise ValueError("evaluator_artifact_binding_mismatch")
    expected_oracle, _ = expected_oracle_identities()
    native = (
        Path(__file__).resolve().parents[2]
        / "model-artifact/runtime/kalah_v1_tablebase"
    )
    tablebase = (
        Path(__file__).resolve().parents[2] / "model-artifact/runtime/kalah_v1_21.kvtb"
    )
    if (
        _sha(native.read_bytes()) != expected_oracle["native_probe_sha256"]
        or _asset_sha(tablebase) != expected_oracle["tablebase_sha256"]
    ):
        raise ValueError("authenticated_native_oracle_asset_mismatch")
    if (
        freeze["oracle"]["native_sha256"] != expected_oracle["native_probe_sha256"]
        or freeze["oracle"]["tablebase_sha256"] != expected_oracle["tablebase_sha256"]
    ):
        raise ValueError("frozen_native_oracle_identity_mismatch")
    if (
        freeze["search"]["simulations"] != 128
        or freeze["search"]["workers"] != 1
        or freeze["search"]["exact_leaf_solving"]
        or freeze["search"]["exact_root_solving"]
    ):
        raise ValueError("search_freeze_mismatch")
    if freeze["diagnostic_routing"]["extended_cutoff"] != 21 or not all(
        verify_routing_boundaries().values()
    ):
        raise ValueError("routing_boundary_contract_mismatch")

    evaluators, _ = build_evaluators(EVALUATOR_ARTIFACTS, expected_evaluators)
    composed = arena.ComposedArtifactEvaluator(
        evaluators["P"],
        evaluators["initializer"],
        policy_source="current",
        value_source="candidate",
    )

    searches = {row["state_hash"]: row for row in _jsonl(data / "search-records.jsonl")}
    oracles = {row["state_hash"]: row for row in _jsonl(data / "oracle-records.jsonl")}
    timings = {row["state_hash"]: row for row in _jsonl(data / "timings.jsonl")}
    timing_rows = _jsonl(data / "timings.jsonl")
    if len(timing_rows) != 512 or any(
        not math.isfinite(float(row[field])) or float(row[field]) < 0
        for row in timing_rows
        for field in ("baseline_warm_decision_ms", "exact_warm_decision_ms")
    ):
        raise ValueError("invalid_or_incomplete_root_timing_measurements")
    if (
        set(searches) != {row["state_hash"] for row in cohort}
        or set(oracles) != set(searches)
        or set(timings) != set(searches)
    ):
        raise ValueError("per_root_evidence_coverage_mismatch")
    reconstructed_metrics = []
    for item in cohort:
        key = item["state_hash"]
        game = KalahGame.from_state(item["state"])
        legal = game.possible_moves()
        if canonical_hash(item["state"]) != key or set(legal) != set(
            oracles[key]["legal_moves"]
        ):
            raise ValueError("canonical_state_or_legal_action_mismatch")
        oracle = oracles[key]
        margins = {
            int(move): int(value) for move, value in oracle["action_margins"].items()
        }
        if oracle["coverage_complete"] is not True or set(margins) != set(legal):
            raise ValueError("oracle_action_coverage_mismatch")
        search = searches[key]
        priors = [float(value) for value in search["priors"]]
        expected_exact = max(
            legal, key=lambda move: (margins[move], priors[move], -move)
        )
        if int(search["exact_action"]) != expected_exact:
            raise ValueError("exact_margin_prior_index_tie_break_mismatch")
        visits = [int(value) for value in search["visits"]]
        stats = {int(row["move"]): row for row in search["child_stats"]}
        max_visits = max(visits[move] for move in legal)
        leaders = [move for move in legal if visits[move] == max_visits]
        expected_baseline = max(
            leaders,
            key=lambda move: (float(stats[move]["q_value"]), priors[move], -move),
        )
        if int(search["baseline_action"]) != expected_baseline:
            raise ValueError("baseline_puct_visit_q_prior_index_tie_break_mismatch")
        if sum(visits) != 128 or any(
            visits[move] for move in range(6) if move not in legal
        ):
            raise ValueError("simulation_budget_or_illegal_visit_mismatch")
        replay = arena.evaluate_artifact_position(
            evaluator=composed,
            state=item["state"],
            simulations=freeze["search"]["simulations"],
            seed=int(search["seed"]),
            c_puct=freeze["search"]["c_puct"],
            search_options=freeze["search"]["search_options"],
            exact_root_solve_threshold=None,
            exact_solve_stone_threshold=None,
        )
        observed_priors = [float(value) for value in replay["policy"]]
        if (
            not all(
                math.isclose(actual, recorded, rel_tol=1e-6, abs_tol=1e-7)
                for actual, recorded in zip(observed_priors, priors, strict=True)
            )
            or replay["visits"] != search["visits"]
            or replay["child_stats"] != search["child_stats"]
            or int(replay["selected_move"]) != int(search["baseline_action"])
        ):
            raise ValueError(f"authenticated_evaluator_search_replay_mismatch:{key}")
        best_margin = max(margins.values())
        reconstructed_metrics.append(
            {
                "state_hash": key,
                "opening_index": item["opening_index"],
                "active_pit_stones": item["active_pit_stones"],
                "baseline_regret": best_margin
                - margins[int(search["baseline_action"])],
                "baseline_wdl_inferior": _wdl(margins[int(search["baseline_action"])])
                < _wdl(best_margin),
                "exact_regret": best_margin - margins[int(search["exact_action"])],
                "exact_wdl_inferior": _wdl(margins[int(search["exact_action"])])
                < _wdl(best_margin),
                "baseline_action": int(search["baseline_action"]),
                "exact_action": int(search["exact_action"]),
            }
        )
    root_rows = _jsonl(data / "root-metrics.jsonl")
    if root_rows != reconstructed_metrics:
        raise ValueError("root_metrics_reconstruction_mismatch")
    analysis = json.loads((data / "analysis.json").read_text())
    if analysis["n_roots"] != 512 or analysis["oracle_full_coverage"] is not True:
        raise ValueError("analysis_coverage_mismatch")
    if analysis["decision"] != "no_cutoff_21_arena_proposal":
        raise ValueError("unexpected_gate_decision")
    expected_analysis = analyze_published_run(data)
    expected_roots = expected_analysis.pop("root_records")
    if expected_roots != root_rows:
        raise ValueError("root_metrics_recomputation_mismatch")
    if expected_analysis != analysis:
        raise ValueError("analysis_latency_interval_or_gate_recomputation_mismatch")
    return {
        "status": "verified",
        "roots": 512,
        "fully_covered_oracle_roots": len(oracles),
        "search_records": len(searches),
        "selected_by_source_side": audit["selected_by_source_side"],
        "decision": analysis["decision"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_ANALYSIS_DATA)
    args = parser.parse_args()
    print(json.dumps(verify(args.data), sort_keys=True, indent=2))
