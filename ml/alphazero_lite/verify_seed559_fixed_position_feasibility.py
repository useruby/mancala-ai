"""Read-only verifier for frozen seed559 feasibility evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import (
    authenticate_runtime,
    canonical_hash,
    expected_oracle_identities,
    load_authenticated_archive,
)
from ml.alphazero_lite.seed559_fixed_position_feasibility import (
    DATA,
    _excluded_hashes,
    _source_candidates,
    _wdl,
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


def verify(data: Path = DATA) -> dict[str, Any]:
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

    searches = {row["state_hash"]: row for row in _jsonl(data / "search-records.jsonl")}
    oracles = {row["state_hash"]: row for row in _jsonl(data / "oracle-records.jsonl")}
    timings = {row["state_hash"]: row for row in _jsonl(data / "timings.jsonl")}
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
    baseline_regret = sum(row["baseline_regret"] for row in reconstructed_metrics) / 512
    baseline_wdl = (
        sum(row["baseline_wdl_inferior"] for row in reconstructed_metrics) / 512
    )
    if (
        baseline_regret != analysis["baseline_mean_exact_margin_regret"]
        or baseline_wdl != analysis["baseline_wdl_inferior_rate"]
    ):
        raise ValueError("aggregate_baseline_metric_reconstruction_mismatch")
    gate = analysis["gate"]
    expected_decision = (
        "no_cutoff_21_arena_proposal"
        if not all(gate.values())
        else "supports_proposing_separate_cutoff_21_arena_only"
    )
    if (
        analysis["decision"] != expected_decision
        or gate["baseline_wdl_inferior_at_least_5_percent"]
        or gate["baseline_regret_at_least_1_seed"]
    ):
        raise ValueError("registered_gate_reconstruction_mismatch")
    native_start = json.loads((data / "native-startup-prewarm.json").read_text())
    if (
        native_start["warm_end_to_end_exact_mean_ms"] > 200
        or native_start["warm_end_to_end_exact_p95_ms"] > 350
    ):
        raise ValueError("exact_latency_gate_mismatch")
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
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(verify(args.data), sort_keys=True, indent=2))
