#!/usr/bin/env python3
"""Balanced-current PUCT consensus-filtered replay smoke test."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_balanced_current_iter2_smoke import (  # noqa: E402
    build_candidate_summary_row,
    load_json,
    validate_replay_rows,
)
from ml.alphazero_lite.run_balanced_opening_puct_replay import (  # noqa: E402
    ASYM_1200_256_BUDGET,
    EQ_1200_BUDGET,
    EQ_768_BUDGET,
    PRIMARY_BUDGET,
    effective_sampling_fractions,
    large_suite_rows,
    markdown_table,
    parse_budget_pairs,
    parse_csv_paths,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    MIN_TRAINABLE_ROWS,
    TARGET_POLICY_MODE,
    TARGET_VALUE_MODE,
    artifact_forward_details,
    build_input_summary,
    collect_evaluation_states,
    kl_divergence,
    margin_bucket,
    partition_batches,
    percentile,
    policy_entropy,
    raw_margin,
    require_existing_file,
    top_policy_move,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    export_checkpoint,
    run_default_gate,
    run_opening_suite_benchmark,
    run_train,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    build_policy_target_from_distribution,
    build_search_options,
    derive_self_play_value_target,
    encode_state,
)

SUMMARY_SCHEMA = "azlite_balanced_current_puct_consensus_filter_v1"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-balanced-current-puct-consensus-filter-results.md"
)
CONSENSUS_AUDIT_FILENAME = "consensus_audit.json"
CONSENSUS_STATE_TABLE_FILENAME = "consensus_state_table.jsonl"
DISAGREEMENT_REPLAY_FILENAME = "consensus_disagreement_replay.jsonl"
STABILITY_REPLAY_FILENAME = "consensus_stability_replay.jsonl"
SUMMARY_FILENAME = "summary_metrics.json"
TARGET_ROWS = 2000
DISAGREEMENT_VISIT_SHARE_THRESHOLD = 0.55
STABILITY_MARGIN_THRESHOLD = 0.05
PREFIX_CAP = 12
TOP_MOVE_CAP = 420
PHASE_CAPS = {"opening": 1200, "mid": 600, "late": 200}
SEAT_CONTEXT_CAPS = {"challenger": 800, "current": 800, "mixed": 400}
PR128_REFERENCE_WORKDIR = Path("/tmp/azlite_balanced_current_iter2")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_balanced_current_puct_consensus"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--expected-init-checkpoint-sha256", required=True)
    parser.add_argument("--generic-bootstrap", required=True)
    parser.add_argument("--random-teacher", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--search-budgets", default="384,768,1200")
    parser.add_argument("--disagreement-weight", type=int, default=8)
    parser.add_argument("--stability-weight", type=int, default=4)
    parser.add_argument(
        "--budget-pairs",
        default="384:256,768:256,768:768,1200:1200,1200:256,256:768",
    )
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def parse_search_budgets(text: str) -> list[int]:
    budgets = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one search budget is required")
    if len(set(budgets)) != len(budgets):
        raise ValueError("search budgets must be unique")
    return budgets


def search_profile(c_puct: float, budgets: list[int]) -> dict[str, Any]:
    base = build_search_options(root_policy_mode="visit_count", tactical_root_bias=0.0)
    profile = {
        "c_puct": float(c_puct),
        "budgets": list(budgets),
        "fpu_mode": str(base["fpu_mode"]),
        "reuse_subtree": bool(base["reuse_subtree"]),
        "normalize_values": bool(base["normalize_values"]),
        "root_policy_mode": str(base["root_policy_mode"]),
        "tactical_root_bias": float(base["tactical_root_bias"]),
        "root_temperature": float(base["root_temperature"]),
        "target_policy_mode": TARGET_POLICY_MODE,
        "target_value_mode": TARGET_VALUE_MODE,
    }
    encoded = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    profile["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return profile


def dominant_seat_context(row: dict[str, Any]) -> str:
    seat_contexts = row.get("seat_contexts", {})
    if not isinstance(seat_contexts, dict) or not seat_contexts:
        return "unknown"
    non_zero = [key for key, value in seat_contexts.items() if int(value) > 0]
    if len(non_zero) > 1:
        return "mixed"
    if len(non_zero) == 1:
        return str(non_zero[0])
    return max(seat_contexts.items(), key=lambda item: (int(item[1]), item[0]))[0]


def legal_mask_is_valid(policy: list[float], legal_moves: list[int]) -> bool:
    if len(policy) != 6:
        return False
    total = sum(float(value) for value in policy)
    if abs(total - 1.0) > 1e-6:
        return False
    legal_set = set(int(move) for move in legal_moves)
    return all(
        move in legal_set or float(probability) <= 1e-6
        for move, probability in enumerate(policy)
    )


def visit_share_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def analyze_consensus_state_batch(
    *,
    batch: list[dict[str, Any]],
    artifact_path: str,
    budgets: list[int],
    c_puct: float,
    seed: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact_path))
    search_options = build_search_options(
        root_policy_mode="visit_count", tactical_root_bias=0.0
    )
    results: list[dict[str, Any]] = []
    for index, state_entry in enumerate(batch):
        game = KalahGame.from_state(state_entry["state"])
        legal_moves = [int(move) for move in game.possible_moves()]
        logits, raw_policy, raw_value = artifact_forward_details(evaluator, game)
        raw_top1 = top_policy_move(raw_policy, legal_moves)
        row: dict[str, Any] = {
            **state_entry,
            "legal_moves": legal_moves,
            "legal_mask_valid": legal_mask_is_valid(raw_policy, legal_moves),
            "raw_logits": [float(value) for value in logits],
            "raw_policy": [float(value) for value in raw_policy],
            "raw_value": float(raw_value),
            "raw_top1": raw_top1,
            "raw_margin": raw_margin(raw_policy, legal_moves),
            "dominant_seat_context": dominant_seat_context(state_entry),
            "searches": {},
        }
        budget_top1s: list[int | None] = []
        for budget_index, simulations in enumerate(budgets):
            search = PUCT(
                evaluator=evaluator,
                simulations=int(simulations),
                c_puct=float(c_puct),
                rng=random.Random(seed + (index * 10_000) + (budget_index * 97)),
                fpu_mode=str(search_options["fpu_mode"]),
                reuse_subtree=bool(search_options["reuse_subtree"]),
                normalize_values=bool(search_options["normalize_values"]),
                root_policy_mode=str(search_options["root_policy_mode"]),
                tactical_root_bias=float(search_options["tactical_root_bias"]),
                root_temperature=float(search_options["root_temperature"]),
            )
            visits, root = search.run(game)
            search_policy = search_policy_from_visits(visits, legal_moves)
            search_top1 = top_policy_move(search_policy, legal_moves)
            legal_raw = np.asarray(
                [raw_policy[move] for move in legal_moves], dtype=np.float64
            )
            legal_search = np.asarray(
                [search_policy[move] for move in legal_moves], dtype=np.float64
            )
            visit_total = int(
                sum(int(round(float(visits[move]))) for move in legal_moves)
            )
            top1_visit_share = (
                float(search_policy[search_top1]) if search_top1 is not None else 0.0
            )
            budget_key = str(simulations)
            row["searches"][budget_key] = {
                "simulations": int(simulations),
                "policy": [float(value) for value in search_policy],
                "value": float(root.q_value if root is not None else 0.0),
                "top1": search_top1,
                "top1_changed": bool(search_top1 != raw_top1),
                "visit_count_total": visit_total,
                "top1_visit_share": float(top1_visit_share),
                "entropy": policy_entropy(search_policy, legal_moves),
                "kl_search_raw": kl_divergence(legal_search, legal_raw),
                "kl_raw_search": kl_divergence(legal_raw, legal_search),
            }
            budget_top1s.append(search_top1)
        first_top1 = budget_top1s[0] if budget_top1s else None
        stable_across_budgets = bool(
            budget_top1s and all(top1 == first_top1 for top1 in budget_top1s)
        )
        budget_384 = row["searches"].get("384")
        budget_1200 = row["searches"].get("1200")
        pr128_style_disagreement = bool(
            budget_384
            and row["legal_mask_valid"]
            and len(legal_moves) > 1
            and budget_384["top1"] != raw_top1
            and float(budget_384["top1_visit_share"])
            >= DISAGREEMENT_VISIT_SHARE_THRESHOLD
        )
        consensus_high_confidence_disagreement = bool(
            stable_across_budgets
            and budget_1200
            and row["legal_mask_valid"]
            and len(legal_moves) > 1
            and budget_1200["top1"] != raw_top1
            and float(budget_1200["top1_visit_share"])
            >= DISAGREEMENT_VISIT_SHARE_THRESHOLD
        )
        consensus_stability_state = bool(
            stable_across_budgets
            and budget_1200
            and row["legal_mask_valid"]
            and len(legal_moves) > 1
            and raw_top1 == budget_1200["top1"]
            and float(budget_1200["top1_visit_share"])
            >= DISAGREEMENT_VISIT_SHARE_THRESHOLD
            and float(row["raw_margin"]) >= STABILITY_MARGIN_THRESHOLD
        )
        row.update(
            {
                "budget_context": {
                    budget: {
                        "raw_top1_disagrees": row["searches"][str(budget)][
                            "top1_changed"
                        ]
                    }
                    for budget in budgets
                },
                "search_top1_stable_across_budgets": stable_across_budgets,
                "stable_search_top1": first_top1 if stable_across_budgets else None,
                "pr128_style_disagreement": pr128_style_disagreement,
                "rejected_by_budget_instability": bool(
                    pr128_style_disagreement and not stable_across_budgets
                ),
                "consensus_high_confidence_disagreement": consensus_high_confidence_disagreement,
                "consensus_stability_state": consensus_stability_state,
            }
        )
        results.append(row)
    return results


def search_policy_from_visits(
    visits: list[float] | np.ndarray, legal_moves: list[int]
) -> list[float]:
    policy = np.zeros(6, dtype=np.float32)
    if not legal_moves:
        return policy.tolist()
    visits_array = np.asarray(visits, dtype=np.float64)
    total = float(np.sum(visits_array[legal_moves]))
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            policy[move] = uniform
        return policy.tolist()
    for move in legal_moves:
        policy[move] = float(visits_array[move] / total)
    return policy.tolist()


def disagreement_rate(rows: list[dict[str, Any]], budget_key: str) -> dict[str, Any]:
    changed = sum(
        1 for row in rows if bool(row["searches"][budget_key]["top1_changed"])
    )
    return {
        "states": len(rows),
        "changed_top1": changed,
        "rate": changed / max(len(rows), 1),
    }


def disagreement_rate_by_group(
    rows: list[dict[str, Any]], group_key: str, budgets: list[int]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return {
        group_name: {
            str(budget): disagreement_rate(group_rows, str(budget))
            for budget in budgets
        }
        for group_name, group_rows in sorted(grouped.items())
    }


def disagreement_rate_by_margin(
    rows: list[dict[str, Any]], budgets: list[int]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[margin_bucket(float(row["raw_margin"]))].append(row)
    return {
        bucket: {
            str(budget): disagreement_rate(group_rows, str(budget))
            for budget in budgets
        }
        for bucket, group_rows in grouped.items()
    }


def summarize_consensus_audit(
    *,
    rows: list[dict[str, Any]],
    budgets: list[int],
    collection_metadata: dict[str, Any],
    input_summary: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    pr128_rows = [row for row in rows if bool(row["pr128_style_disagreement"])]
    rejected_rows = [row for row in rows if bool(row["rejected_by_budget_instability"])]
    consensus_disagreement_rows = [
        row for row in rows if bool(row["consensus_high_confidence_disagreement"])
    ]
    consensus_stability_rows = [
        row for row in rows if bool(row["consensus_stability_state"])
    ]
    stable_count = sum(
        1 for row in rows if bool(row["search_top1_stable_across_budgets"])
    )
    visit_share_by_budget = {
        str(budget): visit_share_summary(
            [float(row["searches"][str(budget)]["top1_visit_share"]) for row in rows]
        )
        for budget in budgets
    }
    top_move_distribution = Counter(
        str(row["searches"][str(max(budgets))]["top1"])
        for row in rows
        if row["searches"][str(max(budgets))]["top1"] is not None
    )
    duplicate_state_count = sum(
        1 for row in rows if int(row.get("occurrence_count", 1)) > 1
    )
    duplicate_state_occurrences = sum(
        max(int(row.get("occurrence_count", 1)) - 1, 0) for row in rows
    )
    return {
        "schema": "azlite_balanced_current_puct_consensus_audit_v1",
        "inputs": {
            "current_artifact_weights_sha256": input_summary[
                "current_artifact_weights"
            ]["actual_sha256"],
            "init_checkpoint_sha256": input_summary["init_checkpoint"]["actual_sha256"],
            "search_profile_hash": profile["sha256"],
            "search_profile": profile,
        },
        "collection_metadata": collection_metadata,
        "metrics": {
            "total_states_analyzed": len(rows),
            "raw_vs_budget_top1_disagreement_rate": {
                str(budget): disagreement_rate(rows, str(budget)) for budget in budgets
            },
            "search_384_768_1200_top1_agreement_rate": {
                "states": len(rows),
                "agreeing_states": stable_count,
                "rate": stable_count / max(len(rows), 1),
            },
            "fraction_of_pr128_style_disagreements_rejected_by_budget_instability": {
                "states": len(pr128_rows),
                "rejected_states": len(rejected_rows),
                "rate": len(rejected_rows) / max(len(pr128_rows), 1),
            },
            "mean_kl_search_vs_raw": {
                str(budget): statistics.fmean(
                    float(row["searches"][str(budget)]["kl_search_raw"]) for row in rows
                )
                if rows
                else 0.0
                for budget in budgets
            },
            "top1_visit_share": visit_share_by_budget,
            "disagreement_rate_by_phase": disagreement_rate_by_group(
                rows, "phase", budgets
            ),
            "disagreement_rate_by_seat_context": disagreement_rate_by_group(
                rows, "dominant_seat_context", budgets
            ),
            "disagreement_rate_by_raw_top1_margin_bucket": disagreement_rate_by_margin(
                rows, budgets
            ),
            "count_consensus_high_confidence_disagreements": len(
                consensus_disagreement_rows
            ),
            "count_consensus_stability_states": len(consensus_stability_rows),
            "top_move_distribution_by_pit": dict(sorted(top_move_distribution.items())),
            "duplicate_state_count": duplicate_state_count,
            "duplicate_state_occurrences": duplicate_state_occurrences,
        },
    }


def selection_sort_key(row: dict[str, Any], *, disagreement: bool) -> tuple[Any, ...]:
    budget_1200 = row["searches"]["1200"]
    disagreement_priority = 0 if disagreement else 1
    return (
        disagreement_priority,
        -float(budget_1200["top1_visit_share"]),
        -float(row["searches"]["1200"]["kl_search_raw"]),
        -float(row["raw_margin"]),
        int(row["move_index"]),
        str(row["state_hash"]),
    )


def select_capped_rows(
    rows: list[dict[str, Any]], *, target_rows: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefix_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    top_move_counts: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    skipped_for_caps: list[dict[str, Any]] = []
    for row in rows:
        prefix_key = str(row.get("opening_prefix_text") or "")
        phase_key = str(row["phase"])
        seat_key = str(row["dominant_seat_context"])
        top_move = int(row["searches"]["1200"]["top1"])
        if prefix_counts[prefix_key] >= PREFIX_CAP:
            skipped_for_caps.append(row)
            continue
        if phase_counts[phase_key] >= PHASE_CAPS.get(phase_key, target_rows):
            skipped_for_caps.append(row)
            continue
        if seat_counts[seat_key] >= SEAT_CONTEXT_CAPS.get(seat_key, target_rows):
            skipped_for_caps.append(row)
            continue
        if top_move_counts[top_move] >= TOP_MOVE_CAP:
            skipped_for_caps.append(row)
            continue
        selected.append(row)
        prefix_counts[prefix_key] += 1
        phase_counts[phase_key] += 1
        seat_counts[seat_key] += 1
        top_move_counts[top_move] += 1
        if len(selected) >= target_rows:
            break
    if len(selected) < target_rows:
        seen_hashes = {str(row["state_hash"]) for row in selected}
        for row in skipped_for_caps:
            state_hash = str(row["state_hash"])
            if state_hash in seen_hashes:
                continue
            selected.append(row)
            seen_hashes.add(state_hash)
            if len(selected) >= target_rows:
                break
    return selected, {
        "target_rows": target_rows,
        "selected_rows": len(selected),
        "prefix_cap": PREFIX_CAP,
        "phase_caps": PHASE_CAPS,
        "seat_context_caps": SEAT_CONTEXT_CAPS,
        "top_move_cap": TOP_MOVE_CAP,
        "phase_counts": dict(phase_counts),
        "seat_context_counts": dict(seat_counts),
        "top_move_counts": {
            str(key): value for key, value in sorted(top_move_counts.items())
        },
    }


def select_consensus_disagreement_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if bool(row["consensus_high_confidence_disagreement"])
        and bool(row["legal_mask_valid"])
        and len(row["legal_moves"]) > 1
    ]
    candidates.sort(key=lambda row: selection_sort_key(row, disagreement=True))
    selected, selection_summary = select_capped_rows(
        candidates, target_rows=TARGET_ROWS
    )
    return selected, {
        **selection_summary,
        "candidate_rows": len(candidates),
        "unique_rows": len({str(row["state_hash"]) for row in selected}),
    }


def select_consensus_stability_rows(
    rows: list[dict[str, Any]], disagreement_hashes: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if bool(row["consensus_stability_state"])
        and bool(row["legal_mask_valid"])
        and len(row["legal_moves"]) > 1
        and str(row["state_hash"]) not in disagreement_hashes
    ]
    candidates.sort(key=lambda row: selection_sort_key(row, disagreement=False))
    selected, selection_summary = select_capped_rows(
        candidates, target_rows=TARGET_ROWS
    )
    return selected, {
        **selection_summary,
        "candidate_rows": len(candidates),
        "unique_rows": len({str(row["state_hash"]) for row in selected}),
        "excluded_overlap_with_disagreement": sum(
            1 for row in rows if str(row["state_hash"]) in disagreement_hashes
        ),
    }


def continue_consensus_state_batch(
    *,
    batch: list[dict[str, Any]],
    artifact_path: str,
    simulations: int,
    c_puct: float,
    seed: int,
    bucket_name: str,
    teacher_source: str,
    selection_tier: str,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact_path))
    search_options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0
    )
    rows: list[dict[str, Any]] = []
    for index, state_entry in enumerate(batch):
        game = KalahGame.from_state(state_entry["state"])
        root_player = int(game.current_player)
        rng = np.random.default_rng(seed + index)
        target_search = state_entry["searches"][str(simulations)]
        search_value = float(target_search["value"])
        policy_target = build_policy_target_from_distribution(
            target_search["policy"], mode=TARGET_POLICY_MODE
        )
        while not game.over():
            legal_moves = game.possible_moves()
            if not legal_moves:
                break
            search = PUCT(
                evaluator=evaluator,
                simulations=int(simulations),
                c_puct=float(c_puct),
                rng=random.Random(int(rng.integers(0, 2**31 - 1))),
                fpu_mode=str(search_options["fpu_mode"]),
                reuse_subtree=bool(search_options["reuse_subtree"]),
                normalize_values=bool(search_options["normalize_values"]),
                root_policy_mode=str(search_options["root_policy_mode"]),
                tactical_root_bias=float(search_options["tactical_root_bias"]),
                root_temperature=float(search_options["root_temperature"]),
            )
            _visits, root = search.run(game)
            move = search.select_root_move(root, legal_moves)
            if not game.move(game.pit_index(move)):
                break
        if game.winner is None:
            outcome = 0.0
        elif int(game.winner) == root_player:
            outcome = 1.0
        else:
            outcome = -1.0
        rows.append(
            {
                "state_hash": state_entry["state_hash"],
                "state": encode_state(state_entry["state"], input_encoding="kalah_v3"),
                "policy": [float(value) for value in policy_target],
                "value": derive_self_play_value_target(
                    outcome_value=outcome,
                    search_value=search_value,
                    move_index=int(state_entry["move_index"]),
                    mode=TARGET_VALUE_MODE,
                ),
                "player": root_player,
                "move_index": int(state_entry["move_index"]),
                "winner": game.winner,
                "legal_moves": [int(move) for move in state_entry["legal_moves"]],
                "top_target_move": target_search["top1"],
                "policy_target_mode": TARGET_POLICY_MODE,
                "policy_target_actual_mode": TARGET_POLICY_MODE,
                "value_target_mode": TARGET_VALUE_MODE,
                "teacher_source": teacher_source,
                "bucket": bucket_name,
                "suite_names": list(state_entry.get("suite_names", [])),
                "suite_name": state_entry["suite_name"],
                "opening_prefix": list(state_entry.get("opening_prefix", [])),
                "phase": state_entry["phase"],
                "selection_tier": selection_tier,
                "search_top1": target_search["top1"],
                "raw_top1": state_entry["raw_top1"],
                "search_top1_visit_share": target_search["top1_visit_share"],
                "raw_margin": state_entry["raw_margin"],
                "kl_search_raw": target_search["kl_search_raw"],
                "continuation_outcome": outcome,
                "continuation_winner": game.winner,
                "budget_consensus": True,
                "target_budget": int(simulations),
                "dominant_seat_context": state_entry["dominant_seat_context"],
            }
        )
    return rows


def build_consensus_replay(
    *,
    selected_rows: list[dict[str, Any]],
    artifact_path: Path,
    simulations: int,
    c_puct: float,
    seed: int,
    workers: int,
    bucket_name: str,
    teacher_source: str,
    selection_tier: str,
) -> list[dict[str, Any]]:
    if not selected_rows:
        return []
    batches = partition_batches(selected_rows, workers)
    replay_rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(batches)))
    ) as pool:
        futures = [
            pool.submit(
                continue_consensus_state_batch,
                batch=batch,
                artifact_path=str(artifact_path),
                simulations=simulations,
                c_puct=c_puct,
                seed=seed + (batch_index * 1000),
                bucket_name=bucket_name,
                teacher_source=teacher_source,
                selection_tier=selection_tier,
            )
            for batch_index, batch in enumerate(batches)
        ]
        for future in futures:
            replay_rows.extend(future.result())
    replay_rows.sort(key=lambda row: str(row["state_hash"]))
    return replay_rows


def gate_targets_from_summary(candidate_rows: list[dict[str, Any]]) -> list[str]:
    targets = ["balanced_current_ref"]
    current_row = next(
        row for row in candidate_rows if row["candidate"] == "balanced_current_ref"
    )
    current_mean = float(
        current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
    )
    for row in candidate_rows:
        name = str(row["candidate"])
        if not name.startswith("consensus_"):
            continue
        mean_ds = float(row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"])
        if mean_ds >= current_mean + 0.05:
            targets.append(name)
    return targets


def discover_pr128_reference() -> dict[str, Any] | None:
    artifact_dir = (
        PR128_REFERENCE_WORKDIR
        / "balanced_iter2_w8s4_policy_head_e1"
        / "artifact_balanced_iter2_w8s4_policy_head_e1"
    )
    checkpoint_path = (
        PR128_REFERENCE_WORKDIR
        / "balanced_iter2_w8s4_policy_head_e1"
        / "checkpoint_epoch1.npz"
    )
    if not (artifact_dir / "weights.json").is_file() or not checkpoint_path.is_file():
        return None
    return {
        "name": "pr128_iter2_e1_ref",
        "report_candidate_name": artifact_dir.name,
        "epochs": 0,
        "checkpoint_path": str(checkpoint_path),
        "artifact_dir": str(artifact_dir),
        "trainable_scope": "none",
        "train_metrics": None,
    }


def candidate_bootstrap_cis(
    suite_rows: dict[str, Any], candidate_name: str, seed: int
) -> dict[str, Any]:
    cis: dict[str, Any] = {}
    for budget_pair in (
        PRIMARY_BUDGET,
        EQ_768_BUDGET,
        EQ_1200_BUDGET,
        ASYM_1200_256_BUDGET,
    ):
        diffs = pooled_per_opening_differences(
            suite_rows=suite_rows,
            candidate_a=candidate_name,
            candidate_b="balanced_current_ref",
            budget_pair=budget_pair,
            metric_key="ds",
        )
        cis[
            f"{candidate_name}_minus_balanced_current_ref_{budget_pair.replace(':', '_')}"
        ] = bootstrap_ci(diffs, seed=seed, samples=DEFAULT_BOOTSTRAP_SAMPLES)
    return cis


def consensus_policy_shift_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        target_search = row["searches"]["1200"]
        result.append(
            {
                "state_hash": str(row["state_hash"]),
                "state": row["state"],
                "legal_moves": [int(move) for move in row["legal_moves"]],
                "phase": str(row["phase"]),
                "raw_margin": float(row["raw_margin"]),
                "raw_top1": row["raw_top1"],
                "search_top1": target_search["top1"],
                "search_policy": target_search["policy"],
            }
        )
    return result


def classify_run(summary: dict[str, Any]) -> str:
    audit = summary["consensus_audit"]
    metrics = audit["metrics"]
    stable_fraction = 1.0 - float(
        metrics["fraction_of_pr128_style_disagreements_rejected_by_budget_instability"][
            "rate"
        ]
    )
    disagreement_rows = int(
        summary["replay_row_counts"].get("consensus_disagreement", 0)
    )
    if stable_fraction < 0.5 or disagreement_rows < MIN_TRAINABLE_ROWS:
        return "search_budget_noise_confirmed"

    trained_rows = [
        row
        for row in summary.get("candidates", [])
        if str(row["candidate"]).startswith("consensus_")
    ]
    if not trained_rows:
        return "search_budget_noise_confirmed"
    current_row = next(
        row
        for row in summary["candidates"]
        if row["candidate"] == "balanced_current_ref"
    )
    for row in trained_rows:
        agg = row["large_suite_aggregate"]
        delta_384 = float(agg[PRIMARY_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
        )
        delta_768 = float(agg[EQ_768_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][EQ_768_BUDGET]["mean_ds"]
        )
        delta_1200 = float(agg[EQ_1200_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][EQ_1200_BUDGET]["mean_ds"]
        )
        delta_1200_256 = float(agg[ASYM_1200_256_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][ASYM_1200_256_BUDGET]["mean_ds"]
        )
        ci_384 = row["bootstrap_cis"][
            f"{row['candidate']}_minus_balanced_current_ref_384_256"
        ]
        gate_ok = row.get("high_search_breakthrough_preserved") is not False
        if (
            delta_384 >= 0.05
            and float(ci_384["lower"]) > 0.01
            and delta_768 >= -0.08
            and delta_1200 >= -0.03
            and delta_1200_256 >= -0.03
            and gate_ok
        ):
            return "consensus_filter_success"

    all_harmful = True
    for row in trained_rows:
        agg = row["large_suite_aggregate"]
        delta_384 = float(agg[PRIMARY_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
        )
        delta_768 = float(agg[EQ_768_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][EQ_768_BUDGET]["mean_ds"]
        )
        delta_1200 = float(agg[EQ_1200_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][EQ_1200_BUDGET]["mean_ds"]
        )
        delta_1200_256 = float(agg[ASYM_1200_256_BUDGET]["mean_ds"]) - float(
            current_row["large_suite_aggregate"][ASYM_1200_256_BUDGET]["mean_ds"]
        )
        if (
            delta_384 >= 0.0
            and delta_768 >= -0.08
            and delta_1200 >= -0.03
            and delta_1200_256 >= -0.03
        ):
            all_harmful = False
    if disagreement_rows >= MIN_TRAINABLE_ROWS and all_harmful:
        return "consensus_targets_still_harmful"

    best = max(
        trained_rows,
        key=lambda row: float(row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]),
    )
    best_delta_384 = float(
        best["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
    ) - float(current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"])
    changed_rate = float(
        best["mined_state_policy_shift"]["top1_changed_rate_vs_promoted_current"]
    )
    if (
        disagreement_rows >= MIN_TRAINABLE_ROWS
        and changed_rate < 0.03
        and best_delta_384 >= -0.01
    ):
        return "update_too_weak"

    raw_vs_1200 = float(metrics["raw_vs_budget_top1_disagreement_rate"]["1200"]["rate"])
    if disagreement_rows < MIN_TRAINABLE_ROWS and raw_vs_1200 <= 0.10:
        return "current_policy_iteration_saturated"
    return "consensus_targets_still_harmful"


def render_report(summary: dict[str, Any]) -> str:
    audit = summary["consensus_audit"]
    metrics = audit["metrics"]
    budget_rows = []
    for budget in ("384", "768", "1200"):
        visit_share = metrics["top1_visit_share"][budget]
        budget_rows.append(
            [
                budget,
                fmt(
                    float(
                        metrics["raw_vs_budget_top1_disagreement_rate"][budget]["rate"]
                    )
                ),
                fmt(float(metrics["mean_kl_search_vs_raw"][budget])),
                fmt(float(visit_share["mean"])),
                fmt(float(visit_share["p50"] or 0.0)),
                fmt(float(visit_share["p90"] or 0.0)),
            ]
        )
    lines = [
        "# AlphaZero-Lite Balanced Current PUCT Consensus Filter Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Inputs",
        "",
        f"- Current artifact weights SHA256: `{summary['inputs']['current_artifact_weights']['actual_sha256']}`",
        f"- Current source checkpoint SHA256: `{summary['inputs']['init_checkpoint']['actual_sha256']}`",
        f"- Search-profile SHA256: `{audit['inputs']['search_profile_hash']}`",
        "",
        "## Consensus Audit",
        "",
        f"- Total states analyzed: `{metrics['total_states_analyzed']}`",
        f"- 384/768/1200 top-1 agreement rate: `{metrics['search_384_768_1200_top1_agreement_rate']['rate']:.4f}`",
        f"- Rejected unstable-disagreement count: `{metrics['fraction_of_pr128_style_disagreements_rejected_by_budget_instability']['rejected_states']}`",
        f"- Consensus high-confidence disagreements: `{metrics['count_consensus_high_confidence_disagreements']}`",
        f"- Consensus stability states: `{metrics['count_consensus_stability_states']}`",
        f"- Duplicate state count: `{metrics['duplicate_state_count']}`",
        f"- Duplicate trajectory count: `{audit['collection_metadata']['duplicate_trajectory_count']}`",
        f"- Duplicate trajectory rate: `{audit['collection_metadata']['duplicate_trajectory_rate']:.4f}`",
        "",
        markdown_table(
            [
                "Budget",
                "Raw-vs-search top-1 disagree",
                "Mean KL(search||raw)",
                "Top-1 share mean",
                "Top-1 share p50",
                "Top-1 share p90",
            ],
            budget_rows,
        ),
        "",
        "## Replay",
        "",
        f"- Final disagreement rows: `{summary['replay_row_counts']['consensus_disagreement']}`",
        f"- Final stability rows: `{summary['replay_row_counts']['consensus_stability']}`",
        f"- Effective replay sampling fractions: `{json.dumps(summary['effective_replay_sampling_fractions'], sort_keys=True)}`",
        "",
    ]
    phase_rows: list[list[Any]] = []
    for phase, phase_data in sorted(metrics["disagreement_rate_by_phase"].items()):
        phase_rows.append(
            [
                phase,
                fmt(float(phase_data["384"]["rate"])),
                fmt(float(phase_data["768"]["rate"])),
                fmt(float(phase_data["1200"]["rate"])),
            ]
        )
    seat_rows: list[list[Any]] = []
    for seat, seat_data in sorted(metrics["disagreement_rate_by_seat_context"].items()):
        seat_rows.append(
            [
                seat,
                fmt(float(seat_data["384"]["rate"])),
                fmt(float(seat_data["768"]["rate"])),
                fmt(float(seat_data["1200"]["rate"])),
            ]
        )
    margin_rows: list[list[Any]] = []
    for bucket, bucket_data in sorted(
        metrics["disagreement_rate_by_raw_top1_margin_bucket"].items()
    ):
        margin_rows.append(
            [
                bucket,
                fmt(float(bucket_data["384"]["rate"])),
                fmt(float(bucket_data["768"]["rate"])),
                fmt(float(bucket_data["1200"]["rate"])),
            ]
        )
    top_move_rows = [
        [pit, count]
        for pit, count in sorted(metrics["top_move_distribution_by_pit"].items())
    ]
    lines.extend(
        [
            "## Audit Breakdown",
            "",
            markdown_table(["Phase", "384", "768", "1200"], phase_rows),
            "",
            markdown_table(["Seat Context", "384", "768", "1200"], seat_rows),
            "",
            markdown_table(["Raw Margin Bucket", "384", "768", "1200"], margin_rows),
            "",
            markdown_table(["Top Move", "States"], top_move_rows),
            "",
        ]
    )
    candidate_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        agg = candidate["large_suite_aggregate"]
        candidate_rows.append(
            [
                candidate["candidate"],
                fmt(float(agg[PRIMARY_BUDGET]["mean_ds"])),
                fmt(
                    float(agg[PRIMARY_BUDGET]["mean_ds"])
                    - float(summary["current_large_mean_384_256"])
                ),
                fmt(
                    float(agg[EQ_768_BUDGET]["mean_ds"])
                    - float(summary["current_large_mean_768_768"])
                ),
                fmt(
                    float(agg[EQ_1200_BUDGET]["mean_ds"])
                    - float(summary["current_large_mean_1200_1200"])
                ),
                fmt(float(candidate["delta_norm_vs_current_checkpoint"])),
                fmt(
                    float(
                        candidate["mined_state_policy_shift"][
                            "top1_changed_rate_vs_promoted_current"
                        ]
                    )
                ),
                fmt(
                    float(
                        candidate["stability_anchor_top1_preserved_rate"][
                            "top1_preserved_rate"
                        ]
                    )
                ),
            ]
        )
    lines.extend(
        [
            "## Candidate Aggregate Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Mean DS 384:256",
                    "Delta 384:256",
                    "Delta 768:768",
                    "Delta 1200:1200",
                    "Delta norm",
                    "Mined top-1 changed",
                    "Stability preserved",
                ],
                candidate_rows,
            ),
            "",
        ]
    )
    loss_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        loss_rows.append(
            [
                candidate["candidate"],
                fmt(float(candidate.get("policy_loss", 0.0)))
                if candidate.get("policy_loss") is not None
                else "n/a",
                fmt(float(candidate.get("value_loss", 0.0)))
                if candidate.get("value_loss") is not None
                else "n/a",
                fmt(float(candidate.get("validation_loss", 0.0)))
                if candidate.get("validation_loss") is not None
                else "n/a",
                candidate.get("checkpoint_sha256", "n/a"),
                candidate.get("artifact_weights_sha256", "n/a"),
            ]
        )
    lines.extend(
        [
            "## Losses And Artifacts",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Policy loss",
                    "Value loss",
                    "Validation loss",
                    "Checkpoint SHA256",
                    "Artifact weights SHA256",
                ],
                loss_rows,
            ),
            "",
        ]
    )
    fixed_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        budget_results = candidate["fixed_large_budget_results"]
        fixed_rows.append(
            [
                candidate["candidate"],
                fmt(float(budget_results[PRIMARY_BUDGET]["ds"])),
                fmt(float(budget_results["768:256"]["ds"])),
                fmt(float(budget_results[EQ_768_BUDGET]["ds"])),
                fmt(float(budget_results[EQ_1200_BUDGET]["ds"])),
                fmt(float(budget_results[ASYM_1200_256_BUDGET]["ds"])),
                fmt(float(budget_results["256:768"]["ds"])),
            ]
        )
    lines.extend(
        [
            "## Fixed Large DS Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "384:256",
                    "768:256",
                    "768:768",
                    "1200:1200",
                    "1200:256",
                    "256:768",
                ],
                fixed_rows,
            ),
            "",
        ]
    )
    heldout_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        heldout = candidate["heldout_summary"]
        heldout_rows.append(
            [
                candidate["candidate"],
                fmt(float(heldout.get("mean_ds_384_256", 0.0)))
                if heldout.get("available")
                else "n/a",
                fmt(float(heldout.get("worst_suite_ds_384_256", 0.0)))
                if heldout.get("available")
                else "n/a",
            ]
        )
    lines.extend(
        [
            "## Held-Out Mean/Worst DS Table",
            "",
            markdown_table(
                ["Candidate", "Held-out mean 384:256", "Held-out worst-suite 384:256"],
                heldout_rows,
            ),
            "",
        ]
    )
    ci_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        for key, ci in sorted(candidate.get("bootstrap_cis", {}).items()):
            ci_rows.append(
                [
                    key,
                    fmt(float(ci["mean"])),
                    fmt(float(ci["lower"])),
                    fmt(float(ci["upper"])),
                    ci["n"],
                ]
            )
    lines.extend(
        [
            "## Bootstrap CI Table",
            "",
            markdown_table(
                ["Comparison", "Mean", "Lower 95%", "Upper 95%", "Openings"],
                ci_rows,
            ),
            "",
            "## P0/P1 Split At 384:256",
            "",
        ]
    )
    split_rows: list[list[Any]] = []
    for candidate in summary["candidates"]:
        agg = candidate["large_suite_aggregate"][PRIMARY_BUDGET]
        split_rows.append(
            [
                candidate["candidate"],
                fmt(float(agg["mean_p0_score"])),
                fmt(float(agg["mean_p1_score"])),
                fmt(abs(float(agg["mean_p0_score"]) - float(agg["mean_p1_score"]))),
                fmt(float(agg["mean_duplicate_trajectory_count"])),
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["Candidate", "Mean P0", "Mean P1", "Gap", "Mean duplicates"],
                split_rows,
            ),
            "",
            "## Gate",
            "",
        ]
    )
    for candidate in summary["candidates"]:
        gate = candidate.get("default_gate")
        lines.append(
            f"- {candidate['candidate']}: `{gate['classification']}`"
            if gate
            else f"- {candidate['candidate']}: `not_run`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_artifact = Path(args.current)
    init_checkpoint = Path(args.init_checkpoint)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    fixed_large_suite = Path(args.fixed_large_suite)
    medium_suite = Path(args.medium_suite)
    heldout_suite_paths = parse_csv_paths(args.heldout_suites)
    budget_pairs = parse_budget_pairs(args.budget_pairs)
    search_budgets = parse_search_budgets(args.search_budgets)
    suite_specs = [("fixed_large", fixed_large_suite)] + [
        (path.stem, path) for path in heldout_suite_paths
    ]

    for path, label in (
        (current_artifact / "weights.json", "current artifact weights"),
        (current_artifact / "metadata.json", "current artifact metadata"),
        (init_checkpoint, "init checkpoint"),
        (generic_bootstrap, "generic bootstrap replay"),
        (random_teacher, "random teacher replay"),
        (fixed_large_suite, "fixed large suite"),
        (medium_suite, "medium suite"),
    ):
        require_existing_file(path, label)
    for path in heldout_suite_paths:
        require_existing_file(path, f"heldout suite {path.name}")

    input_summary = {
        "current_artifact_weights": verify_expected_hash(
            current_artifact / "weights.json",
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact_metadata": build_input_summary(
            current_artifact / "metadata.json"
        ),
        "init_checkpoint": verify_expected_hash(
            init_checkpoint,
            args.expected_init_checkpoint_sha256,
            "init checkpoint",
        ),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "medium_suite": build_input_summary(medium_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suite_paths
        },
    }
    profile = search_profile(args.c_puct, search_budgets)

    state_index, collection_metadata = collect_evaluation_states(
        suite_specs=suite_specs,
        artifact_path=current_artifact,
        c_puct=args.c_puct,
        seed=args.seed,
    )
    unique_states = list(state_index.values())
    state_batches = partition_batches(unique_states, args.workers)
    analyzed_rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, len(state_batches)))
    ) as pool:
        futures = [
            pool.submit(
                analyze_consensus_state_batch,
                batch=batch,
                artifact_path=str(current_artifact),
                budgets=search_budgets,
                c_puct=args.c_puct,
                seed=args.seed + (batch_index * 1000),
            )
            for batch_index, batch in enumerate(state_batches)
        ]
        for future in futures:
            analyzed_rows.extend(future.result())
    analyzed_rows.sort(key=lambda row: str(row["state_hash"]))
    write_jsonl(workdir / CONSENSUS_STATE_TABLE_FILENAME, analyzed_rows)

    consensus_audit = summarize_consensus_audit(
        rows=analyzed_rows,
        budgets=search_budgets,
        collection_metadata=collection_metadata,
        input_summary=input_summary,
        profile=profile,
    )
    write_json(workdir / CONSENSUS_AUDIT_FILENAME, consensus_audit)

    disagreement_selected, disagreement_selection_summary = (
        select_consensus_disagreement_rows(analyzed_rows)
    )
    disagreement_replay_rows = build_consensus_replay(
        selected_rows=disagreement_selected,
        artifact_path=current_artifact,
        simulations=1200,
        c_puct=args.c_puct,
        seed=args.seed + 50_000,
        workers=args.workers,
        bucket_name="consensus_disagreement_replay",
        teacher_source="balanced_current_puct_consensus_disagreement",
        selection_tier="consensus_disagreement",
    )
    write_jsonl(workdir / DISAGREEMENT_REPLAY_FILENAME, disagreement_replay_rows)
    disagreement_validation = validate_replay_rows(disagreement_replay_rows)

    disagreement_hashes = {str(row["state_hash"]) for row in disagreement_replay_rows}
    stability_selected, stability_selection_summary = select_consensus_stability_rows(
        analyzed_rows, disagreement_hashes
    )
    stability_replay_rows = build_consensus_replay(
        selected_rows=stability_selected,
        artifact_path=current_artifact,
        simulations=1200,
        c_puct=args.c_puct,
        seed=args.seed + 75_000,
        workers=args.workers,
        bucket_name="consensus_stability_replay",
        teacher_source="balanced_current_puct_consensus_stability",
        selection_tier="consensus_stability",
    )
    write_jsonl(workdir / STABILITY_REPLAY_FILENAME, stability_replay_rows)
    stability_validation = validate_replay_rows(stability_replay_rows)

    row_counts = {
        "generic_bootstrap": int(input_summary["generic_bootstrap"]["rows"]),
        "random_teacher": int(input_summary["random_teacher"]["rows"]),
        "consensus_disagreement": len(disagreement_replay_rows),
        "consensus_stability": len(stability_replay_rows),
    }
    replay_weights = {
        "generic_bootstrap": 4,
        "random_teacher": 1,
        "consensus_disagreement": int(args.disagreement_weight),
        "consensus_stability": int(args.stability_weight),
    }
    fractions = effective_sampling_fractions(row_counts, replay_weights)

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "budget_pairs": budget_pairs,
        "games_per_opening": args.games_per_opening,
        "inputs": input_summary,
        "consensus_audit": {
            **consensus_audit,
            "disagreement_selection_summary": disagreement_selection_summary,
            "stability_selection_summary": stability_selection_summary,
            "disagreement_replay_validation": disagreement_validation,
            "stability_replay_validation": stability_validation,
        },
        "replay_row_counts": row_counts,
        "replay_weights": replay_weights,
        "effective_replay_sampling_fractions": fractions,
        "guardrails": {
            "promotion": False,
            "overwrite_current": False,
            "architecture_change": False,
            "residual_v4": False,
            "full_network_training": False,
            "last_block_policy_training": False,
            "replay_weight_sweep": False,
            "lr_change": False,
            "classic_mcts_replay": False,
        },
        "candidates": [],
    }

    if (
        len(disagreement_replay_rows) < MIN_TRAINABLE_ROWS
        or len(stability_replay_rows) < MIN_TRAINABLE_ROWS
    ):
        summary["status"] = "aborted_before_training"
        summary["classification"] = classify_run(summary)
        write_json(workdir / SUMMARY_FILENAME, summary)
        REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
        return 1

    candidates: list[dict[str, Any]] = [
        {
            "name": "balanced_current_ref",
            "report_candidate_name": "current",
            "epochs": 0,
            "checkpoint_path": str(init_checkpoint),
            "artifact_dir": str(current_artifact),
            "trainable_scope": "none",
            "train_metrics": None,
        }
    ]
    pr128_ref = discover_pr128_reference()
    if pr128_ref is not None:
        candidates.append(pr128_ref)
    for epochs in (1, 2):
        name = f"consensus_w8s4_policy_head_e{epochs}"
        lane_dir = workdir / name
        lane_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_out = lane_dir / "checkpoint.npz"
        epoch_checkpoint_path = lane_dir / f"checkpoint_epoch{epochs}.npz"
        artifact_dir = lane_dir / f"artifact_{name}"
        metrics_path = lane_dir / "train_metrics.json"
        if (
            epoch_checkpoint_path.is_file()
            and (artifact_dir / "weights.json").is_file()
            and metrics_path.is_file()
        ):
            train_metrics = load_json(metrics_path)
        else:
            train_metrics = run_train(
                data_files=(
                    f"{generic_bootstrap},{random_teacher},{workdir / DISAGREEMENT_REPLAY_FILENAME},{workdir / STABILITY_REPLAY_FILENAME}"
                ),
                replay_weights=f"4,1,{args.disagreement_weight},{args.stability_weight}",
                init_checkpoint=str(init_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                epochs=epochs,
                seed=args.seed,
            )
            export_checkpoint(
                checkpoint_path=str(epoch_checkpoint_path),
                out_dir=str(artifact_dir),
                version=name,
                policy_loss=float(train_metrics.get("policy_loss", 0.0)),
                value_loss=float(train_metrics.get("value_loss", 0.0)),
            )
            write_json(metrics_path, train_metrics)
        candidates.append(
            {
                "name": name,
                "report_candidate_name": artifact_dir.name,
                "epochs": epochs,
                "checkpoint_path": str(epoch_checkpoint_path),
                "artifact_dir": str(artifact_dir),
                "trainable_scope": "policy_head",
                "train_metrics": train_metrics,
                "effective_replay_sampling_fractions": fractions,
            }
        )

    candidate_paths = ",".join(
        str(candidate["artifact_dir"]) for candidate in candidates
    )
    medium_report_path = workdir / "eval_medium" / "temperature_benchmark_report.json"
    if medium_report_path.is_file():
        medium_report = load_json(medium_report_path)
    else:
        medium_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_medium"),
            suite=str(medium_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
    fixed_large_report_path = (
        workdir / "eval_fixed_large" / "temperature_benchmark_report.json"
    )
    if fixed_large_report_path.is_file():
        fixed_large_report = load_json(fixed_large_report_path)
    else:
        fixed_large_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_fixed_large"),
            suite=str(fixed_large_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
    heldout_reports: dict[str, dict[str, Any]] = {}
    for suite_name, suite_path in suite_specs[1:]:
        heldout_report_path = (
            workdir / f"eval_{suite_name}" / "temperature_benchmark_report.json"
        )
        if heldout_report_path.is_file():
            heldout_reports[suite_name] = load_json(heldout_report_path)
        else:
            heldout_reports[suite_name] = run_opening_suite_benchmark(
                workdir=str(workdir / f"eval_{suite_name}"),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.timeout,
            )

    suite_rows = large_suite_rows(
        reports={"fixed_large": fixed_large_report, **heldout_reports},
        candidates=candidates,
    )
    disagreement_policy_rows = consensus_policy_shift_rows(disagreement_selected)
    stability_policy_rows = consensus_policy_shift_rows(stability_selected)

    pre_gate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        pre_gate_rows.append(
            build_candidate_summary_row(
                candidate=candidate,
                init_checkpoint=init_checkpoint,
                medium_report=medium_report,
                fixed_large_report=fixed_large_report,
                heldout_reports=heldout_reports,
                suite_rows=suite_rows,
                budget_pairs=budget_pairs,
                disagreement_policy_rows=disagreement_policy_rows,
                stability_policy_rows=stability_policy_rows,
                seed=args.seed,
                gate_report=None,
            )
        )
    gate_targets = gate_targets_from_summary(pre_gate_rows)
    gate_reports: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["name"] not in gate_targets:
            continue
        gate_report_path = workdir / "gate" / f"{candidate['name']}.json"
        if gate_report_path.is_file():
            gate_reports[str(candidate["name"])] = load_json(gate_report_path)
        else:
            gate_reports[str(candidate["name"])] = run_default_gate(
                candidate_path=str(candidate["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(gate_report_path),
                seed=args.seed,
                workers=args.workers,
            )

    summary_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        row = build_candidate_summary_row(
            candidate=candidate,
            init_checkpoint=init_checkpoint,
            medium_report=medium_report,
            fixed_large_report=fixed_large_report,
            heldout_reports=heldout_reports,
            suite_rows=suite_rows,
            budget_pairs=budget_pairs,
            disagreement_policy_rows=disagreement_policy_rows,
            stability_policy_rows=stability_policy_rows,
            seed=args.seed,
            gate_report=gate_reports.get(str(candidate["name"])),
        )
        row["bootstrap_cis"] = candidate_bootstrap_cis(
            suite_rows, str(candidate["name"]), args.seed
        )
        summary_candidates.append(row)

    current_row = next(
        row for row in summary_candidates if row["candidate"] == "balanced_current_ref"
    )
    summary["candidates"] = summary_candidates
    summary["gate_targets"] = gate_targets
    summary["current_large_mean_384_256"] = float(
        current_row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_ds"]
    )
    summary["current_large_mean_768_768"] = float(
        current_row["large_suite_aggregate"][EQ_768_BUDGET]["mean_ds"]
    )
    summary["current_large_mean_1200_1200"] = float(
        current_row["large_suite_aggregate"][EQ_1200_BUDGET]["mean_ds"]
    )
    summary["classification"] = classify_run(summary)
    write_json(workdir / SUMMARY_FILENAME, summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
