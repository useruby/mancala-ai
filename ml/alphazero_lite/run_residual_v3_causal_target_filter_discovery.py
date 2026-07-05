#!/usr/bin/env python3
"""Discover transparent causal filters for residual_v3 opening targets.

This runner reuses the PR #147 row-level causal audit when available, rebuilds a
selected-row causal dataset with pre-outcome features, evaluates transparent
filters on that selected set, then validates the best deployable filters on a
fresh held-out candidate pool without training or promotion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    DEFAULT_RUNTIME_C_PUCT,
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
)
from ml.alphazero_lite.run_residual_v3_iteration0_target_causal_audit import (  # noqa: E402
    LANE_LABELS,
    SELECTED_RESULTS_FILENAME,
    build_generated_row,
    build_selected_audit_rows,
    forced_task_records,
    init_forced_worker,
    load_json,
    load_jsonl,
    load_target_lanes,
    partition_batches,
    run_forced_audit,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_residual_v3_opening_iteration0_preflight import (  # noqa: E402
    EXPECTED_CURRENT_WEIGHTS_SHA256,
    build_promoted_search_profile,
    canonical_state_hash,
    phase_bucket,
    search_profile_hash,
    sha256_file,
    stable_float,
    validate_guardrails,
)
from ml.alphazero_lite.run_residual_v3_opening_iteration0_train import (  # noqa: E402
    verify_current_artifact,
    verify_preflight_manifest,
)

SCHEMA = "azlite_residual_v3_causal_target_filter_discovery_v1"
DATASET_FILENAME = "causal_row_dataset.jsonl"
DATASET_SUMMARY_FILENAME = "causal_row_dataset_summary.json"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-residual-v3-causal-target-filter-results.md"
)
FILTER_MIN_ROWS = 64
VALIDATION_MIN_ROWS = 128
FILTER_CI_WIDTH_LIMIT = 0.20
HELDOUT_TARGET_ROWS = 1024
MAX_VALIDATED_FILTERS = 3
VALUE_BUCKET_CUTS = [-0.05, 0.0, 0.10, 0.20]
ENTROPY_BUCKET_CUTS = [1.80, 2.10, 2.40]
PRIMARY_BUDGET = 384

_GENERATE_EVALUATOR: arena.ArtifactEvaluator | None = None
_GENERATE_DEFAULT_C_PUCT: float | None = None
_GENERATE_C_PUCT_SCHEDULE: dict[str, float] | None = None
_GENERATE_TACTICAL_ROOT_BIAS: float | None = None
_GENERATE_SEED_BASE: int | None = None


@dataclass(frozen=True)
class FilterDefinition:
    name: str
    description: str
    deployable: bool
    predicate: Callable[[dict[str, Any]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default=EXPECTED_CURRENT_WEIGHTS_SHA256,
    )
    parser.add_argument("--pr147-workdir", required=True)
    parser.add_argument("--target-row-table", required=True)
    parser.add_argument("--selected-rows", required=True)
    parser.add_argument("--canonical-suite", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--continuation-budgets", default="384,768,1200")
    parser.add_argument("--default-c-puct", type=float, default=DEFAULT_RUNTIME_C_PUCT)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_budgets(text: str) -> list[int]:
    budgets = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not budgets:
        raise ValueError("at least one continuation budget is required")
    unique = sorted(set(budgets))
    if len(unique) != len(budgets):
        raise ValueError("continuation budgets must be unique")
    return unique


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def mean_or_zero(values: list[float]) -> float:
    return stable_float(statistics.fmean(values)) if values else 0.0


def bucket_value(value: float, *, cut_points: list[float]) -> str:
    previous = None
    for cut in cut_points:
        if value < cut:
            if previous is None:
                return f"<{cut:.2f}"
            return f"[{previous:.2f},{cut:.2f})"
        previous = cut
    assert previous is not None
    return f">={previous:.2f}"


def normalize_policy(policy: list[float], legal_moves: list[int]) -> list[float]:
    normalized = [0.0] * 6
    if not legal_moves:
        return normalized
    total = sum(float(policy[move]) for move in legal_moves)
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            normalized[move] = stable_float(uniform)
        return normalized
    for move in legal_moves:
        normalized[move] = stable_float(float(policy[move]) / total)
    residual = 1.0 - sum(normalized)
    normalized[max(legal_moves)] = stable_float(normalized[max(legal_moves)] + residual)
    return normalized


def top_share_and_margin(
    policy: list[float], legal_moves: list[int]
) -> tuple[float, float]:
    if not legal_moves:
        return 0.0, 0.0
    ranked = sorted((float(policy[move]) for move in legal_moves), reverse=True)
    top_share = ranked[0]
    margin = 1.0 if len(ranked) < 2 else ranked[0] - ranked[1]
    return stable_float(top_share), stable_float(margin)


def build_selected_state_index(
    selected_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        state_hash = str(row.get("state_hash") or canonical_state_hash(row["state"]))
        search_results = {
            str(item["label"]): item for item in row.get("search_results", [])
        }
        index[state_hash] = {
            "row": row,
            "search_results": search_results,
            "phase": str(row.get("phase_bucket", phase_bucket(row))),
            "seat": int(row.get("side_to_move", row["state"]["current_player"])),
            "opening_prefix_ply": int(row.get("ply", len(row.get("prefix_moves", [])))),
            "first_move_family": str(row.get("first_move_family", "unknown")),
            "selection_tags": [str(tag) for tag in row.get("selection_tags", [])],
            "prefix_moves": [int(move) for move in row.get("prefix_moves", [])],
        }
    return index


def build_row_table_state_index(
    row_table: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    state_index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in row_table:
        state_index[str(row["state_hash"])][str(row["lane_label"])] = row
    return state_index


def build_forced_index(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["state_hash"]), str(record["lane_label"])): record
        for record in records
    }


def build_causal_row(
    *,
    source_name: str,
    row_kind: str,
    state_hash: str,
    lane_label: str,
    state: dict[str, Any],
    lane_row: dict[str, Any],
    lane_state_rows: dict[str, dict[str, Any]],
    forced_record: dict[str, Any],
    state_meta: dict[str, Any] | None,
    budgets: list[int],
    search_profile_hash_value: str,
    current_cpuct_schedule: dict[str, float],
    default_c_puct: float,
) -> dict[str, Any]:
    target_top_move = int(lane_row["target_top_move"])
    legal_mask = [int(value) for value in lane_row["legal_mask"]]
    legal_moves = [index for index, value in enumerate(legal_mask) if value == 1]
    search_row = (
        None if state_meta is None else state_meta["search_results"].get(lane_label)
    )
    if search_row is not None:
        target_policy = normalize_policy(
            [float(value) for value in search_row["search_policy"]],
            [int(move) for move in search_row["legal_moves"]],
        )
        target_top_share = stable_float(target_policy[target_top_move])
        target_margin = stable_float(float(search_row["margin"]))
        effective_c_puct = stable_float(float(search_row["c_puct"]))
    else:
        target_policy = [0.0] * 6
        target_policy[target_top_move] = 1.0
        target_top_share = 1.0
        target_margin = 1.0
        challenger_sims, current_sims = [
            int(part) for part in str(lane_row["target_budget_pair"]).split(":", 1)
        ]
        effective_c_puct = stable_float(
            resolve_budget_cpuct(
                schedule=current_cpuct_schedule,
                challenger_simulations=challenger_sims,
                current_simulations=current_sims,
                default_c_puct=default_c_puct,
            )
        )

    agreement_count = sum(
        1
        for other_lane in LANE_LABELS
        if int(lane_state_rows[other_lane]["target_top_move"]) == target_top_move
    )
    deltas = {
        str(budget): stable_float(
            float(
                forced_record["per_budget"][str(budget)][
                    "target_minus_current_outcome_delta"
                ]
            )
        )
        for budget in budgets
    }
    worst_budget_mean = stable_float(
        min(float(deltas[str(budget)]) for budget in budgets)
    )
    mean_delta = stable_float(
        statistics.fmean(float(deltas[str(budget)]) for budget in budgets)
    )
    promoted_puct_visit_share = None
    current_visits = (
        None if state_meta is None else state_meta["row"].get("current_default_visits")
    )
    if current_visits is not None:
        total_visits = sum(float(current_visits[move]) for move in legal_moves)
        if total_visits > 0.0:
            promoted_puct_visit_share = stable_float(
                float(current_visits[target_top_move]) / total_visits
            )

    return {
        "source": source_name,
        "row_kind": row_kind,
        "selection_rank": None
        if state_meta is None
        else int(state_meta["row"]["selection_rank"]),
        "state_hash": state_hash,
        "state": state,
        "lane_name": lane_label,
        "phase": str(lane_row["phase"]),
        "seat": int(lane_row["seat"]),
        "opening_prefix_ply": int(lane_row["opening_prefix_ply"]),
        "first_move_family": str(lane_row["first_move_family"]),
        "selection_tags": [str(tag) for tag in lane_row.get("selection_tags", [])],
        "selection_tags_key": ",".join(
            sorted(str(tag) for tag in lane_row.get("selection_tags", []))
        ),
        "legal_mask": legal_mask,
        "legal_move_count": len(legal_moves),
        "raw_policy_top_move": lane_row.get("raw_top_move"),
        "current_default_move": lane_row.get("current_default_move"),
        "target_top_move": target_top_move,
        "target_equals_raw_top_move": target_top_move == lane_row.get("raw_top_move"),
        "target_equals_promoted_puct_top_move": target_top_move
        == lane_row.get("current_default_move"),
        "target_disagrees_with_raw": target_top_move != lane_row.get("raw_top_move"),
        "target_disagrees_with_promoted_puct": target_top_move
        != lane_row.get("current_default_move"),
        "all_lanes_agree": bool(lane_row.get("all_lanes_agree", False)),
        "lane_agreement_count": int(agreement_count),
        "pairwise_lane_agreement_count": int(max(0, agreement_count - 1)),
        "target_entropy": stable_float(float(lane_row["target_entropy"])),
        "target_entropy_bucket": bucket_value(
            float(lane_row["target_entropy"]),
            cut_points=ENTROPY_BUCKET_CUTS,
        ),
        "target_value": stable_float(float(lane_row["target_value"])),
        "target_value_bucket": bucket_value(
            float(lane_row["target_value"]),
            cut_points=VALUE_BUCKET_CUTS,
        ),
        "target_top_move_pit": int(target_top_move),
        "target_budget_pair": str(lane_row["target_budget_pair"]),
        "target_top_share": stable_float(target_top_share),
        "target_margin": stable_float(target_margin),
        "raw_top1_margin": None,
        "puct_visit_share": promoted_puct_visit_share,
        "effective_c_puct": effective_c_puct,
        "search_profile_hash": search_profile_hash_value,
        "delta_384": deltas.get("384"),
        "delta_768": deltas.get("768"),
        "delta_1200": deltas.get("1200"),
        "mean_delta": mean_delta,
        "worst_budget_delta": worst_budget_mean,
        "helpful_all": all(float(value) >= 0.0 for value in deltas.values()),
        "harmful_any": any(float(value) <= -1.0 for value in deltas.values()),
        "primary_helpful": float(deltas.get("384", 0.0)) > 0.0,
        "robust_helpful": mean_delta > 0.05
        and all(float(value) >= 0.0 for value in deltas.values()),
        "per_budget": forced_record["per_budget"],
        "state_has_historical_minus_two": False,
    }


def hydrate_selected_dataset(
    *,
    row_table: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    selected_forced: list[dict[str, Any]],
    budgets: list[int],
    search_profile_hash_value: str,
    cpuct_schedule: dict[str, float],
    default_c_puct: float,
) -> list[dict[str, Any]]:
    selected_index = build_selected_state_index(selected_rows)
    row_table_index = build_row_table_state_index(row_table)
    forced_index = build_forced_index(selected_forced)
    dataset_rows: list[dict[str, Any]] = []
    for state_hash, lane_rows in sorted(row_table_index.items()):
        selected_meta = selected_index.get(state_hash)
        if selected_meta is None:
            raise RuntimeError(f"selected state missing from manifest: {state_hash}")
        state = selected_meta["row"]["state"]
        for lane_label in LANE_LABELS:
            lane_row = lane_rows.get(lane_label)
            forced_row = forced_index.get((state_hash, lane_label))
            if lane_row is None or forced_row is None:
                raise RuntimeError(
                    f"missing selected lane payload for {state_hash} {lane_label}"
                )
            row = build_causal_row(
                source_name="selected",
                row_kind="selected",
                state_hash=state_hash,
                lane_label=lane_label,
                state=state,
                lane_row=lane_row,
                lane_state_rows=lane_rows,
                forced_record=forced_row,
                state_meta=selected_meta,
                budgets=budgets,
                search_profile_hash_value=search_profile_hash_value,
                current_cpuct_schedule=cpuct_schedule,
                default_c_puct=default_c_puct,
            )
            search_row = selected_meta["search_results"].get(lane_label)
            raw_search = selected_meta["search_results"].get("sim256_default")
            row["raw_top1_margin"] = (
                None
                if raw_search is None
                else stable_float(float(raw_search["margin"]))
            )
            row["puct_visit_share"] = (
                None
                if search_row is None
                else stable_float(float(search_row["top_share"]))
            )
            dataset_rows.append(row)

    severe_states = {
        row["state_hash"]
        for row in dataset_rows
        if any(float(row[f"delta_{budget}"]) <= -2.0 for budget in (384, 768, 1200))
    }
    for row in dataset_rows:
        row["state_has_historical_minus_two"] = row["state_hash"] in severe_states
    return dataset_rows


def build_dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_lane: dict[str, Any] = {}
    for lane_name in LANE_LABELS:
        lane_rows = [row for row in rows if row["lane_name"] == lane_name]
        by_lane[lane_name] = {
            "rows": len(lane_rows),
            "unique_states": len({row["state_hash"] for row in lane_rows}),
            "mean_delta_384": mean_or_zero(
                [float(row["delta_384"]) for row in lane_rows]
            ),
            "mean_delta_768": mean_or_zero(
                [float(row["delta_768"]) for row in lane_rows]
            ),
            "mean_delta_1200": mean_or_zero(
                [float(row["delta_1200"]) for row in lane_rows]
            ),
            "harmful_any_rate": stable_float(
                sum(1 for row in lane_rows if row["harmful_any"])
                / max(len(lane_rows), 1)
            ),
            "robust_helpful_rate": stable_float(
                sum(1 for row in lane_rows if row["robust_helpful"])
                / max(len(lane_rows), 1)
            ),
        }
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "unique_states": len({row["state_hash"] for row in rows}),
        "rows_by_lane": by_lane,
        "seat_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(str(row["seat"]) for row in rows).items())
        },
        "opening_prefix_ply_distribution": {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row["opening_prefix_ply"]) for row in rows).items()
            )
        },
        "helpful_all_rate": stable_float(
            sum(1 for row in rows if row["helpful_all"]) / max(len(rows), 1)
        ),
        "harmful_any_rate": stable_float(
            sum(1 for row in rows if row["harmful_any"]) / max(len(rows), 1)
        ),
        "robust_helpful_rate": stable_float(
            sum(1 for row in rows if row["robust_helpful"]) / max(len(rows), 1)
        ),
        "severe_harm_state_count": len(
            {row["state_hash"] for row in rows if row["state_has_historical_minus_two"]}
        ),
    }


def bootstrap_worst_budget_ci(
    rows: list[dict[str, Any]], budgets: list[int], seed: int
) -> dict[str, Any]:
    if not rows:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "samples": DEFAULT_BOOTSTRAP_SAMPLES,
            "n": 0,
        }
    per_row = [[float(row[f"delta_{budget}"]) for budget in budgets] for row in rows]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(DEFAULT_BOOTSTRAP_SAMPLES):
        sampled = [per_row[rng.randrange(len(per_row))] for _ in range(len(per_row))]
        means = [
            statistics.fmean(sample[idx] for sample in sampled)
            for idx in range(len(budgets))
        ]
        values.append(float(min(means)))
    values.sort()
    lower_index = int(0.025 * len(values))
    upper_index = min(len(values) - 1, int(0.975 * len(values)))
    mean_value = min(
        statistics.fmean([row[idx] for row in per_row]) for idx in range(len(budgets))
    )
    return {
        "mean": stable_float(mean_value),
        "lower": stable_float(values[lower_index]),
        "upper": stable_float(values[upper_index]),
        "samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "n": len(rows),
    }


def evaluate_filter(
    *,
    definition: FilterDefinition,
    rows: list[dict[str, Any]],
    budgets: list[int],
    seed: int,
) -> dict[str, Any]:
    selected = [row for row in rows if definition.predicate(row)]
    delta_columns = {
        budget: [float(row[f"delta_{budget}"]) for row in selected]
        for budget in budgets
    }
    delta_384_ci = bootstrap_ci(
        delta_columns.get(384, []), seed=seed, samples=DEFAULT_BOOTSTRAP_SAMPLES
    )
    worst_budget_ci = bootstrap_worst_budget_ci(selected, budgets, seed + 1)
    harmful_rate = sum(1 for row in selected if row["harmful_any"]) / max(
        len(selected), 1
    )
    robust_rate = sum(1 for row in selected if row["robust_helpful"]) / max(
        len(selected), 1
    )
    means = {str(budget): mean_or_zero(delta_columns[budget]) for budget in budgets}
    rejected_reasons: list[str] = []
    if len(selected) < FILTER_MIN_ROWS:
        rejected_reasons.append(f"selected rows < {FILTER_MIN_ROWS}")
    if harmful_rate > 0.10:
        rejected_reasons.append("harmful_any rate > 10%")
    if min(float(means[str(budget)]) for budget in budgets) <= 0.0:
        rejected_reasons.append("worst-budget mean <= 0")
    if (
        float(delta_384_ci["upper"]) - float(delta_384_ci["lower"])
        > FILTER_CI_WIDTH_LIMIT
        or float(worst_budget_ci["upper"]) - float(worst_budget_ci["lower"])
        > FILTER_CI_WIDTH_LIMIT
    ):
        rejected_reasons.append("CI too wide to interpret")
    return {
        "filter_name": definition.name,
        "description": definition.description,
        "deployable": definition.deployable,
        "selected_rows": len(selected),
        "selected_unique_states": len({row["state_hash"] for row in selected}),
        "mean_delta_384": means.get("384", 0.0),
        "mean_delta_768": means.get("768", 0.0),
        "mean_delta_1200": means.get("1200", 0.0),
        "worst_budget_mean": stable_float(
            min(float(means[str(budget)]) for budget in budgets)
        )
        if selected
        else 0.0,
        "harmful_any_rate": stable_float(harmful_rate),
        "robust_helpful_rate": stable_float(robust_rate),
        "delta_384_ci": delta_384_ci,
        "worst_budget_ci": worst_budget_ci,
        "seat_coverage": {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row["seat"]) for row in selected).items()
            )
        },
        "opening_ply_coverage": {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row["opening_prefix_ply"]) for row in selected).items()
            )
        },
        "top_move_distribution": {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row["target_top_move"]) for row in selected).items()
            )
        },
        "rejected_reasons": rejected_reasons,
        "passes_selected_gate": not rejected_reasons,
    }


def positive_entropy_buckets(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_entropy_bucket"])].append(row)
    buckets = []
    for bucket_name, members in sorted(grouped.items()):
        if len(members) < 16:
            continue
        mean_384 = statistics.fmean(float(row["delta_384"]) for row in members)
        harmful_rate = sum(1 for row in members if row["harmful_any"]) / len(members)
        if mean_384 > 0.05 and harmful_rate <= 0.10:
            buckets.append(bucket_name)
    return buckets


def build_filter_definitions(rows: list[dict[str, Any]]) -> list[FilterDefinition]:
    positive_entropy = set(positive_entropy_buckets(rows))
    filters = [
        FilterDefinition(
            name="sim1200_default",
            description="lane == sim1200_default",
            deployable=True,
            predicate=lambda row: row["lane_name"] == "sim1200_default",
        ),
        FilterDefinition(
            name="sim1200_value_lt_neg005",
            description="lane == sim1200_default AND target_value < -0.05",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default"
                and float(row["target_value"]) < -0.05
            ),
        ),
        FilterDefinition(
            name="sim1200_entropy_ge_240",
            description="lane == sim1200_default AND target_entropy >= 2.40",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default"
                and float(row["target_entropy"]) >= 2.40
            ),
        ),
        FilterDefinition(
            name="sim1200_ply6",
            description="lane == sim1200_default AND opening_prefix_ply == 6",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default"
                and int(row["opening_prefix_ply"]) == 6
            ),
        ),
        FilterDefinition(
            name="sim1200_disagree_raw_agree_puct",
            description="lane == sim1200_default AND target disagrees with raw but agrees with promoted PUCT",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default"
                and bool(row["target_disagrees_with_raw"])
                and bool(row["target_equals_promoted_puct_top_move"])
            ),
        ),
        FilterDefinition(
            name="lane_agreement_ge_2",
            description="lane agreement count >= 2",
            deployable=True,
            predicate=lambda row: int(row["lane_agreement_count"]) >= 2,
        ),
        FilterDefinition(
            name="exclude_state_any_minus_two",
            description="exclude rows where any lane historically produced a -2.0 delta on the same state",
            deployable=False,
            predicate=lambda row: not bool(row["state_has_historical_minus_two"]),
        ),
        FilterDefinition(
            name="target_value_nonpositive_bucket",
            description="target value bucket in [-0.05, 0.00) or < -0.05",
            deployable=True,
            predicate=lambda row: (
                str(row["target_value_bucket"]) in {"<-0.05", "[-0.05,0.00)"}
            ),
        ),
        FilterDefinition(
            name="positive_entropy_bucket_slice",
            description="target entropy bucket in positive PR #147 slice",
            deployable=True,
            predicate=lambda row, buckets=positive_entropy: (
                str(row["target_entropy_bucket"]) in buckets
            ),
        ),
        FilterDefinition(
            name="sim1200_seat0",
            description="lane == sim1200_default AND seat == 0",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default" and int(row["seat"]) == 0
            ),
        ),
        FilterDefinition(
            name="sim1200_seat1",
            description="lane == sim1200_default AND seat == 1",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default" and int(row["seat"]) == 1
            ),
        ),
        FilterDefinition(
            name="sim1200_seat0_ply6",
            description="lane == sim1200_default AND seat == 0 AND opening_prefix_ply == 6",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default"
                and int(row["seat"]) == 0
                and int(row["opening_prefix_ply"]) == 6
            ),
        ),
        FilterDefinition(
            name="sim1200_seat1_ply6",
            description="lane == sim1200_default AND seat == 1 AND opening_prefix_ply == 6",
            deployable=True,
            predicate=lambda row: (
                row["lane_name"] == "sim1200_default"
                and int(row["seat"]) == 1
                and int(row["opening_prefix_ply"]) == 6
            ),
        ),
    ]
    return filters


def load_source_suite_rows(path: Path, suite_name: str) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["suite_name"] = suite_name
        item["source_row_index"] = index
        item["phase_bucket"] = str(item.get("phase_bucket", phase_bucket(item)))
        item["state_hash"] = canonical_state_hash(item["state"])
        item["side_to_move"] = int(
            item.get("side_to_move", item["state"]["current_player"])
        )
        item["ply"] = int(item.get("ply", len(item.get("prefix_moves", []))))
        item["first_move_family"] = str(item.get("first_move_family", "unknown"))
        item["matches_primary_ply_window"] = 5 <= int(item["ply"]) <= 7
        enriched.append(item)
    return enriched


def build_candidate_pool(
    *,
    canonical_rows: list[dict[str, Any]],
    medium_rows: list[dict[str, Any]],
    large_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_hashes = {
        str(row.get("state_hash") or canonical_state_hash(row["state"]))
        for row in selected_rows
    }
    deduped: dict[str, dict[str, Any]] = {}
    for source_row in [*canonical_rows, *medium_rows, *large_rows]:
        state_hash = str(source_row["state_hash"])
        if state_hash in selected_hashes:
            continue
        if str(source_row["phase_bucket"]) != "early":
            continue
        if int(source_row["ply"]) < 4 or int(source_row["ply"]) > 8:
            continue
        if state_hash not in deduped:
            deduped[state_hash] = source_row
    return list(deduped.values())


def stratified_sample_rows(
    *,
    pool: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    target_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    primary_pool = [row for row in pool if bool(row.get("matches_primary_ply_window"))]
    fallback_pool = [
        row for row in pool if not bool(row.get("matches_primary_ply_window"))
    ]
    if len(primary_pool) + len(fallback_pool) < target_rows:
        raise RuntimeError(
            "held-out candidate pool too small after filtering: "
            f"primary={len(primary_pool)} fallback={len(fallback_pool)}"
        )
    selected_counts: Counter[tuple[int, str, int]] = Counter(
        (
            int(row.get("side_to_move", row["state"]["current_player"])),
            str(row.get("first_move_family", "unknown")),
            int(row.get("ply", len(row.get("prefix_moves", [])))),
        )
        for row in selected_rows
    )
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in primary_pool:
        key = (int(row["side_to_move"]), str(row["first_move_family"]), int(row["ply"]))
        grouped[key].append(row)
    rng = random.Random(seed)
    for members in grouped.values():
        rng.shuffle(members)
    target = min(target_rows, len(primary_pool) + len(fallback_pool))
    quotas: dict[tuple[int, str, int], int] = {}
    fractional: list[tuple[float, tuple[int, str, int]]] = []
    for key, count in selected_counts.items():
        desired = (count / max(len(selected_rows), 1)) * target
        base = min(len(grouped.get(key, [])), int(math.floor(desired)))
        quotas[key] = base
        fractional.append((desired - base, key))
    total = sum(quotas.values())
    for _fraction, key in sorted(fractional, reverse=True):
        if total >= target:
            break
        available = len(grouped.get(key, []))
        if quotas[key] < available:
            quotas[key] += 1
            total += 1
    sampled: list[dict[str, Any]] = []
    used_hashes: set[str] = set()
    for key, quota in quotas.items():
        for row in grouped.get(key, [])[:quota]:
            if row["state_hash"] in used_hashes:
                continue
            sampled.append(row)
            used_hashes.add(str(row["state_hash"]))
    leftovers = [row for row in primary_pool if row["state_hash"] not in used_hashes]
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(sampled) >= target:
            break
        sampled.append(row)
        used_hashes.add(str(row["state_hash"]))
    if len(sampled) < target:
        fallbacks = [
            row for row in fallback_pool if row["state_hash"] not in used_hashes
        ]
        rng.shuffle(fallbacks)
        for row in fallbacks:
            if len(sampled) >= target:
                break
            sampled.append(row)
            used_hashes.add(str(row["state_hash"]))
    sampled.sort(
        key=lambda row: (
            int(row["ply"]),
            int(row["side_to_move"]),
            str(row["first_move_family"]),
            str(row["state_hash"]),
        )
    )
    return sampled[:target]


def init_generate_worker(
    artifact_path: str,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed_base: int,
) -> None:
    global _GENERATE_EVALUATOR, _GENERATE_DEFAULT_C_PUCT, _GENERATE_C_PUCT_SCHEDULE
    global _GENERATE_TACTICAL_ROOT_BIAS, _GENERATE_SEED_BASE
    _GENERATE_EVALUATOR = arena.ArtifactEvaluator(Path(artifact_path))
    _GENERATE_DEFAULT_C_PUCT = float(default_c_puct)
    _GENERATE_C_PUCT_SCHEDULE = dict(cpuct_schedule)
    _GENERATE_TACTICAL_ROOT_BIAS = float(tactical_root_bias)
    _GENERATE_SEED_BASE = int(seed_base)


def generate_row_worker(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if (
        _GENERATE_EVALUATOR is None
        or _GENERATE_DEFAULT_C_PUCT is None
        or _GENERATE_C_PUCT_SCHEDULE is None
        or _GENERATE_TACTICAL_ROOT_BIAS is None
        or _GENERATE_SEED_BASE is None
    ):
        raise RuntimeError("generate worker not initialized")
    index, source_row = task
    return build_generated_row(
        source_row=source_row,
        evaluator=_GENERATE_EVALUATOR,
        default_c_puct=float(_GENERATE_DEFAULT_C_PUCT),
        cpuct_schedule=dict(_GENERATE_C_PUCT_SCHEDULE),
        tactical_root_bias=float(_GENERATE_TACTICAL_ROOT_BIAS),
        seed=int(_GENERATE_SEED_BASE) + int(index),
    )


def generate_heldout_audit_rows(
    *,
    source_rows: list[dict[str, Any]],
    artifact_path: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    workers: int,
    seed: int,
) -> list[dict[str, Any]]:
    tasks = list(enumerate(source_rows))
    if workers <= 1:
        init_generate_worker(
            str(artifact_path),
            default_c_puct,
            cpuct_schedule,
            tactical_root_bias,
            seed,
        )
        return [generate_row_worker(task) for task in tasks]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(int(workers), len(tasks)),
        initializer=init_generate_worker,
        initargs=(
            str(artifact_path),
            float(default_c_puct),
            dict(cpuct_schedule),
            float(tactical_root_bias),
            int(seed),
        ),
    ) as executor:
        return list(executor.map(generate_row_worker, tasks))


def hydrate_heldout_dataset(
    *,
    audit_rows: list[dict[str, Any]],
    budgets: list[int],
    search_profile_hash_value: str,
    cpuct_schedule: dict[str, float],
    default_c_puct: float,
) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        lane_state_rows = {
            lane_name: {
                "phase": audit_row["phase"],
                "seat": audit_row["seat"],
                "opening_prefix_ply": audit_row["opening_prefix_ply"],
                "first_move_family": audit_row["first_move_family"],
                "selection_tags": audit_row["selection_tags"],
                "legal_mask": audit_row["legal_mask"],
                "raw_top_move": audit_row["raw_top_move"],
                "current_default_move": audit_row["current_default_move"],
                "target_top_move": audit_row["lane_details"][lane_name][
                    "target_top_move"
                ],
                "target_entropy": audit_row["lane_details"][lane_name][
                    "target_entropy"
                ],
                "target_value": audit_row["lane_details"][lane_name]["target_value"],
                "target_budget_pair": audit_row["lane_details"][lane_name][
                    "target_budget_pair"
                ],
                "all_lanes_agree": audit_row["all_lanes_agree"],
            }
            for lane_name in LANE_LABELS
        }
        for lane_name in LANE_LABELS:
            lane_row = lane_state_rows[lane_name]
            placeholder_forced = {
                "per_budget": {
                    str(budget): {"target_minus_current_outcome_delta": 0.0}
                    for budget in budgets
                }
            }
            row = build_causal_row(
                source_name="heldout_candidate",
                row_kind="heldout_candidate",
                state_hash=str(audit_row["state_hash"]),
                lane_label=lane_name,
                state=audit_row["state"],
                lane_row=lane_row,
                lane_state_rows=lane_state_rows,
                forced_record=placeholder_forced,
                state_meta=None,
                budgets=budgets,
                search_profile_hash_value=search_profile_hash_value,
                current_cpuct_schedule=cpuct_schedule,
                default_c_puct=default_c_puct,
            )
            row["raw_top1_margin"] = stable_float(float(audit_row["raw_margin"]))
            detail = audit_row["lane_details"][lane_name]
            target_policy = [float(value) for value in detail["target_policy"]]
            row["puct_visit_share"] = stable_float(
                float(target_policy[int(detail["target_top_move"])])
            )
            row.pop("per_budget")
            row.pop("delta_384")
            row.pop("delta_768")
            row.pop("delta_1200")
            row.pop("mean_delta")
            row.pop("worst_budget_delta")
            row.pop("helpful_all")
            row.pop("harmful_any")
            row.pop("primary_helpful")
            row.pop("robust_helpful")
            dataset_rows.append(row)
    return dataset_rows


def build_forced_tasks(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    running_seed = seed
    for row in rows:
        tasks.append(
            {
                "row_kind": row["row_kind"],
                "selection_rank": row.get("selection_rank"),
                "state_hash": row["state_hash"],
                "state": row["state"],
                "lane_label": row["lane_name"],
                "phase": row["phase"],
                "seat": row["seat"],
                "opening_prefix_ply": row["opening_prefix_ply"],
                "first_move_family": row["first_move_family"],
                "selection_tags": row["selection_tags"],
                "target_value": row["target_value"],
                "target_entropy": row["target_entropy"],
                "current_move": int(row["current_default_move"]),
                "target_move": int(row["target_top_move"]),
                "raw_move": row["raw_policy_top_move"],
                "seed": running_seed,
            }
        )
        running_seed += 10000
    return tasks


def run_forced_tasks(
    *,
    tasks: list[dict[str, Any]],
    artifact_path: Path,
    budgets: list[int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    workers: int,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    batches = partition_batches(tasks, workers)
    if len(batches) == 1:
        init_forced_worker(
            str(artifact_path),
            budgets,
            default_c_puct,
            cpuct_schedule,
            tactical_root_bias,
        )
        return forced_task_records(batches[0])
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(batches),
        initializer=init_forced_worker,
        initargs=(
            str(artifact_path),
            budgets,
            float(default_c_puct),
            dict(cpuct_schedule),
            float(tactical_root_bias),
        ),
    ) as executor:
        for batch_rows in executor.map(forced_task_records, batches):
            results.extend(batch_rows)
    results.sort(key=lambda row: (str(row["state_hash"]), str(row["lane_label"])))
    return results


def apply_forced_results(
    rows: list[dict[str, Any]], forced_records: list[dict[str, Any]], budgets: list[int]
) -> list[dict[str, Any]]:
    forced_index = build_forced_index(forced_records)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        record = forced_index[(str(row["state_hash"]), str(row["lane_name"]))]
        merged = dict(row)
        merged["per_budget"] = record["per_budget"]
        merged["delta_384"] = stable_float(
            float(record["per_budget"]["384"]["target_minus_current_outcome_delta"])
        )
        merged["delta_768"] = stable_float(
            float(record["per_budget"]["768"]["target_minus_current_outcome_delta"])
        )
        merged["delta_1200"] = stable_float(
            float(record["per_budget"]["1200"]["target_minus_current_outcome_delta"])
        )
        merged["mean_delta"] = stable_float(
            statistics.fmean(
                float(
                    record["per_budget"][str(budget)][
                        "target_minus_current_outcome_delta"
                    ]
                )
                for budget in budgets
            )
        )
        merged["worst_budget_delta"] = stable_float(
            min(
                float(
                    record["per_budget"][str(budget)][
                        "target_minus_current_outcome_delta"
                    ]
                )
                for budget in budgets
            )
        )
        merged["helpful_all"] = all(
            float(merged[f"delta_{budget}"]) >= 0.0 for budget in (384, 768, 1200)
        )
        merged["harmful_any"] = any(
            float(merged[f"delta_{budget}"]) <= -1.0 for budget in (384, 768, 1200)
        )
        merged["primary_helpful"] = float(merged["delta_384"]) > 0.0
        merged["robust_helpful"] = float(merged["mean_delta"]) > 0.05 and all(
            float(merged[f"delta_{budget}"]) >= 0.0 for budget in (384, 768, 1200)
        )
        enriched.append(merged)
    return enriched


def detect_degenerate_distribution(metrics: dict[str, Any]) -> bool:
    seat_coverage = metrics.get("seat_coverage", {})
    ply_coverage = metrics.get("opening_ply_coverage", {})
    return (
        len([value for value in seat_coverage.values() if value > 0]) < 2
        or len([value for value in ply_coverage.values() if value > 0]) < 2
    )


def classify_results(
    *,
    filter_results: list[dict[str, Any]],
    heldout_results: list[dict[str, Any]],
) -> tuple[str, str, str]:
    selected_pass = [row for row in filter_results if row["passes_selected_gate"]]
    heldout_positive = [
        row
        for row in heldout_results
        if row.get("evaluated")
        and int(row["selected_rows"]) >= VALIDATION_MIN_ROWS
        and float(row["mean_delta_384"]) >= 0.05
        and float(row["delta_384_ci"]["lower"]) > 0.01
        and float(row["worst_budget_mean"]) >= 0.0
        and float(row["harmful_any_rate"]) <= 0.10
        and not detect_degenerate_distribution(row)
    ]
    if heldout_positive:
        non_sim1200 = [
            row
            for row in heldout_positive
            if not row["filter_name"].startswith("sim1200")
        ]
        if non_sim1200:
            return (
                "causal_filter_found",
                "At least one deployable transparent filter validates on held-out rows with positive 384 CI and non-negative worst-budget mean.",
                "next PR may train one small causal-filtered target candidate",
            )
        if len(heldout_positive) == 1 or all(
            row["filter_name"].startswith("sim1200") for row in heldout_positive
        ):
            return (
                "sim1200_context_filter_promising",
                "Only a simple sim1200_default context filter validates on held-out rows.",
                "test exactly one sim1200 filtered training candidate, with no lane sweep",
            )
    if selected_pass and not heldout_positive:
        return (
            "selected_set_overfit",
            "Transparent filters looked promising on the selected rows but failed or shrank below usable size on held-out validation.",
            "stop this selected-row target recipe and redesign selection",
        )
    nondeployable_only = any(
        row["passes_selected_gate"] and not row["deployable"] for row in filter_results
    ) and not any(
        row["passes_selected_gate"] and row["deployable"] for row in filter_results
    )
    if nondeployable_only:
        return (
            "target_generation_needs_rework",
            "Helpful subsets only appear under non-deployable or post-outcome filters, not under deployable pre-outcome features.",
            "improve target generation and search diagnostics before training",
        )
    return (
        "no_predictive_causal_filter",
        "No deployable transparent filter produced a robust held-out positive causal signal.",
        "stop residual-v3 opening iteration-0 target training",
    )


def reconstruct_selected_forced_if_needed(
    *,
    pr147_workdir: Path,
    pr147_summary: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    current_path: Path,
    budgets: list[int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    workers: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected_forced_path = pr147_workdir / SELECTED_RESULTS_FILENAME
    if selected_forced_path.is_file():
        return load_jsonl(selected_forced_path)
    target_lanes_summary = pr147_summary.get("inputs", {}).get("target_lanes", {})
    target_workdirs = [
        Path(payload["dataset_path"]).parents[1]
        for payload in target_lanes_summary.values()
    ]
    if len(target_workdirs) != len(LANE_LABELS):
        raise RuntimeError(
            "cannot reconstruct selected forced outcomes: missing PR #147 target lane workdirs"
        )
    lanes = load_target_lanes(target_workdirs)
    evaluator = arena.ArtifactEvaluator(current_path)
    selected_audit_rows, _validation = build_selected_audit_rows(
        selected_rows=selected_rows,
        lanes=lanes,
        evaluator=evaluator,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        seed=seed,
    )
    selected_forced = run_forced_audit(
        audit_rows=selected_audit_rows,
        artifact_path=current_path,
        budgets=budgets,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        workers=workers,
        seed=seed,
    )
    write_jsonl(selected_forced_path, selected_forced)
    return selected_forced


def write_report(
    *,
    summary: dict[str, Any],
    filter_results: list[dict[str, Any]],
    heldout_results: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# AlphaZero-Lite Residual_v3 Causal Target Filter Results")
    lines.append("")
    lines.append(f"**Classification**: `{summary['classification']}`")
    lines.append("")
    lines.append("## Artifact Hash")
    lines.append("")
    lines.append(f"- current weights SHA256: `{summary['artifact_hash']}`")
    lines.append("")
    lines.append("## Promoted Search Schedule Confirmation")
    lines.append("")
    lines.append(
        f"- default c_puct: `{summary['promoted_search_schedule']['default_c_puct']}`"
    )
    lines.append(
        f"- overrides: `{json.dumps(summary['promoted_search_schedule']['overrides'], sort_keys=True)}`"
    )
    lines.append(
        f"- root_policy_mode: `{summary['search_profile']['root_policy_mode']}`"
    )
    lines.append(
        f"- tactical_root_bias: `{summary['search_profile']['tactical_root_bias']}`"
    )
    lines.append("")
    lines.append("## PR #147 Input Hashes")
    lines.append("")
    for key, payload in sorted(summary["pr147_inputs"].items()):
        if isinstance(payload, dict) and "sha256" in payload:
            lines.append(f"- {key}: `{payload['sha256']}`")
    lines.append("")
    lines.append("## Causal Row Dataset Summary")
    lines.append("")
    dataset_summary = summary["causal_row_dataset_summary"]
    lines.append(f"- rows: `{dataset_summary['rows']}`")
    lines.append(f"- unique states: `{dataset_summary['unique_states']}`")
    lines.append(f"- harmful_any rate: `{dataset_summary['harmful_any_rate']}`")
    lines.append(f"- robust_helpful rate: `{dataset_summary['robust_helpful_rate']}`")
    lines.append("")
    lines.append("## Filter Discovery Table")
    lines.append("")
    discovery_rows = []
    for row in filter_results:
        discovery_rows.append(
            [
                row["filter_name"],
                str(row["selected_rows"]),
                str(row["selected_unique_states"]),
                fmt(float(row["mean_delta_384"])),
                fmt(float(row["worst_budget_mean"])),
                f"{100.0 * float(row['harmful_any_rate']):.1f}%",
                f"[{fmt(float(row['delta_384_ci']['lower']))}, {fmt(float(row['delta_384_ci']['upper']))}]",
                "pass"
                if row["passes_selected_gate"]
                else "; ".join(row["rejected_reasons"]),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Filter",
                "Rows",
                "States",
                "384 Mean",
                "Worst Mean",
                "Harmful",
                "384 CI95",
                "Status",
            ],
            discovery_rows,
        )
    )
    lines.append("")
    lines.append("## Held-Out Filter Validation Table")
    lines.append("")
    heldout_rows = []
    for row in heldout_results:
        status = "not run"
        if row.get("evaluated"):
            status = "validated"
        elif row.get("selected_rows", 0) < VALIDATION_MIN_ROWS:
            status = f"< {VALIDATION_MIN_ROWS} rows"
        heldout_rows.append(
            [
                row["filter_name"],
                str(row["selected_rows"]),
                fmt(float(row.get("mean_delta_384", 0.0))),
                (
                    "n/a"
                    if not row.get("evaluated")
                    else f"[{fmt(float(row['delta_384_ci']['lower']))}, {fmt(float(row['delta_384_ci']['upper']))}]"
                ),
                fmt(float(row.get("worst_budget_mean", 0.0))),
                f"{100.0 * float(row.get('harmful_any_rate', 0.0)):.1f}%",
                status,
            ]
        )
    lines.append(
        markdown_table(
            [
                "Filter",
                "Rows",
                "384 Mean",
                "384 CI95",
                "Worst Mean",
                "Harmful",
                "Status",
            ],
            heldout_rows,
        )
    )
    lines.append("")
    lines.append("## Harmful-Row Analysis")
    lines.append("")
    harmful = sorted(
        selected_rows,
        key=lambda row: (
            float(row["delta_384"]),
            str(row["state_hash"]),
            str(row["lane_name"]),
        ),
    )[:10]
    harmful_rows = [
        [
            row["lane_name"],
            row["state_hash"],
            fmt(float(row["delta_384"])),
            str(row["target_top_move"]),
            str(row["current_default_move"]),
        ]
        for row in harmful
    ]
    lines.append(
        markdown_table(
            ["Lane", "State Hash", "384 Delta", "Target", "Current"], harmful_rows
        )
    )
    lines.append("")
    lines.append("## Helpful-Row Analysis")
    lines.append("")
    helpful = sorted(
        selected_rows,
        key=lambda row: (
            -float(row["delta_384"]),
            str(row["state_hash"]),
            str(row["lane_name"]),
        ),
    )[:10]
    helpful_rows = [
        [
            row["lane_name"],
            row["state_hash"],
            fmt(float(row["delta_384"])),
            str(row["target_top_move"]),
            str(row["current_default_move"]),
        ]
        for row in helpful
    ]
    lines.append(
        markdown_table(
            ["Lane", "State Hash", "384 Delta", "Target", "Current"], helpful_rows
        )
    )
    lines.append("")
    lines.append("## Recommended Next Action")
    lines.append("")
    lines.append(f"- {summary['recommended_next_action']}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_path = Path(args.current)
    pr147_workdir = Path(args.pr147_workdir)
    pr147_summary_path = pr147_workdir / "summary_metrics.json"
    target_row_table_path = Path(args.target_row_table)
    selected_rows_path = Path(args.selected_rows)
    canonical_suite_path = Path(args.canonical_suite)
    medium_suite_path = Path(args.medium_suite)
    fixed_large_suite_path = Path(args.fixed_large_suite)
    budgets = parse_budgets(args.continuation_budgets)
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    manifest_path = selected_rows_path.parent / "iteration0_training_manifest.json"

    manifest, _selected_rows_verified = verify_preflight_manifest(
        manifest_path=manifest_path,
        selected_rows_path=selected_rows_path,
    )
    current_metadata = verify_current_artifact(
        current_path=current_path,
        expected_weights_sha256=str(args.expected_current_weights_sha256),
        manifest=manifest,
    )
    search_profile = build_promoted_search_profile()
    search_profile["default_c_puct"] = float(args.default_c_puct)
    search_profile["c_puct_overrides"] = dict(cpuct_schedule)
    search_profile["tactical_root_bias"] = float(args.tactical_root_bias)
    validate_guardrails(
        search_profile=search_profile,
        model_type=str(current_metadata["architecture"]["model_type"]),
    )
    profile_hash = search_profile_hash(search_profile)

    selected_rows = load_jsonl(selected_rows_path)
    pr147_summary = load_json(pr147_summary_path)
    row_table = load_jsonl(target_row_table_path)
    selected_forced = reconstruct_selected_forced_if_needed(
        pr147_workdir=pr147_workdir,
        pr147_summary=pr147_summary,
        selected_rows=selected_rows,
        current_path=current_path,
        budgets=budgets,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        workers=int(args.workers),
        seed=int(args.seed),
    )

    causal_rows = hydrate_selected_dataset(
        row_table=row_table,
        selected_rows=selected_rows,
        selected_forced=selected_forced,
        budgets=budgets,
        search_profile_hash_value=profile_hash,
        cpuct_schedule=cpuct_schedule,
        default_c_puct=float(args.default_c_puct),
    )
    causal_dataset_summary = build_dataset_summary(causal_rows)
    write_jsonl(workdir / DATASET_FILENAME, causal_rows)
    write_json(workdir / DATASET_SUMMARY_FILENAME, causal_dataset_summary)

    filter_definitions = build_filter_definitions(causal_rows)
    filter_results = [
        evaluate_filter(
            definition=definition,
            rows=causal_rows,
            budgets=budgets,
            seed=int(args.seed) + index,
        )
        for index, definition in enumerate(filter_definitions)
    ]
    selected_pass_filters = [
        row
        for row in filter_results
        if row["passes_selected_gate"] and row["deployable"]
    ]
    selected_pass_filters.sort(
        key=lambda row: (
            -float(row["delta_384_ci"]["lower"]),
            -float(row["mean_delta_384"]),
            -float(row["worst_budget_mean"]),
            -int(row["selected_rows"]),
        )
    )
    validation_filters = selected_pass_filters[:MAX_VALIDATED_FILTERS]
    if not validation_filters:
        fallback_filters = [
            row
            for row in filter_results
            if row["deployable"] and int(row["selected_rows"]) >= FILTER_MIN_ROWS
        ]
        fallback_filters.sort(
            key=lambda row: (
                -float(row["mean_delta_384"]),
                -float(row["delta_384_ci"]["lower"]),
                -int(row["selected_rows"]),
            )
        )
        validation_filters = fallback_filters[:MAX_VALIDATED_FILTERS]

    canonical_rows = load_source_suite_rows(canonical_suite_path, "canonical")
    medium_rows = load_source_suite_rows(medium_suite_path, "medium")
    large_rows = load_source_suite_rows(fixed_large_suite_path, "fixed_large")
    candidate_pool = build_candidate_pool(
        canonical_rows=canonical_rows,
        medium_rows=medium_rows,
        large_rows=large_rows,
        selected_rows=selected_rows,
    )
    heldout_source_rows = stratified_sample_rows(
        pool=candidate_pool,
        selected_rows=selected_rows,
        target_rows=HELDOUT_TARGET_ROWS,
        seed=int(args.seed),
    )
    heldout_audit_rows = generate_heldout_audit_rows(
        source_rows=heldout_source_rows,
        artifact_path=current_path,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        workers=int(args.workers),
        seed=int(args.seed),
    )
    heldout_candidate_rows = hydrate_heldout_dataset(
        audit_rows=heldout_audit_rows,
        budgets=budgets,
        search_profile_hash_value=profile_hash,
        cpuct_schedule=cpuct_schedule,
        default_c_puct=float(args.default_c_puct),
    )

    heldout_results: list[dict[str, Any]] = []
    filter_definition_map = {
        definition.name: definition for definition in filter_definitions
    }
    for index, selected_filter in enumerate(validation_filters):
        definition = filter_definition_map[selected_filter["filter_name"]]
        filtered_rows = [
            row for row in heldout_candidate_rows if definition.predicate(row)
        ]
        base_result = {
            **selected_filter,
            "selected_set_filter_performance": selected_filter,
            "heldout_candidate_rows": len(filtered_rows),
            "selected_rows": len(filtered_rows),
            "selected_unique_states": len({row["state_hash"] for row in filtered_rows}),
            "evaluated": False,
        }
        if len(filtered_rows) < VALIDATION_MIN_ROWS:
            heldout_results.append(base_result)
            continue
        forced_records = run_forced_tasks(
            tasks=build_forced_tasks(
                filtered_rows, seed=int(args.seed) + (index * 100000)
            ),
            artifact_path=current_path,
            budgets=budgets,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            workers=int(args.workers),
        )
        evaluated_rows = apply_forced_results(filtered_rows, forced_records, budgets)
        metrics = evaluate_filter(
            definition=definition,
            rows=evaluated_rows,
            budgets=budgets,
            seed=int(args.seed) + 50000 + index,
        )
        metrics["evaluated"] = True
        metrics["drop_from_selected_to_heldout_rows"] = int(
            selected_filter["selected_rows"]
        ) - int(metrics["selected_rows"])
        metrics["delta_from_selected_to_heldout_mean_384"] = stable_float(
            float(metrics["mean_delta_384"]) - float(selected_filter["mean_delta_384"])
        )
        heldout_results.append(metrics)

    classification, rationale, next_action = classify_results(
        filter_results=filter_results,
        heldout_results=heldout_results,
    )
    summary = {
        "schema": SCHEMA,
        "classification": classification,
        "classification_rationale": rationale,
        "recommended_next_action": next_action,
        "artifact_hash": sha256_file(current_path / "weights.json"),
        "search_profile": search_profile,
        "search_profile_hash": profile_hash,
        "promoted_search_schedule": {
            "default_c_puct": float(args.default_c_puct),
            "overrides": dict(cpuct_schedule),
        },
        "pr147_inputs": {
            "target_row_table": {
                "path": str(target_row_table_path),
                "sha256": sha256_file(target_row_table_path),
            },
            "selected_forced_outcomes": {
                "path": str(pr147_workdir / SELECTED_RESULTS_FILENAME),
                "sha256": sha256_file(pr147_workdir / SELECTED_RESULTS_FILENAME),
            },
            "selected_rows": {
                "path": str(selected_rows_path),
                "sha256": sha256_file(selected_rows_path),
            },
            "canonical_suite": {
                "path": str(canonical_suite_path),
                "sha256": sha256_file(canonical_suite_path),
            },
            "medium_suite": {
                "path": str(medium_suite_path),
                "sha256": sha256_file(medium_suite_path),
            },
            "fixed_large_suite": {
                "path": str(fixed_large_suite_path),
                "sha256": sha256_file(fixed_large_suite_path),
            },
            "pr147_summary": {
                "path": str(pr147_summary_path),
                "sha256": sha256_file(pr147_summary_path),
            },
        },
        "causal_row_dataset_summary": causal_dataset_summary,
        "filter_discovery": filter_results,
        "heldout_candidate_pool": {
            "rows": len(candidate_pool),
            "sampled_rows": len(heldout_source_rows),
            "seat_distribution": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(
                        str(row["side_to_move"]) for row in heldout_source_rows
                    ).items()
                )
            },
            "opening_ply_distribution": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(str(row["ply"]) for row in heldout_source_rows).items()
                )
            },
        },
        "heldout_filter_validation": heldout_results,
    }
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(
        summary=summary,
        filter_results=filter_results,
        heldout_results=heldout_results,
        selected_rows=causal_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
