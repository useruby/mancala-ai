#!/usr/bin/env python3
"""Audit residual_v3 opening iteration-0 target rows with forced continuations.

This runner reconstructs the frozen selected rows and target lanes without
training, validates that the target datasets align with the selected states,
then measures whether forcing each target move causally helps relative to the
current promoted default move under current-vs-current continuations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    DEFAULT_RUNTIME_C_PUCT,
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    top_policy_move,
)
from ml.alphazero_lite.run_residual_v3_opening_iteration0_preflight import (  # noqa: E402
    EXPECTED_CURRENT_WEIGHTS_SHA256,
    SEARCH_SETTINGS,
    build_promoted_search_profile,
    canonical_json,
    canonical_state_hash,
    evaluate_search_setting,
    phase_bucket,
    search_profile_hash,
    sha256_file,
    stable_float,
    validate_guardrails,
)
from ml.alphazero_lite.run_residual_v3_opening_iteration0_train import (  # noqa: E402
    TARGET_MANIFEST_FILENAME,
    TARGET_ROWS_FILENAME,
    verify_current_artifact,
    verify_preflight_manifest,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    outcome_for_player,
)

SUMMARY_SCHEMA = "azlite_residual_v3_iteration0_target_causal_audit_v1"
TARGET_ROW_AUDIT_FILENAME = "target_row_audit.json"
TARGET_ROW_TABLE_FILENAME = "target_row_table.jsonl"
SELECTED_RESULTS_FILENAME = "selected_forced_outcomes.jsonl"
HELDOUT_RESULTS_FILENAME = "heldout_forced_outcomes.jsonl"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-residual-v3-iteration0-target-causal-audit-results.md"
)
FORCED_DEFAULT_BUDGET = 384
PRIMARY_BUDGET_KEY = "384"
LANE_LABELS = [
    "sim768_equal_override",
    "sim384_default",
    "sim768_default",
    "sim1200_default",
]
RAW_POLICY_LABEL = "raw_policy"
CURRENT_DEFAULT_LABEL = "current_default"

_WORKER_EVALUATOR: arena.ArtifactEvaluator | None = None
_WORKER_SEARCH_OPTIONS: dict[str, Any] | None = None
_WORKER_BUDGETS: list[int] | None = None
_WORKER_DEFAULT_C_PUCT: float | None = None
_WORKER_C_PUCT_SCHEDULE: dict[str, float] | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default=EXPECTED_CURRENT_WEIGHTS_SHA256,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selected-rows", required=True)
    parser.add_argument("--target-workdirs", required=True)
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


def legal_mask_for_state(state: dict[str, Any]) -> list[int]:
    legal_moves = set(KalahGame.from_state(state).possible_moves())
    return [1 if move in legal_moves else 0 for move in range(6)]


def legal_moves_from_mask(mask: list[int]) -> list[int]:
    return [index for index, value in enumerate(mask) if int(value) == 1]


def row_top_move(policy: list[float], legal_moves: list[int]) -> int | None:
    return top_policy_move(policy, legal_moves)


def row_phase(entry: dict[str, Any]) -> str:
    if "phase_bucket" in entry:
        return str(entry["phase_bucket"])
    return str(phase_bucket(entry))


def selection_tags_for_row(row: dict[str, Any]) -> list[str]:
    return [str(tag) for tag in row.get("selection_tags", [])]


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


def matrix_counts(
    rows: list[dict[str, Any]], lane_names: list[str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for left in lane_names:
        matrix[left] = {}
        for right in lane_names:
            matrix[left][right] = sum(
                1
                for row in rows
                if row["lane_top_moves"][left] == row["lane_top_moves"][right]
            )
    return matrix


def mean_or_zero(values: list[float]) -> float:
    return stable_float(statistics.fmean(values)) if values else 0.0


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(key))] += 1
    return {name: int(counts[name]) for name in sorted(counts)}


def current_default_move(
    *,
    evaluator: arena.ArtifactEvaluator,
    state: dict[str, Any],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=FORCED_DEFAULT_BUDGET,
        current_simulations=256,
        default_c_puct=float(default_c_puct),
    )
    summary = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=FORCED_DEFAULT_BUDGET,
        seed=seed,
        c_puct=c_puct,
        search_options=arena.build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=float(tactical_root_bias),
        ),
    )
    return {
        "move": None
        if summary.get("selected_move") is None
        else int(summary["selected_move"]),
        "value": stable_float(float(summary.get("value", 0.0))),
        "legal_moves": [int(move) for move in summary.get("legal_moves", [])],
        "visits": [float(value) for value in summary.get("visits", [0.0] * 6)],
        "c_puct": stable_float(c_puct),
    }


def raw_policy_details(
    evaluator: arena.ArtifactEvaluator, state: dict[str, Any]
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    legal_moves = [int(move) for move in game.possible_moves()]
    _logits, policy, value = artifact_forward_details(evaluator, game)
    move = row_top_move(policy, legal_moves)
    ranked = sorted((float(policy[move]) for move in legal_moves), reverse=True)
    margin = 1.0 if len(ranked) < 2 else ranked[0] - ranked[1]
    entropy = 0.0
    for move_idx in legal_moves:
        probability = float(policy[move_idx])
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return {
        "move": None if move is None else int(move),
        "policy": [stable_float(float(probability)) for probability in policy],
        "value": stable_float(float(value)),
        "entropy": stable_float(entropy),
        "margin": stable_float(margin),
        "legal_moves": legal_moves,
    }


def forced_continuation(
    *,
    state: dict[str, Any],
    forced_move: int,
    evaluator: arena.ArtifactEvaluator,
    simulations: int,
    c_puct: float,
    search_options: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    root_player = int(game.current_player)
    if int(forced_move) not in game.possible_moves():
        raise RuntimeError(f"forced move {forced_move} is illegal")
    if not game.move(game.pit_index(int(forced_move))):
        raise RuntimeError(f"failed to play forced move {forced_move}")
    _after_logits, after_policy, after_value = artifact_forward_details(evaluator, game)
    after_value_root = (
        float(after_value)
        if int(game.current_player) == root_player
        else -float(after_value)
    )
    move_counter = 0
    while not game.over():
        legal_moves = [int(move) for move in game.possible_moves()]
        if not legal_moves:
            break
        search = PUCT(
            evaluator=evaluator,
            simulations=int(simulations),
            c_puct=float(c_puct),
            rng=random.Random(int(seed) + move_counter),
            fpu_mode=str(search_options["fpu_mode"]),
            reuse_subtree=bool(search_options["reuse_subtree"]),
            normalize_values=bool(search_options["normalize_values"]),
            root_policy_mode=str(search_options["root_policy_mode"]),
            tactical_root_bias=float(search_options["tactical_root_bias"]),
            root_temperature=float(search_options["root_temperature"]),
        )
        _visits, root = search.run(game)
        selected_move = search.select_root_move(root, legal_moves)
        if not game.move(game.pit_index(int(selected_move))):
            raise RuntimeError(f"failed to play continuation move {selected_move}")
        move_counter += 1
    root_store = int(game.captured_seeds[root_player])
    opponent_store = int(game.captured_seeds[1 - root_player])
    return {
        "forced_move": int(forced_move),
        "post_move_policy": [stable_float(float(value)) for value in after_policy],
        "post_move_value_to_move": stable_float(float(after_value)),
        "post_move_value_root": stable_float(float(after_value_root)),
        "winner": game.winner,
        "outcome_root": stable_float(
            float(outcome_for_player(game.winner, root_player))
        ),
        "root_store": root_store,
        "opponent_store": opponent_store,
        "store_margin_root": root_store - opponent_store,
        "final_state": game.to_state(),
    }


def load_target_lanes(target_workdirs: list[Path]) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for workdir in target_workdirs:
        manifest_path = workdir / "targets" / TARGET_MANIFEST_FILENAME
        dataset_path = workdir / "targets" / TARGET_ROWS_FILENAME
        manifest = load_json(manifest_path)
        rows = load_jsonl(dataset_path)
        lane_name = str(manifest["target_generation"]["preferred_target_lane_label"])
        by_rank = {int(row["selection_rank"]): row for row in rows}
        by_state_hash = {str(row["source_state_hash"]): row for row in rows}
        lanes[lane_name] = {
            "workdir": str(workdir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "manifest": manifest,
            "rows": rows,
            "by_rank": by_rank,
            "by_state_hash": by_state_hash,
        }
    missing = [lane for lane in LANE_LABELS if lane not in lanes]
    if missing:
        raise RuntimeError(f"missing target lanes: {', '.join(missing)}")
    return lanes


def validate_target_row(
    *,
    selected_row: dict[str, Any],
    target_row: dict[str, Any],
    lane_name: str,
) -> dict[str, Any]:
    selected_hash = str(selected_row["state_hash"])
    target_hash = str(target_row["source_state_hash"])
    if selected_hash != target_hash:
        raise RuntimeError(
            f"state hash mismatch for rank {selected_row['selection_rank']} {lane_name}: {selected_hash} != {target_hash}"
        )
    if list(target_row.get("source_prefix_moves", [])) != list(
        selected_row.get("prefix_moves", [])
    ):
        raise RuntimeError(
            f"prefix mismatch for rank {selected_row['selection_rank']} {lane_name}"
        )
    legal_mask = legal_mask_for_state(selected_row["state"])
    legal_moves = legal_moves_from_mask(legal_mask)
    target_legal = [int(move) for move in target_row.get("target_legal_moves", [])]
    if legal_moves != target_legal:
        raise RuntimeError(
            f"legal mask mismatch for rank {selected_row['selection_rank']} {lane_name}"
        )
    target_policy = [float(value) for value in target_row["policy"]]
    target_top_move = row_top_move(target_policy, legal_moves)
    if target_top_move is None or target_top_move not in legal_moves:
        raise RuntimeError(
            f"illegal target top move for rank {selected_row['selection_rank']} {lane_name}"
        )
    return {
        "lane_label": lane_name,
        "legal_mask": legal_mask,
        "legal_moves": legal_moves,
        "target_top_move": int(target_top_move),
        "target_entropy": stable_float(float(target_row["target_entropy"])),
        "target_value": stable_float(float(target_row["value"])),
        "target_policy": [stable_float(float(value)) for value in target_policy],
        "target_budget_pair": str(target_row["target_budget_pair"]),
        "target_role_context": [
            str(value) for value in target_row.get("target_role_context", [])
        ],
        "target_simulations": int(target_row["target_simulations"]),
    }


def build_selected_audit_rows(
    *,
    selected_rows: list[dict[str, Any]],
    lanes: dict[str, dict[str, Any]],
    evaluator: arena.ArtifactEvaluator,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    lane_order_fingerprint: dict[str, list[str]] = {}
    for lane_name in LANE_LABELS:
        lane_order_fingerprint[lane_name] = [
            str(lanes[lane_name]["by_rank"][rank]["source_state_hash"])
            for rank in sorted(lanes[lane_name]["by_rank"])
        ]
    for lane_name in LANE_LABELS[1:]:
        if lane_order_fingerprint[lane_name] != lane_order_fingerprint[LANE_LABELS[0]]:
            raise RuntimeError(
                f"silent state/order mismatch across target lanes: {lane_name}"
            )

    for row_index, selected_row in enumerate(
        sorted(selected_rows, key=lambda row: int(row["selection_rank"]))
    ):
        raw = raw_policy_details(evaluator, selected_row["state"])
        current_default = current_default_move(
            evaluator=evaluator,
            state=selected_row["state"],
            default_c_puct=default_c_puct,
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=tactical_root_bias,
            seed=seed + row_index,
        )
        lane_details: dict[str, dict[str, Any]] = {}
        lane_top_moves: dict[str, int] = {}
        lane_entropies: dict[str, float] = {}
        lane_values: dict[str, float] = {}
        for lane_name in LANE_LABELS:
            target_row = lanes[lane_name]["by_rank"].get(
                int(selected_row["selection_rank"])
            )
            if target_row is None:
                raise RuntimeError(
                    f"missing lane row for rank {selected_row['selection_rank']} {lane_name}"
                )
            detail = validate_target_row(
                selected_row=selected_row,
                target_row=target_row,
                lane_name=lane_name,
            )
            lane_details[lane_name] = detail
            lane_top_moves[lane_name] = int(detail["target_top_move"])
            lane_entropies[lane_name] = float(detail["target_entropy"])
            lane_values[lane_name] = float(detail["target_value"])
        audit_rows.append(
            {
                "row_kind": "selected",
                "selection_rank": int(selected_row["selection_rank"]),
                "state_hash": str(selected_row["state_hash"]),
                "state": selected_row["state"],
                "raw_top_move": raw["move"],
                "raw_policy": raw["policy"],
                "raw_value": raw["value"],
                "raw_entropy": raw["entropy"],
                "raw_margin": raw["margin"],
                "current_default_move": current_default["move"],
                "current_default_value": current_default["value"],
                "current_default_legal_moves": current_default["legal_moves"],
                "current_default_visits": [
                    stable_float(float(value)) for value in current_default["visits"]
                ],
                "phase": row_phase(selected_row),
                "seat": int(
                    selected_row.get(
                        "side_to_move", selected_row["state"]["current_player"]
                    )
                ),
                "opening_prefix_ply": int(
                    selected_row.get("ply", len(selected_row.get("prefix_moves", [])))
                ),
                "first_move_family": str(
                    selected_row.get("first_move_family", "unknown")
                ),
                "selection_tags": selection_tags_for_row(selected_row),
                "prefix_moves": [
                    int(move) for move in selected_row.get("prefix_moves", [])
                ],
                "legal_mask": lane_details[LANE_LABELS[0]]["legal_mask"],
                "legal_moves": lane_details[LANE_LABELS[0]]["legal_moves"],
                "lane_details": lane_details,
                "lane_top_moves": lane_top_moves,
                "lane_entropies": lane_entropies,
                "lane_values": lane_values,
                "all_lanes_agree": len(set(lane_top_moves.values())) == 1,
                "lanes_split": len(set(lane_top_moves.values())) > 1,
            }
        )

    validation = {
        "selected_rows": len(selected_rows),
        "lane_row_counts": {
            lane_name: len(lanes[lane_name]["rows"]) for lane_name in LANE_LABELS
        },
        "state_order_fingerprint": {
            lane_name: hashlib.sha256(
                canonical_json(lane_order_fingerprint[lane_name]).encode("utf-8")
            ).hexdigest()
            for lane_name in LANE_LABELS
        },
    }
    return audit_rows, validation


def build_target_row_table(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in audit_rows:
        for lane_name in LANE_LABELS:
            detail = row["lane_details"][lane_name]
            table.append(
                {
                    "row_kind": row["row_kind"],
                    "selection_rank": row.get("selection_rank"),
                    "state_hash": row["state_hash"],
                    "lane_label": lane_name,
                    "phase": row["phase"],
                    "seat": row["seat"],
                    "opening_prefix_ply": row["opening_prefix_ply"],
                    "first_move_family": row["first_move_family"],
                    "selection_tags": row["selection_tags"],
                    "legal_mask": row["legal_mask"],
                    "raw_top_move": row["raw_top_move"],
                    "current_default_move": row["current_default_move"],
                    "target_top_move": detail["target_top_move"],
                    "target_entropy": detail["target_entropy"],
                    "target_value": detail["target_value"],
                    "target_budget_pair": detail["target_budget_pair"],
                    "target_role_context": detail["target_role_context"],
                    "all_lanes_agree": row["all_lanes_agree"],
                }
            )
    return table


def summarize_target_rows(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lane_agreement_with_raw: dict[str, int] = {}
    lane_agreement_with_current: dict[str, int] = {}
    lane_entropy: dict[str, dict[str, float]] = {}
    lane_values: dict[str, dict[str, float]] = {}
    top_move_distribution: dict[str, dict[str, int]] = {}
    disagreement_count_by_lane: dict[str, int] = {}
    disagreement_384_relevant_count_by_lane: dict[str, int] = {}
    for lane_name in LANE_LABELS:
        entropies = [float(row["lane_entropies"][lane_name]) for row in audit_rows]
        values = [float(row["lane_values"][lane_name]) for row in audit_rows]
        top_moves = [int(row["lane_top_moves"][lane_name]) for row in audit_rows]
        lane_agreement_with_raw[lane_name] = sum(
            1
            for row in audit_rows
            if row["lane_top_moves"][lane_name] == row["raw_top_move"]
        )
        lane_agreement_with_current[lane_name] = sum(
            1
            for row in audit_rows
            if row["lane_top_moves"][lane_name] == row["current_default_move"]
        )
        lane_entropy[lane_name] = {
            "mean": mean_or_zero(entropies),
            "min": stable_float(min(entropies)) if entropies else 0.0,
            "max": stable_float(max(entropies)) if entropies else 0.0,
        }
        lane_values[lane_name] = {
            "mean": mean_or_zero(values),
            "min": stable_float(min(values)) if values else 0.0,
            "max": stable_float(max(values)) if values else 0.0,
        }
        move_counts: Counter[str] = Counter(str(move) for move in top_moves)
        top_move_distribution[lane_name] = {
            key: int(move_counts[key]) for key in sorted(move_counts)
        }
        disagreement_count_by_lane[lane_name] = sum(
            1
            for row in audit_rows
            if row["lane_top_moves"][lane_name] != row["current_default_move"]
        )
        disagreement_384_relevant_count_by_lane[lane_name] = sum(
            1
            for row in audit_rows
            if row["opening_prefix_ply"] in (5, 6, 7)
            and row["lane_top_moves"][lane_name] != row["current_default_move"]
        )
    return {
        "target_top1_agreement_with_current_raw_policy": lane_agreement_with_raw,
        "target_top1_agreement_with_promoted_default_puct": lane_agreement_with_current,
        "lane_vs_lane_target_top1_agreement_matrix": matrix_counts(
            audit_rows, LANE_LABELS
        ),
        "target_entropy_by_lane": lane_entropy,
        "target_value_distribution_by_lane": lane_values,
        "target_top_move_distribution_by_pit": top_move_distribution,
        "disagreement_count_by_lane": disagreement_count_by_lane,
        "disagreement_count_in_384_256_relevant_seat_contexts": disagreement_384_relevant_count_by_lane,
        "rows_where_all_lanes_agree": sum(
            1 for row in audit_rows if row["all_lanes_agree"]
        ),
        "rows_where_lanes_split": sum(1 for row in audit_rows if row["lanes_split"]),
    }


def build_heldout_pool(
    *,
    selected_rows: list[dict[str, Any]],
    medium_suite: list[dict[str, Any]],
    large_suite: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_hashes = {str(row["state_hash"]) for row in selected_rows}
    deduped: dict[str, dict[str, Any]] = {}
    for suite_name, suite_rows in (
        ("medium", medium_suite),
        ("fixed_large", large_suite),
    ):
        for row in suite_rows:
            enriched = dict(row)
            enriched["suite_name"] = suite_name
            enriched["phase_bucket"] = row_phase(enriched)
            enriched["state_hash"] = canonical_state_hash(enriched["state"])
            if enriched["state_hash"] in selected_hashes:
                continue
            if enriched["phase_bucket"] != "early":
                continue
            ply = int(enriched.get("ply", len(enriched.get("prefix_moves", []))))
            if ply < 4 or ply > 8:
                continue
            enriched["matches_primary_ply_window"] = 5 <= ply <= 7
            if enriched["state_hash"] not in deduped:
                deduped[enriched["state_hash"]] = enriched
    return list(deduped.values())


def sample_heldout_rows(
    *,
    pool: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    primary_pool = [row for row in pool if bool(row.get("matches_primary_ply_window"))]
    fallback_pool = [
        row for row in pool if not bool(row.get("matches_primary_ply_window"))
    ]
    if len(primary_pool) + len(fallback_pool) < 256:
        raise RuntimeError(
            "held-out pool too small after filtering: "
            f"primary={len(primary_pool)} fallback={len(fallback_pool)}"
        )
    rng = random.Random(seed)
    target = min(512, len(primary_pool) + len(fallback_pool))
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
        key = (
            int(row.get("side_to_move", row["state"]["current_player"])),
            str(row.get("first_move_family", "unknown")),
            int(row.get("ply", len(row.get("prefix_moves", [])))),
        )
        grouped[key].append(row)
    for members in grouped.values():
        rng.shuffle(members)
    scaled_targets: dict[tuple[int, str, int], int] = {}
    fractional: list[tuple[float, tuple[int, str, int]]] = []
    for key, count in selected_counts.items():
        desired = (count / max(len(selected_rows), 1)) * target
        base = min(len(grouped.get(key, [])), int(math.floor(desired)))
        scaled_targets[key] = base
        fractional.append((desired - base, key))
    current_total = sum(scaled_targets.values())
    for _fraction, key in sorted(fractional, reverse=True):
        if current_total >= target:
            break
        available = len(grouped.get(key, []))
        if scaled_targets[key] < available:
            scaled_targets[key] += 1
            current_total += 1
    sampled: list[dict[str, Any]] = []
    used_hashes: set[str] = set()
    for key, quota in scaled_targets.items():
        for row in grouped.get(key, [])[:quota]:
            state_hash = str(row["state_hash"])
            if state_hash in used_hashes:
                continue
            sampled.append(row)
            used_hashes.add(state_hash)
    leftovers = [
        row for row in primary_pool if str(row["state_hash"]) not in used_hashes
    ]
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(sampled) >= target:
            break
        sampled.append(row)
        used_hashes.add(str(row["state_hash"]))
    if len(sampled) < 256:
        fallback_candidates = [
            row for row in fallback_pool if str(row["state_hash"]) not in used_hashes
        ]
        rng.shuffle(fallback_candidates)
        for row in fallback_candidates:
            if len(sampled) >= max(256, target):
                break
            sampled.append(row)
            used_hashes.add(str(row["state_hash"]))
    elif len(sampled) < target:
        fallback_candidates = [
            row for row in fallback_pool if str(row["state_hash"]) not in used_hashes
        ]
        rng.shuffle(fallback_candidates)
        for row in fallback_candidates:
            if len(sampled) >= target:
                break
            sampled.append(row)
            used_hashes.add(str(row["state_hash"]))
    sampled = sampled[:target]
    sampled.sort(
        key=lambda row: (
            int(row.get("ply", len(row.get("prefix_moves", [])))),
            int(row.get("side_to_move", row["state"]["current_player"])),
            str(row.get("first_move_family", "unknown")),
            str(row["state_hash"]),
        )
    )
    return sampled


def build_generated_row(
    *,
    source_row: dict[str, Any],
    evaluator: arena.ArtifactEvaluator,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    state = source_row["state"]
    raw = raw_policy_details(evaluator, state)
    current_default = current_default_move(
        evaluator=evaluator,
        state=state,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        seed=seed,
    )
    search_profile = {
        "default_c_puct": float(default_c_puct),
        "c_puct_overrides": dict(cpuct_schedule),
        "root_policy_mode": "deterministic",
        "tactical_root_bias": float(tactical_root_bias),
    }
    search_rows = [
        evaluate_search_setting(
            evaluator=evaluator,
            state=state,
            setting=setting,
            search_profile=search_profile,
            seed=seed,
        )
        for setting in SEARCH_SETTINGS
        if str(setting["label"]) != "sim256_default"
    ]
    lane_details: dict[str, dict[str, Any]] = {}
    for search_row in search_rows:
        lane_name = str(search_row["label"])
        policy = [stable_float(float(value)) for value in search_row["search_policy"]]
        legal_moves = [int(move) for move in search_row["legal_moves"]]
        lane_details[lane_name] = {
            "lane_label": lane_name,
            "legal_mask": legal_mask_for_state(state),
            "legal_moves": legal_moves,
            "target_top_move": int(search_row["selected_move"]),
            "target_entropy": stable_float(float(search_row["entropy"])),
            "target_value": stable_float(float(search_row["root_value"])),
            "target_policy": policy,
            "target_budget_pair": str(search_row["budget_pair"]),
            "target_role_context": [
                str(value) for value in search_row.get("role_context", [])
            ],
            "target_simulations": int(search_row["simulations"]),
        }
    lane_top_moves = {
        lane_name: int(detail["target_top_move"])
        for lane_name, detail in lane_details.items()
    }
    return {
        "row_kind": "heldout",
        "selection_rank": None,
        "state_hash": str(source_row["state_hash"]),
        "state": state,
        "raw_top_move": raw["move"],
        "raw_policy": raw["policy"],
        "raw_value": raw["value"],
        "raw_entropy": raw["entropy"],
        "raw_margin": raw["margin"],
        "current_default_move": current_default["move"],
        "current_default_value": current_default["value"],
        "current_default_legal_moves": current_default["legal_moves"],
        "current_default_visits": [
            stable_float(float(value)) for value in current_default["visits"]
        ],
        "phase": row_phase(source_row),
        "seat": int(
            source_row.get("side_to_move", source_row["state"]["current_player"])
        ),
        "opening_prefix_ply": int(
            source_row.get("ply", len(source_row.get("prefix_moves", [])))
        ),
        "first_move_family": str(source_row.get("first_move_family", "unknown")),
        "selection_tags": ["heldout_nearby"],
        "prefix_moves": [int(move) for move in source_row.get("prefix_moves", [])],
        "suite_name": str(source_row.get("suite_name", "unknown")),
        "matches_primary_ply_window": bool(
            source_row.get("matches_primary_ply_window", False)
        ),
        "legal_mask": legal_mask_for_state(state),
        "legal_moves": legal_moves_from_mask(legal_mask_for_state(state)),
        "lane_details": lane_details,
        "lane_top_moves": lane_top_moves,
        "lane_entropies": {
            lane_name: float(detail["target_entropy"])
            for lane_name, detail in lane_details.items()
        },
        "lane_values": {
            lane_name: float(detail["target_value"])
            for lane_name, detail in lane_details.items()
        },
        "all_lanes_agree": len(set(lane_top_moves.values())) == 1,
        "lanes_split": len(set(lane_top_moves.values())) > 1,
    }


def init_forced_worker(
    artifact_path: str,
    budgets: list[int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
) -> None:
    global _WORKER_EVALUATOR, _WORKER_SEARCH_OPTIONS, _WORKER_BUDGETS
    global _WORKER_DEFAULT_C_PUCT, _WORKER_C_PUCT_SCHEDULE
    _WORKER_EVALUATOR = arena.ArtifactEvaluator(Path(artifact_path))
    _WORKER_SEARCH_OPTIONS = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=float(tactical_root_bias),
    )
    _WORKER_BUDGETS = list(budgets)
    _WORKER_DEFAULT_C_PUCT = float(default_c_puct)
    _WORKER_C_PUCT_SCHEDULE = dict(cpuct_schedule)


def forced_task_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        _WORKER_EVALUATOR is None
        or _WORKER_SEARCH_OPTIONS is None
        or _WORKER_BUDGETS is None
        or _WORKER_DEFAULT_C_PUCT is None
        or _WORKER_C_PUCT_SCHEDULE is None
    ):
        raise RuntimeError("forced worker not initialized")
    results: list[dict[str, Any]] = []
    for task in rows:
        state = task["state"]
        lane_name = str(task["lane_label"])
        current_move = int(task["current_move"])
        target_move = int(task["target_move"])
        raw_move = task.get("raw_move")
        per_budget: dict[str, Any] = {}
        for budget_index, budget in enumerate(_WORKER_BUDGETS):
            seed_base = int(task["seed"]) + (budget_index * 1000)
            c_puct = resolve_budget_cpuct(
                schedule=_WORKER_C_PUCT_SCHEDULE,
                challenger_simulations=int(budget),
                current_simulations=int(budget),
                default_c_puct=float(_WORKER_DEFAULT_C_PUCT),
            )
            current_forced = forced_continuation(
                state=state,
                forced_move=current_move,
                evaluator=_WORKER_EVALUATOR,
                simulations=int(budget),
                c_puct=c_puct,
                search_options=_WORKER_SEARCH_OPTIONS,
                seed=seed_base,
            )
            target_forced = forced_continuation(
                state=state,
                forced_move=target_move,
                evaluator=_WORKER_EVALUATOR,
                simulations=int(budget),
                c_puct=c_puct,
                search_options=_WORKER_SEARCH_OPTIONS,
                seed=seed_base + 500,
            )
            raw_forced = None
            if raw_move is not None and int(raw_move) != int(current_move):
                raw_forced = forced_continuation(
                    state=state,
                    forced_move=int(raw_move),
                    evaluator=_WORKER_EVALUATOR,
                    simulations=int(budget),
                    c_puct=c_puct,
                    search_options=_WORKER_SEARCH_OPTIONS,
                    seed=seed_base + 900,
                )
            delta = float(
                target_forced["outcome_root"] - current_forced["outcome_root"]
            )
            per_budget[str(budget)] = {
                "c_puct": stable_float(c_puct),
                "current_forced": current_forced,
                "target_forced": target_forced,
                "raw_policy_forced": raw_forced,
                "target_minus_current_outcome_delta": stable_float(delta),
                "target_wins_vs_current": delta > 0.0,
                "target_ties_vs_current": delta == 0.0,
                "target_loses_vs_current": delta < 0.0,
            }
        budget_keys = [str(budget) for budget in _WORKER_BUDGETS]
        results.append(
            {
                "row_kind": str(task["row_kind"]),
                "state_hash": str(task["state_hash"]),
                "selection_rank": task.get("selection_rank"),
                "lane_label": lane_name,
                "phase": str(task["phase"]),
                "seat": int(task["seat"]),
                "opening_prefix_ply": int(task["opening_prefix_ply"]),
                "first_move_family": str(task["first_move_family"]),
                "selection_tags": list(task["selection_tags"]),
                "target_value": stable_float(float(task["target_value"])),
                "target_entropy": stable_float(float(task["target_entropy"])),
                "current_move": current_move,
                "target_move": target_move,
                "raw_move": raw_move,
                "per_budget": per_budget,
                "result_changes_across_budget": any(
                    per_budget[budget_keys[index]]["target_minus_current_outcome_delta"]
                    != per_budget[budget_keys[0]]["target_minus_current_outcome_delta"]
                    for index in range(1, len(budget_keys))
                ),
            }
        )
    return results


def partition_batches(
    items: list[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    worker_count = max(1, min(int(workers), len(items)))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    for index, item in enumerate(items):
        buckets[index % worker_count].append(item)
    return [bucket for bucket in buckets if bucket]


def run_forced_audit(
    *,
    audit_rows: list[dict[str, Any]],
    artifact_path: Path,
    budgets: list[int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    workers: int,
    seed: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    running_seed = seed
    for row in audit_rows:
        current_move = row["current_default_move"]
        if current_move is None:
            continue
        for lane_name in LANE_LABELS:
            target_move = row["lane_top_moves"][lane_name]
            tasks.append(
                {
                    "row_kind": row["row_kind"],
                    "selection_rank": row.get("selection_rank"),
                    "state_hash": row["state_hash"],
                    "state": row["state"],
                    "lane_label": lane_name,
                    "phase": row["phase"],
                    "seat": row["seat"],
                    "opening_prefix_ply": row["opening_prefix_ply"],
                    "first_move_family": row["first_move_family"],
                    "selection_tags": row["selection_tags"],
                    "target_value": row["lane_values"][lane_name],
                    "target_entropy": row["lane_entropies"][lane_name],
                    "current_move": int(current_move),
                    "target_move": int(target_move),
                    "raw_move": row["raw_top_move"],
                    "seed": running_seed,
                }
            )
            running_seed += 10000
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
    results.sort(
        key=lambda row: (
            str(row["row_kind"]),
            10**9 if row["selection_rank"] is None else int(row["selection_rank"]),
            str(row["state_hash"]),
            str(row["lane_label"]),
        )
    )
    return results


def summarize_forced_rows(
    records: list[dict[str, Any]], budgets: list[int]
) -> dict[str, Any]:
    by_lane: dict[str, dict[str, Any]] = {}
    for lane_name in LANE_LABELS:
        lane_records = [row for row in records if row["lane_label"] == lane_name]
        budget_rows: dict[str, Any] = {}
        for budget in budgets:
            budget_key = str(budget)
            diffs = [
                float(
                    row["per_budget"][budget_key]["target_minus_current_outcome_delta"]
                )
                for row in lane_records
            ]
            ci = bootstrap_ci(
                diffs,
                seed=42 + abs(hash((lane_name, budget_key))) % 100000,
                samples=DEFAULT_BOOTSTRAP_SAMPLES,
            )
            budget_rows[budget_key] = {
                **ci,
                "wins": sum(1 for diff in diffs if diff > 0.0),
                "ties": sum(1 for diff in diffs if diff == 0.0),
                "losses": sum(1 for diff in diffs if diff < 0.0),
            }
        by_lane[lane_name] = budget_rows
    return by_lane


def grouped_summary(
    records: list[dict[str, Any]], budgets: list[int], key_name: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row[key_name])].append(row)
    result: dict[str, Any] = {}
    for group_name, group_rows in sorted(grouped.items()):
        result[group_name] = summarize_forced_rows(group_rows, budgets)
    return result


def grouped_summary_bucketed(
    records: list[dict[str, Any]], budgets: list[int], key_fn
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(key_fn(row))].append(row)
    result: dict[str, Any] = {}
    for group_name, group_rows in sorted(grouped.items()):
        result[group_name] = summarize_forced_rows(group_rows, budgets)
    return result


def extreme_forced_rows(
    records: list[dict[str, Any]],
    *,
    helpful: bool,
    budget_key: str = PRIMARY_BUDGET_KEY,
    limit: int = 25,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (
            float(row["per_budget"][budget_key]["target_minus_current_outcome_delta"]),
            -float(row["target_entropy"]),
            str(row["state_hash"]),
            str(row["lane_label"]),
        ),
        reverse=helpful,
    )
    rows: list[dict[str, Any]] = []
    for row in ordered:
        delta = float(
            row["per_budget"][budget_key]["target_minus_current_outcome_delta"]
        )
        if helpful and delta <= 0.0:
            continue
        if not helpful and delta >= 0.0:
            continue
        budget = row["per_budget"][budget_key]
        rows.append(
            {
                "row_kind": row["row_kind"],
                "selection_rank": row.get("selection_rank"),
                "state_hash": row["state_hash"],
                "lane_label": row["lane_label"],
                "phase": row["phase"],
                "seat": row["seat"],
                "opening_prefix_ply": row["opening_prefix_ply"],
                "target_entropy": row["target_entropy"],
                "target_value": row["target_value"],
                "current_move": row["current_move"],
                "target_move": row["target_move"],
                "raw_move": row["raw_move"],
                "outcome_delta": stable_float(delta),
                "current_outcome": budget["current_forced"]["outcome_root"],
                "target_outcome": budget["target_forced"]["outcome_root"],
                "current_margin": budget["current_forced"]["store_margin_root"],
                "target_margin": budget["target_forced"]["store_margin_root"],
            }
        )
        if len(rows) >= limit:
            break
    return rows


def best_lane_signal(
    summary: dict[str, Any], budget_key: str
) -> tuple[str | None, dict[str, Any] | None]:
    best_name = None
    best_row = None
    best_mean = -10.0
    for lane_name in LANE_LABELS:
        row = summary.get(lane_name, {}).get(budget_key)
        if row is None:
            continue
        mean = float(row["mean"])
        if mean > best_mean:
            best_mean = mean
            best_name = lane_name
            best_row = row
    return best_name, best_row


def classify_audit(
    *,
    selected_summary: dict[str, Any],
    heldout_summary: dict[str, Any],
    selected_seat_summary: dict[str, Any],
    budgets: list[int],
) -> tuple[str, str]:
    helpful_selected = []
    helpful_heldout = []
    selected_lane_positive_across_budgets = False
    for lane_name in LANE_LABELS:
        lane_selected_good = True
        lane_heldout_good = True
        any_selected_03 = False
        any_positive_768_or_1200 = False
        lane_selected_positive_across_budgets = True
        for budget in budgets:
            budget_key = str(budget)
            selected_row = selected_summary[lane_name][budget_key]
            heldout_row = heldout_summary[lane_name][budget_key]
            if (
                float(selected_row["mean"]) >= 0.03
                and float(selected_row["lower"]) > 0.0
            ):
                any_selected_03 = True
            else:
                lane_selected_positive_across_budgets = False
            if (
                budget in (768, 1200)
                and float(selected_row["mean"]) > 0.0
                and float(selected_row["lower"]) > 0.0
            ):
                any_positive_768_or_1200 = True
            if not (
                float(selected_row["mean"]) >= 0.05
                and float(selected_row["lower"]) > 0.0
            ):
                lane_selected_good = False
            if not (
                float(heldout_row["mean"]) >= 0.05 and float(heldout_row["lower"]) > 0.0
            ):
                lane_heldout_good = False
        if lane_selected_positive_across_budgets:
            selected_lane_positive_across_budgets = True
        if any_selected_03:
            helpful_selected.append(lane_name)
        if lane_selected_good:
            helpful_selected.append(f"strong:{lane_name}")
        if lane_heldout_good:
            helpful_heldout.append(lane_name)
        if any_positive_768_or_1200:
            seat0 = (
                selected_seat_summary.get("0", {})
                .get(lane_name, {})
                .get(PRIMARY_BUDGET_KEY)
            )
            seat1 = (
                selected_seat_summary.get("1", {})
                .get(lane_name, {})
                .get(PRIMARY_BUDGET_KEY)
            )
            seat_rows = [row for row in (seat0, seat1) if row is not None]
            if seat_rows and all(
                float(row["upper"]) <= 0.0 or float(row["mean"]) < 0.03
                for row in seat_rows
            ):
                return (
                    "target_lane_context_misaligned",
                    "Targets help at higher equal-budget continuations but not in the 384-relevant seat split.",
                )
    if not helpful_selected or not selected_lane_positive_across_budgets:
        return (
            "target_rows_not_causally_helpful",
            "No target lane clears the required selected-row causal bar across continuation budgets with consistently positive confidence intervals.",
        )
    strong_selected = [
        name.split(":", 1)[1] for name in helpful_selected if name.startswith("strong:")
    ]
    if strong_selected and not helpful_heldout:
        return (
            "selected_rows_overfit",
            "Some target lanes help on the exact selected rows but the advantage does not hold on nearby held-out states.",
        )
    strong_both = [
        lane
        for lane in LANE_LABELS
        if lane in strong_selected and lane in helpful_heldout
    ]
    if strong_both:
        return (
            "targets_good_training_blocked",
            "At least one lane is causally helpful on both selected and nearby held-out states, so the failure is likely in update/generalization.",
        )
    sim1200_selected = selected_summary["sim1200_default"][PRIMARY_BUDGET_KEY]
    sim768_selected = selected_summary["sim768_equal_override"][PRIMARY_BUDGET_KEY]
    if (
        float(sim1200_selected["mean"]) - float(sim768_selected["mean"]) >= 0.03
        and float(sim1200_selected["mean"]) > 0.0
    ):
        return (
            "sim1200_lane_promising_only_if",
            "The sim1200_default lane is materially better in forced continuations even though the trained candidates did not win.",
        )
    return (
        "target_rows_not_causally_helpful",
        "Targets do not meet the causal-helpfulness bar needed to justify another supervised iteration-0 training pass.",
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    body = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        body.append("| " + " | ".join(row) + " |")
    return "\n".join(body)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def write_report(
    *,
    summary: dict[str, Any],
    selected_audit_rows: list[dict[str, Any]],
    heldout_audit_rows: list[dict[str, Any]],
    selected_forced: list[dict[str, Any]],
    heldout_forced: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# AlphaZero-Lite Residual_v3 Iteration-0 Target Causal Audit Results")
    lines.append("")
    lines.append(f"**Classification**: `{summary['classification']}`")
    lines.append("")
    lines.append("## Artifact Hash")
    lines.append("")
    lines.append(f"- current weights SHA256: `{summary['artifact_hash']}`")
    lines.append("")
    lines.append("## Promoted Search Schedule Confirmation")
    lines.append("")
    schedule = summary["promoted_search_schedule"]
    lines.append(f"- default c_puct: `{schedule['default_c_puct']}`")
    lines.append(f"- overrides: `{json.dumps(schedule['overrides'], sort_keys=True)}`")
    lines.append(
        f"- root_policy_mode: `{summary['search_profile']['root_policy_mode']}`"
    )
    lines.append(
        f"- tactical_root_bias: `{summary['search_profile']['tactical_root_bias']}`"
    )
    lines.append("")
    lines.append("## Input Hashes")
    lines.append("")
    for key, payload in sorted(summary["inputs"].items()):
        if isinstance(payload, dict) and "sha256" in payload:
            lines.append(f"- {key}: `{payload['sha256']}`")
    lines.append("")
    lines.append("## Target-Row Validation Table")
    lines.append("")
    validation = summary["target_row_validation"]
    rows = [
        [
            lane,
            str(validation["lane_row_counts"][lane]),
            validation["state_order_fingerprint"][lane],
        ]
        for lane in LANE_LABELS
    ]
    lines.append(markdown_table(["Lane", "Rows", "State Order Fingerprint"], rows))
    lines.append("")
    lines.append("## Target-Lane Agreement Matrix")
    lines.append("")
    agreement = summary["target_row_metrics"][
        "lane_vs_lane_target_top1_agreement_matrix"
    ]
    matrix_rows = [
        [left, *[str(agreement[left][right]) for right in LANE_LABELS]]
        for left in LANE_LABELS
    ]
    lines.append(markdown_table(["Lane", *LANE_LABELS], matrix_rows))
    lines.append("")
    lines.append("## Selected-Row Forced Outcome Table")
    lines.append("")
    selected_rows = []
    for lane_name in LANE_LABELS:
        row = summary["selected_forced_summary"][lane_name]
        selected_rows.append(
            [
                lane_name,
                fmt(float(row["384"]["mean"])),
                f"[{fmt(float(row['384']['lower']))}, {fmt(float(row['384']['upper']))}]",
                fmt(float(row["768"]["mean"])),
                f"[{fmt(float(row['768']['lower']))}, {fmt(float(row['768']['upper']))}]",
                fmt(float(row["1200"]["mean"])),
                f"[{fmt(float(row['1200']['lower']))}, {fmt(float(row['1200']['upper']))}]",
            ]
        )
    lines.append(
        markdown_table(
            [
                "Lane",
                "384 Mean",
                "384 CI95",
                "768 Mean",
                "768 CI95",
                "1200 Mean",
                "1200 CI95",
            ],
            selected_rows,
        )
    )
    lines.append("")
    lines.append("## Held-Out Nearby Forced Outcome Table")
    lines.append("")
    heldout_sampling = summary["heldout_sampling"]
    lines.append(
        "- held-out rows: "
        f"`{heldout_sampling['rows']}` "
        f"(`ply 5-7={heldout_sampling['primary_ply_window_rows']}`, "
        f"`fallback ply 4/8={heldout_sampling['fallback_rows']}`)"
    )
    lines.append("")
    heldout_rows = []
    for lane_name in LANE_LABELS:
        row = summary["heldout_forced_summary"][lane_name]
        heldout_rows.append(
            [
                lane_name,
                fmt(float(row["384"]["mean"])),
                f"[{fmt(float(row['384']['lower']))}, {fmt(float(row['384']['upper']))}]",
                fmt(float(row["768"]["mean"])),
                f"[{fmt(float(row['768']['lower']))}, {fmt(float(row['768']['upper']))}]",
                fmt(float(row["1200"]["mean"])),
                f"[{fmt(float(row['1200']['lower']))}, {fmt(float(row['1200']['upper']))}]",
            ]
        )
    lines.append(
        markdown_table(
            [
                "Lane",
                "384 Mean",
                "384 CI95",
                "768 Mean",
                "768 CI95",
                "1200 Mean",
                "1200 CI95",
            ],
            heldout_rows,
        )
    )
    lines.append("")
    lines.append("## Bootstrap 95% CI For Target Minus Current Outcome Delta")
    lines.append("")
    lines.append(
        "Selected rows and held-out rows use the same bootstrap procedure over per-row deltas."
    )
    lines.append("")
    lines.append("## Forced Outcome By Seat Context")
    lines.append("")
    seat_rows = []
    for seat, seat_payload in sorted(summary["selected_forced_by_seat"].items()):
        for lane_name in LANE_LABELS:
            row = seat_payload[lane_name][PRIMARY_BUDGET_KEY]
            seat_rows.append(
                [
                    f"selected seat {seat}",
                    lane_name,
                    fmt(float(row["mean"])),
                    f"[{fmt(float(row['lower']))}, {fmt(float(row['upper']))}]",
                ]
            )
    for seat, seat_payload in sorted(summary["heldout_forced_by_seat"].items()):
        for lane_name in LANE_LABELS:
            row = seat_payload[lane_name][PRIMARY_BUDGET_KEY]
            seat_rows.append(
                [
                    f"heldout seat {seat}",
                    lane_name,
                    fmt(float(row["mean"])),
                    f"[{fmt(float(row['lower']))}, {fmt(float(row['upper']))}]",
                ]
            )
    lines.append(markdown_table(["Group", "Lane", "384 Mean", "384 CI95"], seat_rows))
    lines.append("")
    lines.append("## Forced Outcome By Opening Prefix Ply")
    lines.append("")
    ply_rows = []
    for ply, payload in sorted(
        summary["selected_forced_by_opening_prefix_ply"].items()
    ):
        for lane_name in LANE_LABELS:
            row = payload[lane_name][PRIMARY_BUDGET_KEY]
            ply_rows.append([f"selected ply {ply}", lane_name, fmt(float(row["mean"]))])
    lines.append(markdown_table(["Group", "Lane", "384 Mean"], ply_rows))
    lines.append("")
    lines.append("## Forced Outcome By Target Entropy Bucket")
    lines.append("")
    entropy_rows = []
    for bucket_name, payload in sorted(
        summary["selected_forced_by_target_entropy_bucket"].items()
    ):
        for lane_name in LANE_LABELS:
            row = payload[lane_name][PRIMARY_BUDGET_KEY]
            entropy_rows.append([bucket_name, lane_name, fmt(float(row["mean"]))])
    lines.append(markdown_table(["Bucket", "Lane", "384 Mean"], entropy_rows))
    lines.append("")
    lines.append("## Forced Outcome By Target Value Bucket")
    lines.append("")
    value_rows = []
    for bucket_name, payload in sorted(
        summary["selected_forced_by_target_value_bucket"].items()
    ):
        for lane_name in LANE_LABELS:
            row = payload[lane_name][PRIMARY_BUDGET_KEY]
            value_rows.append([bucket_name, lane_name, fmt(float(row["mean"]))])
    lines.append(markdown_table(["Bucket", "Lane", "384 Mean"], value_rows))
    lines.append("")
    lines.append("## First 25 Most Harmful Target Rows")
    lines.append("")
    harmful_rows = [
        [
            str(row["selection_rank"]),
            row["lane_label"],
            row["state_hash"],
            fmt(float(row["outcome_delta"])),
            str(row["current_move"]),
            str(row["target_move"]),
        ]
        for row in summary["first_25_most_harmful_target_rows"]
    ]
    lines.append(
        markdown_table(
            ["Rank", "Lane", "State Hash", "Delta", "Current", "Target"], harmful_rows
        )
    )
    lines.append("")
    lines.append("## First 25 Most Helpful Target Rows")
    lines.append("")
    helpful_rows = [
        [
            str(row["selection_rank"]),
            row["lane_label"],
            row["state_hash"],
            fmt(float(row["outcome_delta"])),
            str(row["current_move"]),
            str(row["target_move"]),
        ]
        for row in summary["first_25_most_helpful_target_rows"]
    ]
    lines.append(
        markdown_table(
            ["Rank", "Lane", "State Hash", "Delta", "Current", "Target"], helpful_rows
        )
    )
    lines.append("")
    lines.append("## Final Classification")
    lines.append("")
    lines.append(f"- classification: `{summary['classification']}`")
    lines.append(f"- rationale: {summary['classification_rationale']}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_path = Path(args.current)
    manifest_path = Path(args.manifest)
    selected_rows_path = Path(args.selected_rows)
    medium_suite_path = Path(args.medium_suite)
    fixed_large_suite_path = Path(args.fixed_large_suite)
    target_workdirs = [
        Path(path) for path in str(args.target_workdirs).split(",") if path
    ]
    budgets = parse_budgets(args.continuation_budgets)
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)

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

    selected_rows = load_jsonl(selected_rows_path)
    target_lanes = load_target_lanes(target_workdirs)
    evaluator = arena.ArtifactEvaluator(current_path)

    selected_audit_rows, validation = build_selected_audit_rows(
        selected_rows=selected_rows,
        lanes=target_lanes,
        evaluator=evaluator,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        seed=int(args.seed),
    )
    target_row_table = build_target_row_table(selected_audit_rows)
    target_row_metrics = summarize_target_rows(selected_audit_rows)

    medium_suite = load_jsonl(medium_suite_path)
    fixed_large_suite = load_jsonl(fixed_large_suite_path)
    heldout_pool = build_heldout_pool(
        selected_rows=selected_rows,
        medium_suite=medium_suite,
        large_suite=fixed_large_suite,
    )
    heldout_source_rows = sample_heldout_rows(
        pool=heldout_pool,
        selected_rows=selected_rows,
        seed=int(args.seed),
    )
    heldout_audit_rows = [
        build_generated_row(
            source_row=row,
            evaluator=evaluator,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=int(args.seed) + index,
        )
        for index, row in enumerate(heldout_source_rows)
    ]

    selected_forced = run_forced_audit(
        audit_rows=selected_audit_rows,
        artifact_path=current_path,
        budgets=budgets,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        workers=int(args.workers),
        seed=int(args.seed),
    )
    heldout_forced = run_forced_audit(
        audit_rows=heldout_audit_rows,
        artifact_path=current_path,
        budgets=budgets,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        workers=int(args.workers),
        seed=int(args.seed) + 1_000_000,
    )

    selected_summary = summarize_forced_rows(selected_forced, budgets)
    heldout_summary = summarize_forced_rows(heldout_forced, budgets)
    selected_seat_summary = grouped_summary(selected_forced, budgets, "seat")
    heldout_seat_summary = grouped_summary(heldout_forced, budgets, "seat")
    selected_ply_summary = grouped_summary(
        selected_forced, budgets, "opening_prefix_ply"
    )
    selected_entropy_summary = grouped_summary_bucketed(
        selected_forced,
        budgets,
        lambda row: bucket_value(
            float(row["target_entropy"]), cut_points=[1.8, 2.1, 2.4]
        ),
    )
    selected_value_summary = grouped_summary_bucketed(
        selected_forced,
        budgets,
        lambda row: bucket_value(
            float(row["target_value"]), cut_points=[-0.05, 0.0, 0.1, 0.2]
        ),
    )

    classification, rationale = classify_audit(
        selected_summary=selected_summary,
        heldout_summary=heldout_summary,
        selected_seat_summary=selected_seat_summary,
        budgets=budgets,
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "classification": classification,
        "classification_rationale": rationale,
        "artifact_hash": sha256_file(current_path / "weights.json"),
        "search_profile": search_profile,
        "search_profile_hash": search_profile_hash(search_profile),
        "promoted_search_schedule": schedule_definition(
            default_c_puct=float(args.default_c_puct),
            schedule=cpuct_schedule,
        ),
        "inputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "selected_rows": {
                "path": str(selected_rows_path),
                "sha256": sha256_file(selected_rows_path),
            },
            "medium_suite": {
                "path": str(medium_suite_path),
                "sha256": sha256_file(medium_suite_path),
            },
            "fixed_large_suite": {
                "path": str(fixed_large_suite_path),
                "sha256": sha256_file(fixed_large_suite_path),
            },
            "current_weights": {
                "path": str(current_path / "weights.json"),
                "sha256": sha256_file(current_path / "weights.json"),
            },
            "target_lanes": {
                lane_name: {
                    "manifest_path": target_lanes[lane_name]["manifest_path"],
                    "manifest_sha256": target_lanes[lane_name]["manifest_sha256"],
                    "dataset_path": target_lanes[lane_name]["dataset_path"],
                    "dataset_sha256": target_lanes[lane_name]["dataset_sha256"],
                }
                for lane_name in LANE_LABELS
            },
        },
        "target_row_validation": validation,
        "target_row_metrics": target_row_metrics,
        "selected_row_count": len(selected_audit_rows),
        "heldout_row_count": len(heldout_audit_rows),
        "heldout_sampling": {
            "rows": len(heldout_audit_rows),
            "primary_ply_window_rows": sum(
                1
                for row in heldout_audit_rows
                if bool(row.get("matches_primary_ply_window"))
            ),
            "fallback_rows": sum(
                1
                for row in heldout_audit_rows
                if not bool(row.get("matches_primary_ply_window"))
            ),
        },
        "selected_forced_summary": selected_summary,
        "heldout_forced_summary": heldout_summary,
        "selected_forced_by_seat": selected_seat_summary,
        "heldout_forced_by_seat": heldout_seat_summary,
        "selected_forced_by_opening_prefix_ply": selected_ply_summary,
        "selected_forced_by_target_entropy_bucket": selected_entropy_summary,
        "selected_forced_by_target_value_bucket": selected_value_summary,
        "first_25_most_harmful_target_rows": extreme_forced_rows(
            selected_forced, helpful=False
        ),
        "first_25_most_helpful_target_rows": extreme_forced_rows(
            selected_forced, helpful=True
        ),
    }

    write_json(
        workdir / TARGET_ROW_AUDIT_FILENAME,
        {
            "schema": SUMMARY_SCHEMA,
            "validation": validation,
            "metrics": target_row_metrics,
        },
    )
    write_jsonl(workdir / TARGET_ROW_TABLE_FILENAME, target_row_table)
    write_jsonl(workdir / SELECTED_RESULTS_FILENAME, selected_forced)
    write_jsonl(workdir / HELDOUT_RESULTS_FILENAME, heldout_forced)
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(
        summary=summary,
        selected_audit_rows=selected_audit_rows,
        heldout_audit_rows=heldout_audit_rows,
        selected_forced=selected_forced,
        heldout_forced=heldout_forced,
    )
    print(f"[audit] summary={workdir / SUMMARY_FILENAME}")
    print(f"[audit] report={REPORT_PATH}")
    print(f"[audit] classification={classification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
