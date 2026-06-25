#!/usr/bin/env python3
"""Counterfactual validation of balanced-current PUCT consensus targets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_balanced_opening_puct_replay import (  # noqa: E402
    markdown_table,
    parse_csv_paths,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    build_input_summary,
    margin_bucket,
    read_jsonl,
    require_existing_file,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    outcome_for_player,
)

SUMMARY_SCHEMA = "azlite_balanced_current_puct_target_counterfactual_v1"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-balanced-current-puct-target-counterfactual-results.md"
)
SUMMARY_FILENAME = "summary_metrics.json"
DISAGREEMENT_RESULTS_FILENAME = "disagreement_counterfactual_results.jsonl"
STABILITY_RESULTS_FILENAME = "stability_counterfactual_results.jsonl"
MIN_DISAGREEMENT_STATES = 512
DISAGREEMENT_VISIT_SHARE_THRESHOLD = 0.55
STABILITY_MARGIN_THRESHOLD = 0.05


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_balanced_current_puct_target_counterfactual"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--consensus-state-table", required=True)
    parser.add_argument("--consensus-audit", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--sample-disagreement-states", type=int, default=1024)
    parser.add_argument("--sample-stability-states", type=int, default=512)
    parser.add_argument("--continuation-budgets", default="384,1200")
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_budgets(text: str) -> list[int]:
    budgets = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not budgets:
        raise ValueError("at least one continuation budget is required")
    if len(set(budgets)) != len(budgets):
        raise ValueError("continuation budgets must be unique")
    return sorted(budgets)


def partition_batches(
    items: list[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    workers = max(1, min(workers, len(items) or 1))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for index, item in enumerate(items):
        buckets[index % workers].append(item)
    return [bucket for bucket in buckets if bucket]


def suite_bucket(row: dict[str, Any]) -> str:
    return "fixed_large" if str(row.get("suite_name")) == "fixed_large" else "heldout"


def visit_share_bucket(share: float) -> str:
    if share < 0.65:
        return "0.55 <= share < 0.65"
    if share < 0.75:
        return "0.65 <= share < 0.75"
    if share < 0.85:
        return "0.75 <= share < 0.85"
    return "share >= 0.85"


def search_profile(c_puct: float, budgets: list[int]) -> dict[str, Any]:
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0
    )
    profile = {
        "c_puct": float(c_puct),
        "budgets": list(budgets),
        "fpu_mode": str(options["fpu_mode"]),
        "reuse_subtree": bool(options["reuse_subtree"]),
        "normalize_values": bool(options["normalize_values"]),
        "root_policy_mode": str(options["root_policy_mode"]),
        "tactical_root_bias": float(options["tactical_root_bias"]),
        "root_temperature": float(options["root_temperature"]),
    }
    encoded = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    profile["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return profile


def unique_rows_by_state_hash(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_hash = str(row["state_hash"])
        existing = unique.get(state_hash)
        if existing is None:
            unique[state_hash] = row
            continue
        if int(row.get("occurrence_count", 0)) > int(
            existing.get("occurrence_count", 0)
        ):
            unique[state_hash] = row
    return list(unique.values())


def disagreement_candidate(row: dict[str, Any]) -> bool:
    search_384 = row.get("searches", {}).get("384", {})
    search_768 = row.get("searches", {}).get("768", {})
    search_1200 = row.get("searches", {}).get("1200", {})
    stable_top1 = row.get("stable_search_top1")
    return bool(
        row.get("legal_mask_valid")
        and row.get("raw_top1") is not None
        and stable_top1 is not None
        and len(row.get("legal_moves", [])) > 1
        and row.get("raw_top1") != stable_top1
        and row.get("search_top1_stable_across_budgets")
        and search_384.get("top1") == stable_top1
        and search_768.get("top1") == stable_top1
        and search_1200.get("top1") == stable_top1
        and float(search_1200.get("top1_visit_share", 0.0))
        >= DISAGREEMENT_VISIT_SHARE_THRESHOLD
    )


def stability_candidate(row: dict[str, Any]) -> bool:
    search_384 = row.get("searches", {}).get("384", {})
    search_768 = row.get("searches", {}).get("768", {})
    search_1200 = row.get("searches", {}).get("1200", {})
    stable_top1 = row.get("stable_search_top1")
    raw_top1 = row.get("raw_top1")
    return bool(
        row.get("legal_mask_valid")
        and raw_top1 is not None
        and stable_top1 is not None
        and len(row.get("legal_moves", [])) > 1
        and row.get("search_top1_stable_across_budgets")
        and raw_top1 == stable_top1
        and search_384.get("top1") == stable_top1
        and search_768.get("top1") == stable_top1
        and search_1200.get("top1") == stable_top1
        and float(search_1200.get("top1_visit_share", 0.0))
        >= DISAGREEMENT_VISIT_SHARE_THRESHOLD
        and float(row.get("raw_margin", 0.0)) >= STABILITY_MARGIN_THRESHOLD
    )


def stratification_key(row: dict[str, Any]) -> tuple[Any, ...]:
    search_1200 = row["searches"]["1200"]
    return (
        suite_bucket(row),
        str(row.get("suite_name", "")),
        str(row.get("dominant_seat_context", "unknown")),
        str(row.get("phase", "unknown")),
        margin_bucket(float(row.get("raw_margin", 0.0))),
        visit_share_bucket(float(search_1200.get("top1_visit_share", 0.0))),
        bool(row.get("poor_384_256_p0_game", False)),
        int(search_1200.get("top1", -1)),
    )


def shuffled_group_queues(
    rows: list[dict[str, Any]], rng: random.Random
) -> dict[tuple[Any, ...], deque[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[stratification_key(row)].append(row)
    queues: dict[tuple[Any, ...], deque[dict[str, Any]]] = {}
    for key, members in grouped.items():
        rng.shuffle(members)
        queues[key] = deque(members)
    return queues


def round_robin_sample(
    rows: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    *,
    skip_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    skip_hashes = set() if skip_hashes is None else set(skip_hashes)
    queues = shuffled_group_queues(rows, rng)
    ordered_keys = list(queues)
    rng.shuffle(ordered_keys)
    selected: list[dict[str, Any]] = []
    selected_hashes = set(skip_hashes)
    while ordered_keys and len(selected) < target:
        next_keys: list[tuple[Any, ...]] = []
        for key in ordered_keys:
            queue = queues[key]
            while queue and str(queue[0]["state_hash"]) in selected_hashes:
                queue.popleft()
            if not queue:
                continue
            row = queue.popleft()
            state_hash = str(row["state_hash"])
            if state_hash in selected_hashes:
                continue
            selected.append(row)
            selected_hashes.add(state_hash)
            if queue:
                next_keys.append(key)
            if len(selected) >= target:
                break
        ordered_keys = next_keys
    return selected


def select_disagreement_rows(
    rows: list[dict[str, Any]], target: int, rng: random.Random
) -> list[dict[str, Any]]:
    poor_rows = [row for row in rows if bool(row.get("poor_384_256_p0_game"))]
    poor_target = min(len(poor_rows), max(128, target // 6))
    selected = round_robin_sample(poor_rows, poor_target, rng)
    selected_hashes = {str(row["state_hash"]) for row in selected}
    selected.extend(
        round_robin_sample(
            rows, target - len(selected), rng, skip_hashes=selected_hashes
        )
    )
    return selected[:target]


def second_best_move(
    policy: list[float], legal_moves: list[int], *, exclude_move: int
) -> int | None:
    candidates = [move for move in legal_moves if int(move) != int(exclude_move)]
    if not candidates:
        return None
    return min(candidates, key=lambda move: (-float(policy[move]), move))


def stability_alternative_move(row: dict[str, Any]) -> tuple[int, str]:
    legal_moves = [int(move) for move in row["legal_moves"]]
    shared_top1 = int(row["raw_top1"])
    raw_second = second_best_move(
        row["raw_policy"], legal_moves, exclude_move=shared_top1
    )
    if raw_second is not None:
        return raw_second, "raw_second_best"
    puct_second = second_best_move(
        row["searches"]["1200"]["policy"], legal_moves, exclude_move=shared_top1
    )
    if puct_second is not None:
        return puct_second, "puct_second_best"
    for move in sorted(legal_moves):
        if move != shared_top1:
            return int(move), "fallback_legal"
    raise RuntimeError(f"no alternative legal move for {row['state_hash']}")


def root_value_for_player(
    value: float, *, to_move_player: int, root_player: int
) -> float:
    return float(value) if int(to_move_player) == int(root_player) else -float(value)


def forced_continuation(
    *,
    state: dict[str, Any],
    forced_move: int,
    evaluator: ArtifactEvaluator,
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
    after_value_root = root_value_for_player(
        float(after_value),
        to_move_player=int(game.current_player),
        root_player=root_player,
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
            rng=random.Random(seed + move_counter),
            fpu_mode=str(search_options["fpu_mode"]),
            reuse_subtree=bool(search_options["reuse_subtree"]),
            normalize_values=bool(search_options["normalize_values"]),
            root_policy_mode=str(search_options["root_policy_mode"]),
            tactical_root_bias=float(search_options["tactical_root_bias"]),
            root_temperature=float(search_options["root_temperature"]),
        )
        _visits, root = search.run(game)
        selected_move = search.select_root_move(root, legal_moves)
        if not game.move(game.pit_index(selected_move)):
            raise RuntimeError(f"failed to play continuation move {selected_move}")
        move_counter += 1
    root_store = int(game.captured_seeds[root_player])
    opponent_store = int(game.captured_seeds[1 - root_player])
    return {
        "forced_move": int(forced_move),
        "post_move_policy": [float(value) for value in after_policy],
        "post_move_value_to_move": float(after_value),
        "post_move_value_root": float(after_value_root),
        "winner": game.winner,
        "outcome_root": float(outcome_for_player(game.winner, root_player)),
        "root_store": root_store,
        "opponent_store": opponent_store,
        "store_margin_root": root_store - opponent_store,
        "final_state": game.to_state(),
    }


def analyze_counterfactual_batch(
    *,
    batch: list[dict[str, Any]],
    artifact_path: str,
    budgets: list[int],
    c_puct: float,
    seed: int,
    stability_mode: bool,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact_path))
    search_options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0
    )
    results: list[dict[str, Any]] = []
    for index, row in enumerate(batch):
        state_hash = str(row["state_hash"])
        base: dict[str, Any] = {
            "state_hash": state_hash,
            "suite_name": str(row["suite_name"]),
            "suite_bucket": suite_bucket(row),
            "suite_names": list(row.get("suite_names", [])),
            "opening_prefix": list(row.get("opening_prefix", [])),
            "opening_prefix_text": str(row.get("opening_prefix_text", "")),
            "phase": str(row.get("phase", "unknown")),
            "dominant_seat_context": str(row.get("dominant_seat_context", "unknown")),
            "side_to_move": int(
                row.get("side_to_move", row["state"]["current_player"])
            ),
            "challenger_player": int(row.get("challenger_player", 0)),
            "poor_384_256_p0_game": bool(row.get("poor_384_256_p0_game", False)),
            "legal_moves": [int(move) for move in row["legal_moves"]],
            "raw_top1": int(row["raw_top1"]),
            "puct_top1": int(row["searches"]["1200"]["top1"]),
            "raw_policy": [float(value) for value in row["raw_policy"]],
            "puct_policy_1200": [
                float(value) for value in row["searches"]["1200"]["policy"]
            ],
            "raw_margin": float(row["raw_margin"]),
            "puct_visit_share_1200": float(row["searches"]["1200"]["top1_visit_share"]),
            "comparison_kind": "stability" if stability_mode else "disagreement",
            "pre_move_raw_value": float(row["raw_value"]),
            "per_budget": {},
        }
        alternative_move: int | None = None
        alternative_move_int = 0
        if stability_mode:
            alternative_move, alternative_source = stability_alternative_move(row)
            assert alternative_move is not None
            alternative_move_int = int(alternative_move)
            base["shared_top1"] = int(row["raw_top1"])
            base["alternative_move"] = alternative_move_int
            base["alternative_move_source"] = alternative_source
        for budget_index, budget in enumerate(budgets):
            budget_key = str(budget)
            base_seed = seed + (index * 100_000) + (budget_index * 1_000)
            pre_move_search_value = float(
                row.get("searches", {}).get(budget_key, {}).get("value", 0.0)
            )
            if stability_mode:
                shared_result = forced_continuation(
                    state=row["state"],
                    forced_move=int(row["raw_top1"]),
                    evaluator=evaluator,
                    simulations=int(budget),
                    c_puct=float(c_puct),
                    search_options=search_options,
                    seed=base_seed,
                )
                alternative_result = forced_continuation(
                    state=row["state"],
                    forced_move=alternative_move_int,
                    evaluator=evaluator,
                    simulations=int(budget),
                    c_puct=float(c_puct),
                    search_options=search_options,
                    seed=base_seed + 500,
                )
                delta_outcome = float(
                    shared_result["outcome_root"] - alternative_result["outcome_root"]
                )
                budget_result = {
                    "pre_move_search_value": pre_move_search_value,
                    "shared_forced": shared_result,
                    "alternative_forced": alternative_result,
                    "outcome_delta": delta_outcome,
                    "shared_beats_alternative": delta_outcome > 0.0,
                    "shared_ties_alternative": delta_outcome == 0.0,
                }
            else:
                raw_result = forced_continuation(
                    state=row["state"],
                    forced_move=int(row["raw_top1"]),
                    evaluator=evaluator,
                    simulations=int(budget),
                    c_puct=float(c_puct),
                    search_options=search_options,
                    seed=base_seed,
                )
                puct_result = forced_continuation(
                    state=row["state"],
                    forced_move=int(row["searches"]["1200"]["top1"]),
                    evaluator=evaluator,
                    simulations=int(budget),
                    c_puct=float(c_puct),
                    search_options=search_options,
                    seed=base_seed + 500,
                )
                delta_outcome = float(
                    puct_result["outcome_root"] - raw_result["outcome_root"]
                )
                budget_result = {
                    "pre_move_search_value": pre_move_search_value,
                    "raw_forced": raw_result,
                    "puct_forced": puct_result,
                    "outcome_delta": delta_outcome,
                    "puct_beats_raw": delta_outcome > 0.0,
                    "puct_ties_raw": delta_outcome == 0.0,
                }
            base["per_budget"][budget_key] = budget_result
        if len(budgets) > 1:
            first = base["per_budget"][str(budgets[0])]
            last = base["per_budget"][str(budgets[-1])]
            if stability_mode:
                base["result_differs_by_budget"] = bool(
                    first["shared_forced"]["outcome_root"]
                    != last["shared_forced"]["outcome_root"]
                    or first["alternative_forced"]["outcome_root"]
                    != last["alternative_forced"]["outcome_root"]
                    or first["outcome_delta"] != last["outcome_delta"]
                )
            else:
                base["result_differs_by_budget"] = bool(
                    first["raw_forced"]["outcome_root"]
                    != last["raw_forced"]["outcome_root"]
                    or first["puct_forced"]["outcome_root"]
                    != last["puct_forced"]["outcome_root"]
                    or first["outcome_delta"] != last["outcome_delta"]
                )
        else:
            base["result_differs_by_budget"] = False
        results.append(base)
    return results


def counts_by(
    rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(key_fn(row))] += 1
    return dict(sorted(counts.items()))


def paired_summary(records: list[dict[str, Any]], budget_key: str) -> dict[str, Any]:
    diffs = [
        float(record["per_budget"][budget_key]["outcome_delta"]) for record in records
    ]
    ci = bootstrap_ci(diffs, seed=42, samples=DEFAULT_BOOTSTRAP_SAMPLES)
    wins = losses = ties = 0
    for diff in diffs:
        if diff > 0.0:
            wins += 1
        elif diff < 0.0:
            losses += 1
        else:
            ties += 1
    return {
        **ci,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def grouped_paired_summary(
    records: list[dict[str, Any]],
    budget_key: str,
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(key_fn(record))].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for key, members in sorted(grouped.items()):
        summary[key] = paired_summary(members, budget_key)
    return summary


def budget_comparison(
    records: list[dict[str, Any]], budgets: list[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for budget in budgets:
        budget_key = str(budget)
        result[budget_key] = paired_summary(records, budget_key)
    if len(budgets) > 1:
        first_key = str(budgets[0])
        last_key = str(budgets[-1])
        diff_records = []
        for record in records:
            diff_records.append(
                float(record["per_budget"][last_key]["outcome_delta"])
                - float(record["per_budget"][first_key]["outcome_delta"])
            )
        result["delta_between_budgets"] = bootstrap_ci(
            diff_records, seed=42, samples=DEFAULT_BOOTSTRAP_SAMPLES
        )
        result["states_with_any_difference"] = sum(
            1 for record in records if bool(record.get("result_differs_by_budget"))
        )
    return result


def extreme_rows(
    records: list[dict[str, Any]], budget_key: str, *, helpful: bool, limit: int = 25
) -> list[dict[str, Any]]:
    sorted_records = sorted(
        records,
        key=lambda record: (
            float(record["per_budget"][budget_key]["outcome_delta"]),
            float(record["puct_visit_share_1200"]),
            -float(record["raw_margin"]),
            str(record["state_hash"]),
        ),
        reverse=helpful,
    )
    rows: list[dict[str, Any]] = []
    for record in sorted_records:
        budget = record["per_budget"][budget_key]
        row = {
            "state_hash": record["state_hash"],
            "suite_name": record["suite_name"],
            "phase": record["phase"],
            "seat_context": record["dominant_seat_context"],
            "poor_384_256_p0_game": record["poor_384_256_p0_game"],
            "legal_moves": list(record["legal_moves"]),
            "raw_top1": record["raw_top1"],
            "puct_top1": record["puct_top1"],
            "raw_policy": list(record["raw_policy"]),
            "puct_policy_1200": list(record["puct_policy_1200"]),
            "puct_visits_1200": {
                str(move): int(round(float(record["puct_policy_1200"][move]) * 1200))
                for move in record["legal_moves"]
            },
            "raw_margin": float(record["raw_margin"]),
            "puct_visit_share_1200": float(record["puct_visit_share_1200"]),
            "outcome_delta": float(budget["outcome_delta"]),
            "raw_forced_outcome": float(budget["raw_forced"]["outcome_root"]),
            "puct_forced_outcome": float(budget["puct_forced"]["outcome_root"]),
            "raw_forced_margin": int(budget["raw_forced"]["store_margin_root"]),
            "puct_forced_margin": int(budget["puct_forced"]["store_margin_root"]),
        }
        if helpful and row["outcome_delta"] <= 0.0:
            continue
        if not helpful and row["outcome_delta"] >= 0.0:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def stability_extremes(
    records: list[dict[str, Any]], budget_key: str, *, helpful: bool, limit: int = 10
) -> list[dict[str, Any]]:
    sorted_records = sorted(
        records,
        key=lambda record: float(record["per_budget"][budget_key]["outcome_delta"]),
        reverse=helpful,
    )
    rows: list[dict[str, Any]] = []
    for record in sorted_records:
        budget = record["per_budget"][budget_key]
        delta = float(budget["outcome_delta"])
        if helpful and delta <= 0.0:
            continue
        if not helpful and delta >= 0.0:
            continue
        rows.append(
            {
                "state_hash": record["state_hash"],
                "suite_name": record["suite_name"],
                "phase": record["phase"],
                "shared_top1": int(record["shared_top1"]),
                "alternative_move": int(record["alternative_move"]),
                "alternative_move_source": str(record["alternative_move_source"]),
                "outcome_delta": delta,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def classify_result(
    primary_summary: dict[str, Any],
    poor_summary: dict[str, Any] | None,
    secondary_summary: dict[str, Any] | None,
) -> tuple[str, str]:
    mean = float(primary_summary["mean"])
    lower = float(primary_summary["lower"])
    upper = float(primary_summary["upper"])
    poor_mean = None if poor_summary is None else float(poor_summary["mean"])
    poor_upper = None if poor_summary is None else float(poor_summary["upper"])
    secondary_mean = (
        None if secondary_summary is None else float(secondary_summary["mean"])
    )
    if mean <= -0.03 or upper < 0.0:
        return (
            "puct_targets_bad_for_eval_states",
            "PUCT-forced continuations lose to raw-forced on the evaluation states themselves.",
        )
    if (
        mean > 0.0
        and poor_mean is not None
        and poor_mean <= -0.03
        and (poor_upper is not None and poor_upper < 0.0)
    ) or (mean > 0.0 and secondary_mean is not None and secondary_mean < 0.0):
        return (
            "puct_targets_context_misaligned",
            "PUCT helps overall but breaks in the high-risk P0 or lower-budget contexts.",
        )
    if mean >= 0.03 and lower > 0.01:
        return (
            "puct_targets_good_but_training_harmful",
            "PUCT-forced continuations are better than raw-forced, so the regression likely comes from the update format rather than the targets.",
        )
    return (
        "counterfactual_inconclusive",
        "The paired CI still crosses zero or context slices disagree, so more sample or budget is needed before training again.",
    )


def paired_table_rows(summary: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for name, values in summary.items():
        if not isinstance(values, dict) or "wins" not in values:
            continue
        rows.append(
            [
                str(name),
                str(values["n"]),
                str(values["wins"]),
                str(values["ties"]),
                str(values["losses"]),
                fmt(values["mean"]),
                fmt(values["lower"]),
                fmt(values["upper"]),
            ]
        )
    return rows


def extreme_markdown_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    output: list[list[str]] = []
    for row in rows:
        raw_policy_text = ",".join(f"{value:.3f}" for value in row["raw_policy"])
        puct_visits_text = ",".join(
            f"{move}:{visits}" for move, visits in row["puct_visits_1200"].items()
        )
        output.append(
            [
                str(row["state_hash"]),
                str(row["suite_name"]),
                str(row["phase"]),
                str(row["seat_context"]),
                str(row["poor_384_256_p0_game"]),
                ",".join(str(move) for move in row["legal_moves"]),
                raw_policy_text,
                puct_visits_text,
                str(row["raw_top1"]),
                str(row["puct_top1"]),
                fmt(row["raw_margin"]),
                fmt(row["puct_visit_share_1200"]),
                fmt(row["raw_forced_outcome"]),
                fmt(row["puct_forced_outcome"]),
                str(row["raw_forced_margin"]),
                str(row["puct_forced_margin"]),
                fmt(row["outcome_delta"]),
            ]
        )
    return output


def write_report(
    *,
    report_path: Path,
    classification: str,
    rationale: str,
    summary: dict[str, Any],
) -> None:
    primary_budget = str(summary["results"]["primary_budget"])
    primary_disagreement = summary["results"]["disagreement"]["by_budget"][
        primary_budget
    ]
    primary_stability = summary["results"]["stability"]["by_budget"][primary_budget]
    poor_table = summary["results"]["disagreement"]["by_poor_384_256_p0"][
        primary_budget
    ]
    lines = [
        "# AlphaZero-Lite Balanced Current PUCT Target Counterfactual Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{classification}`",
        "",
        f"{rationale}",
        "",
        "## Inputs",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                [
                    "Current artifact weights SHA256",
                    summary["inputs"]["current_weights"]["actual_sha256"],
                ],
                [
                    "Consensus mining profile SHA256",
                    summary["inputs"]["consensus_search_profile_hash"],
                ],
                [
                    "Continuation search profile SHA256",
                    summary["inputs"]["continuation_search_profile"]["sha256"],
                ],
            ],
        ),
        "",
        "## Sampled State Counts",
        "",
        markdown_table(
            ["Bucket", "Count"],
            [
                [
                    "Valid disagreement candidates",
                    str(summary["sampling"]["disagreement_candidates"]),
                ],
                [
                    "Sampled disagreement states",
                    str(summary["sampling"]["disagreement_sampled"]),
                ],
                [
                    "Valid stability candidates",
                    str(summary["sampling"]["stability_candidates"]),
                ],
                [
                    "Sampled stability states",
                    str(summary["sampling"]["stability_sampled"]),
                ],
            ],
        ),
        "",
        "## Disagreement-State Stratification",
        "",
        markdown_table(
            ["Group", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "suite_bucket"
                ].items()
            ],
        ),
        "",
        markdown_table(
            ["Seat Context", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "seat_context"
                ].items()
            ],
        ),
        "",
        markdown_table(
            ["Phase", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "phase"
                ].items()
            ],
        ),
        "",
        markdown_table(
            ["Raw Margin Bucket", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "raw_margin_bucket"
                ].items()
            ],
        ),
        "",
        markdown_table(
            ["PUCT Visit Share Bucket", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "visit_share_bucket"
                ].items()
            ],
        ),
        "",
        markdown_table(
            ["PUCT Top Move", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "puct_top1"
                ].items()
            ],
        ),
        "",
        markdown_table(
            ["Poor 384:256 P0 Context", "Count"],
            [
                [key, str(value)]
                for key, value in summary["sampling"]["disagreement_stratification"][
                    "poor_384_256_p0_game"
                ].items()
            ],
        ),
        "",
        "## PUCT-Forced Versus Raw-Forced",
        "",
        markdown_table(
            ["Budget", "N", "Wins", "Ties", "Losses", "Mean", "Lower 95%", "Upper 95%"],
            paired_table_rows(summary["results"]["disagreement"]["by_budget"]),
        ),
        "",
        f"Primary-budget mean outcome delta (`PUCT - raw`) at `{primary_budget}`: {fmt(primary_disagreement['mean'])}",
        "",
        "## Outcome Delta By Phase",
        "",
        markdown_table(
            ["Phase", "N", "Wins", "Ties", "Losses", "Mean", "Lower 95%", "Upper 95%"],
            paired_table_rows(
                summary["results"]["disagreement"]["by_phase"][primary_budget]
            ),
        ),
        "",
        "## Outcome Delta By Seat Context",
        "",
        markdown_table(
            [
                "Seat Context",
                "N",
                "Wins",
                "Ties",
                "Losses",
                "Mean",
                "Lower 95%",
                "Upper 95%",
            ],
            paired_table_rows(
                summary["results"]["disagreement"]["by_seat_context"][primary_budget]
            ),
        ),
        "",
        "## Outcome Delta By Raw Margin Bucket",
        "",
        markdown_table(
            [
                "Raw Margin Bucket",
                "N",
                "Wins",
                "Ties",
                "Losses",
                "Mean",
                "Lower 95%",
                "Upper 95%",
            ],
            paired_table_rows(
                summary["results"]["disagreement"]["by_raw_margin_bucket"][
                    primary_budget
                ]
            ),
        ),
        "",
        "## Outcome Delta By PUCT Visit-Share Bucket",
        "",
        markdown_table(
            [
                "Visit Share Bucket",
                "N",
                "Wins",
                "Ties",
                "Losses",
                "Mean",
                "Lower 95%",
                "Upper 95%",
            ],
            paired_table_rows(
                summary["results"]["disagreement"]["by_visit_share_bucket"][
                    primary_budget
                ]
            ),
        ),
        "",
        "## Outcome Delta By 384:256 P0 Context",
        "",
        markdown_table(
            [
                "Poor 384:256 P0",
                "N",
                "Wins",
                "Ties",
                "Losses",
                "Mean",
                "Lower 95%",
                "Upper 95%",
            ],
            paired_table_rows(poor_table),
        ),
        "",
        "## Outcome Delta By PUCT Top Move",
        "",
        markdown_table(
            [
                "Top Move",
                "N",
                "Wins",
                "Ties",
                "Losses",
                "Mean",
                "Lower 95%",
                "Upper 95%",
            ],
            paired_table_rows(
                summary["results"]["disagreement"]["by_puct_top1"][primary_budget]
            ),
        ),
        "",
        "## Continuation Budget Comparison",
        "",
        markdown_table(
            ["Budget", "N", "Wins", "Ties", "Losses", "Mean", "Lower 95%", "Upper 95%"],
            paired_table_rows(summary["results"]["disagreement"]["by_budget"]),
        ),
        "",
    ]
    budget_delta = summary["results"]["disagreement"]["by_budget"].get(
        "delta_between_budgets"
    )
    if budget_delta is not None:
        lines.extend(
            [
                f"Primary-minus-lower continuation mean delta change: {fmt(budget_delta['mean'])} "
                f"[{fmt(budget_delta['lower'])}, {fmt(budget_delta['upper'])}]",
                "",
                "States with any continuation-budget result difference: "
                f"{summary['results']['disagreement']['by_budget']['states_with_any_difference']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Stability-Control Sanity",
            "",
            markdown_table(
                [
                    "Budget",
                    "N",
                    "Wins",
                    "Ties",
                    "Losses",
                    "Mean",
                    "Lower 95%",
                    "Upper 95%",
                ],
                paired_table_rows(summary["results"]["stability"]["by_budget"]),
            ),
            "",
            f"Primary-budget shared-top1 control mean delta (`shared - alternative`) at `{primary_budget}`: {fmt(primary_stability['mean'])}",
            "",
            "## Largest Harmful PUCT Disagreements",
            "",
            markdown_table(
                [
                    "State Hash",
                    "Suite",
                    "Phase",
                    "Seat",
                    "Poor P0",
                    "Legal",
                    "Raw Policy",
                    "PUCT Visits 1200",
                    "Raw",
                    "PUCT",
                    "Raw Margin",
                    "PUCT Share",
                    "Raw Outcome",
                    "PUCT Outcome",
                    "Raw Margin Final",
                    "PUCT Margin Final",
                    "Delta",
                ],
                extreme_markdown_rows(
                    summary["results"]["disagreement"]["largest_harmful"][
                        primary_budget
                    ]
                ),
            ),
            "",
            "## Largest Helpful PUCT Disagreements",
            "",
            markdown_table(
                [
                    "State Hash",
                    "Suite",
                    "Phase",
                    "Seat",
                    "Poor P0",
                    "Legal",
                    "Raw Policy",
                    "PUCT Visits 1200",
                    "Raw",
                    "PUCT",
                    "Raw Margin",
                    "PUCT Share",
                    "Raw Outcome",
                    "PUCT Outcome",
                    "Raw Margin Final",
                    "PUCT Margin Final",
                    "Delta",
                ],
                extreme_markdown_rows(
                    summary["results"]["disagreement"]["largest_helpful"][
                        primary_budget
                    ]
                ),
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_dir = Path(args.current)
    current_weights = current_dir / "weights.json"
    consensus_table_path = Path(args.consensus_state_table)
    consensus_audit_path = Path(args.consensus_audit)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    continuation_budgets = parse_budgets(args.continuation_budgets)
    continuation_profile = search_profile(args.c_puct, continuation_budgets)

    require_existing_file(current_weights, "current weights")
    require_existing_file(consensus_table_path, "consensus state table")
    require_existing_file(consensus_audit_path, "consensus audit")
    require_existing_file(fixed_large_suite, "fixed large suite")
    for suite_path in heldout_suites:
        require_existing_file(suite_path, f"heldout suite {suite_path.name}")

    current_hash = verify_expected_hash(
        current_weights,
        args.expected_current_weights_sha256,
        "current weights",
    )
    consensus_audit = load_json(consensus_audit_path)
    consensus_rows = unique_rows_by_state_hash(read_jsonl(consensus_table_path))

    disagreement_rows = [row for row in consensus_rows if disagreement_candidate(row)]
    stability_rows = [row for row in consensus_rows if stability_candidate(row)]
    if len(disagreement_rows) < MIN_DISAGREEMENT_STATES:
        raise RuntimeError(
            f"only {len(disagreement_rows)} valid disagreement states found; need at least {MIN_DISAGREEMENT_STATES}"
        )

    rng = random.Random(args.seed)
    disagreement_sample = select_disagreement_rows(
        disagreement_rows,
        min(args.sample_disagreement_states, len(disagreement_rows)),
        rng,
    )
    stability_sample = round_robin_sample(
        stability_rows,
        min(args.sample_stability_states, len(stability_rows)),
        rng,
    )

    disagreement_batches = partition_batches(disagreement_sample, args.workers)
    stability_batches = partition_batches(stability_sample, args.workers)

    disagreement_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(disagreement_batches) or 1
    ) as executor:
        futures = [
            executor.submit(
                analyze_counterfactual_batch,
                batch=batch,
                artifact_path=str(current_dir),
                budgets=continuation_budgets,
                c_puct=args.c_puct,
                seed=args.seed,
                stability_mode=False,
            )
            for batch in disagreement_batches
        ]
        for future in concurrent.futures.as_completed(futures):
            disagreement_results.extend(future.result())
    disagreement_results.sort(key=lambda row: str(row["state_hash"]))

    stability_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(stability_batches) or 1
    ) as executor:
        futures = [
            executor.submit(
                analyze_counterfactual_batch,
                batch=batch,
                artifact_path=str(current_dir),
                budgets=continuation_budgets,
                c_puct=args.c_puct,
                seed=args.seed + 50_000,
                stability_mode=True,
            )
            for batch in stability_batches
        ]
        for future in concurrent.futures.as_completed(futures):
            stability_results.extend(future.result())
    stability_results.sort(key=lambda row: str(row["state_hash"]))

    disagreement_results_path = workdir / DISAGREEMENT_RESULTS_FILENAME
    stability_results_path = workdir / STABILITY_RESULTS_FILENAME
    write_jsonl(disagreement_results_path, disagreement_results)
    write_jsonl(stability_results_path, stability_results)

    primary_budget = max(continuation_budgets)
    primary_budget_key = str(primary_budget)
    disagreement_by_budget = budget_comparison(
        disagreement_results, continuation_budgets
    )
    stability_by_budget = budget_comparison(stability_results, continuation_budgets)
    disagreement_by_phase = {
        str(budget): grouped_paired_summary(
            disagreement_results, str(budget), lambda row: str(row["phase"])
        )
        for budget in continuation_budgets
    }
    disagreement_by_seat = {
        str(budget): grouped_paired_summary(
            disagreement_results,
            str(budget),
            lambda row: str(row["dominant_seat_context"]),
        )
        for budget in continuation_budgets
    }
    disagreement_by_margin = {
        str(budget): grouped_paired_summary(
            disagreement_results,
            str(budget),
            lambda row: margin_bucket(float(row["raw_margin"])),
        )
        for budget in continuation_budgets
    }
    disagreement_by_visit_share = {
        str(budget): grouped_paired_summary(
            disagreement_results,
            str(budget),
            lambda row: visit_share_bucket(float(row["puct_visit_share_1200"])),
        )
        for budget in continuation_budgets
    }
    disagreement_by_poor = {
        str(budget): grouped_paired_summary(
            disagreement_results,
            str(budget),
            lambda row: str(bool(row["poor_384_256_p0_game"])),
        )
        for budget in continuation_budgets
    }
    disagreement_by_top1 = {
        str(budget): grouped_paired_summary(
            disagreement_results,
            str(budget),
            lambda row: str(int(row["puct_top1"])),
        )
        for budget in continuation_budgets
    }

    poor_primary = disagreement_by_poor[primary_budget_key].get("True")
    secondary_summary = None
    if len(continuation_budgets) > 1:
        secondary_summary = disagreement_by_budget[str(min(continuation_budgets))]
    classification, rationale = classify_result(
        disagreement_by_budget[primary_budget_key], poor_primary, secondary_summary
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "classification": classification,
        "classification_rationale": rationale,
        "inputs": {
            "current_weights": current_hash,
            "consensus_search_profile_hash": consensus_audit.get("inputs", {}).get(
                "search_profile_hash"
            ),
            "consensus_search_profile": consensus_audit.get("inputs", {}).get(
                "search_profile", {}
            ),
            "continuation_search_profile": continuation_profile,
            "consensus_state_table": build_input_summary(consensus_table_path),
            "consensus_audit": build_input_summary(consensus_audit_path),
            "fixed_large_suite": build_input_summary(fixed_large_suite),
            "heldout_suites": [build_input_summary(path) for path in heldout_suites],
        },
        "sampling": {
            "disagreement_candidates": len(disagreement_rows),
            "disagreement_sampled": len(disagreement_sample),
            "stability_candidates": len(stability_rows),
            "stability_sampled": len(stability_sample),
            "disagreement_stratification": {
                "suite_bucket": counts_by(
                    disagreement_sample, lambda row: suite_bucket(row)
                ),
                "suite_name": counts_by(
                    disagreement_sample, lambda row: str(row["suite_name"])
                ),
                "seat_context": counts_by(
                    disagreement_sample,
                    lambda row: str(row.get("dominant_seat_context", "unknown")),
                ),
                "phase": counts_by(disagreement_sample, lambda row: str(row["phase"])),
                "raw_margin_bucket": counts_by(
                    disagreement_sample,
                    lambda row: margin_bucket(float(row["raw_margin"])),
                ),
                "visit_share_bucket": counts_by(
                    disagreement_sample,
                    lambda row: visit_share_bucket(
                        float(row["searches"]["1200"]["top1_visit_share"])
                    ),
                ),
                "poor_384_256_p0_game": counts_by(
                    disagreement_sample,
                    lambda row: str(bool(row["poor_384_256_p0_game"])),
                ),
                "puct_top1": counts_by(
                    disagreement_sample,
                    lambda row: str(int(row["searches"]["1200"]["top1"])),
                ),
            },
        },
        "outputs": {
            "workdir": str(workdir),
            "disagreement_results": str(disagreement_results_path),
            "stability_results": str(stability_results_path),
            "report": str(REPORT_PATH),
        },
        "results": {
            "primary_budget": primary_budget,
            "disagreement": {
                "by_budget": disagreement_by_budget,
                "by_phase": disagreement_by_phase,
                "by_seat_context": disagreement_by_seat,
                "by_raw_margin_bucket": disagreement_by_margin,
                "by_visit_share_bucket": disagreement_by_visit_share,
                "by_poor_384_256_p0": disagreement_by_poor,
                "by_puct_top1": disagreement_by_top1,
                "largest_harmful": {
                    str(budget): extreme_rows(
                        disagreement_results, str(budget), helpful=False
                    )
                    for budget in continuation_budgets
                },
                "largest_helpful": {
                    str(budget): extreme_rows(
                        disagreement_results, str(budget), helpful=True
                    )
                    for budget in continuation_budgets
                },
            },
            "stability": {
                "by_budget": stability_by_budget,
                "largest_helpful": {
                    str(budget): stability_extremes(
                        stability_results, str(budget), helpful=True
                    )
                    for budget in continuation_budgets
                },
                "largest_harmful": {
                    str(budget): stability_extremes(
                        stability_results, str(budget), helpful=False
                    )
                    for budget in continuation_budgets
                },
            },
        },
    }

    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(
        report_path=REPORT_PATH,
        classification=classification,
        rationale=rationale,
        summary=summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
