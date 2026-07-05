#!/usr/bin/env python3
"""Trace and classify current 384:256 failure families without training.

This runner measures current-vs-current failures under the promoted runtime
profile, captures disadvantaged-side root telemetry for 384:256 opening-suite
 games, probes counterfactual alternatives with forced continuations, and
 classifies the dominant failure families using transparent rules.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
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
    budget_pair_label,
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
)
from ml.alphazero_lite.kalah_rules import (  # noqa: E402
    KalahGame,
    move_consequence_for_state,
    move_consequence_table,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
)
from ml.alphazero_lite.run_residual_v3_iteration0_target_causal_audit import (  # noqa: E402
    forced_continuation,
    load_jsonl,
    partition_batches,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_residual_v3_opening_iteration0_preflight import (  # noqa: E402
    EXPECTED_CURRENT_WEIGHTS_SHA256,
    build_promoted_search_profile,
    canonical_json,
    canonical_state_hash,
    sha256_file,
    stable_float,
    validate_guardrails,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    build_eval_search_options,
    build_search_profile,
    outcome_for_player,
)

SCHEMA = "azlite_current_384_256_failure_family_trace_v1"
ROOT_TRACE_FILENAME = "root_trace_384_256.jsonl"
GAME_OUTCOMES_FILENAME = "game_outcomes_384_256.jsonl"
COUNTERFACTUAL_FILENAME = "counterfactual_decision_probes.jsonl"
SUMMARY_FILENAME = "summary_metrics.json"
DEDUP_SUMMARY_FILENAME = "deduplicated_rankings.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-current-384-256-failure-family-trace-results.md"
)
PRIMARY_IMPROVEMENT_THRESHOLD = 0.25
MAX_PROBES = 1024
MIN_PROBES = 512
PER_PREFIX_PROBE_CAP = 4
SEARCH_BUDGET_FAMILY = "search_budget_limited"
POLICY_FAMILY = "policy_prior_trap"
VALUE_FAMILY = "value_miscalibrated_root"
TACTICAL_MISS_FAMILY = "tactical_capture_miss"
TACTICAL_BLUNDER_FAMILY = "tactical_capture_blunder"
SEAT_FAMILY = "seat_context_specific"
OPENING_FAMILY = "opening_family_specific"
NO_LOCAL_FAMILY = "no_local_better_move_found"

_TRACE_EVALUATOR: arena.ArtifactEvaluator | None = None
_TRACE_SEARCH_OPTIONS: dict[str, Any] | None = None
_TRACE_CHALLENGER_SIMS: int | None = None
_TRACE_CURRENT_SIMS: int | None = None
_TRACE_C_PUCT: float | None = None
_TRACE_PROFILE_HASH: str | None = None

_PROBE_EVALUATOR: arena.ArtifactEvaluator | None = None
_PROBE_SEARCH_OPTIONS: dict[str, Any] | None = None
_PROBE_BUDGETS: list[int] | None = None
_PROBE_DEFAULT_C_PUCT: float | None = None
_PROBE_C_PUCT_SCHEDULE: dict[str, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default=EXPECTED_CURRENT_WEIGHTS_SHA256,
    )
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--budget-pair", default="384:256")
    parser.add_argument("--counterfactual-continuation-budgets", default="384,768,1200")
    parser.add_argument("--default-c-puct", type=float, default=DEFAULT_RUNTIME_C_PUCT)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_budget_pair(text: str) -> tuple[int, int]:
    challenger_text, current_text = str(text).split(":", 1)
    challenger = int(challenger_text)
    current = int(current_text)
    if challenger <= 0 or current <= 0:
        raise ValueError("budget pair values must be positive")
    return challenger, current


def parse_budgets(text: str) -> list[int]:
    budgets = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not budgets:
        raise ValueError("at least one continuation budget is required")
    unique = sorted(set(budgets))
    if unique != budgets:
        raise ValueError("continuation budgets must be unique and sorted")
    return unique


def existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"missing path: {value}")
    return path


def verify_current_weights_sha256(current_path: Path, expected_sha256: str) -> None:
    weights_path = current_path / "weights.json"
    metadata_path = current_path / "metadata.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing current artifact weights: {weights_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing current artifact metadata: {metadata_path}")
    actual_sha256 = sha256_file(weights_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "current artifact weights hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def mean_or_zero(values: list[float]) -> float:
    return stable_float(statistics.fmean(values)) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return stable_float(statistics.median(values)) if values else 0.0


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def phase_name_for_game(game: KalahGame, *, total_ply: int) -> str:
    if total_ply <= 8:
        return "opening"
    phase = arena.phase_bucket_for_game(game)
    if phase == "early":
        return "opening"
    return phase


def policy_map(policy: list[float], legal_moves: list[int]) -> dict[str, float]:
    return {str(move): stable_float(float(policy[move])) for move in legal_moves}


def top_move_from_policy(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), int(move)))


def visit_share_map(visits: list[float], legal_moves: list[int]) -> dict[int, float]:
    total = sum(float(visits[move]) for move in legal_moves)
    if total <= 0.0:
        uniform = 1.0 / max(len(legal_moves), 1)
        return {int(move): stable_float(uniform) for move in legal_moves}
    return {
        int(move): stable_float(float(visits[move]) / total) for move in legal_moves
    }


def top_visit_share_and_margin(
    visits: list[float], legal_moves: list[int]
) -> tuple[float, float]:
    shares = sorted(visit_share_map(visits, legal_moves).values(), reverse=True)
    if not shares:
        return 0.0, 0.0
    top_share = shares[0]
    margin = 1.0 if len(shares) < 2 else shares[0] - shares[1]
    return stable_float(top_share), stable_float(margin)


def selection_entry_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    breakdown = summary.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def child_stat_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(entry["move"]): entry
        for entry in list(summary.get("child_stats") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def store_diff_for_player(game: KalahGame, player: int) -> int:
    return int(game.captured_seeds[player] - game.captured_seeds[1 - player])


def after_state_for_move(state: dict[str, Any], move: int) -> dict[str, Any] | None:
    game = KalahGame.from_state(state)
    if move not in game.possible_moves():
        return None
    if not game.move(game.pit_index(move)):
        return None
    return game.to_state()


def opponent_tactical_flags(after_state: dict[str, Any] | None) -> dict[str, bool]:
    if after_state is None:
        return {
            "opponent_immediate_capture_available": False,
            "opponent_immediate_extra_turn_available": False,
        }
    table = move_consequence_table(after_state)
    return {
        "opponent_immediate_capture_available": any(
            bool(row["produces_capture"]) for row in table if bool(row["legal"])
        ),
        "opponent_immediate_extra_turn_available": any(
            bool(row["gives_extra_turn"]) for row in table if bool(row["legal"])
        ),
    }


def immediate_tactical_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in move_consequence_table(state)
        if bool(row["legal"])
        and (bool(row["produces_capture"]) or bool(row["gives_extra_turn"]))
    ]


def select_tactical_candidate(trace_row: dict[str, Any]) -> dict[str, Any] | None:
    selected_move = int(trace_row["selected_move"])
    consequences = list(trace_row["legal_move_consequences"])
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for consequence in consequences:
        if not bool(consequence["legal"]):
            continue
        move = int(consequence["move_index"])
        if move == selected_move:
            continue
        after_state = after_state_for_move(trace_row["state"], move)
        opponent_flags = opponent_tactical_flags(after_state)
        if not (
            bool(consequence["produces_capture"])
            or bool(consequence["gives_extra_turn"])
            or (
                (
                    trace_row["selected_allows_opponent_capture"]
                    or trace_row["selected_allows_opponent_extra_turn"]
                )
                and not opponent_flags["opponent_immediate_capture_available"]
                and not opponent_flags["opponent_immediate_extra_turn_available"]
            )
        ):
            continue
        priority = (
            1.0 if bool(consequence["gives_extra_turn"]) else 0.0,
            1.0 if bool(consequence["produces_capture"]) else 0.0,
            float(consequence["store_delta_immediate"]),
            1.0 if not opponent_flags["opponent_immediate_capture_available"] else 0.0,
            1.0
            if not opponent_flags["opponent_immediate_extra_turn_available"]
            else 0.0,
            float(trace_row["raw_policy_distribution"][str(move)]),
            -float(move),
        )
        candidate = {
            "move": move,
            "consequence": consequence,
            **opponent_flags,
        }
        candidates.append((priority, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def suite_name_for_path(path: Path) -> str:
    return path.stem


def load_suite_payload(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    return {
        "suite_name": suite_name_for_path(path),
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
    }


def init_trace_worker(
    artifact_path: str,
    challenger_simulations: int,
    current_simulations: int,
    c_puct: float,
    profile_hash: str,
    tactical_root_bias: float,
) -> None:
    global _TRACE_EVALUATOR, _TRACE_SEARCH_OPTIONS, _TRACE_CHALLENGER_SIMS
    global _TRACE_CURRENT_SIMS, _TRACE_C_PUCT, _TRACE_PROFILE_HASH
    _TRACE_EVALUATOR = arena.ArtifactEvaluator(Path(artifact_path))
    _TRACE_SEARCH_OPTIONS = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=float(tactical_root_bias),
    )
    _TRACE_CHALLENGER_SIMS = int(challenger_simulations)
    _TRACE_CURRENT_SIMS = int(current_simulations)
    _TRACE_C_PUCT = float(c_puct)
    _TRACE_PROFILE_HASH = str(profile_hash)


def trace_game_task(task: dict[str, Any]) -> dict[str, Any]:
    if (
        _TRACE_EVALUATOR is None
        or _TRACE_SEARCH_OPTIONS is None
        or _TRACE_CHALLENGER_SIMS is None
        or _TRACE_CURRENT_SIMS is None
        or _TRACE_C_PUCT is None
        or _TRACE_PROFILE_HASH is None
    ):
        raise RuntimeError("trace worker not initialized")

    opening_entry = dict(task["opening_entry"])
    challenger_player = int(task["challenger_player"])
    disadvantaged_player = 1 - challenger_player
    game = KalahGame.from_state(opening_entry["state"])
    total_ply = int(
        opening_entry.get("ply", len(opening_entry.get("prefix_moves", [])))
    )
    game_moves: list[int] = []
    trace_rows: list[dict[str, Any]] = []
    move_index = 0

    while not game.over():
        legal_moves = [int(move) for move in game.possible_moves()]
        if not legal_moves:
            break
        acting_player = int(game.current_player)
        acting_role = "challenger" if acting_player == challenger_player else "current"
        simulations = (
            int(_TRACE_CHALLENGER_SIMS)
            if acting_role == "challenger"
            else int(_TRACE_CURRENT_SIMS)
        )
        state = game.to_state()
        summary = arena.evaluate_artifact_position(
            evaluator=_TRACE_EVALUATOR,
            state=state,
            simulations=simulations,
            seed=int(task["seed"]) + move_index,
            c_puct=float(_TRACE_C_PUCT),
            search_options=dict(_TRACE_SEARCH_OPTIONS),
        )
        selected_move = summary.get("selected_move")
        if selected_move is None:
            break
        selected_move = int(selected_move)
        if acting_player == disadvantaged_player:
            _raw_logits, raw_policy, raw_value = artifact_forward_details(
                _TRACE_EVALUATOR,
                game.clone(),
            )
            raw_top_move = top_move_from_policy(raw_policy, legal_moves)
            selection_entries = selection_entry_map(summary)
            child_stats = child_stat_map(summary)
            selected_consequence = move_consequence_for_state(state, selected_move)
            next_state = after_state_for_move(state, selected_move)
            opponent_flags = opponent_tactical_flags(next_state)
            selected_visit_share, top2_visit_margin = top_visit_share_and_margin(
                [float(value) for value in list(summary.get("visits") or [0.0] * 6)],
                legal_moves,
            )
            trace_rows.append(
                {
                    "suite_name": str(task["suite_name"]),
                    "suite_sha256": str(task["suite_sha256"]),
                    "game_index": int(task["game_index"]),
                    "opening_index": int(task["opening_index"]),
                    "opening_prefix": [
                        int(move) for move in opening_entry.get("prefix_moves", [])
                    ],
                    "opening_prefix_text": ",".join(
                        str(move) for move in opening_entry.get("prefix_moves", [])
                    ),
                    "first_move_family": str(
                        opening_entry.get("first_move_family", "unknown")
                    ),
                    "initial_suite_ply": int(opening_entry.get("ply", 0)),
                    "state": state,
                    "state_hash": canonical_state_hash(state),
                    "ply": int(total_ply),
                    "decision_index": len(trace_rows),
                    "side_to_move": int(acting_player),
                    "challenger_player": int(challenger_player),
                    "disadvantaged_player": int(disadvantaged_player),
                    "seat_context": (
                        f"challenger_player_{challenger_player}_disadvantaged_player_{disadvantaged_player}"
                    ),
                    "phase": phase_name_for_game(game, total_ply=total_ply),
                    "legal_moves": legal_moves,
                    "selected_move": int(selected_move),
                    "raw_policy_top_move": None
                    if raw_top_move is None
                    else int(raw_top_move),
                    "raw_policy_distribution": policy_map(raw_policy, legal_moves),
                    "raw_policy_entropy_bits": stable_float(
                        -sum(
                            float(raw_policy[move])
                            * math.log(float(raw_policy[move]), 2)
                            for move in legal_moves
                            if float(raw_policy[move]) > 0.0
                        )
                    ),
                    "raw_policy_value": stable_float(float(raw_value)),
                    "puct_visit_counts": {
                        str(move): int(
                            float(list(summary.get("visits") or [0.0] * 6)[move])
                        )
                        for move in legal_moves
                    },
                    "puct_visit_share": {
                        str(move): share
                        for move, share in visit_share_map(
                            [
                                float(value)
                                for value in list(summary.get("visits") or [0.0] * 6)
                            ],
                            legal_moves,
                        ).items()
                    },
                    "child_q_values": {
                        str(move): stable_float(
                            float(child_stats.get(move, {}).get("q_value", 0.0))
                        )
                        for move in legal_moves
                    },
                    "child_priors": {
                        str(move): stable_float(
                            float(selection_entries.get(move, {}).get("prior", 0.0))
                        )
                        for move in legal_moves
                    },
                    "child_u_values": {
                        str(move): stable_float(
                            float(
                                selection_entries.get(move, {}).get("u_component", 0.0)
                            )
                        )
                        for move in legal_moves
                    },
                    "child_selection_scores": {
                        str(move): stable_float(
                            float(
                                selection_entries.get(move, {}).get(
                                    "selection_score", 0.0
                                )
                            )
                        )
                        for move in legal_moves
                    },
                    "root_value": stable_float(float(summary.get("value", 0.0))),
                    "search_root_q_value": stable_float(
                        float(summary.get("search_root_value", 0.0))
                    )
                    if summary.get("search_root_value") is not None
                    else None,
                    "selected_move_visit_share": stable_float(selected_visit_share),
                    "top2_visit_margin": stable_float(top2_visit_margin),
                    "effective_c_puct": stable_float(float(_TRACE_C_PUCT)),
                    "search_profile_hash": str(_TRACE_PROFILE_HASH),
                    "selected_grants_extra_turn": bool(
                        selected_consequence["gives_extra_turn"]
                    ),
                    "selected_performs_capture": bool(
                        selected_consequence["produces_capture"]
                    ),
                    "selected_capture_count": int(
                        selected_consequence["capture_count"]
                    ),
                    "selected_allows_opponent_capture": bool(
                        opponent_flags["opponent_immediate_capture_available"]
                    ),
                    "selected_allows_opponent_extra_turn": bool(
                        opponent_flags["opponent_immediate_extra_turn_available"]
                    ),
                    "store_diff_before": int(
                        store_diff_for_player(game, acting_player)
                    ),
                    "store_diff_after": (
                        None
                        if next_state is None
                        else int(
                            store_diff_for_player(
                                KalahGame.from_state(next_state), acting_player
                            )
                        )
                    ),
                    "selected_move_consequence": selected_consequence,
                    "legal_move_consequences": move_consequence_table(state),
                    "selection_breakdown": summary.get("selection_breakdown") or {},
                    "visit_snapshots": list(summary.get("visit_snapshots") or []),
                }
            )
        game_moves.append(selected_move)
        if not game.move(game.pit_index(selected_move)):
            break
        total_ply += 1
        move_index += 1

    challenger_store = int(game.captured_seeds[challenger_player])
    disadvantaged_store = int(game.captured_seeds[disadvantaged_player])
    if challenger_store > disadvantaged_store:
        winner = "challenger"
        challenger_score = 1.0
    elif challenger_store < disadvantaged_store:
        winner = "current"
        challenger_score = 0.0
    else:
        winner = "draw"
        challenger_score = 0.5
    disadvantaged_outcome = stable_float(
        float(outcome_for_player(game.winner, disadvantaged_player))
    )
    ds_contribution = stable_float(
        challenger_score if challenger_player == 0 else -challenger_score
    )

    outcome_row = {
        "suite_name": str(task["suite_name"]),
        "suite_sha256": str(task["suite_sha256"]),
        "game_index": int(task["game_index"]),
        "opening_index": int(task["opening_index"]),
        "opening_prefix": [int(move) for move in opening_entry.get("prefix_moves", [])],
        "opening_prefix_text": ",".join(
            str(move) for move in opening_entry.get("prefix_moves", [])
        ),
        "first_move_family": str(opening_entry.get("first_move_family", "unknown")),
        "challenger_player": int(challenger_player),
        "disadvantaged_player": int(disadvantaged_player),
        "winner": winner,
        "challenger_store": int(challenger_store),
        "disadvantaged_store": int(disadvantaged_store),
        "margin_for_challenger": int(challenger_store - disadvantaged_store),
        "margin_for_disadvantaged": int(disadvantaged_store - challenger_store),
        "challenger_score": stable_float(challenger_score),
        "disadvantaged_outcome": disadvantaged_outcome,
        "ds_contribution": ds_contribution,
        "game_length": int(total_ply),
        "trajectory": ",".join(str(move) for move in game_moves),
        "root_decision_count": len(trace_rows),
    }
    for row in trace_rows:
        row["final_winner"] = winner
        row["final_margin_for_challenger"] = int(challenger_store - disadvantaged_store)
        row["final_margin_for_disadvantaged"] = int(
            disadvantaged_store - challenger_store
        )
        row["disadvantaged_outcome"] = disadvantaged_outcome
        row["ds_contribution"] = ds_contribution
    return {"game_outcome": outcome_row, "root_traces": trace_rows}


def init_probe_worker(
    artifact_path: str,
    budgets: list[int],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
) -> None:
    global _PROBE_EVALUATOR, _PROBE_SEARCH_OPTIONS, _PROBE_BUDGETS
    global _PROBE_DEFAULT_C_PUCT, _PROBE_C_PUCT_SCHEDULE
    _PROBE_EVALUATOR = arena.ArtifactEvaluator(Path(artifact_path))
    _PROBE_SEARCH_OPTIONS = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=float(tactical_root_bias),
    )
    _PROBE_BUDGETS = list(budgets)
    _PROBE_DEFAULT_C_PUCT = float(default_c_puct)
    _PROBE_C_PUCT_SCHEDULE = dict(cpuct_schedule)


def probe_task_records(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        _PROBE_EVALUATOR is None
        or _PROBE_SEARCH_OPTIONS is None
        or _PROBE_BUDGETS is None
        or _PROBE_DEFAULT_C_PUCT is None
        or _PROBE_C_PUCT_SCHEDULE is None
    ):
        raise RuntimeError("probe worker not initialized")
    results: list[dict[str, Any]] = []
    for task in tasks:
        state = dict(task["state"])
        legal_moves = [int(move) for move in task["legal_moves"]]
        move_summaries: dict[str, Any] = {}
        root_probes: dict[str, Any] = {}
        for root_budget in [768, 1200]:
            c_puct = resolve_budget_cpuct(
                schedule=_PROBE_C_PUCT_SCHEDULE,
                challenger_simulations=root_budget,
                current_simulations=root_budget,
                default_c_puct=float(_PROBE_DEFAULT_C_PUCT),
            )
            summary = arena.evaluate_artifact_position(
                evaluator=_PROBE_EVALUATOR,
                state=state,
                simulations=int(root_budget),
                seed=int(task["seed"]) + (root_budget * 10),
                c_puct=float(c_puct),
                search_options=dict(_PROBE_SEARCH_OPTIONS),
            )
            root_probes[str(root_budget)] = {
                "selected_move": summary.get("selected_move"),
                "root_value": stable_float(float(summary.get("value", 0.0))),
                "c_puct": stable_float(float(c_puct)),
                "selection_breakdown": summary.get("selection_breakdown") or {},
            }

        move_labels: dict[str, int] = {"current_384_move": int(task["selected_move"])}
        move_768 = root_probes["768"]["selected_move"]
        move_1200 = root_probes["1200"]["selected_move"]
        if move_768 is not None:
            move_labels["deeper_search_768_move"] = int(move_768)
        if move_1200 is not None:
            move_labels["deeper_search_1200_move"] = int(move_1200)
        raw_policy_move = task.get("raw_policy_top_move")
        if raw_policy_move is not None:
            move_labels["raw_policy_move"] = int(raw_policy_move)
        tactical_move = task.get("tactical_candidate_move")
        if tactical_move is not None:
            move_labels["tactical_candidate_move"] = int(tactical_move)

        unique_moves: dict[int, list[str]] = defaultdict(list)
        for label, move in move_labels.items():
            if move in legal_moves:
                unique_moves[int(move)].append(str(label))

        for move, labels in unique_moves.items():
            budgets: dict[str, Any] = {}
            outcomes: list[float] = []
            margins: list[float] = []
            for budget_index, budget in enumerate(_PROBE_BUDGETS):
                c_puct = resolve_budget_cpuct(
                    schedule=_PROBE_C_PUCT_SCHEDULE,
                    challenger_simulations=int(budget),
                    current_simulations=int(budget),
                    default_c_puct=float(_PROBE_DEFAULT_C_PUCT),
                )
                result = forced_continuation(
                    state=state,
                    forced_move=int(move),
                    evaluator=_PROBE_EVALUATOR,
                    simulations=int(budget),
                    c_puct=float(c_puct),
                    search_options=dict(_PROBE_SEARCH_OPTIONS),
                    seed=int(task["seed"]) + (budget_index * 1000) + (move * 100),
                )
                budgets[str(budget)] = {
                    "c_puct": stable_float(float(c_puct)),
                    "outcome_root": stable_float(float(result["outcome_root"])),
                    "store_margin_root": int(result["store_margin_root"]),
                    "winner": result["winner"],
                }
                outcomes.append(float(result["outcome_root"]))
                margins.append(float(result["store_margin_root"]))
            move_summaries[str(move)] = {
                "move": int(move),
                "labels": sorted(labels),
                "forced_budgets": budgets,
                "mean_outcome": mean_or_zero(outcomes),
                "median_outcome": median_or_zero(outcomes),
                "mean_store_margin": mean_or_zero(margins),
            }

        current_move_summary = move_summaries[str(task["selected_move"])]
        for move_summary in move_summaries.values():
            deltas = {}
            margin_deltas = {}
            for budget in _PROBE_BUDGETS:
                budget_key = str(budget)
                deltas[budget_key] = stable_float(
                    float(move_summary["forced_budgets"][budget_key]["outcome_root"])
                    - float(
                        current_move_summary["forced_budgets"][budget_key][
                            "outcome_root"
                        ]
                    )
                )
                margin_deltas[budget_key] = stable_float(
                    float(
                        move_summary["forced_budgets"][budget_key]["store_margin_root"]
                    )
                    - float(
                        current_move_summary["forced_budgets"][budget_key][
                            "store_margin_root"
                        ]
                    )
                )
            move_summary["outcome_delta_vs_current"] = deltas
            move_summary["store_margin_delta_vs_current"] = margin_deltas
            move_summary["mean_outcome_delta_vs_current"] = mean_or_zero(
                list(deltas.values())
            )
            move_summary["median_outcome_delta_vs_current"] = median_or_zero(
                list(deltas.values())
            )

        deeper_candidates = []
        for label in ("deeper_search_768_move", "deeper_search_1200_move"):
            move = move_labels.get(label)
            if move is None or move == int(task["selected_move"]):
                continue
            deeper_candidates.append(move_summaries[str(move)])
        deeper_choice = None
        if deeper_candidates:
            deeper_candidates.sort(
                key=lambda row: (
                    float(row["mean_outcome_delta_vs_current"]),
                    float(row["forced_budgets"]["1200"]["outcome_root"]),
                    -float(row["move"]),
                ),
                reverse=True,
            )
            deeper_choice = deeper_candidates[0]

        results.append(
            {
                "trace_id": str(task["trace_id"]),
                "state_hash": str(task["state_hash"]),
                "state": state,
                "suite_name": str(task["suite_name"]),
                "opening_prefix": list(task["opening_prefix"]),
                "opening_prefix_text": str(task["opening_prefix_text"]),
                "first_move_family": str(task["first_move_family"]),
                "ply": int(task["ply"]),
                "challenger_player": int(task["challenger_player"]),
                "disadvantaged_player": int(task["disadvantaged_player"]),
                "seat_context": str(task["seat_context"]),
                "phase": str(task["phase"]),
                "selected_move": int(task["selected_move"]),
                "raw_policy_top_move": task.get("raw_policy_top_move"),
                "tactical_candidate_move": task.get("tactical_candidate_move"),
                "selected_allows_opponent_capture": bool(
                    task["selected_allows_opponent_capture"]
                ),
                "selected_allows_opponent_extra_turn": bool(
                    task["selected_allows_opponent_extra_turn"]
                ),
                "selected_grants_extra_turn": bool(task["selected_grants_extra_turn"]),
                "selected_performs_capture": bool(task["selected_performs_capture"]),
                "selected_move_consequence": dict(task["selected_move_consequence"]),
                "selected_visit_share": stable_float(
                    float(task["selected_move_visit_share"])
                ),
                "top2_visit_margin": stable_float(float(task["top2_visit_margin"])),
                "raw_policy_distribution": dict(task["raw_policy_distribution"]),
                "child_q_values": dict(task["child_q_values"]),
                "child_priors": dict(task["child_priors"]),
                "deeper_root_probes": root_probes,
                "move_summaries": move_summaries,
                "deeper_search_move": deeper_choice,
                "disadvantaged_outcome": stable_float(
                    float(task["disadvantaged_outcome"])
                ),
                "final_margin_for_disadvantaged": int(
                    task["final_margin_for_disadvantaged"]
                ),
            }
        )
    return results


def suspiciousness_score(trace_row: dict[str, Any]) -> float:
    raw_top_move = trace_row.get("raw_policy_top_move")
    selected_move = int(trace_row["selected_move"])
    raw_selected_prior = float(trace_row["raw_policy_distribution"][str(selected_move)])
    best_nonselected_prior = max(
        (
            float(prior)
            for move, prior in trace_row["raw_policy_distribution"].items()
            if int(move) != selected_move
        ),
        default=0.0,
    )
    best_nonselected_q = max(
        (
            float(q)
            for move, q in trace_row["child_q_values"].items()
            if int(move) != selected_move
        ),
        default=0.0,
    )
    selected_q = float(trace_row["child_q_values"].get(str(selected_move), 0.0))
    score = 0.0
    if float(trace_row["disadvantaged_outcome"]) < 0.0:
        score += 5.0
    if int(trace_row["final_margin_for_disadvantaged"]) < 0:
        score += 3.0
    score += max(0.0, 0.30 - float(trace_row["top2_visit_margin"])) * 4.0
    if raw_top_move is not None and int(raw_top_move) != selected_move:
        score += 1.5 + (best_nonselected_prior * 2.0)
    if best_nonselected_q > selected_q + 0.05:
        score += 1.5 + (best_nonselected_q - selected_q)
    if trace_row["selected_allows_opponent_capture"]:
        score += 2.0
    if trace_row["selected_allows_opponent_extra_turn"]:
        score += 1.5
    if any(
        bool(row["gives_extra_turn"]) or bool(row["produces_capture"])
        for row in trace_row["legal_move_consequences"]
        if bool(row["legal"]) and int(row["move_index"]) != selected_move
    ):
        score += 1.0
    score += max(0.0, raw_selected_prior - 0.40)
    return score


def select_probe_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        trace_rows,
        key=lambda row: (
            suspiciousness_score(row),
            -abs(int(row["final_margin_for_disadvantaged"])),
            -float(row["selected_move_visit_share"]),
            -int(row["ply"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    per_prefix_counts: Counter[str] = Counter()
    for row in ranked:
        prefix = str(row["opening_prefix_text"])
        if per_prefix_counts[prefix] >= PER_PREFIX_PROBE_CAP:
            continue
        selected.append(row)
        per_prefix_counts[prefix] += 1
        if len(selected) >= MAX_PROBES:
            break
    if len(selected) < MIN_PROBES:
        seen = {str(row["trace_id"]) for row in selected}
        for row in ranked:
            if str(row["trace_id"]) in seen:
                continue
            selected.append(row)
            seen.add(str(row["trace_id"]))
            if len(selected) >= min(MAX_PROBES, len(ranked)):
                break
    return selected


def build_probe_tasks(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        tactical_candidate = select_tactical_candidate(row)
        tasks.append(
            {
                "trace_id": str(row["trace_id"]),
                "state_hash": str(row["state_hash"]),
                "state": dict(row["state"]),
                "suite_name": str(row["suite_name"]),
                "opening_prefix": list(row["opening_prefix"]),
                "opening_prefix_text": str(row["opening_prefix_text"]),
                "first_move_family": str(row["first_move_family"]),
                "ply": int(row["ply"]),
                "challenger_player": int(row["challenger_player"]),
                "disadvantaged_player": int(row["disadvantaged_player"]),
                "seat_context": str(row["seat_context"]),
                "phase": str(row["phase"]),
                "legal_moves": [int(move) for move in row["legal_moves"]],
                "selected_move": int(row["selected_move"]),
                "raw_policy_top_move": row.get("raw_policy_top_move"),
                "tactical_candidate_move": (
                    None
                    if tactical_candidate is None
                    else int(tactical_candidate["move"])
                ),
                "selected_allows_opponent_capture": bool(
                    row["selected_allows_opponent_capture"]
                ),
                "selected_allows_opponent_extra_turn": bool(
                    row["selected_allows_opponent_extra_turn"]
                ),
                "selected_grants_extra_turn": bool(row["selected_grants_extra_turn"]),
                "selected_performs_capture": bool(row["selected_performs_capture"]),
                "selected_move_consequence": dict(row["selected_move_consequence"]),
                "selected_move_visit_share": stable_float(
                    float(row["selected_move_visit_share"])
                ),
                "top2_visit_margin": stable_float(float(row["top2_visit_margin"])),
                "raw_policy_distribution": dict(row["raw_policy_distribution"]),
                "child_q_values": dict(row["child_q_values"]),
                "child_priors": dict(row["child_priors"]),
                "disadvantaged_outcome": stable_float(
                    float(row["disadvantaged_outcome"])
                ),
                "final_margin_for_disadvantaged": int(
                    row["final_margin_for_disadvantaged"]
                ),
                "seed": int(seed) + (index * 10000),
            }
        )
    return tasks


def run_probe_tasks(
    tasks: list[dict[str, Any]],
    *,
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
        init_probe_worker(
            str(artifact_path),
            budgets,
            default_c_puct,
            cpuct_schedule,
            tactical_root_bias,
        )
        return probe_task_records(batches[0])
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(batches),
        initializer=init_probe_worker,
        initargs=(
            str(artifact_path),
            budgets,
            float(default_c_puct),
            dict(cpuct_schedule),
            float(tactical_root_bias),
        ),
    ) as executor:
        for batch_rows in executor.map(probe_task_records, batches):
            results.extend(batch_rows)
    results.sort(key=lambda row: str(row["trace_id"]))
    return results


def classify_probe_families(probe: dict[str, Any]) -> dict[str, Any]:
    selected_move = int(probe["selected_move"])
    selected_raw_prior = float(probe["raw_policy_distribution"][str(selected_move)])
    selected_q = float(probe["child_q_values"].get(str(selected_move), 0.0))
    move_summaries = probe["move_summaries"]
    deeper_choice = probe.get("deeper_search_move")
    tactical_move = probe.get("tactical_candidate_move")
    raw_policy_move = probe.get("raw_policy_top_move")
    families: dict[str, dict[str, Any]] = {}
    best_local_improvement = 0.0
    best_label = None
    for move_summary in move_summaries.values():
        if int(move_summary["move"]) == selected_move:
            continue
        improvement = float(move_summary["mean_outcome_delta_vs_current"])
        if improvement > best_local_improvement:
            best_local_improvement = improvement
            best_label = ",".join(move_summary["labels"])

    if deeper_choice is not None:
        move_768 = probe["deeper_root_probes"]["768"]["selected_move"]
        move_1200 = probe["deeper_root_probes"]["1200"]["selected_move"]
        delta_768 = float(deeper_choice["outcome_delta_vs_current"]["768"])
        delta_1200 = float(deeper_choice["outcome_delta_vs_current"]["1200"])
        if (
            int(deeper_choice["move"]) != selected_move
            and delta_768 >= PRIMARY_IMPROVEMENT_THRESHOLD
            and (move_768 == move_1200 or delta_1200 >= delta_768)
        ):
            families[SEARCH_BUDGET_FAMILY] = {
                "improvement": stable_float(
                    float(deeper_choice["mean_outcome_delta_vs_current"])
                ),
                "evidence_move": int(deeper_choice["move"]),
                "move_768": move_768,
                "move_1200": move_1200,
            }

    if raw_policy_move is not None and str(int(raw_policy_move)) in move_summaries:
        raw_summary = move_summaries[str(int(raw_policy_move))]
        raw_q = float(probe["child_q_values"].get(str(int(raw_policy_move)), 0.0))
        if (
            selected_raw_prior >= 0.40
            and int(raw_policy_move) != selected_move
            and float(raw_summary["mean_outcome_delta_vs_current"])
            >= PRIMARY_IMPROVEMENT_THRESHOLD
            and float(raw_summary["mean_outcome_delta_vs_current"])
            >= best_local_improvement - 1e-9
        ):
            families[POLICY_FAMILY] = {
                "improvement": stable_float(
                    float(raw_summary["mean_outcome_delta_vs_current"])
                ),
                "evidence_move": int(raw_policy_move),
                "selected_raw_prior": stable_float(selected_raw_prior),
                "alternative_raw_prior": stable_float(
                    float(probe["raw_policy_distribution"][str(int(raw_policy_move))])
                ),
                "selected_q": stable_float(selected_q),
                "alternative_q": stable_float(raw_q),
            }

    helpful_q_candidates = []
    for move_summary in move_summaries.values():
        move = int(move_summary["move"])
        if move == selected_move:
            continue
        improvement = float(move_summary["mean_outcome_delta_vs_current"])
        alt_q = float(probe["child_q_values"].get(str(move), 0.0))
        if improvement >= PRIMARY_IMPROVEMENT_THRESHOLD and selected_q > alt_q + 0.05:
            helpful_q_candidates.append((improvement, move, alt_q))
    if helpful_q_candidates:
        helpful_q_candidates.sort(reverse=True)
        improvement, move, alt_q = helpful_q_candidates[0]
        families[VALUE_FAMILY] = {
            "improvement": stable_float(improvement),
            "evidence_move": int(move),
            "selected_q": stable_float(selected_q),
            "alternative_q": stable_float(alt_q),
        }

    if tactical_move is not None and str(int(tactical_move)) in move_summaries:
        tactical_summary = move_summaries[str(int(tactical_move))]
        tactical_consequence = move_consequence_for_state(
            probe["state"], int(tactical_move)
        )
        tactical_improvement = float(tactical_summary["mean_outcome_delta_vs_current"])
        if (
            tactical_improvement >= PRIMARY_IMPROVEMENT_THRESHOLD
            and not probe["selected_grants_extra_turn"]
            and not probe["selected_performs_capture"]
            and (
                bool(tactical_consequence["gives_extra_turn"])
                or bool(tactical_consequence["produces_capture"])
            )
        ):
            families[TACTICAL_MISS_FAMILY] = {
                "improvement": stable_float(tactical_improvement),
                "evidence_move": int(tactical_move),
            }
        tactical_after_state = after_state_for_move(probe["state"], int(tactical_move))
        tactical_opponent_flags = opponent_tactical_flags(tactical_after_state)
        if (
            tactical_improvement >= PRIMARY_IMPROVEMENT_THRESHOLD
            and (
                probe["selected_allows_opponent_capture"]
                or probe["selected_allows_opponent_extra_turn"]
            )
            and not tactical_opponent_flags["opponent_immediate_capture_available"]
            and not tactical_opponent_flags["opponent_immediate_extra_turn_available"]
        ):
            families[TACTICAL_BLUNDER_FAMILY] = {
                "improvement": stable_float(tactical_improvement),
                "evidence_move": int(tactical_move),
            }

    if not families and best_local_improvement < PRIMARY_IMPROVEMENT_THRESHOLD:
        families[NO_LOCAL_FAMILY] = {
            "improvement": stable_float(best_local_improvement),
            "evidence_move_labels": best_label,
        }

    return {
        "trace_id": str(probe["trace_id"]),
        "families": families,
        "best_local_improvement": stable_float(best_local_improvement),
    }


def aggregate_family_rows(
    probes: list[dict[str, Any]],
    classifications: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    probe_by_id = {str(probe["trace_id"]): probe for probe in probes}
    family_state_ids: dict[str, set[str]] = defaultdict(set)
    family_improvements: dict[str, list[float]] = defaultdict(list)
    family_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_openings: dict[str, set[str]] = defaultdict(set)
    for trace_id, classification in classifications.items():
        probe = probe_by_id[trace_id]
        for family_name, evidence in classification["families"].items():
            family_state_ids[family_name].add(trace_id)
            family_openings[family_name].add(str(probe["opening_prefix_text"]))
            family_improvements[family_name].append(
                float(evidence.get("improvement", 0.0))
            )
            family_examples[family_name].append(
                {
                    "trace_id": trace_id,
                    "state_hash": str(probe["state_hash"]),
                    "opening_prefix_text": str(probe["opening_prefix_text"]),
                    "ply": int(probe["ply"]),
                    "challenger_player": int(probe["challenger_player"]),
                    "selected_move": int(probe["selected_move"]),
                    "evidence_move": evidence.get("evidence_move"),
                    "improvement": stable_float(
                        float(evidence.get("improvement", 0.0))
                    ),
                }
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for family_name, trace_ids in family_state_ids.items():
        improvements = family_improvements[family_name]
        ci = bootstrap_ci(
            improvements,
            seed=42 + abs(hash(family_name)) % 100000,
            samples=DEFAULT_BOOTSTRAP_SAMPLES,
        )
        aggregate[family_name] = {
            "state_count": int(len(trace_ids)),
            "unique_opening_count": int(len(family_openings[family_name])),
            "mean_forced_improvement": mean_or_zero(improvements),
            "median_forced_improvement": median_or_zero(improvements),
            "bootstrap_ci": {
                "mean": stable_float(float(ci["mean"])),
                "lower": stable_float(float(ci["lower"])),
                "upper": stable_float(float(ci["upper"])),
                "n": int(ci["n"]),
            },
            "harmful_blunder_rate": stable_float(
                sum(
                    1
                    for value in improvements
                    if value >= PRIMARY_IMPROVEMENT_THRESHOLD
                )
                / max(len(improvements), 1)
            ),
            "representative_examples": sorted(
                family_examples[family_name],
                key=lambda row: float(row["improvement"]),
                reverse=True,
            )[:5],
            "trace_ids": sorted(trace_ids),
        }
    return aggregate


def annotate_concentration_families(
    probes: list[dict[str, Any]],
    classifications: dict[str, dict[str, Any]],
) -> None:
    bad_probe_ids = [
        str(probe["trace_id"])
        for probe in probes
        if float(classifications[str(probe["trace_id"])]["best_local_improvement"])
        >= 0.0
    ]
    if not bad_probe_ids:
        return
    probe_by_id = {str(probe["trace_id"]): probe for probe in probes}
    seat_counts = Counter(
        str(probe_by_id[trace_id]["seat_context"]) for trace_id in bad_probe_ids
    )
    dominant_seat, dominant_seat_count = seat_counts.most_common(1)[0]
    if dominant_seat_count / len(bad_probe_ids) >= 0.65:
        for probe in probes:
            if str(probe["seat_context"]) == dominant_seat:
                classifications[str(probe["trace_id"])]["families"].setdefault(
                    SEAT_FAMILY,
                    {
                        "improvement": classifications[str(probe["trace_id"])][
                            "best_local_improvement"
                        ]
                    },
                )
    family_counts = Counter(
        str(probe_by_id[trace_id]["first_move_family"]) for trace_id in bad_probe_ids
    )
    dominant_families = []
    running = 0
    for family_name, count in family_counts.most_common(3):
        dominant_families.append(family_name)
        running += count
        if running / len(bad_probe_ids) >= 0.50:
            break
    if dominant_families and running / len(bad_probe_ids) >= 0.50:
        dominant_family_set = set(dominant_families)
        for probe in probes:
            if str(probe["first_move_family"]) in dominant_family_set:
                classifications[str(probe["trace_id"])]["families"].setdefault(
                    OPENING_FAMILY,
                    {
                        "improvement": classifications[str(probe["trace_id"])][
                            "best_local_improvement"
                        ]
                    },
                )


def overlap_matrix(family_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    names = sorted(family_rows)
    matrix: dict[str, dict[str, int]] = {}
    for left in names:
        matrix[left] = {}
        left_ids = set(family_rows[left]["trace_ids"])
        for right in names:
            matrix[left][right] = int(
                len(left_ids & set(family_rows[right]["trace_ids"]))
            )
    return matrix


def build_top_costly_decisions(
    probes: list[dict[str, Any]],
    classifications: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        trace_id = str(probe["trace_id"])
        rows.append(
            {
                "trace_id": trace_id,
                "state_hash": str(probe["state_hash"]),
                "opening_prefix_text": str(probe["opening_prefix_text"]),
                "ply": int(probe["ply"]),
                "challenger_player": int(probe["challenger_player"]),
                "selected_move": int(probe["selected_move"]),
                "best_local_improvement": stable_float(
                    float(classifications[trace_id]["best_local_improvement"])
                ),
                "families": sorted(classifications[trace_id]["families"]),
                "final_margin_for_disadvantaged": int(
                    probe["final_margin_for_disadvantaged"]
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["best_local_improvement"]),
            -abs(int(row["final_margin_for_disadvantaged"])),
            -int(row["ply"]),
        ),
        reverse=True,
    )
    return rows[:25]


def top_family_rows(
    probes: list[dict[str, Any]],
    classifications: dict[str, dict[str, Any]],
    family_names: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        trace_id = str(probe["trace_id"])
        families = set(classifications[trace_id]["families"])
        if not families.intersection(family_names):
            continue
        rows.append(
            {
                "trace_id": trace_id,
                "state_hash": str(probe["state_hash"]),
                "opening_prefix_text": str(probe["opening_prefix_text"]),
                "best_local_improvement": stable_float(
                    float(classifications[trace_id]["best_local_improvement"])
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["best_local_improvement"]),
            str(row["opening_prefix_text"]),
            str(row["trace_id"]),
        ),
        reverse=True,
    )
    return rows[:25]


def best_local_improvement_for(
    trace_id: str, classifications: dict[str, dict[str, Any]]
) -> float:
    return float(classifications[str(trace_id)]["best_local_improvement"])


def deduplicated_rankings(
    probes: list[dict[str, Any]],
    classifications: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def row_score(row: dict[str, Any]) -> tuple[float, int, str]:
        return (
            best_local_improvement_for(str(row["trace_id"]), classifications),
            -abs(int(row["final_margin_for_disadvantaged"])),
            str(row["trace_id"]),
        )

    best_by_state: dict[str, dict[str, Any]] = {}
    best_by_prefix: dict[str, dict[str, Any]] = {}
    for probe in probes:
        state_hash = str(probe["state_hash"])
        prefix = str(probe["opening_prefix_text"])
        if state_hash not in best_by_state or row_score(probe) > row_score(
            best_by_state[state_hash]
        ):
            best_by_state[state_hash] = probe
        if prefix not in best_by_prefix or row_score(probe) > row_score(
            best_by_prefix[prefix]
        ):
            best_by_prefix[prefix] = probe

    state_rows = sorted(best_by_state.values(), key=row_score, reverse=True)
    prefix_rows = sorted(best_by_prefix.values(), key=row_score, reverse=True)

    family_counts: Counter[str] = Counter()
    family_improvements: dict[str, list[float]] = defaultdict(list)
    for probe in state_rows:
        trace_id = str(probe["trace_id"])
        for family_name, evidence in classifications[trace_id]["families"].items():
            family_counts[family_name] += 1
            family_improvements[family_name].append(
                float(evidence.get("improvement", 0.0))
            )

    return {
        "summary": {
            "original_probes": int(len(probes)),
            "unique_states": int(len(state_rows)),
            "unique_prefixes": int(len(prefix_rows)),
            "original_improvable_rows_ge_0_25": int(
                sum(
                    1
                    for probe in probes
                    if best_local_improvement_for(
                        str(probe["trace_id"]), classifications
                    )
                    >= PRIMARY_IMPROVEMENT_THRESHOLD
                )
            ),
            "unique_state_improvable_ge_0_25": int(
                sum(
                    1
                    for probe in state_rows
                    if best_local_improvement_for(
                        str(probe["trace_id"]), classifications
                    )
                    >= PRIMARY_IMPROVEMENT_THRESHOLD
                )
            ),
        },
        "top_25_unique_exact_states": [
            {
                "trace_id": str(probe["trace_id"]),
                "state_hash": str(probe["state_hash"]),
                "opening_prefix_text": str(probe["opening_prefix_text"]),
                "challenger_player": int(probe["challenger_player"]),
                "selected_move": int(probe["selected_move"]),
                "best_local_improvement": stable_float(
                    best_local_improvement_for(str(probe["trace_id"]), classifications)
                ),
                "families": sorted(classifications[str(probe["trace_id"])]["families"]),
            }
            for probe in state_rows[:25]
        ],
        "top_25_unique_opening_prefixes": [
            {
                "trace_id": str(probe["trace_id"]),
                "state_hash": str(probe["state_hash"]),
                "opening_prefix_text": str(probe["opening_prefix_text"]),
                "challenger_player": int(probe["challenger_player"]),
                "selected_move": int(probe["selected_move"]),
                "best_local_improvement": stable_float(
                    best_local_improvement_for(str(probe["trace_id"]), classifications)
                ),
                "families": sorted(classifications[str(probe["trace_id"])]["families"]),
            }
            for probe in prefix_rows[:25]
        ],
        "family_counts_after_exact_state_dedup": {
            family_name: {
                "state_count": int(family_counts[family_name]),
                "mean_improvement": mean_or_zero(family_improvements[family_name]),
            }
            for family_name in sorted(family_counts)
        },
    }


def outcome_summary(game_rows: list[dict[str, Any]]) -> dict[str, Any]:
    disadvantaged_outcomes = [float(row["disadvantaged_outcome"]) for row in game_rows]
    ds_contributions = [float(row["ds_contribution"]) for row in game_rows]
    by_seat: dict[str, list[float]] = defaultdict(list)
    for row in game_rows:
        by_seat[f"challenger_player_{row['challenger_player']}"].append(
            float(row["disadvantaged_outcome"])
        )
    return {
        "games": int(len(game_rows)),
        "mean_disadvantaged_outcome": mean_or_zero(disadvantaged_outcomes),
        "mean_ds_contribution": mean_or_zero(ds_contributions),
        "seat_context_mean_outcome": {
            key: mean_or_zero(values) for key, values in sorted(by_seat.items())
        },
    }


def root_trace_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = Counter(str(row["phase"]) for row in trace_rows)
    seats = Counter(str(row["seat_context"]) for row in trace_rows)
    return {
        "rows": int(len(trace_rows)),
        "phase_counts": {name: int(phases[name]) for name in sorted(phases)},
        "seat_context_counts": {name: int(seats[name]) for name in sorted(seats)},
        "mean_selected_visit_share": mean_or_zero(
            [float(row["selected_move_visit_share"]) for row in trace_rows]
        ),
        "mean_top2_visit_margin": mean_or_zero(
            [float(row["top2_visit_margin"]) for row in trace_rows]
        ),
        "selected_allows_opponent_capture_rate": stable_float(
            sum(
                1 for row in trace_rows if bool(row["selected_allows_opponent_capture"])
            )
            / max(len(trace_rows), 1)
        ),
        "selected_allows_opponent_extra_turn_rate": stable_float(
            sum(
                1
                for row in trace_rows
                if bool(row["selected_allows_opponent_extra_turn"])
            )
            / max(len(trace_rows), 1)
        ),
    }


def summarize_counterfactuals(probes: list[dict[str, Any]]) -> dict[str, Any]:
    better = []
    deeper = []
    tactical = []
    raw = []
    for probe in probes:
        best = 0.0
        deeper_choice = probe.get("deeper_search_move")
        if deeper_choice is not None:
            deeper.append(float(deeper_choice["mean_outcome_delta_vs_current"]))
        tactical_move = probe.get("tactical_candidate_move")
        if (
            tactical_move is not None
            and str(int(tactical_move)) in probe["move_summaries"]
        ):
            tactical.append(
                float(
                    probe["move_summaries"][str(int(tactical_move))][
                        "mean_outcome_delta_vs_current"
                    ]
                )
            )
        raw_policy_move = probe.get("raw_policy_top_move")
        if (
            raw_policy_move is not None
            and str(int(raw_policy_move)) in probe["move_summaries"]
        ):
            raw.append(
                float(
                    probe["move_summaries"][str(int(raw_policy_move))][
                        "mean_outcome_delta_vs_current"
                    ]
                )
            )
        for move_summary in probe["move_summaries"].values():
            if int(move_summary["move"]) == int(probe["selected_move"]):
                continue
            best = max(best, float(move_summary["mean_outcome_delta_vs_current"]))
        better.append(best)
    return {
        "probed_states": int(len(probes)),
        "states_with_any_improvement_ge_0_25": int(
            sum(1 for value in better if value >= PRIMARY_IMPROVEMENT_THRESHOLD)
        ),
        "mean_best_improvement": mean_or_zero(better),
        "deeper_search_mean_delta": mean_or_zero(deeper),
        "tactical_mean_delta": mean_or_zero(tactical),
        "raw_policy_mean_delta": mean_or_zero(raw),
    }


def choose_primary_classification(
    family_rows: dict[str, dict[str, Any]],
    probes: list[dict[str, Any]],
) -> tuple[str, str]:
    probed_count = max(len(probes), 1)
    search_share = (
        family_rows.get(SEARCH_BUDGET_FAMILY, {}).get("state_count", 0) / probed_count
    )
    tactical_count = family_rows.get(TACTICAL_MISS_FAMILY, {}).get(
        "state_count", 0
    ) + family_rows.get(TACTICAL_BLUNDER_FAMILY, {}).get("state_count", 0)
    tactical_share = tactical_count / probed_count
    policy_share = (
        family_rows.get(POLICY_FAMILY, {}).get("state_count", 0) / probed_count
    )
    value_share = family_rows.get(VALUE_FAMILY, {}).get("state_count", 0) / probed_count

    def ci_positive(name: str) -> bool:
        row = family_rows.get(name, {})
        return float((row.get("bootstrap_ci") or {}).get("lower", 0.0)) > 0.0

    if search_share >= 0.30 and ci_positive(SEARCH_BUDGET_FAMILY):
        return (
            "search_budget_limited_primary",
            "next PR should test selective deeper-search or confidence-triggered budget escalation",
        )
    if policy_share >= 0.20 and ci_positive(POLICY_FAMILY):
        return (
            "policy_prior_trap_primary",
            "next PR should test a targeted root-prior dampening transform on the detected motif",
        )
    if tactical_share >= 0.20 and (
        ci_positive(TACTICAL_MISS_FAMILY) or ci_positive(TACTICAL_BLUNDER_FAMILY)
    ):
        return (
            "tactical_guard_candidate",
            "next PR should test a diagnostic tactical root guard for capture and extra-turn motifs",
        )
    if value_share >= 0.20 and ci_positive(VALUE_FAMILY):
        return (
            "value_q_miscalibration_primary",
            "next PR should inspect value backup and child-afterstate value usage instead of fitting another mild value transform",
        )
    return (
        "diffuse_failure_no_single_intervention",
        "stop local target/intervention work and consider broader architecture or training-objective changes",
    )


def render_report(
    *,
    artifact_hash: str,
    runtime_profile: dict[str, Any],
    suite_hashes: dict[str, str],
    game_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    probe_summary: dict[str, Any],
    family_rows: dict[str, dict[str, Any]],
    overlap: dict[str, dict[str, int]],
    top_costly: list[dict[str, Any]],
    top_search_budget: list[dict[str, Any]],
    top_tactical: list[dict[str, Any]],
    dedup_summary: dict[str, Any],
    primary_classification: str,
    recommendation: str,
) -> str:
    family_table = markdown_table(
        [
            "Family",
            "States",
            "Openings",
            "Mean imp",
            "Median imp",
            "CI95",
            "Rate",
        ],
        [
            [
                name,
                str(int(row["state_count"])),
                str(int(row["unique_opening_count"])),
                fmt(float(row["mean_forced_improvement"])),
                fmt(float(row["median_forced_improvement"])),
                (
                    f"[{fmt(float(row['bootstrap_ci']['lower']))}, "
                    f"{fmt(float(row['bootstrap_ci']['upper']))}]"
                ),
                fmt(float(row["harmful_blunder_rate"])),
            ]
            for name, row in sorted(
                family_rows.items(),
                key=lambda item: item[1]["state_count"],
                reverse=True,
            )
        ],
    )
    overlap_headers = ["Family", *sorted(overlap)]
    overlap_rows = [
        [left, *[str(int(overlap[left][right])) for right in sorted(overlap)]]
        for left in sorted(overlap)
    ]
    overlap_table = markdown_table(overlap_headers, overlap_rows) if overlap else "n/a"
    top_costly_table = markdown_table(
        ["Trace", "State", "Prefix", "Ply", "Seat", "Selected", "Best imp", "Families"],
        [
            [
                row["trace_id"],
                row["state_hash"][:12],
                row["opening_prefix_text"],
                str(int(row["ply"])),
                str(int(row["challenger_player"])),
                str(int(row["selected_move"])),
                fmt(float(row["best_local_improvement"])),
                ",".join(row["families"]),
            ]
            for row in top_costly
        ],
    )
    search_budget_table = markdown_table(
        ["Trace", "State", "Prefix", "Best imp"],
        [
            [
                row["trace_id"],
                row["state_hash"][:12],
                row["opening_prefix_text"],
                fmt(float(row["best_local_improvement"])),
            ]
            for row in top_search_budget
        ],
    )
    tactical_table = markdown_table(
        ["Trace", "State", "Prefix", "Best imp"],
        [
            [
                row["trace_id"],
                row["state_hash"][:12],
                row["opening_prefix_text"],
                fmt(float(row["best_local_improvement"])),
            ]
            for row in top_tactical
        ],
    )
    dedup_state_table = markdown_table(
        ["Trace", "State", "Prefix", "Seat", "Selected", "Best imp", "Families"],
        [
            [
                row["trace_id"],
                row["state_hash"][:12],
                row["opening_prefix_text"],
                str(int(row["challenger_player"])),
                str(int(row["selected_move"])),
                fmt(float(row["best_local_improvement"])),
                ",".join(row["families"]),
            ]
            for row in dedup_summary["top_25_unique_exact_states"]
        ],
    )
    dedup_prefix_table = markdown_table(
        ["Trace", "State", "Prefix", "Seat", "Selected", "Best imp", "Families"],
        [
            [
                row["trace_id"],
                row["state_hash"][:12],
                row["opening_prefix_text"],
                str(int(row["challenger_player"])),
                str(int(row["selected_move"])),
                fmt(float(row["best_local_improvement"])),
                ",".join(row["families"]),
            ]
            for row in dedup_summary["top_25_unique_opening_prefixes"]
        ],
    )
    dedup_family_table = markdown_table(
        ["Family", "State count", "Mean imp"],
        [
            [
                family_name,
                str(int(row["state_count"])),
                fmt(float(row["mean_improvement"])),
            ]
            for family_name, row in sorted(
                dedup_summary["family_counts_after_exact_state_dedup"].items(),
                key=lambda item: item[1]["state_count"],
                reverse=True,
            )
        ],
    )
    suite_lines = "\n".join(
        f"- {name}: `{sha}`" for name, sha in sorted(suite_hashes.items())
    )
    return "\n".join(
        [
            "# AlphaZero-Lite Current 384:256 Failure Family Trace Results",
            "",
            f"**Classification**: `{primary_classification}`",
            "",
            "## Artifact Hash",
            "",
            f"- current weights SHA256: `{artifact_hash}`",
            "",
            "## Promoted Runtime Profile Confirmation",
            "",
            f"- runtime profile: `{canonical_json(runtime_profile)}`",
            "",
            "## Suite Hashes",
            "",
            suite_lines,
            "",
            "## 384:256 Outcome Summary",
            "",
            f"- games: `{game_summary['games']}`",
            f"- mean disadvantaged outcome: `{fmt(float(game_summary['mean_disadvantaged_outcome']))}`",
            f"- mean DS contribution: `{fmt(float(game_summary['mean_ds_contribution']))}`",
            f"- seat context means: `{canonical_json(game_summary['seat_context_mean_outcome'])}`",
            "",
            "## Root Trace Telemetry Summary",
            "",
            f"- rows: `{trace_summary['rows']}`",
            f"- phase counts: `{canonical_json(trace_summary['phase_counts'])}`",
            f"- seat context counts: `{canonical_json(trace_summary['seat_context_counts'])}`",
            f"- mean selected visit share: `{fmt(float(trace_summary['mean_selected_visit_share']))}`",
            f"- mean top-2 visit margin: `{fmt(float(trace_summary['mean_top2_visit_margin']))}`",
            f"- selected allows opponent capture rate: `{fmt(float(trace_summary['selected_allows_opponent_capture_rate']))}`",
            f"- selected allows opponent extra-turn rate: `{fmt(float(trace_summary['selected_allows_opponent_extra_turn_rate']))}`",
            "",
            "## Counterfactual Probe Summary",
            "",
            f"- probed states: `{probe_summary['probed_states']}`",
            f"- states with any improvement >= +0.25: `{probe_summary['states_with_any_improvement_ge_0_25']}`",
            f"- mean best improvement: `{fmt(float(probe_summary['mean_best_improvement']))}`",
            f"- deeper-search mean delta: `{fmt(float(probe_summary['deeper_search_mean_delta']))}`",
            f"- tactical mean delta: `{fmt(float(probe_summary['tactical_mean_delta']))}`",
            f"- raw-policy mean delta: `{fmt(float(probe_summary['raw_policy_mean_delta']))}`",
            "",
            "## Failure-Family Taxonomy Table",
            "",
            family_table,
            "",
            "## Family Overlap Matrix",
            "",
            overlap_table,
            "",
            "## Top 25 Most Costly Root Decisions",
            "",
            top_costly_table,
            "",
            "## Top 25 Search-Budget-Limited States",
            "",
            search_budget_table,
            "",
            "## Top 25 Tactical Capture/Extra-Turn Failures",
            "",
            tactical_table,
            "",
            "## Recommended Next Intervention",
            "",
            f"- {recommendation}",
            "",
            "## Deduplication Addendum",
            "",
            f"- original probes: `{dedup_summary['summary']['original_probes']}`",
            f"- unique exact states: `{dedup_summary['summary']['unique_states']}`",
            f"- unique opening prefixes: `{dedup_summary['summary']['unique_prefixes']}`",
            f"- original improvable rows >= +0.25: `{dedup_summary['summary']['original_improvable_rows_ge_0_25']}`",
            f"- unique exact states improvable >= +0.25: `{dedup_summary['summary']['unique_state_improvable_ge_0_25']}`",
            "",
            "### Top 25 Unique Exact States",
            "",
            dedup_state_table,
            "",
            "### Top 25 Unique Opening Prefixes",
            "",
            dedup_prefix_table,
            "",
            "### Family Counts After Exact-State Dedup",
            "",
            dedup_family_table,
        ]
    )


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_path = existing_path(args.current)
    verify_current_weights_sha256(
        current_path,
        str(args.expected_current_weights_sha256),
    )
    search_profile = build_promoted_search_profile()
    validate_guardrails(search_profile=search_profile, model_type="residual_v3")

    challenger_simulations, current_simulations = parse_budget_pair(args.budget_pair)
    continuation_budgets = parse_budgets(args.counterfactual_continuation_budgets)
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    effective_c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=challenger_simulations,
        current_simulations=current_simulations,
        default_c_puct=float(args.default_c_puct),
    )
    runtime_profile = {
        "artifact": str(current_path),
        "budget_pair": budget_pair_label(challenger_simulations, current_simulations),
        "default_c_puct": float(args.default_c_puct),
        "effective_c_puct_384_256": stable_float(float(effective_c_puct)),
        "c_puct_schedule": {
            "default_c_puct": float(args.default_c_puct),
            "overrides": cpuct_schedule,
        },
        "root_policy_mode": "deterministic",
        "root_prior_transform": None,
        "value_transform": None,
        "tactical_root_bias": float(args.tactical_root_bias),
    }
    profile = build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=max(challenger_simulations, current_simulations),
        c_puct=float(effective_c_puct),
        search_options=build_eval_search_options(
            root_policy_mode="deterministic",
            tactical_root_bias=float(args.tactical_root_bias),
        ),
        extra_fields={
            "challenger_simulations": int(challenger_simulations),
            "current_simulations": int(current_simulations),
        },
    )

    fixed_large = load_suite_payload(existing_path(args.fixed_large_suite))
    medium_suite_path = existing_path(args.medium_suite)
    medium_sha = sha256_file(medium_suite_path)
    heldout_paths = [
        existing_path(part.strip())
        for part in str(args.heldout_suites).split(",")
        if part.strip()
    ]
    heldout_suites = [load_suite_payload(path) for path in heldout_paths]
    suites = [fixed_large, *heldout_suites]
    suite_hashes = {payload["suite_name"]: str(payload["sha256"]) for payload in suites}
    suite_hashes[medium_suite_path.stem] = medium_sha

    trace_tasks: list[dict[str, Any]] = []
    game_index = 0
    for suite in suites:
        for opening_index, opening_entry in enumerate(suite["rows"]):
            for challenger_player in (0, 1):
                trace_tasks.append(
                    {
                        "suite_name": str(suite["suite_name"]),
                        "suite_sha256": str(suite["sha256"]),
                        "opening_index": int(opening_index),
                        "opening_entry": opening_entry,
                        "challenger_player": int(challenger_player),
                        "game_index": int(game_index),
                        "seed": int(args.seed) + (game_index * 100),
                    }
                )
                game_index += 1

    trace_results: list[dict[str, Any]] = []
    trace_batches = partition_batches(trace_tasks, args.workers)
    if len(trace_batches) == 1:
        init_trace_worker(
            str(current_path),
            challenger_simulations,
            current_simulations,
            effective_c_puct,
            str(profile["hash"]),
            float(args.tactical_root_bias),
        )
        for task in trace_batches[0]:
            trace_results.append(trace_game_task(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(trace_batches),
            initializer=init_trace_worker,
            initargs=(
                str(current_path),
                int(challenger_simulations),
                int(current_simulations),
                float(effective_c_puct),
                str(profile["hash"]),
                float(args.tactical_root_bias),
            ),
        ) as executor:
            flat_tasks = [task for batch in trace_batches for task in batch]
            for result in executor.map(trace_game_task, flat_tasks):
                trace_results.append(result)

    game_rows = sorted(
        [result["game_outcome"] for result in trace_results],
        key=lambda row: (str(row["suite_name"]), int(row["game_index"])),
    )
    root_trace_rows = []
    for result in trace_results:
        root_trace_rows.extend(result["root_traces"])
    for index, row in enumerate(
        sorted(
            root_trace_rows,
            key=lambda entry: (
                str(entry["suite_name"]),
                int(entry["game_index"]),
                int(entry["ply"]),
                int(entry["decision_index"]),
            ),
        )
    ):
        row["trace_id"] = f"trace-{index:05d}"

    write_jsonl(workdir / ROOT_TRACE_FILENAME, root_trace_rows)
    write_jsonl(workdir / GAME_OUTCOMES_FILENAME, game_rows)

    probe_rows = select_probe_rows(root_trace_rows)
    probe_tasks = build_probe_tasks(probe_rows, seed=int(args.seed))
    probe_results = run_probe_tasks(
        probe_tasks,
        artifact_path=current_path,
        budgets=continuation_budgets,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        workers=int(args.workers),
    )
    write_jsonl(workdir / COUNTERFACTUAL_FILENAME, probe_results)

    classifications = {
        str(result["trace_id"]): classify_probe_families(result)
        for result in probe_results
    }
    annotate_concentration_families(probe_results, classifications)
    family_rows = aggregate_family_rows(probe_results, classifications)
    overlap = overlap_matrix(family_rows)
    top_costly = build_top_costly_decisions(probe_results, classifications)
    top_search_budget = top_family_rows(
        probe_results,
        classifications,
        {SEARCH_BUDGET_FAMILY},
    )
    top_tactical = top_family_rows(
        probe_results,
        classifications,
        {TACTICAL_MISS_FAMILY, TACTICAL_BLUNDER_FAMILY},
    )
    dedup_summary = deduplicated_rankings(probe_results, classifications)
    primary_classification, recommendation = choose_primary_classification(
        family_rows,
        probe_results,
    )

    summary = {
        "schema": SCHEMA,
        "artifact_hash": str(args.expected_current_weights_sha256),
        "runtime_profile": runtime_profile,
        "suite_hashes": suite_hashes,
        "game_outcome_summary": outcome_summary(game_rows),
        "root_trace_summary": root_trace_summary(root_trace_rows),
        "counterfactual_probe_summary": summarize_counterfactuals(probe_results),
        "family_rows": {
            name: {key: value for key, value in row.items() if key != "trace_ids"}
            for name, row in sorted(family_rows.items())
        },
        "family_overlap": overlap,
        "top_costly_root_decisions": top_costly,
        "deduplicated_rankings": dedup_summary,
        "primary_classification": primary_classification,
        "recommendation": recommendation,
        "artifacts": {
            "root_trace": str(workdir / ROOT_TRACE_FILENAME),
            "game_outcomes": str(workdir / GAME_OUTCOMES_FILENAME),
            "counterfactual_probes": str(workdir / COUNTERFACTUAL_FILENAME),
            "deduplicated_rankings": str(workdir / DEDUP_SUMMARY_FILENAME),
        },
    }
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_json(workdir / DEDUP_SUMMARY_FILENAME, dedup_summary)

    report_text = render_report(
        artifact_hash=str(args.expected_current_weights_sha256),
        runtime_profile=runtime_profile,
        suite_hashes=suite_hashes,
        game_summary=summary["game_outcome_summary"],
        trace_summary=summary["root_trace_summary"],
        probe_summary=summary["counterfactual_probe_summary"],
        family_rows=family_rows,
        overlap=overlap,
        top_costly=top_costly,
        top_search_budget=top_search_budget,
        top_tactical=top_tactical,
        dedup_summary=dedup_summary,
        primary_classification=primary_classification,
        recommendation=recommendation,
    )
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
