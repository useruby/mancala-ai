#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_incumbent_proxy_disagreement_value_backup_audit import (
    child_state_from_move,
    load_json,
    round_float,
    state_to_root_perspective_value,
    write_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_PRIOR_REPORT_PATH = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-residual-reference-adjudication-results.md"
)
DEFAULT_SUMMARY_OUT = Path(
    "/tmp/azlite_incumbent_proxy_teacher_policy_decision/"
    "teacher_policy_decision_summary.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-teacher-policy-decision-results.md"
)
SCHEMA = "azlite_incumbent_proxy_teacher_policy_decision_v1"
FAMILY = "incumbent_proxy_disagreement"
ROOT_SELECTION_PRESSURE_ROWS = (
    "incumbent_proxy_disagreement-007",
    "incumbent_proxy_disagreement-009",
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-018",
    "incumbent_proxy_disagreement-021",
    "incumbent_proxy_disagreement-023",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-032",
    "incumbent_proxy_disagreement-033",
)
VALUE_HEAD_CANDIDATE_ROWS = (
    "incumbent_proxy_disagreement-003",
    "incumbent_proxy_disagreement-012",
    "incumbent_proxy_disagreement-020",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-027",
    "incumbent_proxy_disagreement-035",
)
CONTROL_ROWS = (
    "incumbent_proxy_disagreement-008",
    "incumbent_proxy_disagreement-026",
    "incumbent_proxy_disagreement-028",
    "incumbent_proxy_disagreement-029",
)
EXCLUDED_ROWS = ("incumbent_proxy_disagreement-010",)
ROW_GROUP_BY_ID = {
    **{row_id: "root_selection_pressure" for row_id in ROOT_SELECTION_PRESSURE_ROWS},
    **{row_id: "value_head_candidate" for row_id in VALUE_HEAD_CANDIDATE_ROWS},
    **{row_id: "control" for row_id in CONTROL_ROWS},
    **{row_id: "excluded_unstable" for row_id in EXCLUDED_ROWS},
}
ROW_IDS = tuple(sorted(ROW_GROUP_BY_ID))
ROOT_CLASSIC_BUDGETS = (1200, 2400, 5000, 10000)
ROOT_CLASSIC_SEEDS = (11, 23, 37, 42, 101, 202, 303)
ROOT_PUCT_BUDGETS = (384, 1200, 2400, 5000)
HIGH_SIM_PUCT_BUDGETS = (384, 1200, 2400, 5000, 10000)
CHILD_CLASSIC_BUDGETS = (2400, 5000)
CHILD_CLASSIC_SEEDS = (11, 23, 37, 42, 101)
CHILD_PUCT_BUDGETS = (1200, 5000)
CONTINUATION_CLASSIC_BUDGET = 1200
CONTINUATION_PUCT_BUDGET = 1200
CONTINUATION_COUNT = 8
CONTINUATION_MAX_PLIES = 200
C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
STABILITY_FRACTION = 0.71
SMALL_DELTA = 0.02


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--prior-report-path", type=Path, default=DEFAULT_PRIOR_REPORT_PATH
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--continuations", type=int, default=CONTINUATION_COUNT)
    parser.add_argument("--skip-high-sim-10000", action="store_true")
    return parser.parse_args(argv)


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def canonical_reference_rows(
    reference_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = list(reference_payload.get("rows") or [])
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        by_id[str(row["id"])] = row
    return by_id


def remaining_seed_count(state: dict[str, Any]) -> int:
    return int(sum(KalahGame.from_state(state).pits))


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return round_float(mean), round_float(std)


def mean_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        rounded = round_float(mean)
        return rounded, rounded, rounded
    std = statistics.stdev(values)
    margin = 1.96 * (std / math.sqrt(len(values)))
    return round_float(mean), round_float(mean - margin), round_float(mean + margin)


def counter_to_text(counter: Counter[int]) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{move}:{count}" for move, count in sorted(counter.items()))


def majority_from_counter(
    counter: Counter[int], total: int
) -> tuple[int | None, float | None]:
    if not counter or total <= 0:
        return None, None
    move, count = max(counter.items(), key=lambda item: (int(item[1]), -int(item[0])))
    return int(move), round_float(count / total)


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(value) for value in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / total)


def q_by_move_from_puct(result: dict[str, Any]) -> dict[int, float]:
    return {
        int(child["move"]): round_float(float(child.get("q_value", 0.0))) or 0.0
        for child in list(result.get("child_stats") or [])
        if child.get("move") is not None
    }


def prior_by_move_from_puct(result: dict[str, Any]) -> dict[int, float]:
    legal_moves = [int(move) for move in list(result.get("legal_moves") or [])]
    policy = list(result.get("policy") or [])
    return {
        move: round_float(float(policy[move])) or 0.0
        for move in legal_moves
        if move < len(policy)
    }


def top_margin_from_q_map(q_by_move: dict[int, float]) -> float | None:
    if len(q_by_move) < 2:
        return None
    ordered = sorted(
        q_by_move.items(), key=lambda item: (-float(item[1]), int(item[0]))
    )
    return round_float(float(ordered[0][1]) - float(ordered[1][1]))


def terminal_state_raw_value(state: dict[str, Any], root_player: int) -> float:
    game = KalahGame.from_state(state)
    settled = game.captured_seeds.copy()
    for player in (0, 1):
        start = player * 6
        settled[player] += sum(game.pits[start : start + 6])
    if settled[root_player] > settled[1 - root_player]:
        return 1.0
    if settled[root_player] < settled[1 - root_player]:
        return -1.0
    return 0.0


def classic_root_run(
    state: dict[str, Any], *, budget: int, seed: int
) -> dict[str, Any]:
    summary = ClassicMCTS(
        KalahGame.from_state(state), simulations=int(budget), seed=int(seed)
    ).root_summary()
    visits_by_move: dict[int, int] = {}
    q_by_move: dict[int, float] = {}
    for child in list(summary.get("child_stats") or []):
        move = int(child["move"])
        visits_by_move[move] = int(child.get("visits", 0))
        q_by_move[move] = (
            round_float((2.0 * float(child.get("win_rate", 0.0))) - 1.0) or 0.0
        )
    total_visits = sum(visits_by_move.values())
    visit_share_by_move = {
        move: round_float(count / total_visits) if total_visits > 0 else None
        for move, count in visits_by_move.items()
    }
    selected_move = summary.get("selected_move")
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "visits_by_move": visits_by_move,
        "visit_share_by_move": visit_share_by_move,
        "q_by_move": q_by_move,
        "top1_margin": top_margin_from_q_map(q_by_move),
    }


def classic_child_run(
    child_state: dict[str, Any], *, budget: int, seed: int, root_player: int
) -> dict[str, Any]:
    game = KalahGame.from_state(child_state)
    summary = ClassicMCTS(game, simulations=int(budget), seed=int(seed)).root_summary()
    selected_move = summary.get("selected_move")
    raw_value = None
    q_by_move: dict[int, float] = {}
    visits_by_move: dict[int, int] = {}
    for child in list(summary.get("child_stats") or []):
        move = int(child["move"])
        visits_by_move[move] = int(child.get("visits", 0))
        q_by_move[move] = (
            round_float((2.0 * float(child.get("win_rate", 0.0))) - 1.0) or 0.0
        )
    if selected_move is not None:
        raw_value = q_by_move.get(int(selected_move), 0.0)
    elif game.over():
        raw_value = terminal_state_raw_value(
            child_state, int(child_state["current_player"])
        )
    root_value = None
    if raw_value is not None:
        root_value = state_to_root_perspective_value(
            raw_value=float(raw_value), state=child_state, root_player=root_player
        )
    total_visits = sum(visits_by_move.values())
    visit_share_by_move = {
        move: round_float(count / total_visits) if total_visits > 0 else None
        for move, count in visits_by_move.items()
    }
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "raw_value": round_float(raw_value),
        "root_value": round_float(root_value),
        "visits_by_move": visits_by_move,
        "visit_share_by_move": visit_share_by_move,
        "q_by_move": q_by_move,
        "top1_margin": top_margin_from_q_map(q_by_move),
    }


def deterministic_puct_run(
    evaluator: ArtifactEvaluator, state: dict[str, Any], *, budget: int
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=17,
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    legal_moves = [int(move) for move in list(result.get("legal_moves") or [])]
    visits = [float(value) for value in list(result.get("visits") or [])]
    selected_move = result.get("selected_move")
    q_by_move = q_by_move_from_puct(result)
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "legal_moves": legal_moves,
        "visit_share_by_move": {
            move: visit_share(visits, move) for move in legal_moves
        },
        "q_by_move": q_by_move,
        "prior_by_move": prior_by_move_from_puct(result),
        "root_value": round_float(float(result.get("value", 0.0))),
        "top1_margin": top_margin_from_q_map(q_by_move),
    }


def puct_child_run(
    evaluator: ArtifactEvaluator,
    child_state: dict[str, Any],
    *,
    budget: int,
    root_player: int,
) -> dict[str, Any]:
    result = deterministic_puct_run(evaluator, child_state, budget=int(budget))
    root_value = state_to_root_perspective_value(
        raw_value=float(result["root_value"] or 0.0),
        state=child_state,
        root_player=root_player,
    )
    return {
        **result,
        "root_value": round_float(root_value),
    }


def tablebase_preferred_move(
    tablebase: EndgameTablebase, state: dict[str, Any], *, root_player: int
) -> tuple[float | None, int | None]:
    game = KalahGame.from_state(state)
    value = tablebase.lookup_cached(game, root_player)
    if value is None:
        value = tablebase.lookup(game, root_player)
    if value is None:
        return None, None
    legal_moves = game.possible_moves()
    if not legal_moves:
        return round_float((2.0 * float(value)) - 1.0), None
    best_move = None
    best_value = None
    for move in legal_moves:
        child_game = KalahGame.from_state(child_state_from_move(state, move))
        child_value = tablebase.lookup_cached(child_game, root_player)
        if child_value is None:
            child_value = tablebase.lookup(child_game, root_player)
        if child_value is None:
            return round_float((2.0 * float(value)) - 1.0), None
        if (
            best_value is None
            or float(child_value) > float(best_value)
            or (
                float(child_value) == float(best_value)
                and (best_move is None or int(move) < int(best_move))
            )
        ):
            best_value = float(child_value)
            best_move = int(move)
    return round_float((2.0 * float(value)) - 1.0), best_move


def parse_prior_report(path: Path) -> dict[str, dict[str, int | None]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, int | None]] = {}
    section = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## 4. Root ClassicMCTS adjudication"):
            section = "classic"
            continue
        if line.startswith("## 7. PUCT/artifact teacher comparison"):
            section = "puct"
            continue
        if line.startswith("## "):
            section = None
            continue
        if not line.startswith("| incumbent_proxy_disagreement-"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if section == "classic" and len(parts) >= 11:
            row_id = parts[0]
            budget = int(parts[1])
            majority_move = None if parts[5] == "-" else int(parts[5])
            previous = rows.setdefault(row_id, {})
            current_budget = previous.get("classic_budget")
            if current_budget is None or budget >= int(current_budget):
                previous["classic_budget"] = budget
                previous["classic_majority_move"] = majority_move
        elif section == "puct" and len(parts) >= 9:
            row_id = parts[0]
            budget = int(parts[1])
            selected_move = None if parts[3] == "-" else int(parts[3])
            previous = rows.setdefault(row_id, {})
            current_budget = previous.get("puct_budget")
            if current_budget is None or budget >= int(current_budget):
                previous["puct_budget"] = budget
                previous["puct_selected_move"] = selected_move
    return rows


def row_consequences(
    state: dict[str, Any], legal_moves: list[int]
) -> list[dict[str, Any]]:
    consequences = []
    for move in legal_moves:
        immediate = move_consequence_for_state(state, move)
        child_state = child_state_from_move(state, move)
        consequences.append(
            {
                "move": int(move),
                "gives_extra_turn": bool(immediate["gives_extra_turn"]),
                "produces_capture": bool(immediate["produces_capture"]),
                "capture_count": int(immediate["capture_count"]),
                "immediate_store_delta": int(immediate["store_delta_immediate"]),
                "side_to_move_after": int(child_state["current_player"]),
                "game_over_after_move": bool(KalahGame.from_state(child_state).over()),
                "remaining_seed_count": remaining_seed_count(child_state),
            }
        )
    return consequences


def validate_rows(
    *,
    suite_by_id: dict[str, Any],
    reference_by_id: dict[str, dict[str, Any]],
    prior_report: dict[str, dict[str, int | None]],
    tablebase: EndgameTablebase,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    validation_table: list[dict[str, Any]] = []
    for row_id in ROW_IDS:
        suite_row = suite_by_id.get(row_id)
        reference_row = reference_by_id.get(row_id)
        if suite_row is None or reference_row is None:
            raise ValueError(f"missing suite/reference row for {row_id}")
        state = dict(suite_row.state)
        reference_state = dict(reference_row["state"])
        canonical_match = canonical_state_key(state) == canonical_state_key(
            reference_state
        )
        legal_moves = list(suite_row.legal_moves)
        reference_move = int(reference_row["reference_move"])
        consequences = row_consequences(state, legal_moves)
        consequence_by_move = {int(row["move"]): row for row in consequences}
        tablebase_root_value, tablebase_root_move = tablebase_preferred_move(
            tablebase, state, root_player=int(state["current_player"])
        )
        row_payload = {
            "row_id": row_id,
            "row_group": ROW_GROUP_BY_ID[row_id],
            "suite_state": state,
            "canonical_state_hash": canonical_state_key(state),
            "active_reference_move": reference_move,
            "legal_moves": legal_moves,
            "reference_unstable": bool(reference_row.get("reference_unstable", False)),
            "excluded_unstable": row_id in EXCLUDED_ROWS,
            "canonical_state_match": canonical_match,
            "pr45_classic_majority_move": prior_report.get(row_id, {}).get(
                "classic_majority_move"
            ),
            "pr45_puct_selected_move": prior_report.get(row_id, {}).get(
                "puct_selected_move"
            ),
            "move_consequences": consequences,
            "move_consequence_by_move": consequence_by_move,
            "remaining_seed_count": remaining_seed_count(state),
            "tablebase_root_available": tablebase_root_value is not None,
            "tablebase_root_value": tablebase_root_value,
            "tablebase_root_move": tablebase_root_move,
        }
        valid_rows.append(row_payload)
        validation_table.append(
            {
                "row_id": row_id,
                "row_group": ROW_GROUP_BY_ID[row_id],
                "active_reference_move": reference_move,
                "reference_move_legal": reference_move in legal_moves,
                "canonical_state_match": canonical_match,
                "reference_unstable": bool(
                    reference_row.get("reference_unstable", False)
                ),
                "excluded_unstable": row_id in EXCLUDED_ROWS,
                "remaining_seed_count": row_payload["remaining_seed_count"],
                "tablebase_root_available": tablebase_root_value is not None,
                "notes": (
                    "row 010 kept excluded/unstable by audit policy"
                    if row_id in EXCLUDED_ROWS
                    else "validated against active references"
                ),
            }
        )
    return valid_rows, validation_table


def summarize_classic_root(row: dict[str, Any]) -> dict[str, Any]:
    runs_by_budget: dict[str, dict[str, Any]] = {}
    majority_by_budget: dict[int, int | None] = {}
    stable_budgets: list[int] = []
    for budget in ROOT_CLASSIC_BUDGETS:
        selected_counter: Counter[int] = Counter()
        top1_margins: list[float] = []
        reference_margins: list[float] = []
        seed_runs: list[dict[str, Any]] = []
        for seed in ROOT_CLASSIC_SEEDS:
            run = classic_root_run(
                row["suite_state"], budget=int(budget), seed=int(seed)
            )
            selected_move = run["selected_move"]
            if selected_move is not None:
                selected_counter[int(selected_move)] += 1
            if run["top1_margin"] is not None:
                top1_margins.append(float(run["top1_margin"]))
            if (
                selected_move is not None
                and row["active_reference_move"] in run["q_by_move"]
                and selected_move in run["q_by_move"]
            ):
                reference_margins.append(
                    float(run["q_by_move"].get(int(selected_move), 0.0))
                    - float(
                        run["q_by_move"].get(int(row["active_reference_move"]), 0.0)
                    )
                )
            seed_runs.append(run)
        majority_move, majority_fraction = majority_from_counter(
            selected_counter, len(ROOT_CLASSIC_SEEDS)
        )
        if (
            majority_move is not None
            and majority_fraction is not None
            and float(majority_fraction) >= STABILITY_FRACTION
        ):
            stable_budgets.append(int(budget))
        majority_by_budget[int(budget)] = majority_move
        runs_by_budget[str(budget)] = {
            "budget": int(budget),
            "seed_runs": seed_runs,
            "selected_counter": dict(selected_counter),
            "majority_move": majority_move,
            "majority_fraction": majority_fraction,
            "visit_share_by_move": {
                str(move): mean_std(
                    [
                        float(seed_run["visit_share_by_move"].get(move) or 0.0)
                        for seed_run in seed_runs
                    ]
                )[0]
                for move in row["legal_moves"]
            },
            "q_by_move": {
                str(move): mean_std(
                    [
                        float(seed_run["q_by_move"].get(move) or 0.0)
                        for seed_run in seed_runs
                    ]
                )[0]
                for move in row["legal_moves"]
            },
            "top1_margin_mean": mean_std(top1_margins)[0],
            "selected_minus_reference_mean": mean_std(reference_margins)[0],
        }
    stable_move = None
    stable = False
    high_budget_moves = [
        majority_by_budget.get(budget) for budget in (2400, 5000, 10000)
    ]
    if high_budget_moves and all(
        move == high_budget_moves[0] for move in high_budget_moves
    ):
        stable_move = high_budget_moves[0]
        stable = stable_move is not None
    selected_move = (
        stable_move or runs_by_budget[str(max(ROOT_CLASSIC_BUDGETS))]["majority_move"]
    )
    return {
        "selected_move": selected_move,
        "stable": stable,
        "stable_budgets": stable_budgets,
        "majority_by_budget": majority_by_budget,
        "runs_by_budget": runs_by_budget,
        "stability_note": (
            "high budgets agree across 2400/5000/10000"
            if stable
            else "high-budget seed majorities remain mixed"
        ),
    }


def summarize_puct_root(
    evaluator: ArtifactEvaluator, row: dict[str, Any], budgets: tuple[int, ...]
) -> dict[str, Any]:
    runs_by_budget: dict[str, dict[str, Any]] = {}
    selected_sequence: list[int | None] = []
    for budget in budgets:
        run = deterministic_puct_run(evaluator, row["suite_state"], budget=int(budget))
        selected_sequence.append(run["selected_move"])
        runs_by_budget[str(budget)] = run
    selected_move = runs_by_budget[str(max(budgets))]["selected_move"]
    stable = all(move == selected_sequence[0] for move in selected_sequence)
    return {
        "selected_move": selected_move,
        "stable": stable,
        "selected_sequence": selected_sequence,
        "runs_by_budget": runs_by_budget,
        "stability_note": (
            "deterministic selection unchanged across budgets"
            if stable
            else "deterministic selection changes across budgets"
        ),
    }


def summarize_child_classic(
    child_state: dict[str, Any], *, root_player: int
) -> dict[str, Any]:
    runs_by_budget: dict[str, dict[str, Any]] = {}
    for budget in CHILD_CLASSIC_BUDGETS:
        root_values: list[float] = []
        top1_margins: list[float] = []
        selected_counter: Counter[int] = Counter()
        seed_runs: list[dict[str, Any]] = []
        for seed in CHILD_CLASSIC_SEEDS:
            run = classic_child_run(
                child_state, budget=int(budget), seed=int(seed), root_player=root_player
            )
            seed_runs.append(run)
            if run["root_value"] is not None:
                root_values.append(float(run["root_value"]))
            if run["top1_margin"] is not None:
                top1_margins.append(float(run["top1_margin"]))
            if run["selected_move"] is not None:
                selected_counter[int(run["selected_move"])] += 1
        runs_by_budget[str(budget)] = {
            "budget": int(budget),
            "seed_runs": seed_runs,
            "selected_counter": dict(selected_counter),
            "selected_move": majority_from_counter(
                selected_counter, len(CHILD_CLASSIC_SEEDS)
            )[0],
            "selected_fraction": majority_from_counter(
                selected_counter, len(CHILD_CLASSIC_SEEDS)
            )[1],
            "root_value_mean": mean_std(root_values)[0],
            "root_value_std": mean_std(root_values)[1],
            "top1_margin_mean": mean_std(top1_margins)[0],
        }
    selected_moves = [
        runs_by_budget[str(budget)]["selected_move"] for budget in CHILD_CLASSIC_BUDGETS
    ]
    stable = all(move == selected_moves[0] for move in selected_moves)
    return {
        "selected_move": runs_by_budget[str(max(CHILD_CLASSIC_BUDGETS))][
            "selected_move"
        ],
        "stable": stable,
        "runs_by_budget": runs_by_budget,
        "root_value_mean": mean_std(
            [
                float(runs_by_budget[str(budget)]["root_value_mean"])
                for budget in CHILD_CLASSIC_BUDGETS
                if runs_by_budget[str(budget)]["root_value_mean"] is not None
            ]
        )[0],
        "root_value_std": mean_std(
            [
                float(runs_by_budget[str(budget)]["root_value_mean"])
                for budget in CHILD_CLASSIC_BUDGETS
                if runs_by_budget[str(budget)]["root_value_mean"] is not None
            ]
        )[1],
        "top1_margin_mean": mean_std(
            [
                float(runs_by_budget[str(budget)]["top1_margin_mean"])
                for budget in CHILD_CLASSIC_BUDGETS
                if runs_by_budget[str(budget)]["top1_margin_mean"] is not None
            ]
        )[0],
    }


def summarize_child_puct(
    evaluator: ArtifactEvaluator, child_state: dict[str, Any], *, root_player: int
) -> dict[str, Any]:
    runs_by_budget: dict[str, dict[str, Any]] = {}
    for budget in CHILD_PUCT_BUDGETS:
        run = puct_child_run(
            evaluator, child_state, budget=int(budget), root_player=root_player
        )
        runs_by_budget[str(budget)] = run
    selected_moves = [
        runs_by_budget[str(budget)]["selected_move"] for budget in CHILD_PUCT_BUDGETS
    ]
    stable = all(move == selected_moves[0] for move in selected_moves)
    root_values = [
        float(runs_by_budget[str(budget)]["root_value"])
        for budget in CHILD_PUCT_BUDGETS
        if runs_by_budget[str(budget)]["root_value"] is not None
    ]
    top1_margins = [
        float(runs_by_budget[str(budget)]["top1_margin"])
        for budget in CHILD_PUCT_BUDGETS
        if runs_by_budget[str(budget)]["top1_margin"] is not None
    ]
    return {
        "selected_move": runs_by_budget[str(max(CHILD_PUCT_BUDGETS))]["selected_move"],
        "stable": stable,
        "runs_by_budget": runs_by_budget,
        "root_value_mean": mean_std(root_values)[0],
        "root_value_std": mean_std(root_values)[1],
        "top1_margin_mean": mean_std(top1_margins)[0],
    }


def resolve_preferred_move(
    classic_child: dict[str, Any],
    puct_child: dict[str, Any],
    classic_move: int,
    puct_move: int,
) -> tuple[str, str]:
    classic_value = classic_child["root_value_mean"]
    puct_value = puct_child["root_value_mean"]
    if classic_value is None or puct_value is None:
        return "inconclusive", "missing child values"
    delta = float(classic_value) - float(puct_value)
    if delta > SMALL_DELTA:
        return "classic", f"child delta {delta:.4f} favors move {classic_move}"
    if delta < -SMALL_DELTA:
        return "puct", f"child delta {delta:.4f} favors move {puct_move}"
    return "split", f"child delta {delta:.4f} remains near zero"


def select_classic_move(state: dict[str, Any], *, budget: int, seed: int) -> int | None:
    return ClassicMCTS(
        KalahGame.from_state(state), simulations=int(budget), seed=int(seed)
    ).choose_move()


def select_puct_move(
    evaluator: ArtifactEvaluator, state: dict[str, Any], *, budget: int, seed: int
) -> int | None:
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    move = result.get("selected_move")
    return None if move is None else int(move)


def continuation_game(
    *,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    first_move: int,
    policy: str,
    seed: int,
) -> float:
    game = KalahGame.from_state(state)
    root_player = int(game.current_player)
    if not game.move(game.pit_index(int(first_move))):
        raise ValueError("illegal continuation first_move")
    ply = 1
    while not game.over() and ply < CONTINUATION_MAX_PLIES:
        position = game.to_state()
        if policy == "classic":
            move = select_classic_move(
                position, budget=CONTINUATION_CLASSIC_BUDGET, seed=int(seed + ply)
            )
        else:
            move = select_puct_move(
                evaluator,
                position,
                budget=CONTINUATION_PUCT_BUDGET,
                seed=int(seed + ply),
            )
        if move is None:
            break
        if not game.move(game.pit_index(int(move))):
            raise ValueError("illegal continuation move")
        ply += 1
    return round_float(terminal_state_raw_value(game.to_state(), root_player)) or 0.0


def continuation_summary(
    *,
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
    classic_move: int,
    puct_move: int,
    continuations: int,
) -> list[dict[str, Any]]:
    results = []
    for policy in ("classic", "puct"):
        branch_classic: list[float] = []
        branch_puct: list[float] = []
        deltas: list[float] = []
        positive = 0
        negative = 0
        neutral = 0
        for index in range(int(continuations)):
            seed = 1000 + (index * 97)
            classic_outcome = continuation_game(
                evaluator=evaluator,
                state=row["suite_state"],
                first_move=int(classic_move),
                policy=policy,
                seed=seed,
            )
            puct_outcome = continuation_game(
                evaluator=evaluator,
                state=row["suite_state"],
                first_move=int(puct_move),
                policy=policy,
                seed=seed,
            )
            delta = float(classic_outcome) - float(puct_outcome)
            branch_classic.append(float(classic_outcome))
            branch_puct.append(float(puct_outcome))
            deltas.append(delta)
            if delta > 0.0:
                positive += 1
            elif delta < 0.0:
                negative += 1
            else:
                neutral += 1
        delta_mean, ci_low, ci_high = mean_ci(deltas)
        mean_classic = mean_std(branch_classic)[0]
        mean_puct = mean_std(branch_puct)[0]
        if (delta_mean or 0.0) > SMALL_DELTA:
            decision = "classic_branch_better"
        elif (delta_mean or 0.0) < -SMALL_DELTA:
            decision = "puct_branch_better"
        else:
            decision = "neutral_or_split"
        results.append(
            {
                "row_id": row["row_id"],
                "continuation_policy": policy,
                "branch_classic_or_reference_outcome_mean": mean_classic,
                "branch_puct_outcome_mean": mean_puct,
                "outcome_delta_classic_minus_puct": delta_mean,
                "continuations": int(continuations),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "decision": decision,
                "positive_count": positive,
                "negative_count": negative,
                "neutral_count": neutral,
                "notes": f"continuation budget={CONTINUATION_CLASSIC_BUDGET if policy == 'classic' else CONTINUATION_PUCT_BUDGET}",
            }
        )
    return results


def classify_row(
    *,
    row: dict[str, Any],
    classic_root: dict[str, Any],
    puct_root: dict[str, Any],
    high_sim_puct_root: dict[str, Any],
    tablebase_move: int | None,
    child_summary: dict[str, Any] | None,
    rollout_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_move = int(row["active_reference_move"])
    classic_move = classic_root["selected_move"]
    puct_move = puct_root["selected_move"]
    high_sim_move = high_sim_puct_root["selected_move"]
    if row["excluded_unstable"]:
        return {
            "row_decision": "unstable_or_inconclusive",
            "preferred_teacher": "excluded",
            "preferred_move": reference_move,
            "recommended_use": "exclude_from_targets_and_keep_diagnostic_only",
            "evidence_summary": "row 010 remains excluded/unstable regardless of downstream comparisons",
        }
    if tablebase_move is not None:
        return {
            "row_decision": "exact_or_tablebase_resolved",
            "preferred_teacher": "exact_tablebase",
            "preferred_move": int(tablebase_move),
            "recommended_use": "use exact result where available",
            "evidence_summary": f"tablebase resolves root move to {tablebase_move}",
        }
    if classic_move is None or puct_move is None:
        return {
            "row_decision": "unstable_or_inconclusive",
            "preferred_teacher": "none",
            "preferred_move": reference_move,
            "recommended_use": "diagnostic_only",
            "evidence_summary": "missing root move from one teacher",
        }
    if classic_move == puct_move == reference_move:
        return {
            "row_decision": "classic_reference_confirmed",
            "preferred_teacher": "classic_mcts",
            "preferred_move": reference_move,
            "recommended_use": "safe_control_or_reference_row",
            "evidence_summary": "ClassicMCTS, artifact PUCT, and active reference all agree",
        }
    child_pref = "inconclusive"
    child_note = "no child adjudication needed"
    if child_summary is not None:
        child_pref = child_summary["combined_preference"]
        child_note = child_summary["combined_note"]
    rollout_pref_map = {
        row_payload["continuation_policy"]: row_payload["decision"]
        for row_payload in rollout_rows
    }
    classic_rollout = rollout_pref_map.get("classic")
    puct_rollout = rollout_pref_map.get("puct")
    if child_pref == "classic":
        if (
            classic_rollout == "puct_branch_better"
            or puct_rollout == "puct_branch_better"
        ):
            return {
                "row_decision": "teacher_policy_dependent",
                "preferred_teacher": "split",
                "preferred_move": reference_move
                if classic_move == reference_move
                else classic_move,
                "recommended_use": "diagnostic_only_until_teacher_policy_split",
                "evidence_summary": f"child favors classic/reference but rollout conflict remains; {child_note}",
            }
        return {
            "row_decision": "classic_reference_confirmed",
            "preferred_teacher": "classic_mcts",
            "preferred_move": reference_move
            if classic_move == reference_move
            else classic_move,
            "recommended_use": (
                "classic_target_or_preservation_row"
                if row["row_group"] != "root_selection_pressure"
                else "confirmed_root_selection_pressure_diagnostic"
            ),
            "evidence_summary": f"downstream child evidence favors classic/reference; {child_note}",
        }
    if child_pref == "puct":
        if (
            classic_rollout == "classic_branch_better"
            or puct_rollout == "classic_branch_better"
        ):
            return {
                "row_decision": "teacher_policy_dependent",
                "preferred_teacher": "split",
                "preferred_move": int(puct_move),
                "recommended_use": "diagnostic_only_until_teacher_policy_split",
                "evidence_summary": f"child favors PUCT but rollout conflict remains; {child_note}",
            }
        return {
            "row_decision": "puct_reference_preferred",
            "preferred_teacher": "artifact_puct",
            "preferred_move": int(puct_move),
            "recommended_use": "do_not_train_until_reference_policy_changes",
            "evidence_summary": f"downstream child evidence favors PUCT; {child_note}",
        }
    if (
        classic_rollout == "classic_branch_better"
        and puct_rollout == "puct_branch_better"
    ):
        return {
            "row_decision": "teacher_policy_dependent",
            "preferred_teacher": "split",
            "preferred_move": reference_move
            if classic_move == reference_move
            else classic_move,
            "recommended_use": "diagnostic_only_until_teacher_policy_split",
            "evidence_summary": "paired continuations flip preference by continuation policy",
        }
    if (
        classic_rollout == "classic_branch_better"
        and puct_rollout == "classic_branch_better"
    ):
        return {
            "row_decision": "classic_reference_confirmed",
            "preferred_teacher": "classic_mcts",
            "preferred_move": reference_move
            if classic_move == reference_move
            else classic_move,
            "recommended_use": (
                "classic_target_or_preservation_row"
                if row["row_group"] != "root_selection_pressure"
                else "confirmed_root_selection_pressure_diagnostic"
            ),
            "evidence_summary": "paired continuations support classic/reference under both policies",
        }
    if classic_rollout == "puct_branch_better" and puct_rollout == "puct_branch_better":
        return {
            "row_decision": "puct_reference_preferred",
            "preferred_teacher": "artifact_puct",
            "preferred_move": int(
                high_sim_move if high_sim_move is not None else puct_move
            ),
            "recommended_use": "do_not_train_until_reference_policy_changes",
            "evidence_summary": "paired continuations support the PUCT-selected branch under both policies",
        }
    return {
        "row_decision": "unstable_or_inconclusive",
        "preferred_teacher": "none",
        "preferred_move": reference_move,
        "recommended_use": "exclude_from_training_and_keep_diagnostic_only",
        "evidence_summary": f"teacher evidence remains mixed; classic={classic_move} puct={puct_move} high_sim={high_sim_move}; {child_note}",
    }


def family_decision(row_decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["row_decision"]) for row in row_decisions)
    root_rows = [
        row
        for row in row_decisions
        if row.get("row_group") == "root_selection_pressure"
        and row.get("row_decision") == "classic_reference_confirmed"
    ]
    if (
        counts.get("puct_reference_preferred", 0) > 0
        and counts.get("classic_reference_confirmed", 0) > 0
    ):
        decision = "teacher_policy_split_required"
        next_action = "split rows into Classic-teacher, PUCT-teacher, and excluded diagnostic buckets before training."
    elif counts.get("puct_reference_preferred", 0) > 0:
        decision = "puct_teacher_should_define_references"
        next_action = "build a non-mutating PUCT-teacher reference artifact and rerun corrected rebaseline."
    elif (
        counts.get("unstable_or_inconclusive", 0) >= 3
        and counts.get("classic_reference_confirmed", 0) < 10
    ):
        decision = "references_not_trainable_yet"
        next_action = (
            "exclude incumbent_proxy_disagreement and mine the next non-opening family."
        )
    elif root_rows:
        decision = "root_selection_diagnostics_before_training"
        next_action = "run PUCT root cpuct/prior/value calibration diagnostics, not value training."
    else:
        decision = "classic_teacher_should_define_references"
        next_action = "keep ClassicMCTS references and diagnose PUCT root selection/prior/value behavior before training."
    return {
        "family_decision": decision,
        "classic_reference_confirmed_count": int(
            counts.get("classic_reference_confirmed", 0)
        ),
        "puct_reference_preferred_count": int(
            counts.get("puct_reference_preferred", 0)
        ),
        "teacher_policy_dependent_count": int(
            counts.get("teacher_policy_dependent", 0)
        ),
        "exact_or_tablebase_resolved_count": int(
            counts.get("exact_or_tablebase_resolved", 0)
        ),
        "unstable_or_inconclusive_count": int(
            counts.get("unstable_or_inconclusive", 0)
        ),
        "next_action": next_action,
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Incumbent Proxy Disagreement Teacher Policy Decision Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        "- No replay artifacts were created.",
        "- Active reference fixtures stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        "",
        "## 2. Why PR #45 blocked training",
        "",
        "- PR #45 concluded that this family still mixed root-selection pressure with value-head failures, while ClassicMCTS and deterministic artifact PUCT disagreed on several active references.",
        "- The exact follow-up required before any training was to decide which teacher should define future references and training targets.",
        "- This audit answers that question without mutating fixtures or generating any training artifacts.",
        "",
        "## 3. Teacher definitions",
        "",
        "- ClassicMCTS teacher: budgets `1200, 2400, 5000, 10000` across seeds `11,23,37,42,101,202,303`.",
        "- Deterministic artifact PUCT teacher: budgets `384, 1200, 2400, 5000` with `fpu_mode=parent_q`, `reuse_subtree=true`, `normalize_values=true`, `root_policy_mode=deterministic`, `tactical_root_bias=0.1`, `c_puct=1.25`.",
        f"- High-sim artifact PUCT teacher: same deterministic settings with highest budget `{summary['high_sim_max_budget']}`.",
        "- Exact/tablebase teacher: only used where the repo tablebase can solve the root or compared child states.",
        f"- Paired continuations: `{summary['continuations']}` paired rollouts per unresolved row and per continuation policy.",
        "- The rollout count was runtime-bounded below the nominal 16-continuation default to keep this pre-training audit tractable.",
        "",
        "## 4. Row set and validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "row_group",
                "active_reference_move",
                "reference_move_legal",
                "canonical_state_match",
                "reference_unstable",
                "excluded_unstable",
                "remaining_seed_count",
                "tablebase_root_available",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["row_group"],
                    str(row["active_reference_move"]),
                    format_bool(row["reference_move_legal"]),
                    format_bool(row["canonical_state_match"]),
                    format_bool(row["reference_unstable"]),
                    format_bool(row["excluded_unstable"]),
                    str(row["remaining_seed_count"]),
                    format_bool(row["tablebase_root_available"]),
                    row["notes"],
                ]
                for row in summary["validation_table"]
            ],
        )
    )
    lines.extend(["", "## 5. Teacher agreement matrix", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "active_reference_move",
                "classic_selected_move",
                "puct_selected_move",
                "high_sim_puct_selected_move",
                "tablebase_selected_move",
                "classic_reference_agreement",
                "puct_reference_agreement",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["active_reference_move"]),
                    str(
                        row["classic_selected_move"]
                        if row["classic_selected_move"] is not None
                        else "-"
                    ),
                    str(
                        row["puct_selected_move"]
                        if row["puct_selected_move"] is not None
                        else "-"
                    ),
                    str(
                        row["high_sim_puct_selected_move"]
                        if row["high_sim_puct_selected_move"] is not None
                        else "-"
                    ),
                    str(
                        row["tablebase_selected_move"]
                        if row["tablebase_selected_move"] is not None
                        else "-"
                    ),
                    format_bool(row["classic_reference_agreement"]),
                    format_bool(row["puct_reference_agreement"]),
                    row["notes"],
                ]
                for row in summary["teacher_agreement_table"]
            ],
        )
    )
    lines.extend(["", "## 6. Child-afterstate outcome adjudication", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "classic_or_reference_move",
                "puct_selected_move",
                "classic_child_value_root",
                "puct_child_value_root",
                "classic_minus_puct_child_value",
                "tablebase_value_if_available",
                "decision",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["classic_or_reference_move"]),
                    str(row["puct_selected_move"]),
                    format_float(row["classic_child_value_root"]),
                    format_float(row["puct_child_value_root"]),
                    format_float(row["classic_minus_puct_child_value"]),
                    format_float(row["tablebase_value_if_available"]),
                    row["decision"],
                    row["notes"],
                ]
                for row in summary["child_outcome_table"]
            ],
        )
    )
    lines.extend(["", "## 7. Paired continuation rollouts", ""])
    if summary["paired_rollout_table"]:
        lines.extend(
            markdown_table(
                [
                    "row_id",
                    "continuation_policy",
                    "branch_classic_or_reference_outcome_mean",
                    "branch_puct_outcome_mean",
                    "outcome_delta_classic_minus_puct",
                    "continuations",
                    "ci_low",
                    "ci_high",
                    "decision",
                    "notes",
                ],
                [
                    [
                        row["row_id"],
                        row["continuation_policy"],
                        format_float(row["branch_classic_or_reference_outcome_mean"]),
                        format_float(row["branch_puct_outcome_mean"]),
                        format_float(row["outcome_delta_classic_minus_puct"]),
                        str(row["continuations"]),
                        format_float(row["ci_low"]),
                        format_float(row["ci_high"]),
                        row["decision"],
                        row["notes"],
                    ]
                    for row in summary["paired_rollout_table"]
                ],
            )
        )
    else:
        lines.append(
            "No paired continuations were required after child-afterstate adjudication."
        )
    lines.extend(["", "## 8. Row-level teacher decisions", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "row_decision",
                "preferred_teacher",
                "preferred_move",
                "active_reference_move",
                "recommended_use",
                "evidence_summary",
            ],
            [
                [
                    row["row_id"],
                    row["row_decision"],
                    row["preferred_teacher"],
                    str(row["preferred_move"]),
                    str(row["active_reference_move"]),
                    row["recommended_use"],
                    row["evidence_summary"],
                ]
                for row in summary["row_decision_table"]
            ],
        )
    )
    lines.extend(["", "## 9. Family-level teacher policy decision", ""])
    lines.extend(
        markdown_table(
            [
                "family_decision",
                "classic_reference_confirmed_count",
                "puct_reference_preferred_count",
                "teacher_policy_dependent_count",
                "exact_or_tablebase_resolved_count",
                "unstable_or_inconclusive_count",
                "next_action",
            ],
            [
                [
                    summary["family_decision_table"]["family_decision"],
                    str(
                        summary["family_decision_table"][
                            "classic_reference_confirmed_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "puct_reference_preferred_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "teacher_policy_dependent_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "exact_or_tablebase_resolved_count"
                        ]
                    ),
                    str(
                        summary["family_decision_table"][
                            "unstable_or_inconclusive_count"
                        ]
                    ),
                    summary["family_decision_table"]["next_action"],
                ]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 10. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['family_decision_table']['next_action']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id = canonical_reference_rows(load_json(args.reference_path))
    prior_report = parse_prior_report(args.prior_report_path)
    evaluator = ArtifactEvaluator(args.current_artifact)
    tablebase = EndgameTablebase()
    valid_rows, validation_table = validate_rows(
        suite_by_id=suite_by_id,
        reference_by_id=reference_by_id,
        prior_report=prior_report,
        tablebase=tablebase,
    )
    high_sim_budgets = (
        tuple(budget for budget in HIGH_SIM_PUCT_BUDGETS if budget < 10000)
        if args.skip_high_sim_10000
        else HIGH_SIM_PUCT_BUDGETS
    )

    teacher_agreement_table: list[dict[str, Any]] = []
    child_outcome_table: list[dict[str, Any]] = []
    paired_rollout_table: list[dict[str, Any]] = []
    row_decision_table: list[dict[str, Any]] = []
    rows_summary: list[dict[str, Any]] = []

    for row in valid_rows:
        classic_root = summarize_classic_root(row)
        puct_root = summarize_puct_root(evaluator, row, ROOT_PUCT_BUDGETS)
        high_sim_puct_root = summarize_puct_root(evaluator, row, high_sim_budgets)
        classic_move = classic_root["selected_move"]
        puct_move = puct_root["selected_move"]
        high_sim_move = high_sim_puct_root["selected_move"]
        tablebase_move = row["tablebase_root_move"]
        teacher_agreement_table.append(
            {
                "row_id": row["row_id"],
                "active_reference_move": row["active_reference_move"],
                "classic_selected_move": classic_move,
                "puct_selected_move": puct_move,
                "high_sim_puct_selected_move": high_sim_move,
                "tablebase_selected_move": tablebase_move,
                "classic_reference_agreement": classic_move
                == row["active_reference_move"],
                "puct_reference_agreement": puct_move == row["active_reference_move"],
                "notes": "; ".join(
                    [
                        classic_root["stability_note"],
                        puct_root["stability_note"],
                        (
                            f"PR45 classic={row['pr45_classic_majority_move']} puct={row['pr45_puct_selected_move']}"
                            if row["pr45_classic_majority_move"] is not None
                            or row["pr45_puct_selected_move"] is not None
                            else "no PR45 teacher metadata parsed"
                        ),
                    ]
                ),
            }
        )

        child_summary = None
        rollout_rows: list[dict[str, Any]] = []
        if (
            classic_move is not None
            and puct_move is not None
            and classic_move != puct_move
        ):
            classic_or_reference_move = (
                int(classic_move)
                if classic_root["stable"]
                else int(row["active_reference_move"])
            )
            classic_child_state = child_state_from_move(
                row["suite_state"], classic_or_reference_move
            )
            puct_child_state = child_state_from_move(row["suite_state"], int(puct_move))
            classic_child_classic = summarize_child_classic(
                classic_child_state,
                root_player=int(row["suite_state"]["current_player"]),
            )
            puct_child_classic = summarize_child_classic(
                puct_child_state, root_player=int(row["suite_state"]["current_player"])
            )
            classic_child_puct = summarize_child_puct(
                evaluator,
                classic_child_state,
                root_player=int(row["suite_state"]["current_player"]),
            )
            puct_child_puct = summarize_child_puct(
                evaluator,
                puct_child_state,
                root_player=int(row["suite_state"]["current_player"]),
            )
            classic_pref, classic_note = resolve_preferred_move(
                classic_child_classic,
                puct_child_classic,
                classic_or_reference_move,
                int(puct_move),
            )
            puct_pref, puct_note = resolve_preferred_move(
                classic_child_puct,
                puct_child_puct,
                classic_or_reference_move,
                int(puct_move),
            )
            if classic_pref == puct_pref and classic_pref in {"classic", "puct"}:
                combined_preference = classic_pref
                combined_note = (
                    f"classic-child: {classic_note}; puct-child: {puct_note}"
                )
            elif classic_pref in {"classic", "puct"} and puct_pref == "split":
                combined_preference = classic_pref
                combined_note = f"classic-child decisive; puct-child near-zero split; {classic_note}; {puct_note}"
            elif puct_pref in {"classic", "puct"} and classic_pref == "split":
                combined_preference = puct_pref
                combined_note = f"puct-child decisive; classic-child near-zero split; {classic_note}; {puct_note}"
            else:
                combined_preference = "split"
                combined_note = (
                    f"classic-child: {classic_note}; puct-child: {puct_note}"
                )
            child_summary = {
                "classic_or_reference_move": classic_or_reference_move,
                "puct_selected_move": int(puct_move),
                "classic_child_value_root": classic_child_classic["root_value_mean"],
                "puct_child_value_root": puct_child_classic["root_value_mean"],
                "classic_minus_puct_child_value": None
                if classic_child_classic["root_value_mean"] is None
                or puct_child_classic["root_value_mean"] is None
                else round_float(
                    float(classic_child_classic["root_value_mean"])
                    - float(puct_child_classic["root_value_mean"])
                ),
                "tablebase_value_if_available": None,
                "decision": combined_preference,
                "notes": combined_note,
                "combined_preference": combined_preference,
                "combined_note": combined_note,
            }
            child_outcome_table.append(
                {
                    "row_id": row["row_id"],
                    "classic_or_reference_move": classic_or_reference_move,
                    "puct_selected_move": int(puct_move),
                    "classic_child_value_root": child_summary[
                        "classic_child_value_root"
                    ],
                    "puct_child_value_root": child_summary["puct_child_value_root"],
                    "classic_minus_puct_child_value": child_summary[
                        "classic_minus_puct_child_value"
                    ],
                    "tablebase_value_if_available": child_summary[
                        "tablebase_value_if_available"
                    ],
                    "decision": child_summary["decision"],
                    "notes": child_summary["notes"],
                }
            )
            if combined_preference == "split":
                rollout_rows = continuation_summary(
                    evaluator=evaluator,
                    row=row,
                    classic_move=classic_or_reference_move,
                    puct_move=int(puct_move),
                    continuations=int(args.continuations),
                )
                paired_rollout_table.extend(rollout_rows)

        classification = classify_row(
            row=row,
            classic_root=classic_root,
            puct_root=puct_root,
            high_sim_puct_root=high_sim_puct_root,
            tablebase_move=tablebase_move,
            child_summary=child_summary,
            rollout_rows=rollout_rows,
        )
        row_decision = {
            "row_id": row["row_id"],
            "row_group": row["row_group"],
            "row_decision": classification["row_decision"],
            "preferred_teacher": classification["preferred_teacher"],
            "preferred_move": classification["preferred_move"],
            "active_reference_move": row["active_reference_move"],
            "recommended_use": classification["recommended_use"],
            "evidence_summary": classification["evidence_summary"],
        }
        row_decision_table.append(row_decision)
        rows_summary.append(
            {
                **row,
                "classic_root": classic_root,
                "puct_root": puct_root,
                "high_sim_puct_root": high_sim_puct_root,
                "child_summary": child_summary,
                "rollout_rows": rollout_rows,
                "row_decision": row_decision,
            }
        )

    family_decision_table = family_decision(row_decision_table)
    summary = {
        "schema": SCHEMA,
        "family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "prior_report_path": str(args.prior_report_path),
        },
        "continuations": int(args.continuations),
        "high_sim_max_budget": int(max(high_sim_budgets)),
        "validation_table": validation_table,
        "teacher_agreement_table": teacher_agreement_table,
        "child_outcome_table": child_outcome_table,
        "paired_rollout_table": paired_rollout_table,
        "row_decision_table": row_decision_table,
        "family_decision_table": family_decision_table,
        "rows": rows_summary,
    }
    write_json(args.summary_out, summary)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
