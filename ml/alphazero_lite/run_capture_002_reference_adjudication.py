#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.build_forensic_references import finalize_reference_row
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    row_map_from_reference,
)
from ml.alphazero_lite.self_play import build_eval_search_options


ROOT_ROW_ID = "capture_available-002"
CONTROL_ROW_IDS = (
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
DOCUMENTED_OLD_REFERENCE_MOVES = {
    "capture_available-002": 4,
    "capture_available-003": 1,
    "capture_available-006": 2,
    "capture_available-007": 1,
    "capture_available-008": 1,
}
DISPUTED_MOVE = 2
OLD_REFERENCE_MOVE_002 = 4
CLASSIC_BUDGETS = (1200, 2400, 5000, 10000)
OPTIONAL_CLASSIC_BUDGET = 30000
PUCT_BUDGETS = (384, 1200, 2400, 5000)
SEEDS = (11, 23, 37, 42, 101, 202, 303)
DEFAULT_FIXTURE_PATH = Path(
    "/home/alex/Mancala/ai/ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
)
DEFAULT_OLD_REFERENCE_ARTIFACT = Path(
    "/home/alex/Mancala/ai/ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_CURRENT_ARTIFACT = Path(
    "/home/alex/Mancala/ai/storage/ai/alphazero_lite/current"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_capture_002_reference_adjudication")
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "capture_002_reference_adjudication_summary.json"
)
DEFAULT_REPORT_PATH = Path(
    "/home/alex/Mancala/ai/docs/alphazero-lite-capture-002-reference-adjudication-results.md"
)
DEFAULT_CANDIDATE_ARTIFACT_PATH = (
    DEFAULT_OUTPUT_DIR / "incumbent_forensic_suite_v1_references_adjudicated.json"
)
DEFAULT_PATCH_ARTIFACT_PATH = (
    DEFAULT_OUTPUT_DIR / "incumbent_forensic_suite_v1_references_adjudicated_patch.json"
)
SCHEMA = "azlite_capture_002_reference_adjudication_v1"
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument(
        "--old-reference-artifact",
        type=Path,
        default=DEFAULT_OLD_REFERENCE_ARTIFACT,
    )
    parser.add_argument(
        "--current-artifact",
        type=Path,
        default=DEFAULT_CURRENT_ARTIFACT,
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--candidate-artifact-out",
        type=Path,
        default=DEFAULT_CANDIDATE_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--patch-artifact-out",
        type=Path,
        default=DEFAULT_PATCH_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--max-projected-30000-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument("--skip-30000", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_suite_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"suite at {path} must be a JSON list")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            rows[str(row["id"])] = row
    return rows


def round_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def q_from_win_rate(win_rate: float) -> float:
    return (2.0 * float(win_rate)) - 1.0


def centered_from_probability(probability: float) -> float:
    return (2.0 * float(probability)) - 1.0


def aggregate_probability(child_stats: list[dict[str, Any]]) -> float | None:
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        return None
    weighted_sum = sum(
        float(child.get("win_rate", 0.0)) * int(child.get("visits", 0))
        for child in child_stats
    )
    return weighted_sum / float(total_visits)


def selected_move_from_q_map(q_by_move: dict[int, float]) -> int | None:
    if not q_by_move:
        return None
    return max(q_by_move, key=lambda move: (float(q_by_move[move]), -int(move)))


def top1_margin(q_by_move: dict[int, float]) -> float | None:
    if len(q_by_move) < 2:
        return None
    ordered = sorted(q_by_move.values(), reverse=True)
    return float(ordered[0] - ordered[1])


def state_to_root_perspective_value(
    *, raw_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(raw_value)
        if int(state["current_player"]) == int(root_player)
        else -float(raw_value)
    )


def child_state_from_move(root_state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(move))
    if not succeeded:
        raise ValueError(f"illegal move {move}")
    return game.to_state()


def child_value_conversion_note(child_state: dict[str, Any], root_player: int) -> str:
    if int(child_state["current_player"]) == int(root_player):
        return "identity"
    return "sign_flip"


def row_ids_from_capture_family(rows: dict[str, dict[str, Any]]) -> list[str]:
    capture_ids = [row_id for row_id in rows if row_id.startswith("capture_available-")]
    return sorted(capture_ids)


def old_reference_move_for(
    row_id: str, old_reference_rows: dict[str, dict[str, Any]]
) -> int | None:
    if row_id in DOCUMENTED_OLD_REFERENCE_MOVES:
        return int(DOCUMENTED_OLD_REFERENCE_MOVES[row_id])
    row = old_reference_rows.get(row_id)
    if row is None or row.get("reference_move") is None:
        return None
    return int(row["reference_move"])


def merge_row_sources(
    *, fixture_rows: dict[str, dict[str, Any]], fallback_rows: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    merged = {row_id: copy.deepcopy(row) for row_id, row in fixture_rows.items()}
    for row_id, row in fallback_rows.items():
        merged.setdefault(row_id, copy.deepcopy(row))
    return merged


def remaining_seeds(state: dict[str, Any]) -> int:
    return int(sum(state["player_pits"]) + sum(state["opponent_pits"]))


def classic_run_for_state(
    *, state: dict[str, Any], budget: int, seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    search = ClassicMCTS(
        KalahGame.from_state(state), simulations=int(budget), seed=int(seed)
    )
    summary = search.root_summary()
    duration_ms = (time.perf_counter() - started) * 1000.0
    q_values: dict[int, float] = {}
    visits: dict[int, int] = {}
    for child in summary.get("child_stats") or []:
        move = int(child["move"])
        visits[move] = int(child.get("visits", 0))
        q_values[move] = q_from_win_rate(float(child.get("win_rate", 0.0)))
    return {
        "selected_move": summary.get("selected_move"),
        "child_stats": list(summary.get("child_stats") or []),
        "visits_by_move": visits,
        "q_by_move": q_values,
        "top1_margin": top1_margin(q_values),
        "teacher_probability": aggregate_probability(
            list(summary.get("child_stats") or [])
        ),
        "duration_ms": duration_ms,
    }


def puct_run_for_state(
    *,
    evaluator: ArtifactEvaluator,
    artifact_path: Path,
    state: dict[str, Any],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=artifact_path,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=1.25,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    q_values = {
        int(entry["move"]): float(entry.get("q_value", 0.0))
        for entry in result.get("child_stats") or []
    }
    visits = {
        int(entry["move"]): int(entry.get("visits", 0))
        for entry in result.get("child_stats") or []
    }
    return {
        "selected_move": result.get("selected_move"),
        "child_stats": list(result.get("child_stats") or []),
        "visits_by_move": visits,
        "q_by_move": q_values,
        "top1_margin": top1_margin(q_values),
        "root_value": float(result.get("value", 0.0)),
    }


def majority_summary(moves: list[int | None]) -> dict[str, Any]:
    observed = [int(move) for move in moves if move is not None]
    counts = Counter(observed)
    ordered = sorted(counts)
    if not observed:
        return {
            "observed_reference_moves": [],
            "majority_move": None,
            "majority_fraction": None,
            "stable": False,
        }
    majority_move, majority_count = max(
        counts.items(), key=lambda item: (int(item[1]), -int(item[0]))
    )
    return {
        "observed_reference_moves": ordered,
        "majority_move": int(majority_move),
        "majority_fraction": round_float(majority_count / len(observed)),
        "stable": len(ordered) == 1,
    }


def tablebase_check(
    *, tablebase: EndgameTablebase, state: dict[str, Any], root_player: int
) -> dict[str, Any]:
    seeds = remaining_seeds(state)
    available = seeds <= tablebase.MAX_SOLVED_SEEDS
    probability = None
    centered = None
    if available:
        probability = tablebase.lookup(KalahGame.from_state(state), int(root_player))
        centered = (
            None
            if probability is None
            else centered_from_probability(float(probability))
        )
    return {
        "tablebase_available": bool(available),
        "remaining_seeds": int(seeds),
        "root_perspective_probability": round_float(probability),
        "root_perspective_value": round_float(centered),
    }


def row_move_consequence_summary(
    *,
    row_id: str,
    row: dict[str, Any],
    old_reference_move: int | None,
    current_selected_move: int | None,
) -> list[dict[str, Any]]:
    state = dict(row["state"])
    root_player = int(state["current_player"])
    rows = []
    for move in row["legal_moves"]:
        consequence = move_consequence_for_state(state, int(move))
        child_state = child_state_from_move(state, int(move))
        rows.append(
            {
                "row_id": row_id,
                "move": int(move),
                "old_reference_move": old_reference_move,
                "current_known_selected_move": current_selected_move,
                "gives_extra_turn": bool(consequence["gives_extra_turn"]),
                "produces_capture": bool(consequence["produces_capture"]),
                "capture_count": int(consequence["capture_count"]),
                "immediate_store_delta": int(consequence["store_delta_immediate"]),
                "side_to_move_after": int(child_state["current_player"]),
                "game_over_after_move": bool(consequence["game_over_after_move"]),
                "remaining_seeds_after_move": int(remaining_seeds(child_state)),
                "child_tablebase_solvable": remaining_seeds(child_state)
                <= EndgameTablebase.MAX_SOLVED_SEEDS,
                "child_state": child_state,
                "perspective_conversion": child_value_conversion_note(
                    child_state, root_player
                ),
            }
        )
    return rows


def estimate_30000_feasible(
    *,
    classic_rows: list[dict[str, Any]],
    row_count: int,
    seed_count: int,
    threshold_seconds: float,
) -> tuple[bool, str]:
    reference_rows = [
        row
        for row in classic_rows
        if int(row["budget"]) == 10000 and row.get("duration_ms") is not None
    ]
    if not reference_rows:
        return False, "30000 skipped: no completed 10000-budget timing sample"
    ms_per_sim = statistics.fmean(
        float(row["duration_ms"]) / float(row["budget"]) for row in reference_rows
    )
    projected_seconds = (
        ms_per_sim
        * float(OPTIONAL_CLASSIC_BUDGET)
        * float(row_count * seed_count)
        / 1000.0
    )
    if projected_seconds > float(threshold_seconds):
        return (
            False,
            "30000 skipped: projected root ClassicMCTS runtime "
            f"{round(projected_seconds, 1)}s exceeds threshold {round(threshold_seconds, 1)}s",
        )
    return (
        True,
        f"30000 included: projected root ClassicMCTS runtime {round(projected_seconds, 1)}s",
    )


def budget_consistency(
    stability_rows: list[dict[str, Any]],
    *,
    target_move: int,
    minimum_budget: int,
) -> bool:
    relevant = [
        row for row in stability_rows if int(row["budget"]) >= int(minimum_budget)
    ]
    if not relevant:
        return False
    return all(
        row.get("stable") and row.get("majority_move") == int(target_move)
        for row in relevant
    )


def child_consistency(
    child_stability_rows: list[dict[str, Any]], *, prefer_move: int, minimum_budget: int
) -> bool:
    relevant = [
        row for row in child_stability_rows if int(row["budget"]) >= int(minimum_budget)
    ]
    if not relevant:
        return False
    expected = "move_2_gt_move_4" if int(prefer_move) == 2 else "move_4_gt_move_2"
    return all(
        row.get("ordering") == expected and row.get("stable") for row in relevant
    )


def classify_002(
    *,
    classic_stability_rows: list[dict[str, Any]],
    child_stability_rows: list[dict[str, Any]],
    puct_stability_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    classic_move2 = budget_consistency(
        classic_stability_rows, target_move=2, minimum_budget=5000
    )
    classic_move4 = budget_consistency(
        classic_stability_rows, target_move=4, minimum_budget=5000
    )
    child_move2 = child_consistency(
        child_stability_rows, prefer_move=2, minimum_budget=5000
    )
    child_move4 = child_consistency(
        child_stability_rows, prefer_move=4, minimum_budget=5000
    )
    puct_move2 = budget_consistency(
        puct_stability_rows, target_move=2, minimum_budget=1200
    )
    if classic_move2 and child_move2:
        return (
            "reference_should_flip_to_move_2",
            "update forensic reference artifacts and remove capture_available-002 from model-failure guard status, then rerun prior local diagnostics with corrected reference labels",
        )
    if classic_move4 and child_move4 and not puct_move2:
        return (
            "old_reference_confirmed_move_4",
            "audit why PR #30 child-afterstate teacher ordering reported move_2_gt_move_4, focusing on child decomposition, teacher value aggregation, or perspective conversion",
        )
    if classic_move4 and puct_move2:
        return (
            "puct_teacher_divergence",
            "compare rollout teacher versus PUCT teacher assumptions before any training",
        )
    if any(
        not row.get("stable")
        for row in classic_stability_rows
        if int(row["budget"]) >= 5000
    ):
        return (
            "reference_unstable",
            "mark 002 as unstable and exclude it from hard pass/fail gates, replacing it with a more stable row from the same failure family",
        )
    return (
        "still_inconclusive",
        "do not train on 002; build a broader stable failure-family suite and drop 002 as a single-row target",
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def format_json_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Capture 002 Reference Adjudication Results",
        "",
        "## Context",
        "",
        "- No training, no arena, no promotion, and no model-artifact changes were run.",
        f"- Summary artifact: `{summary['summary_path']}`.",
        f"- Candidate reference artifact: `{summary['candidate_reference_artifact']['path']}`.",
        f"- Patch artifact: `{summary['candidate_reference_artifact'].get('patch_path') or '-'}`.",
        "",
        "## Why this audit was needed",
        "",
        "- PR #30 showed that the current root search, child-afterstate ordering, and direct child PUCT ordering all preferred move `2` over old reference move `4` on `capture_available-002`.",
        "- Earlier policy-target work still treated `002` as a model/search failure against the older move-`4` label.",
        "- This run checks whether `002` is a true failure or a bad/unstable reference target.",
        "",
        "## Row and move consequence summary",
        "",
    ]

    lines.extend(
        markdown_table(
            [
                "row_id",
                "state_source",
                "legal_moves",
                "old_reference_move",
                "current_known_selected_move",
                "old_reference_state_matches_row",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["state_source"],
                    format_json_cell(row["legal_moves"]),
                    format_json_cell(row["old_reference_move"]),
                    format_json_cell(row["current_known_selected_move"]),
                    format_json_cell(row["old_reference_state_matches_row"]),
                    row["notes"] or "-",
                ]
                for row in summary["row_overview"]
            ],
        )
    )
    lines.append("")
    lines.extend(
        markdown_table(
            [
                "row_id",
                "move",
                "gives_extra_turn",
                "produces_capture",
                "capture_count",
                "immediate_store_delta",
                "side_to_move_after",
                "game_over_after_move",
                "remaining_seeds_after_move",
                "child_tablebase_solvable",
            ],
            [
                [
                    row["row_id"],
                    str(row["move"]),
                    format_json_cell(row["gives_extra_turn"]),
                    format_json_cell(row["produces_capture"]),
                    str(row["capture_count"]),
                    str(row["immediate_store_delta"]),
                    str(row["side_to_move_after"]),
                    format_json_cell(row["game_over_after_move"]),
                    str(row["remaining_seeds_after_move"]),
                    format_json_cell(row["child_tablebase_solvable"]),
                ]
                for row in summary["move_consequence_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "For `capture_available-002`:",
            f"- Move `2` child state: `{json.dumps(summary['capture_002_focus']['move_2_child_state'], sort_keys=True)}`",
            f"- Move `4` child state: `{json.dumps(summary['capture_002_focus']['move_4_child_state'], sort_keys=True)}`",
            f"- Move `2` perspective conversion: `{summary['capture_002_focus']['move_2_perspective_conversion']}`",
            f"- Move `4` perspective conversion: `{summary['capture_002_focus']['move_4_perspective_conversion']}`",
            "",
            "## Root ClassicMCTS multi-budget/multi-seed adjudication",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seed",
                "selected_move",
                "old_reference_move",
                "disputed_move",
                "visits_move_2",
                "visits_move_4",
                "value_move_2",
                "value_move_4",
                "value_4_minus_2",
                "top1_margin",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seed"]),
                    format_json_cell(row["selected_move"]),
                    format_json_cell(row["old_reference_move"]),
                    str(row["disputed_move"]),
                    format_json_cell(row["visits_move_2"]),
                    format_json_cell(row["visits_move_4"]),
                    format_json_cell(row["value_move_2"]),
                    format_json_cell(row["value_move_4"]),
                    format_json_cell(row["value_4_minus_2"]),
                    format_json_cell(row["top1_margin"]),
                    row["notes"] or "-",
                ]
                for row in summary["root_classic_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- 30000-budget status: `{summary['root_classic_30000_note']}`.",
            "",
            "Stability summary:",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seeds",
                "observed_reference_moves",
                "majority_move",
                "majority_fraction",
                "old_reference_move",
                "stable",
                "decision",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    format_json_cell(row["seeds"]),
                    format_json_cell(row["observed_reference_moves"]),
                    format_json_cell(row["majority_move"]),
                    format_json_cell(row["majority_fraction"]),
                    format_json_cell(row["old_reference_move"]),
                    format_json_cell(row["stable"]),
                    row["decision"],
                ]
                for row in summary["root_classic_stability"]
            ],
        )
    )
    lines.extend(["", "## Child-afterstate adjudication", ""])
    lines.extend(
        markdown_table(
            [
                "child_from_move",
                "budget",
                "seed",
                "child_value_raw",
                "child_value_root_perspective",
                "child_selected_move",
                "notes",
            ],
            [
                [
                    str(row["child_from_move"]),
                    str(row["budget"]),
                    str(row["seed"]),
                    format_json_cell(row["child_value_raw"]),
                    format_json_cell(row["child_value_root_perspective"]),
                    format_json_cell(row["child_selected_move"]),
                    row["notes"],
                ]
                for row in summary["child_classic_rows"]
            ],
        )
    )
    lines.extend(["", "Child stability:", ""])
    lines.extend(
        markdown_table(
            [
                "budget",
                "seeds",
                "observed_move2_root_values",
                "observed_move4_root_values",
                "ordering",
                "stable",
                "decomposition_agrees_with_root",
            ],
            [
                [
                    str(row["budget"]),
                    format_json_cell(row["seeds"]),
                    format_json_cell(row["move_2_root_values"]),
                    format_json_cell(row["move_4_root_values"]),
                    row["ordering"],
                    format_json_cell(row["stable"]),
                    format_json_cell(row["decomposition_agrees_with_root"]),
                ]
                for row in summary["child_classic_stability"]
            ],
        )
    )
    lines.extend(["", "## Tablebase availability", ""])
    lines.extend(
        markdown_table(
            [
                "state_label",
                "tablebase_available",
                "remaining_seeds",
                "root_perspective_value",
                "notes",
            ],
            [
                [
                    row["state_label"],
                    format_json_cell(row["tablebase_available"]),
                    str(row["remaining_seeds"]),
                    format_json_cell(row["root_perspective_value"]),
                    row["notes"],
                ]
                for row in summary["tablebase_rows"]
            ],
        )
    )
    lines.extend(["", "## PUCT/artifact teacher comparison", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seed",
                "selected_move",
                "old_reference_move",
                "disputed_move",
                "visits_move_2",
                "visits_move_4",
                "value_move_2",
                "value_move_4",
                "value_4_minus_2",
                "top1_margin",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seed"]),
                    format_json_cell(row["selected_move"]),
                    format_json_cell(row["old_reference_move"]),
                    str(row["disputed_move"]),
                    format_json_cell(row["visits_move_2"]),
                    format_json_cell(row["visits_move_4"]),
                    format_json_cell(row["value_move_2"]),
                    format_json_cell(row["value_move_4"]),
                    format_json_cell(row["value_4_minus_2"]),
                    format_json_cell(row["top1_margin"]),
                    row["notes"] or "-",
                ]
                for row in summary["root_puct_rows"]
            ],
        )
    )
    lines.extend(["", "PUCT stability:", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seeds",
                "observed_reference_moves",
                "majority_move",
                "majority_fraction",
                "old_reference_move",
                "stable",
                "decision",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    format_json_cell(row["seeds"]),
                    format_json_cell(row["observed_reference_moves"]),
                    format_json_cell(row["majority_move"]),
                    format_json_cell(row["majority_fraction"]),
                    format_json_cell(row["old_reference_move"]),
                    format_json_cell(row["stable"]),
                    row["decision"],
                ]
                for row in summary["root_puct_stability"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Candidate reference artifact",
            "",
            f"- Artifact scope: `{summary['candidate_reference_artifact']['scope']}`.",
            f"- Included row ids: `{json.dumps(summary['candidate_reference_artifact']['included_row_ids'])}`.",
            f"- Missing requested incumbent rows: `{json.dumps(summary['candidate_reference_artifact']['missing_requested_rows'])}`.",
            "",
            "## Old vs adjudicated reference comparison",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "old_reference_move",
                "adjudicated_reference_move",
                "reference_unstable",
                "observed_reference_moves",
                "decision",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    format_json_cell(row["old_reference_move"]),
                    format_json_cell(row["adjudicated_reference_move"]),
                    format_json_cell(row["reference_unstable"]),
                    format_json_cell(row["observed_reference_moves"]),
                    row["decision"],
                    row["notes"],
                ]
                for row in summary["reference_comparison_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `capture_available-002` classification: `{summary['classification_002']}`.",
            f"- Highest-budget ClassicMCTS majority move for `002`: `{summary['capture_002_highest_budget_classic_majority']}`.",
            f"- Highest-budget PUCT majority move for `002`: `{summary['capture_002_highest_budget_puct_majority']}`.",
            f"- Highest-budget child-afterstate ordering for `002`: `{summary['capture_002_highest_budget_child_ordering']}`.",
            "",
            "## Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (
        args.summary_out,
        args.report_out,
        args.candidate_artifact_out,
        args.patch_artifact_out,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    fixture_rows = load_suite_rows(args.fixture_path)
    old_reference_artifact = load_json(args.old_reference_artifact)
    old_reference_rows = row_map_from_reference(old_reference_artifact)
    old_suite_path = Path(str(old_reference_artifact["suite_path"]))
    old_suite_rows = load_suite_rows(old_suite_path)
    merged_rows = merge_row_sources(
        fixture_rows=fixture_rows, fallback_rows=old_suite_rows
    )

    audited_row_ids = [ROOT_ROW_ID, *CONTROL_ROW_IDS]
    audited_rows: dict[str, dict[str, Any]] = {}
    for row_id in audited_row_ids:
        row = merged_rows.get(row_id)
        if row is None:
            raise ValueError(f"missing audited row: {row_id}")
        audited_rows[row_id] = copy.deepcopy(row)

    evaluator = ArtifactEvaluator(args.current_artifact)
    row_overview = []
    move_consequence_rows = []
    capture_002_focus: dict[str, Any] = {}

    current_known_selected_by_row: dict[str, int | None] = {}
    for row_id, row in audited_rows.items():
        probe = puct_run_for_state(
            evaluator=evaluator,
            artifact_path=args.current_artifact,
            state=dict(row["state"]),
            budget=1200,
            seed=42,
        )
        current_known_selected_by_row[row_id] = probe.get("selected_move")

    for row_id, row in audited_rows.items():
        old_reference_row = old_reference_rows.get(row_id)
        old_reference_move = old_reference_move_for(row_id, old_reference_rows)
        old_reference_state_matches_row = None
        notes: list[str] = []
        if old_reference_row is not None:
            old_reference_state_matches_row = canonical_state_key(
                dict(old_reference_row["state"])
            ) == canonical_state_key(dict(row["state"]))
            if not old_reference_state_matches_row:
                notes.append("old_reference_state_mismatch")
        if row_id not in fixture_rows:
            notes.append("row_loaded_from_old_suite_fallback")
        row_overview.append(
            {
                "row_id": row_id,
                "state_source": "fixture"
                if row_id in fixture_rows
                else "old_reference_suite_fallback",
                "legal_moves": list(row["legal_moves"]),
                "old_reference_move": old_reference_move,
                "current_known_selected_move": current_known_selected_by_row[row_id],
                "old_reference_state_matches_row": old_reference_state_matches_row,
                "notes": ", ".join(notes),
            }
        )
        move_consequence_rows.extend(
            row_move_consequence_summary(
                row_id=row_id,
                row=row,
                old_reference_move=(
                    int(old_reference_move) if old_reference_move is not None else None
                ),
                current_selected_move=current_known_selected_by_row[row_id],
            )
        )
        if row_id == ROOT_ROW_ID:
            move_2_child = child_state_from_move(dict(row["state"]), DISPUTED_MOVE)
            move_4_child = child_state_from_move(
                dict(row["state"]), OLD_REFERENCE_MOVE_002
            )
            root_player = int(row["state"]["current_player"])
            capture_002_focus = {
                "move_2_child_state": move_2_child,
                "move_4_child_state": move_4_child,
                "move_2_perspective_conversion": child_value_conversion_note(
                    move_2_child, root_player
                ),
                "move_4_perspective_conversion": child_value_conversion_note(
                    move_4_child, root_player
                ),
            }

    root_classic_rows: list[dict[str, Any]] = []
    classic_raw_results: dict[tuple[str, int, int], dict[str, Any]] = {}
    for budget in CLASSIC_BUDGETS:
        for row_id, row in audited_rows.items():
            old_reference_row = old_reference_rows.get(row_id)
            old_reference_move = old_reference_move_for(row_id, old_reference_rows)
            for seed in SEEDS:
                result = classic_run_for_state(
                    state=dict(row["state"]), budget=budget, seed=seed
                )
                classic_raw_results[(row_id, budget, seed)] = result
                q_by_move = result["q_by_move"]
                visits = result["visits_by_move"]
                notes = []
                if old_reference_row is not None and canonical_state_key(
                    dict(old_reference_row["state"])
                ) != canonical_state_key(dict(row["state"])):
                    notes.append("old_reference_state_mismatch")
                root_classic_rows.append(
                    {
                        "row_id": row_id,
                        "budget": int(budget),
                        "seed": int(seed),
                        "selected_move": result["selected_move"],
                        "old_reference_move": old_reference_move,
                        "disputed_move": DISPUTED_MOVE,
                        "visits_move_2": visits.get(DISPUTED_MOVE),
                        "visits_move_4": visits.get(OLD_REFERENCE_MOVE_002),
                        "value_move_2": round_float(q_by_move.get(DISPUTED_MOVE)),
                        "value_move_4": round_float(
                            q_by_move.get(OLD_REFERENCE_MOVE_002)
                        ),
                        "value_4_minus_2": round_float(
                            (q_by_move.get(OLD_REFERENCE_MOVE_002) or 0.0)
                            - (q_by_move.get(DISPUTED_MOVE) or 0.0)
                        )
                        if DISPUTED_MOVE in q_by_move
                        and OLD_REFERENCE_MOVE_002 in q_by_move
                        else None,
                        "top1_margin": round_float(result["top1_margin"]),
                        "notes": ", ".join(notes),
                        "duration_ms": round_float(result["duration_ms"]),
                    }
                )

    include_30000 = False
    root_classic_30000_note = "30000 skipped by default"
    if args.skip_30000:
        root_classic_30000_note = "30000 skipped by --skip-30000"
    else:
        include_30000, root_classic_30000_note = estimate_30000_feasible(
            classic_rows=root_classic_rows,
            row_count=len(audited_rows),
            seed_count=len(SEEDS),
            threshold_seconds=float(args.max_projected_30000_seconds),
        )
    completed_classic_budgets = [int(budget) for budget in CLASSIC_BUDGETS]
    if include_30000:
        completed_classic_budgets.append(OPTIONAL_CLASSIC_BUDGET)
        for row_id, row in audited_rows.items():
            old_reference_row = old_reference_rows.get(row_id)
            old_reference_move = old_reference_move_for(row_id, old_reference_rows)
            for seed in SEEDS:
                result = classic_run_for_state(
                    state=dict(row["state"]), budget=OPTIONAL_CLASSIC_BUDGET, seed=seed
                )
                classic_raw_results[(row_id, OPTIONAL_CLASSIC_BUDGET, seed)] = result
                q_by_move = result["q_by_move"]
                visits = result["visits_by_move"]
                root_classic_rows.append(
                    {
                        "row_id": row_id,
                        "budget": int(OPTIONAL_CLASSIC_BUDGET),
                        "seed": int(seed),
                        "selected_move": result["selected_move"],
                        "old_reference_move": old_reference_move,
                        "disputed_move": DISPUTED_MOVE,
                        "visits_move_2": visits.get(DISPUTED_MOVE),
                        "visits_move_4": visits.get(OLD_REFERENCE_MOVE_002),
                        "value_move_2": round_float(q_by_move.get(DISPUTED_MOVE)),
                        "value_move_4": round_float(
                            q_by_move.get(OLD_REFERENCE_MOVE_002)
                        ),
                        "value_4_minus_2": round_float(
                            (q_by_move.get(OLD_REFERENCE_MOVE_002) or 0.0)
                            - (q_by_move.get(DISPUTED_MOVE) or 0.0)
                        )
                        if DISPUTED_MOVE in q_by_move
                        and OLD_REFERENCE_MOVE_002 in q_by_move
                        else None,
                        "top1_margin": round_float(result["top1_margin"]),
                        "notes": "-",
                        "duration_ms": round_float(result["duration_ms"]),
                    }
                )

    root_classic_stability = []
    for row_id in audited_rows:
        old_reference_move = old_reference_move_for(row_id, old_reference_rows)
        for budget in completed_classic_budgets:
            selected = [
                classic_raw_results[(row_id, budget, seed)]["selected_move"]
                for seed in SEEDS
            ]
            summary = majority_summary(selected)
            if old_reference_move is None:
                decision = "no_old_reference"
            elif summary["stable"] and summary["majority_move"] == int(
                old_reference_move
            ):
                decision = "old_target_still_valid"
            elif summary["stable"]:
                decision = "old_target_invalid"
            else:
                decision = "reference_unstable"
            root_classic_stability.append(
                {
                    "row_id": row_id,
                    "budget": int(budget),
                    "seeds": list(SEEDS),
                    "observed_reference_moves": summary["observed_reference_moves"],
                    "majority_move": summary["majority_move"],
                    "majority_fraction": summary["majority_fraction"],
                    "old_reference_move": old_reference_move,
                    "stable": summary["stable"],
                    "decision": decision,
                }
            )

    root_state_002 = dict(audited_rows[ROOT_ROW_ID]["state"])
    root_player_002 = int(root_state_002["current_player"])
    child_states_002 = {
        DISPUTED_MOVE: child_state_from_move(root_state_002, DISPUTED_MOVE),
        OLD_REFERENCE_MOVE_002: child_state_from_move(
            root_state_002, OLD_REFERENCE_MOVE_002
        ),
    }
    child_classic_rows = []
    child_raw: dict[tuple[int, int, int], dict[str, Any]] = {}
    for budget in completed_classic_budgets:
        for child_from_move, child_state in child_states_002.items():
            for seed in SEEDS:
                result = classic_run_for_state(
                    state=dict(child_state), budget=budget, seed=seed
                )
                raw_centered = centered_from_probability(
                    float(result["teacher_probability"])
                )
                root_centered = state_to_root_perspective_value(
                    raw_value=raw_centered,
                    state=child_state,
                    root_player=root_player_002,
                )
                child_raw[(child_from_move, budget, seed)] = {
                    **result,
                    "child_value_raw": raw_centered,
                    "child_value_root_perspective": root_centered,
                }
                child_classic_rows.append(
                    {
                        "child_from_move": int(child_from_move),
                        "budget": int(budget),
                        "seed": int(seed),
                        "child_value_raw": round_float(raw_centered),
                        "child_value_root_perspective": round_float(root_centered),
                        "child_selected_move": result["selected_move"],
                        "notes": "root-perspective child ClassicMCTS adjudication",
                    }
                )

    child_classic_stability = []
    for budget in completed_classic_budgets:
        move_2_root_values = [
            round_float(
                child_raw[(DISPUTED_MOVE, budget, seed)]["child_value_root_perspective"]
            )
            for seed in SEEDS
        ]
        move_4_root_values = [
            round_float(
                child_raw[(OLD_REFERENCE_MOVE_002, budget, seed)][
                    "child_value_root_perspective"
                ]
            )
            for seed in SEEDS
        ]
        orderings = []
        for seed in SEEDS:
            move_2_value = float(
                child_raw[(DISPUTED_MOVE, budget, seed)]["child_value_root_perspective"]
            )
            move_4_value = float(
                child_raw[(OLD_REFERENCE_MOVE_002, budget, seed)][
                    "child_value_root_perspective"
                ]
            )
            if move_4_value > move_2_value:
                orderings.append("move_4_gt_move_2")
            elif move_4_value < move_2_value:
                orderings.append("move_2_gt_move_4")
            else:
                orderings.append("tie")
        ordering_summary = majority_summary(
            [
                {"move_2_gt_move_4": 2, "move_4_gt_move_2": 4, "tie": 0}[value]
                for value in orderings
            ]
        )
        majority_ordering = {
            2: "move_2_gt_move_4",
            4: "move_4_gt_move_2",
            0: "tie",
        }.get(ordering_summary["majority_move"], "mixed")
        root_budget_summary = next(
            row
            for row in root_classic_stability
            if row["row_id"] == ROOT_ROW_ID and int(row["budget"]) == int(budget)
        )
        child_classic_stability.append(
            {
                "budget": int(budget),
                "seeds": list(SEEDS),
                "move_2_root_values": move_2_root_values,
                "move_4_root_values": move_4_root_values,
                "ordering": majority_ordering,
                "stable": len(set(orderings)) == 1,
                "decomposition_agrees_with_root": (
                    (
                        majority_ordering == "move_2_gt_move_4"
                        and root_budget_summary["majority_move"] == 2
                    )
                    or (
                        majority_ordering == "move_4_gt_move_2"
                        and root_budget_summary["majority_move"] == 4
                    )
                ),
            }
        )

    tablebase = EndgameTablebase()
    tablebase_rows = []
    for row_id, row in audited_rows.items():
        tb = tablebase_check(
            tablebase=tablebase,
            state=dict(row["state"]),
            root_player=int(row["state"]["current_player"]),
        )
        tablebase_rows.append(
            {
                "state_label": row_id,
                **tb,
                "notes": "root state",
            }
        )
    for move, state in child_states_002.items():
        tb = tablebase_check(
            tablebase=tablebase, state=state, root_player=root_player_002
        )
        tablebase_rows.append(
            {
                "state_label": f"capture_available-002 child_after_move_{move}",
                **tb,
                "notes": "002 child afterstate",
            }
        )

    root_puct_rows = []
    puct_raw_results: dict[tuple[str, int, int], dict[str, Any]] = {}
    for budget in PUCT_BUDGETS:
        for row_id, row in audited_rows.items():
            old_reference_move = old_reference_rows.get(row_id, {}).get(
                "reference_move"
            )
            for seed in SEEDS:
                result = puct_run_for_state(
                    evaluator=evaluator,
                    artifact_path=args.current_artifact,
                    state=dict(row["state"]),
                    budget=budget,
                    seed=seed,
                )
                puct_raw_results[(row_id, budget, seed)] = result
                q_by_move = result["q_by_move"]
                visits = result["visits_by_move"]
                root_puct_rows.append(
                    {
                        "row_id": row_id,
                        "budget": int(budget),
                        "seed": int(seed),
                        "selected_move": result["selected_move"],
                        "old_reference_move": old_reference_move,
                        "disputed_move": DISPUTED_MOVE,
                        "visits_move_2": visits.get(DISPUTED_MOVE),
                        "visits_move_4": visits.get(OLD_REFERENCE_MOVE_002),
                        "value_move_2": round_float(q_by_move.get(DISPUTED_MOVE)),
                        "value_move_4": round_float(
                            q_by_move.get(OLD_REFERENCE_MOVE_002)
                        ),
                        "value_4_minus_2": round_float(
                            (q_by_move.get(OLD_REFERENCE_MOVE_002) or 0.0)
                            - (q_by_move.get(DISPUTED_MOVE) or 0.0)
                        )
                        if DISPUTED_MOVE in q_by_move
                        and OLD_REFERENCE_MOVE_002 in q_by_move
                        else None,
                        "top1_margin": round_float(result["top1_margin"]),
                        "notes": "current artifact deterministic PUCT",
                    }
                )

    root_puct_stability = []
    for row_id in audited_rows:
        old_reference_move = old_reference_move_for(row_id, old_reference_rows)
        for budget in PUCT_BUDGETS:
            selected = [
                puct_raw_results[(row_id, budget, seed)]["selected_move"]
                for seed in SEEDS
            ]
            summary = majority_summary(selected)
            if old_reference_move is None:
                decision = "no_old_reference"
            elif summary["stable"] and summary["majority_move"] == int(
                old_reference_move
            ):
                decision = "old_target_still_valid"
            elif summary["stable"]:
                decision = "old_target_invalid"
            else:
                decision = "reference_unstable"
            root_puct_stability.append(
                {
                    "row_id": row_id,
                    "budget": int(budget),
                    "seeds": list(SEEDS),
                    "observed_reference_moves": summary["observed_reference_moves"],
                    "majority_move": summary["majority_move"],
                    "majority_fraction": summary["majority_fraction"],
                    "old_reference_move": old_reference_move,
                    "stable": summary["stable"],
                    "decision": decision,
                }
            )

    classification_002, recommended_next_action = classify_002(
        classic_stability_rows=[
            row for row in root_classic_stability if row["row_id"] == ROOT_ROW_ID
        ],
        child_stability_rows=child_classic_stability,
        puct_stability_rows=[
            row for row in root_puct_stability if row["row_id"] == ROOT_ROW_ID
        ],
    )

    highest_completed_classic_budget = max(completed_classic_budgets)
    highest_completed_puct_budget = max(PUCT_BUDGETS)
    highest_budget_classic_002 = next(
        row
        for row in root_classic_stability
        if row["row_id"] == ROOT_ROW_ID
        and int(row["budget"]) == highest_completed_classic_budget
    )
    highest_budget_puct_002 = next(
        row
        for row in root_puct_stability
        if row["row_id"] == ROOT_ROW_ID
        and int(row["budget"]) == highest_completed_puct_budget
    )
    highest_budget_child_002 = next(
        row
        for row in child_classic_stability
        if int(row["budget"]) == highest_completed_classic_budget
    )

    reference_comparison_rows = []
    changed_or_unstable_rows = []
    for row_id in audited_rows:
        old_reference_row = old_reference_rows.get(row_id)
        old_reference_move = old_reference_move_for(row_id, old_reference_rows)
        highest_budget_row = next(
            row
            for row in root_classic_stability
            if row["row_id"] == row_id
            and int(row["budget"]) == highest_completed_classic_budget
        )
        adjudicated_move = highest_budget_row["majority_move"]
        reference_unstable = not bool(highest_budget_row["stable"])
        notes = []
        if old_reference_row is not None:
            same_state = canonical_state_key(
                dict(old_reference_row["state"])
            ) == canonical_state_key(dict(audited_rows[row_id]["state"]))
            if not same_state:
                notes.append("old_reference_state_mismatch")
        if row_id == ROOT_ROW_ID:
            notes.append(classification_002)
        if old_reference_move is None:
            decision = "no_old_reference"
        elif reference_unstable:
            decision = "reference_unstable"
        elif adjudicated_move == int(old_reference_move):
            decision = "old_target_still_valid"
        else:
            decision = "old_target_invalid"
            changed_or_unstable_rows.append(row_id)
        if reference_unstable:
            changed_or_unstable_rows.append(row_id)
        reference_comparison_rows.append(
            {
                "row_id": row_id,
                "old_reference_move": old_reference_move,
                "adjudicated_reference_move": adjudicated_move,
                "reference_unstable": reference_unstable,
                "observed_reference_moves": highest_budget_row[
                    "observed_reference_moves"
                ],
                "decision": decision,
                "notes": ", ".join(notes) or "-",
            }
        )

    candidate_rows = []
    candidate_included_row_ids = []
    missing_requested_rows = []
    for row_id in audited_row_ids:
        fixture_row = fixture_rows.get(row_id)
        if fixture_row is None:
            missing_requested_rows.append(row_id)
            continue
        seed_samples = []
        for seed in SEEDS:
            result = classic_raw_results[
                (row_id, highest_completed_classic_budget, seed)
            ]
            seed_samples.append(
                {
                    "seed": int(seed),
                    "reference_move": int(result["selected_move"]),
                    "teacher_value": centered_from_probability(
                        float(result["teacher_probability"])
                    ),
                    "child_stats": list(result["child_stats"]),
                }
            )
        candidate_rows.append(
            finalize_reference_row(
                row_id=row_id,
                canonical_state=canonical_state_key(dict(fixture_row["state"])),
                state=dict(fixture_row["state"]),
                seed_samples=seed_samples,
            )
        )
        candidate_included_row_ids.append(row_id)

    candidate_artifact = {
        "schema": "azlite_forensic_references_v1",
        "suite_path": str(args.fixture_path),
        "reference": {
            "policy_simulations": int(highest_completed_classic_budget),
            "value_simulations": int(highest_completed_classic_budget),
            "sample_seeds": list(SEEDS),
        },
        "meta": {
            "generated_by": SCHEMA,
            "scope": "audited_subset",
            "missing_requested_rows": missing_requested_rows,
        },
        "rows": candidate_rows,
    }
    args.candidate_artifact_out.write_text(
        json.dumps(candidate_artifact, indent=2), encoding="utf-8"
    )

    patch_path = None
    if changed_or_unstable_rows == [ROOT_ROW_ID] or set(changed_or_unstable_rows) == {
        ROOT_ROW_ID
    }:
        patch_rows = [row for row in candidate_rows if row["id"] == ROOT_ROW_ID]
        patch_artifact = {
            "schema": "azlite_forensic_references_v1",
            "suite_path": str(args.fixture_path),
            "reference": candidate_artifact["reference"],
            "meta": {
                "generated_by": SCHEMA,
                "scope": "changed_rows_only",
                "classification_002": classification_002,
            },
            "rows": patch_rows,
        }
        args.patch_artifact_out.write_text(
            json.dumps(patch_artifact, indent=2), encoding="utf-8"
        )
        patch_path = str(args.patch_artifact_out)

    summary = {
        "schema": SCHEMA,
        "summary_path": str(args.summary_out),
        "report_path": str(args.report_out),
        "fixture_path": str(args.fixture_path),
        "old_reference_artifact": str(args.old_reference_artifact),
        "old_reference_suite_path": str(old_suite_path),
        "audited_row_ids": audited_row_ids,
        "row_overview": row_overview,
        "move_consequence_rows": move_consequence_rows,
        "capture_002_focus": capture_002_focus,
        "root_classic_rows": root_classic_rows,
        "root_classic_stability": root_classic_stability,
        "root_classic_30000_note": root_classic_30000_note,
        "child_classic_rows": child_classic_rows,
        "child_classic_stability": child_classic_stability,
        "tablebase_rows": tablebase_rows,
        "root_puct_rows": root_puct_rows,
        "root_puct_stability": root_puct_stability,
        "reference_comparison_rows": reference_comparison_rows,
        "classification_002": classification_002,
        "recommended_next_action": recommended_next_action,
        "capture_002_highest_budget_classic_majority": highest_budget_classic_002[
            "majority_move"
        ],
        "capture_002_highest_budget_puct_majority": highest_budget_puct_002[
            "majority_move"
        ],
        "capture_002_highest_budget_child_ordering": highest_budget_child_002[
            "ordering"
        ],
        "candidate_reference_artifact": {
            "path": str(args.candidate_artifact_out),
            "patch_path": patch_path,
            "scope": "audited_subset",
            "included_row_ids": candidate_included_row_ids,
            "missing_requested_rows": missing_requested_rows,
        },
    }
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
