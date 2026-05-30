#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.hard_state_teacher_labeling import run_teacher_label
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.self_play import (
    Node,
    PUCT,
    build_eval_search_options,
    terminal_value,
)

ROOT_ROW_ID = "capture_available-002"
PRESERVATION_ROW_IDS = (
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
REFERENCE_MOVE = 4
WRONG_MOVE = 2
TEACHER_BUDGETS = (384, 1200, 2400)
OPTIONAL_TEACHER_BUDGETS = (5000,)
CHILD_PUCT_BUDGETS = (64, 128, 384, 1200)
ROOT_TRACE_BUDGETS = (64, 128, 384, 1200)
COUNTERFACTUAL_BUDGETS = (384, 1200)
SEED = 17
C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
DEFAULT_CURRENT_ARTIFACT = Path(
    "/home/alex/Mancala/ai/storage/ai/alphazero_lite/current"
)
DEFAULT_GUARDED_W2_ARTIFACT = Path(
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_FIXTURE_PATH = Path(
    "/home/alex/Mancala/ai/ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_capture_002_child_afterstate_value_backup_audit")
DEFAULT_SUMMARY_PATH = (
    DEFAULT_OUTPUT_DIR / "capture_002_child_afterstate_value_backup_audit_summary.json"
)
DEFAULT_REPORT_PATH = Path(
    "/home/alex/Mancala/ai/docs/alphazero-lite-capture-002-child-afterstate-value-backup-audit-results.md"
)
SCHEMA = "azlite_capture_002_child_afterstate_value_backup_audit_v1"


@dataclass(frozen=True)
class AuditState:
    name: str
    state: dict
    root_perspective_sign: float
    notes: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--guarded-w2-artifact", type=Path, default=DEFAULT_GUARDED_W2_ARTIFACT
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpuct", type=float, default=C_PUCT)
    parser.add_argument("--skip-optional-5000", action="store_true")
    return parser.parse_args(argv)


def load_fixture_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("fixture must contain a JSON list")
    rows: dict[str, dict] = {}
    for row in payload:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            rows[str(row["id"])] = row
    return rows


def round_float(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def safe_entropy(distribution: dict[str, float]) -> float | None:
    if not distribution:
        return None
    total = float(sum(float(value) for value in distribution.values()))
    if total <= 0.0:
        return None
    probabilities = [
        float(value) / total for value in distribution.values() if value > 0.0
    ]
    if not probabilities:
        return 0.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    max_entropy = math.log(len(distribution)) if len(distribution) > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


def move_entry(selection_breakdown: dict | None, move: int) -> dict | None:
    if not isinstance(selection_breakdown, dict):
        return None
    for entry in selection_breakdown.get("moves") or []:
        if int(entry.get("move", -1)) == int(move):
            return entry
    return None


def q_by_move(child_stats: list[dict] | None, move: int) -> float | None:
    if not isinstance(child_stats, list):
        return None
    for entry in child_stats:
        if int(entry.get("move", -1)) == int(move):
            return float(entry.get("q_value", 0.0))
    return None


def visits_by_move(visits: list[float] | None, move: int) -> float | None:
    if not isinstance(visits, list) or move >= len(visits):
        return None
    return float(visits[move])


def legal_policy_mass(policy: list[float], legal_moves: list[int]) -> float:
    return float(sum(float(policy[move]) for move in legal_moves if move < len(policy)))


def policy_rank(policy: list[float], move: int, legal_moves: list[int]) -> int | None:
    legal = [candidate for candidate in legal_moves if candidate < len(policy)]
    if move not in legal:
        return None
    ranked = sorted(
        legal, key=lambda candidate: (-float(policy[candidate]), int(candidate))
    )
    for index, candidate in enumerate(ranked, start=1):
        if int(candidate) == int(move):
            return index
    return None


def selected_move_from_policy(
    policy: list[float], legal_moves: list[int]
) -> int | None:
    if not legal_moves:
        return None
    return max(legal_moves, key=lambda move: (float(policy[move]), -int(move)))


def state_to_root_perspective_value(
    *, raw_value: float, state: dict, root_player: int
) -> float:
    return (
        float(raw_value)
        if int(state["current_player"]) == int(root_player)
        else -float(raw_value)
    )


def child_state_from_move(root_state: dict, move: int) -> dict:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(move))
    if not succeeded:
        raise ValueError(f"illegal move {move} for state")
    return game.to_state()


def immediate_store_delta(root_state: dict, child_state: dict, root_player: int) -> int:
    if root_player == 0:
        return int(child_state["player_store"]) - int(root_state["player_store"])
    return int(child_state["opponent_store"]) - int(root_state["opponent_store"])


def root_player_store_name(root_player: int) -> str:
    return "player_store" if root_player == 0 else "opponent_store"


def opponent_store_name(root_player: int) -> str:
    return "opponent_store" if root_player == 0 else "player_store"


def child_state_table_row(root_state: dict, move: int, root_player: int) -> dict:
    consequence = move_consequence_for_state(root_state, move)
    child_state = child_state_from_move(root_state, move)
    same_player = int(child_state["current_player"]) == int(root_player)
    conversion = (
        "identity: child raw value already uses root-player perspective"
        if same_player
        else "sign flip: child raw value is opponent-to-move perspective, negate for root"
    )
    note = (
        "extra turn keeps perspective"
        if same_player
        else "turn handoff requires sign flip"
    )
    return {
        "move": int(move),
        "child_state": child_state,
        "legal_moves_after": list(KalahGame.from_state(child_state).possible_moves()),
        "gives_extra_turn": bool(consequence["gives_extra_turn"]),
        "side_to_move_after": int(child_state["current_player"]),
        "immediate_store_delta": int(
            immediate_store_delta(root_state, child_state, root_player)
        ),
        "capture_count": int(consequence["capture_count"]),
        "game_over_after_move": bool(consequence["game_over_after_move"]),
        "root_perspective_conversion": conversion,
        "perspective_used_for_value_evaluation": int(child_state["current_player"]),
        "backup_conversion_back_to_root": "negate on parent-child player change"
        if not same_player
        else "no negation on same-player extra turn",
        "notes": note,
    }


def build_audit_states(root_state: dict, root_player: int) -> dict[str, AuditState]:
    child_2 = child_state_from_move(root_state, WRONG_MOVE)
    child_4 = child_state_from_move(root_state, REFERENCE_MOVE)
    return {
        "root": AuditState(
            name="root",
            state=copy.deepcopy(root_state),
            root_perspective_sign=1.0,
            notes="root value already aligned with root player",
        ),
        "child_after_move_2": AuditState(
            name="child_after_move_2",
            state=child_2,
            root_perspective_sign=1.0
            if int(child_2["current_player"]) == int(root_player)
            else -1.0,
            notes="move 2 extra-turn child"
            if int(child_2["current_player"]) == int(root_player)
            else "move 2 handoff child",
        ),
        "child_after_move_4": AuditState(
            name="child_after_move_4",
            state=child_4,
            root_perspective_sign=1.0
            if int(child_4["current_player"]) == int(root_player)
            else -1.0,
            notes="move 4 extra-turn child"
            if int(child_4["current_player"]) == int(root_player)
            else "move 4 handoff child",
        ),
    }


def raw_neural_audit_for_artifact(
    *,
    artifact_label: str,
    evaluator: ArtifactEvaluator,
    audit_states: dict[str, AuditState],
) -> dict:
    rows = []
    root_row = None
    child_values: dict[str, float] = {}
    for audit_state in audit_states.values():
        game = KalahGame.from_state(audit_state.state)
        policy, raw_value = evaluator.evaluate(game)
        legal_moves = game.possible_moves()
        raw_policy = [float(value) for value in policy.tolist()]
        root_value = state_to_root_perspective_value(
            raw_value=raw_value,
            state=audit_state.state,
            root_player=int(audit_states["root"].state["current_player"]),
        )
        row = {
            "artifact": artifact_label,
            "state": audit_state.name,
            "raw_policy_distribution": raw_policy,
            "raw_value": round_float(raw_value),
            "root_perspective_value": round_float(root_value),
            "legal_policy_mass": round_float(
                legal_policy_mass(raw_policy, legal_moves)
            ),
            "policy_top_move": selected_move_from_policy(raw_policy, legal_moves),
            "policy_move_2": round_float(
                raw_policy[WRONG_MOVE] if WRONG_MOVE < len(raw_policy) else 0.0
            ),
            "policy_move_4": round_float(
                raw_policy[REFERENCE_MOVE] if REFERENCE_MOVE < len(raw_policy) else 0.0
            ),
            "policy_rank_move_2": policy_rank(raw_policy, WRONG_MOVE, legal_moves),
            "policy_rank_move_4": policy_rank(raw_policy, REFERENCE_MOVE, legal_moves),
            "notes": audit_state.notes,
        }
        if audit_state.name == "root":
            root_row = row
        else:
            child_values[audit_state.name] = root_value
        rows.append(row)

    child_diff = child_values["child_after_move_4"] - child_values["child_after_move_2"]
    for row in rows:
        row["value_child4_minus_child2"] = round_float(child_diff)
    assert root_row is not None
    return {"artifact": artifact_label, "rows": rows}


def teacher_child_audit(
    *,
    child_move: int,
    child_state: dict,
    root_player: int,
    budgets: tuple[int, ...],
    seed: int,
) -> list[dict]:
    child_rows = []
    for budget in budgets:
        label = run_teacher_label(
            child_state,
            teacher_budget=int(budget),
            teacher_mode="classic_mcts",
            seed=int(seed),
        )
        policy = [float(value) for value in label["policy"]]
        child_selected_move = selected_move_from_policy(
            policy, KalahGame.from_state(child_state).possible_moves()
        )
        child_value_raw = float(label["value"])
        child_value_root = state_to_root_perspective_value(
            raw_value=child_value_raw,
            state=child_state,
            root_player=root_player,
        )
        search = ClassicMCTS(
            KalahGame.from_state(child_state), simulations=int(budget), seed=int(seed)
        )
        summary = search.root_summary()
        child_stats = list(summary.get("child_stats") or [])
        q_summary = {
            str(int(entry["move"])): round_float((2.0 * float(entry["win_rate"])) - 1.0)
            for entry in child_stats
            if int(entry.get("visits", 0)) > 0
        }
        child_rows.append(
            {
                "child_move": int(child_move),
                "budget": int(budget),
                "child_selected_move": child_selected_move,
                "child_value_raw": round_float(child_value_raw),
                "child_value_root_perspective": round_float(child_value_root),
                "visits": {
                    str(int(entry["move"])): int(entry["visits"])
                    for entry in child_stats
                },
                "q_summary": q_summary,
                "top_policy_move": child_selected_move,
                "notes": "teacher search from child afterstate",
            }
        )
    return child_rows


def puct_child_audit(
    *,
    artifact_label: str,
    artifact_path: Path,
    child_move: int,
    child_state: dict,
    root_player: int,
    budgets: tuple[int, ...],
    seed: int,
    c_puct: float,
) -> list[dict]:
    evaluator = ArtifactEvaluator(artifact_path)
    rows = []
    for simulations in budgets:
        result = evaluate_artifact_position(
            artifact_path=artifact_path,
            evaluator=evaluator,
            state=child_state,
            simulations=int(simulations),
            seed=int(seed),
            c_puct=float(c_puct),
            search_options=dict(SEARCH_OPTIONS),
            ablation_mode="full",
        )
        visit_distribution = {
            str(move): int(visits_by_move(result.get("visits"), move) or 0)
            for move in result.get("legal_moves") or []
        }
        q_summary = {
            str(int(entry["move"])): round_float(float(entry.get("q_value", 0.0)))
            for entry in result.get("child_stats") or []
        }
        rows.append(
            {
                "artifact": artifact_label,
                "child_move": int(child_move),
                "simulations": int(simulations),
                "child_selected_move": result.get("selected_move"),
                "child_value_raw": round_float(float(result.get("value", 0.0))),
                "child_value_root_perspective": round_float(
                    state_to_root_perspective_value(
                        raw_value=float(result.get("value", 0.0)),
                        state=child_state,
                        root_player=root_player,
                    )
                ),
                "visit_distribution": visit_distribution,
                "visit_entropy": round_float(safe_entropy(visit_distribution)),
                "q_summary": q_summary,
                "notes": "direct artifact PUCT from child afterstate",
            }
        )
    return rows


class InstrumentedDiagnosticPUCT(PUCT):
    def __init__(
        self, *args, child_value_overrides: dict[int, float] | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.child_value_overrides = dict(child_value_overrides or {})
        self.root_move_by_state: dict[str, int] = {}
        self._current_backup_source = "neural"
        self.backup_events: list[dict] = []

    def register_root_children(self, root: Node) -> None:
        for move, child in root.children.items():
            self.root_move_by_state[self._state_key(child.game)] = int(move)

    def _state_key(self, game: KalahGame) -> str:
        return json.dumps(game.to_state(), sort_keys=True, separators=(",", ":"))

    def _search(self, node: Node) -> float:
        terminal = terminal_value(node.game)
        if terminal is not None:
            self._current_backup_source = "terminal"
            return terminal

        if not node.expanded:
            _, value = self._expand(
                node,
                apply_dirichlet=False,
                dirichlet_alpha=None,
                dirichlet_epsilon=0.0,
                is_root=False,
            )
            root_move = self.root_move_by_state.get(self._state_key(node.game))
            if root_move is not None and root_move in self.child_value_overrides:
                value = float(self.child_value_overrides[root_move])
                self._current_backup_source = f"override_root_child_{root_move}"
            else:
                self._current_backup_source = "neural"
            return value

        if not node.children:
            self._current_backup_source = "leaf_no_children"
            return 0.0

        child = self._select_child(node)
        value = self._search(child)
        sign_flipped = child.game.current_player != node.game.current_player
        if sign_flipped:
            value = -value
        child.visit_count += 1
        child.value_sum += value
        root_move = self.root_move_by_state.get(self._state_key(child.game))
        if root_move is not None:
            self.backup_events.append(
                {
                    "root_move": int(root_move),
                    "source": self._current_backup_source,
                    "value_after_sign_to_parent": round_float(value),
                    "sign_flipped_to_parent": bool(sign_flipped),
                }
            )
        return value


def root_trace_snapshot(
    *,
    artifact_label: str,
    root: Node,
    search: InstrumentedDiagnosticPUCT,
    simulation_index: int,
    move_2: int,
    move_4: int,
) -> dict:
    selection_breakdown = search._root_selection_breakdown(root)
    move2_entry = move_entry(selection_breakdown, move_2) or {}
    move4_entry = move_entry(selection_breakdown, move_4) or {}
    selected_move = selection_breakdown.get("selected_move")
    notes = []
    if float(move2_entry.get("selection_score", -1e9)) > float(
        move4_entry.get("selection_score", -1e9)
    ):
        notes.append("move_2_score_gt_move_4")
    if float(move2_entry.get("q_value", -1e9)) > float(
        move4_entry.get("q_value", -1e9)
    ):
        notes.append("move_2_q_gt_move_4")
    if selected_move == move_2:
        notes.append("selected_move_2")
    if selected_move == move_4:
        notes.append("selected_move_4")
    latest_source = None
    for event in reversed(search.backup_events):
        if event["root_move"] in {move_2, move_4}:
            latest_source = event["source"]
            break
    return {
        "artifact": artifact_label,
        "simulations": int(simulation_index),
        "selected_move": selected_move,
        "n_move_2": int(
            visits_by_move(
                search._build_root_visit_snapshot(
                    root, simulation_index=simulation_index
                )["visits"],
                move_2,
            )
            or 0
        ),
        "n_move_4": int(
            visits_by_move(
                search._build_root_visit_snapshot(
                    root, simulation_index=simulation_index
                )["visits"],
                move_4,
            )
            or 0
        ),
        "p_move_2": round_float(float(move2_entry.get("prior", 0.0))),
        "p_move_4": round_float(float(move4_entry.get("prior", 0.0))),
        "q_move_2": round_float(float(move2_entry.get("q_value", 0.0))),
        "q_move_4": round_float(float(move4_entry.get("q_value", 0.0))),
        "u_move_2": round_float(float(move2_entry.get("u_component", 0.0))),
        "u_move_4": round_float(float(move4_entry.get("u_component", 0.0))),
        "score_move_2": round_float(float(move2_entry.get("selection_score", 0.0))),
        "score_move_4": round_float(float(move4_entry.get("selection_score", 0.0))),
        "score_margin_2_minus_4": round_float(
            float(move2_entry.get("selection_score", 0.0))
            - float(move4_entry.get("selection_score", 0.0))
        ),
        "q_margin_4_minus_2": round_float(
            float(move4_entry.get("q_value", 0.0))
            - float(move2_entry.get("q_value", 0.0))
        ),
        "backup_value_source": latest_source,
        "notes": ", ".join(notes),
    }


def run_instrumented_root_trace(
    *,
    evaluator: ArtifactEvaluator,
    root_state: dict,
    simulations: int,
    seed: int,
    c_puct: float,
    root_prior_override=None,
    child_value_overrides: dict[int, float] | None = None,
) -> dict:
    game = KalahGame.from_state(root_state)
    search = InstrumentedDiagnosticPUCT(
        evaluator=evaluator,
        simulations=int(simulations),
        c_puct=float(c_puct),
        rng=random.Random(int(seed)),
        fpu_mode=str(SEARCH_OPTIONS["fpu_mode"]),
        reuse_subtree=bool(SEARCH_OPTIONS["reuse_subtree"]),
        normalize_values=bool(SEARCH_OPTIONS["normalize_values"]),
        root_policy_mode=str(SEARCH_OPTIONS["root_policy_mode"]),
        tactical_root_bias=float(SEARCH_OPTIONS["tactical_root_bias"]),
        ablation_mode="full",
        root_prior_override=root_prior_override,
        child_value_overrides=child_value_overrides,
    )
    root = search._root_for(game)
    search._last_visit_snapshots = []
    search._expand(
        root,
        apply_dirichlet=False,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        is_root=True,
    )
    search.register_root_children(root)
    trace_rows = []
    first_score_overtake = None
    first_visit_overtake = None
    first_q_support_move_2 = None
    first_q_support_move_4 = None
    for simulation_index in range(1, int(simulations) + 1):
        value = search._search(root)
        root.visit_count += 1
        root.value_sum += value
        snapshot = root_trace_snapshot(
            artifact_label="",
            root=root,
            search=search,
            simulation_index=simulation_index,
            move_2=WRONG_MOVE,
            move_4=REFERENCE_MOVE,
        )
        trace_rows.append(snapshot)
        if (
            first_score_overtake is None
            and (snapshot["score_margin_2_minus_4"] or -1.0) > 0.0
        ):
            first_score_overtake = simulation_index
        if first_visit_overtake is None and snapshot["n_move_2"] > snapshot["n_move_4"]:
            first_visit_overtake = simulation_index
        if first_q_support_move_2 is None and (snapshot["q_move_2"] or -1e9) > (
            snapshot["q_move_4"] or -1e9
        ):
            first_q_support_move_2 = simulation_index
        if first_q_support_move_4 is None and (snapshot["q_move_4"] or -1e9) > (
            snapshot["q_move_2"] or -1e9
        ):
            first_q_support_move_4 = simulation_index
    search._last_root = root
    final_summary = search.root_summary()
    final_breakdown = final_summary.get("selection_breakdown") or {}
    return {
        "selected_move": final_summary.get("selected_move"),
        "trace_rows": trace_rows,
        "root_summary": final_summary,
        "selection_breakdown": final_breakdown,
        "first_score_overtake_simulation": first_score_overtake,
        "first_visit_overtake_simulation": first_visit_overtake,
        "first_q_support_move_2_simulation": first_q_support_move_2,
        "first_q_support_move_4_simulation": first_q_support_move_4,
        "backup_events": list(search.backup_events),
    }


def equalize_root_2_and_4_override(
    *, game, legal_moves: list[int], priors: np.ndarray
) -> np.ndarray:
    adjusted = np.asarray(priors, dtype=np.float32).copy()
    if WRONG_MOVE in legal_moves and REFERENCE_MOVE in legal_moves:
        target = max(float(adjusted[WRONG_MOVE]), float(adjusted[REFERENCE_MOVE]))
        adjusted[WRONG_MOVE] = target
        adjusted[REFERENCE_MOVE] = target
    normalized = np.zeros_like(adjusted)
    if legal_moves:
        normalized[legal_moves] = adjusted[legal_moves]
        total = float(np.sum(normalized[legal_moves]))
        if total > 0.0:
            normalized[legal_moves] /= total
        else:
            normalized[legal_moves] = 1.0 / len(legal_moves)
    return normalized.astype(np.float32)


def teacher_child_root_values(summary: dict) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in summary:
        budget = int(row["budget"])
        del budget
        values[int(row["child_move"])] = float(row["child_value_root_perspective"])
    return values


def latest_budget_value(rows: list[dict], child_move: int) -> float:
    matching = [row for row in rows if int(row["child_move"]) == int(child_move)]
    matching.sort(key=lambda row: int(row["budget"]))
    return float(matching[-1]["child_value_root_perspective"])


def latest_child_puct_value(rows: list[dict], *, child_move: int) -> float:
    matching = [row for row in rows if int(row["child_move"]) == int(child_move)]
    matching.sort(key=lambda row: int(row["simulations"]))
    return float(matching[-1]["child_value_root_perspective"])


def annotate_trace_rows(
    *,
    trace_rows: list[dict],
    neural_child_values: dict[int, float],
    teacher_child_values: dict[int, float],
    child_puct_values: dict[int, float],
    artifact_label: str,
) -> list[dict]:
    annotated = []
    for row in trace_rows:
        enriched = dict(row)
        enriched["artifact"] = artifact_label
        enriched["neural_child_value_move_2"] = round_float(
            neural_child_values[WRONG_MOVE]
        )
        enriched["neural_child_value_move_4"] = round_float(
            neural_child_values[REFERENCE_MOVE]
        )
        enriched["teacher_child_value_move_2"] = round_float(
            teacher_child_values[WRONG_MOVE]
        )
        enriched["teacher_child_value_move_4"] = round_float(
            teacher_child_values[REFERENCE_MOVE]
        )
        enriched["child_puct_value_move_2"] = round_float(child_puct_values[WRONG_MOVE])
        enriched["child_puct_value_move_4"] = round_float(
            child_puct_values[REFERENCE_MOVE]
        )
        annotated.append(enriched)
    return annotated


def counterfactual_rows_for_artifact(
    *,
    artifact_label: str,
    evaluator: ArtifactEvaluator,
    root_state: dict,
    teacher_child_values: dict[int, float],
    neural_child_values: dict[int, float],
    seed: int,
    c_puct: float,
) -> list[dict]:
    interventions = {
        "original": {"root_prior_override": None, "child_value_overrides": None},
        "root_prior_equalized": {
            "root_prior_override": equalize_root_2_and_4_override,
            "child_value_overrides": None,
        },
        "child_value_override_teacher": {
            "root_prior_override": None,
            "child_value_overrides": {
                WRONG_MOVE: float(teacher_child_values[WRONG_MOVE]),
                REFERENCE_MOVE: float(teacher_child_values[REFERENCE_MOVE]),
            },
        },
        "child_value_override_neural_swapped": {
            "root_prior_override": None,
            "child_value_overrides": {
                WRONG_MOVE: float(neural_child_values[REFERENCE_MOVE]),
                REFERENCE_MOVE: float(neural_child_values[WRONG_MOVE]),
            },
        },
        "root_q_init_teacher": {
            "root_prior_override": None,
            "child_value_overrides": {
                WRONG_MOVE: float(teacher_child_values[WRONG_MOVE]),
                REFERENCE_MOVE: float(teacher_child_values[REFERENCE_MOVE]),
            },
        },
    }
    rows = []
    for intervention_name, intervention in interventions.items():
        for simulations in COUNTERFACTUAL_BUDGETS:
            trace = run_instrumented_root_trace(
                evaluator=evaluator,
                root_state=root_state,
                simulations=simulations,
                seed=seed,
                c_puct=c_puct,
                root_prior_override=intervention["root_prior_override"],
                child_value_overrides=intervention["child_value_overrides"],
            )
            breakdown = trace["selection_breakdown"]
            move2 = move_entry(breakdown, WRONG_MOVE) or {}
            move4 = move_entry(breakdown, REFERENCE_MOVE) or {}
            selected_move = trace["selected_move"]
            if intervention_name == "root_q_init_teacher":
                notes = "implemented as root-child first-expansion override; clean root-Q preseed hook not present"
            else:
                notes = "diagnostic-only intervention"
            rows.append(
                {
                    "artifact": artifact_label,
                    "intervention": intervention_name,
                    "simulations": int(simulations),
                    "selected_move": selected_move,
                    "selected_is_reference": selected_move == REFERENCE_MOVE,
                    "n_move_2": int(
                        visits_by_move(
                            trace["root_summary"]
                            .get("visit_snapshots", [{}])[-1]
                            .get("visits", []),
                            WRONG_MOVE,
                        )
                        or 0
                    )
                    if trace["root_summary"].get("visit_snapshots")
                    else int((move2.get("visit_count") or 0)),
                    "n_move_4": int(
                        visits_by_move(
                            trace["root_summary"]
                            .get("visit_snapshots", [{}])[-1]
                            .get("visits", []),
                            REFERENCE_MOVE,
                        )
                        or 0
                    )
                    if trace["root_summary"].get("visit_snapshots")
                    else int((move4.get("visit_count") or 0)),
                    "q_move_2": round_float(float(move2.get("q_value", 0.0))),
                    "q_move_4": round_float(float(move4.get("q_value", 0.0))),
                    "score_margin_2_minus_4": round_float(
                        float(move2.get("selection_score", 0.0))
                        - float(move4.get("selection_score", 0.0))
                    ),
                    "decision": "selected_reference"
                    if selected_move == REFERENCE_MOVE
                    else "selected_non_reference",
                    "notes": notes,
                }
            )
    return rows


def child_value_ordering(rows: list[dict], *, value_field: str) -> str | None:
    by_move = {int(row["child_move"]): float(row[value_field]) for row in rows}
    if REFERENCE_MOVE not in by_move or WRONG_MOVE not in by_move:
        return None
    if by_move[REFERENCE_MOVE] > by_move[WRONG_MOVE]:
        return "move_4_gt_move_2"
    if by_move[REFERENCE_MOVE] < by_move[WRONG_MOVE]:
        return "move_2_gt_move_4"
    return "tie"


def classify_audit(
    *,
    teacher_ordering: str | None,
    neural_ordering: str | None,
    child_puct_ordering: str | None,
    root_selected_move_current: int | None,
    counterfactual_rows_current: list[dict],
) -> tuple[str, str]:
    teacher_override_flips = any(
        row["intervention"] == "child_value_override_teacher"
        and row["selected_is_reference"]
        for row in counterfactual_rows_current
    )
    prior_equalization_flips = any(
        row["intervention"] == "root_prior_equalized" and row["selected_is_reference"]
        for row in counterfactual_rows_current
    )

    if teacher_ordering != "move_4_gt_move_2":
        return (
            "teacher_decomposition_disagreement",
            "rebuild 002 reference with deeper exact/teacher search and do not treat move 4 as settled",
        )
    if teacher_override_flips:
        return (
            "value_backup_causal",
            "value backup / value target calibration lane",
        )
    if prior_equalization_flips and not teacher_override_flips:
        return (
            "prior_pressure_causal",
            "policy-prior calibration or PUCT cpuct/root exploration calibration, not value training",
        )
    if neural_ordering == "move_2_gt_move_4":
        return (
            "value_head_child_afterstate_miscalibration",
            "value-target calibration experiment on child-afterstate positions, not replay weighting",
        )
    if (
        teacher_ordering == "move_4_gt_move_2"
        and neural_ordering == "move_4_gt_move_2"
        and child_puct_ordering == "move_4_gt_move_2"
        and root_selected_move_current == WRONG_MOVE
    ):
        return (
            "root_selection_pressure_bug_or_prior_lock",
            "root PUCT selection/backup implementation audit, especially U term, visit initialization, and extra-turn side handling",
        )
    if (
        teacher_ordering == "move_4_gt_move_2"
        and child_puct_ordering == "move_2_gt_move_4"
    ):
        return (
            "puct_child_search_value_mismatch",
            "audit child-node policy/value expansion and backup perspective",
        )
    return (
        "trace_inconclusive",
        "instrument full node-level backup/perspective logs",
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def format_json_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def build_report(summary: dict) -> str:
    lines = [
        "# AlphaZero-lite Capture 002 Child-Afterstate Value/Backup Audit Results",
        "",
        "## Context",
        "",
        "- No production training, no arena, and no promotion were run.",
        "- Focus row: `capture_available-002`.",
        "- Preservation controls loaded: `capture_available-003`, `006`, `007`, `008`.",
        f"- Summary artifact: `{summary['summary_path']}`",
        "",
        "## Root and child state extraction",
        "",
    ]
    child_rows = summary["root_and_child_state_extraction"]["child_state_rows"]
    lines.extend(
        markdown_table(
            [
                "move",
                "gives_extra_turn",
                "side_to_move_after",
                "immediate_store_delta",
                "capture_count",
                "game_over_after_move",
                "root_perspective_conversion",
                "notes",
            ],
            [
                [
                    str(row["move"]),
                    str(row["gives_extra_turn"]).lower(),
                    str(row["side_to_move_after"]),
                    str(row["immediate_store_delta"]),
                    str(row["capture_count"]),
                    str(row["game_over_after_move"]).lower(),
                    row["root_perspective_conversion"],
                    row["notes"],
                ]
                for row in child_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Perspective and backup convention check",
            "",
            f"- Root player is `{summary['root_and_child_state_extraction']['root_player']}`.",
            f"- Move `2` child side to move: `{summary['root_and_child_state_extraction']['move_2_side_to_move_after']}`.",
            f"- Move `4` child side to move: `{summary['root_and_child_state_extraction']['move_4_side_to_move_after']}`.",
            f"- Neural value is evaluated from `state.current_player` perspective and converted back to root perspective with sign `{summary['perspective_and_backup_check']['conversion_rule']}`.",
            f"- PUCT backup convention check: `{summary['perspective_and_backup_check']['puct_backup_rule']}`.",
            f"- Extra-turn audit finding: `{summary['perspective_and_backup_check']['extra_turn_note']}`.",
            "",
            "## Raw neural value audit",
            "",
        ]
    )
    raw_rows = []
    for artifact in summary["raw_neural_value_audit"]:
        for row in artifact["rows"]:
            raw_rows.append(
                [
                    artifact["artifact"],
                    row["state"],
                    format_json_cell(row["raw_value"]),
                    format_json_cell(row["root_perspective_value"]),
                    format_json_cell(row["policy_top_move"]),
                    format_json_cell(row["policy_move_2"]),
                    format_json_cell(row["policy_move_4"]),
                    format_json_cell(row["value_child4_minus_child2"]),
                    row["notes"],
                ]
            )
    lines.extend(
        markdown_table(
            [
                "artifact",
                "state",
                "raw_value",
                "root_perspective_value",
                "policy_top_move",
                "policy_move_2",
                "policy_move_4",
                "value_child4_minus_child2",
                "notes",
            ],
            raw_rows,
        )
    )
    lines.extend(["", "## Child-afterstate teacher audit", ""])
    teacher_rows = []
    for row in summary["teacher_child_audit"]:
        teacher_rows.append(
            [
                str(row["child_move"]),
                str(row["budget"]),
                format_json_cell(row["child_selected_move"]),
                format_json_cell(row["child_value_raw"]),
                format_json_cell(row["child_value_root_perspective"]),
                format_json_cell(row["visits"]),
                format_json_cell(row["q_summary"]),
                format_json_cell(row["child4_minus_child2_root_value"]),
                row["notes"],
            ]
        )
    lines.extend(
        markdown_table(
            [
                "child_move",
                "budget",
                "child_selected_move",
                "child_value_raw",
                "child_value_root_perspective",
                "visits",
                "q_summary",
                "child4_minus_child2_root_value",
                "notes",
            ],
            teacher_rows,
        )
    )
    lines.extend(["", "## Child-afterstate PUCT audit", ""])
    child_puct_rows = []
    for row in summary["child_puct_audit"]:
        child_puct_rows.append(
            [
                row["artifact"],
                str(row["child_move"]),
                str(row["simulations"]),
                format_json_cell(row["child_selected_move"]),
                format_json_cell(row["child_value_raw"]),
                format_json_cell(row["child_value_root_perspective"]),
                format_json_cell(row["visit_entropy"]),
                format_json_cell(row["q_summary"]),
                format_json_cell(row["child4_minus_child2_root_value"]),
                row["notes"],
            ]
        )
    lines.extend(
        markdown_table(
            [
                "artifact",
                "child_move",
                "simulations",
                "child_selected_move",
                "child_value_raw",
                "child_value_root_perspective",
                "visit_entropy",
                "q_summary",
                "child4_minus_child2_root_value",
                "notes",
            ],
            child_puct_rows,
        )
    )
    lines.extend(["", "## Root selection-score trace with child-value annotations", ""])
    trace_rows = []
    for row in summary["root_trace_with_child_annotations"]:
        trace_rows.append(
            [
                row["artifact"],
                str(row["simulations"]),
                format_json_cell(row["selected_move"]),
                str(row["n_move_2"]),
                str(row["n_move_4"]),
                format_json_cell(row["p_move_2"]),
                format_json_cell(row["p_move_4"]),
                format_json_cell(row["q_move_2"]),
                format_json_cell(row["q_move_4"]),
                format_json_cell(row["u_move_2"]),
                format_json_cell(row["u_move_4"]),
                format_json_cell(row["score_move_2"]),
                format_json_cell(row["score_move_4"]),
                format_json_cell(row["score_margin_2_minus_4"]),
                format_json_cell(row["q_margin_4_minus_2"]),
                row["notes"] or "-",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "artifact",
                "simulations",
                "selected_move",
                "n_move_2",
                "n_move_4",
                "p_move_2",
                "p_move_4",
                "q_move_2",
                "q_move_4",
                "u_move_2",
                "u_move_4",
                "score_move_2",
                "score_move_4",
                "score_margin_2_minus_4",
                "q_margin_4_minus_2",
                "notes",
            ],
            trace_rows,
        )
    )
    lines.extend(["", "## Counterfactual root interventions", ""])
    counterfactual_rows = []
    for row in summary["counterfactual_root_interventions"]:
        counterfactual_rows.append(
            [
                row["artifact"],
                row["intervention"],
                str(row["simulations"]),
                format_json_cell(row["selected_move"]),
                str(row["selected_is_reference"]).lower(),
                str(row["n_move_2"]),
                str(row["n_move_4"]),
                format_json_cell(row["q_move_2"]),
                format_json_cell(row["q_move_4"]),
                format_json_cell(row["score_margin_2_minus_4"]),
                row["decision"],
                row["notes"],
            ]
        )
    lines.extend(
        markdown_table(
            [
                "artifact",
                "intervention",
                "simulations",
                "selected_move",
                "selected_is_reference",
                "n_move_2",
                "n_move_4",
                "q_move_2",
                "q_move_4",
                "score_margin_2_minus_4",
                "decision",
                "notes",
            ],
            counterfactual_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Final classification: `{summary['classification']}`.",
            f"- Current artifact root selected move at the largest traced budget: `{summary['current_artifact_root_selected_move']}`.",
            f"- Teacher child ordering: `{summary['teacher_child_ordering']}`.",
            f"- Current neural child ordering: `{summary['current_neural_child_ordering']}`.",
            f"- Current child PUCT ordering: `{summary['current_child_puct_ordering']}`.",
            "",
            "## Recommended next action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_out = args.summary_out
    report_out = args.report_out
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    fixture_rows = load_fixture_rows(args.fixture_path)
    root_row = fixture_rows[ROOT_ROW_ID]
    preservation_rows = {
        row_id: fixture_rows[row_id]
        for row_id in PRESERVATION_ROW_IDS
        if row_id in fixture_rows
    }
    root_state = copy.deepcopy(root_row["state"])
    root_player = int(root_state["current_player"])
    root_legal_moves = list(root_row["legal_moves"])
    child_rows = [
        child_state_table_row(root_state, WRONG_MOVE, root_player),
        child_state_table_row(root_state, REFERENCE_MOVE, root_player),
    ]
    audit_states = build_audit_states(root_state, root_player)

    artifact_specs = [{"label": "current", "path": args.current_artifact}]
    if args.guarded_w2_artifact.exists():
        artifact_specs.append({"label": "guarded-w2", "path": args.guarded_w2_artifact})

    evaluators = {
        spec["label"]: ArtifactEvaluator(spec["path"]) for spec in artifact_specs
    }

    raw_neural_value_audit = [
        raw_neural_audit_for_artifact(
            artifact_label=spec["label"],
            evaluator=evaluators[spec["label"]],
            audit_states=audit_states,
        )
        for spec in artifact_specs
    ]

    teacher_budgets = (
        TEACHER_BUDGETS
        if args.skip_optional_5000
        else (TEACHER_BUDGETS + OPTIONAL_TEACHER_BUDGETS)
    )
    teacher_child_audit_rows = []
    for child_move, child_name in (
        (WRONG_MOVE, "child_after_move_2"),
        (REFERENCE_MOVE, "child_after_move_4"),
    ):
        teacher_child_audit_rows.extend(
            teacher_child_audit(
                child_move=child_move,
                child_state=audit_states[child_name].state,
                root_player=root_player,
                budgets=teacher_budgets,
                seed=args.seed,
            )
        )
    teacher_by_budget = {int(row["budget"]): {} for row in teacher_child_audit_rows}
    for row in teacher_child_audit_rows:
        teacher_by_budget[int(row["budget"])][int(row["child_move"])] = float(
            row["child_value_root_perspective"]
        )
    for row in teacher_child_audit_rows:
        pair = teacher_by_budget[int(row["budget"])]
        if REFERENCE_MOVE in pair and WRONG_MOVE in pair:
            row["child4_minus_child2_root_value"] = round_float(
                pair[REFERENCE_MOVE] - pair[WRONG_MOVE]
            )
        else:
            row["child4_minus_child2_root_value"] = None

    child_puct_rows = []
    for spec in artifact_specs:
        for child_move, child_name in (
            (WRONG_MOVE, "child_after_move_2"),
            (REFERENCE_MOVE, "child_after_move_4"),
        ):
            child_puct_rows.extend(
                puct_child_audit(
                    artifact_label=spec["label"],
                    artifact_path=spec["path"],
                    child_move=child_move,
                    child_state=audit_states[child_name].state,
                    root_player=root_player,
                    budgets=CHILD_PUCT_BUDGETS,
                    seed=args.seed,
                    c_puct=args.cpuct,
                )
            )
    child_puct_by_artifact_and_budget: dict[tuple[str, int], dict[int, float]] = {}
    for row in child_puct_rows:
        key = (str(row["artifact"]), int(row["simulations"]))
        child_puct_by_artifact_and_budget.setdefault(key, {})[
            int(row["child_move"])
        ] = float(row["child_value_root_perspective"])
    for row in child_puct_rows:
        pair = child_puct_by_artifact_and_budget[
            (str(row["artifact"]), int(row["simulations"]))
        ]
        row["child4_minus_child2_root_value"] = (
            round_float(pair[REFERENCE_MOVE] - pair[WRONG_MOVE])
            if REFERENCE_MOVE in pair and WRONG_MOVE in pair
            else None
        )

    root_trace_rows = []
    root_trace_summaries = []
    counterfactual_rows = []
    current_root_selected_move = None
    current_neural_child_values: dict[int, float] = {}
    current_child_puct_values: dict[int, float] = {}
    for raw_artifact in raw_neural_value_audit:
        if raw_artifact["artifact"] == "current":
            for row in raw_artifact["rows"]:
                if row["state"] == "child_after_move_2":
                    current_neural_child_values[WRONG_MOVE] = float(
                        row["root_perspective_value"]
                    )
                if row["state"] == "child_after_move_4":
                    current_neural_child_values[REFERENCE_MOVE] = float(
                        row["root_perspective_value"]
                    )

    teacher_latest_values = {
        WRONG_MOVE: latest_budget_value(teacher_child_audit_rows, WRONG_MOVE),
        REFERENCE_MOVE: latest_budget_value(teacher_child_audit_rows, REFERENCE_MOVE),
    }

    for spec in artifact_specs:
        evaluator = evaluators[spec["label"]]
        artifact_child_puct_values = {
            WRONG_MOVE: latest_child_puct_value(
                [row for row in child_puct_rows if row["artifact"] == spec["label"]],
                child_move=WRONG_MOVE,
            ),
            REFERENCE_MOVE: latest_child_puct_value(
                [row for row in child_puct_rows if row["artifact"] == spec["label"]],
                child_move=REFERENCE_MOVE,
            ),
        }
        if spec["label"] == "current":
            current_child_puct_values = dict(artifact_child_puct_values)
        for simulations in ROOT_TRACE_BUDGETS:
            trace = run_instrumented_root_trace(
                evaluator=evaluator,
                root_state=root_state,
                simulations=simulations,
                seed=args.seed,
                c_puct=args.cpuct,
            )
            if spec["label"] == "current" and simulations == max(ROOT_TRACE_BUDGETS):
                current_root_selected_move = trace["selected_move"]
            root_trace_summaries.append(
                {
                    "artifact": spec["label"],
                    "simulations": int(simulations),
                    "selected_move": trace["selected_move"],
                    "first_score_overtake_simulation": trace[
                        "first_score_overtake_simulation"
                    ],
                    "first_visit_overtake_simulation": trace[
                        "first_visit_overtake_simulation"
                    ],
                    "first_q_support_move_2_simulation": trace[
                        "first_q_support_move_2_simulation"
                    ],
                    "first_q_support_move_4_simulation": trace[
                        "first_q_support_move_4_simulation"
                    ],
                }
            )
            root_trace_rows.extend(
                annotate_trace_rows(
                    trace_rows=trace["trace_rows"],
                    neural_child_values={
                        WRONG_MOVE: next(
                            float(row["root_perspective_value"])
                            for row in next(
                                a
                                for a in raw_neural_value_audit
                                if a["artifact"] == spec["label"]
                            )["rows"]
                            if row["state"] == "child_after_move_2"
                        ),
                        REFERENCE_MOVE: next(
                            float(row["root_perspective_value"])
                            for row in next(
                                a
                                for a in raw_neural_value_audit
                                if a["artifact"] == spec["label"]
                            )["rows"]
                            if row["state"] == "child_after_move_4"
                        ),
                    },
                    teacher_child_values=teacher_latest_values,
                    child_puct_values=artifact_child_puct_values,
                    artifact_label=spec["label"],
                )
            )
        counterfactual_rows.extend(
            counterfactual_rows_for_artifact(
                artifact_label=spec["label"],
                evaluator=evaluator,
                root_state=root_state,
                teacher_child_values=teacher_latest_values,
                neural_child_values={
                    WRONG_MOVE: next(
                        float(row["root_perspective_value"])
                        for row in next(
                            a
                            for a in raw_neural_value_audit
                            if a["artifact"] == spec["label"]
                        )["rows"]
                        if row["state"] == "child_after_move_2"
                    ),
                    REFERENCE_MOVE: next(
                        float(row["root_perspective_value"])
                        for row in next(
                            a
                            for a in raw_neural_value_audit
                            if a["artifact"] == spec["label"]
                        )["rows"]
                        if row["state"] == "child_after_move_4"
                    ),
                },
                seed=args.seed,
                c_puct=args.cpuct,
            )
        )

    current_neural_ordering = (
        "move_4_gt_move_2"
        if current_neural_child_values[REFERENCE_MOVE]
        > current_neural_child_values[WRONG_MOVE]
        else "move_2_gt_move_4"
        if current_neural_child_values[REFERENCE_MOVE]
        < current_neural_child_values[WRONG_MOVE]
        else "tie"
    )
    teacher_child_ordering = (
        "move_4_gt_move_2"
        if teacher_latest_values[REFERENCE_MOVE] > teacher_latest_values[WRONG_MOVE]
        else "move_2_gt_move_4"
        if teacher_latest_values[REFERENCE_MOVE] < teacher_latest_values[WRONG_MOVE]
        else "tie"
    )
    current_child_puct_ordering = (
        "move_4_gt_move_2"
        if current_child_puct_values[REFERENCE_MOVE]
        > current_child_puct_values[WRONG_MOVE]
        else "move_2_gt_move_4"
        if current_child_puct_values[REFERENCE_MOVE]
        < current_child_puct_values[WRONG_MOVE]
        else "tie"
    )

    classification, recommended_next_action = classify_audit(
        teacher_ordering=teacher_child_ordering,
        neural_ordering=current_neural_ordering,
        child_puct_ordering=current_child_puct_ordering,
        root_selected_move_current=current_root_selected_move,
        counterfactual_rows_current=[
            row for row in counterfactual_rows if row["artifact"] == "current"
        ],
    )

    summary = {
        "schema": SCHEMA,
        "row_id": ROOT_ROW_ID,
        "summary_path": str(summary_out),
        "report_path": str(report_out),
        "guardrails": {
            "trained_production_model": False,
            "ran_arena": False,
            "promoted_model": False,
        },
        "artifacts": [
            {"label": spec["label"], "path": str(spec["path"])}
            for spec in artifact_specs
        ],
        "root_and_child_state_extraction": {
            "root_player": root_player,
            "legal_moves_at_root": root_legal_moves,
            "move_2_side_to_move_after": child_rows[0]["side_to_move_after"],
            "move_4_side_to_move_after": child_rows[1]["side_to_move_after"],
            "child_state_rows": child_rows,
            "preservation_controls_loaded": sorted(preservation_rows.keys()),
        },
        "perspective_and_backup_check": {
            "conversion_rule": "+1 when child current_player == root_player else -1",
            "puct_backup_rule": "_search negates returned child value exactly when child.game.current_player != parent.game.current_player",
            "extra_turn_note": "move 2 keeps the same player to move in this position; move 4 hands off and therefore must flip sign back to root perspective",
        },
        "raw_neural_value_audit": raw_neural_value_audit,
        "teacher_child_audit": teacher_child_audit_rows,
        "child_puct_audit": child_puct_rows,
        "root_trace_summaries": root_trace_summaries,
        "root_trace_with_child_annotations": root_trace_rows,
        "counterfactual_root_interventions": counterfactual_rows,
        "teacher_child_ordering": teacher_child_ordering,
        "current_neural_child_ordering": current_neural_ordering,
        "current_child_puct_ordering": current_child_puct_ordering,
        "current_artifact_root_selected_move": current_root_selected_move,
        "classification": classification,
        "recommended_next_action": recommended_next_action,
    }

    summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_out),
                "report_path": str(report_out),
                "classification": classification,
                "recommended_next_action": recommended_next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
