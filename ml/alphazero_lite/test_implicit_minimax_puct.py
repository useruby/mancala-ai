from __future__ import annotations

from ml.alphazero_lite.gumbel_root_search import run_puct_root_search
from ml.alphazero_lite.implicit_minimax_puct import (
    ImplicitMinimaxPUCT,
    ImplicitNode,
    run_implicit_minimax_puct,
)
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.self_play import HeuristicEvaluator, standard_start_state


def game(player=0):
    return KalahGame.from_state(standard_start_state() | {"current_player": player})


def test_zero_lambda_is_byte_equivalent_to_exact_budget_puct():
    baseline = run_puct_root_search(
        game(), HeuristicEvaluator(), seed=274101, budget=48
    )
    candidate = run_implicit_minimax_puct(
        game(), HeuristicEvaluator(), seed=274101, budget=48, lambda_=0
    )
    assert candidate.visits == baseline.visits
    assert candidate.q_values == baseline.q_values
    assert candidate.selected_move == baseline.selected_move
    assert candidate.evaluator_calls == baseline.evaluator_calls == 48


def test_q_backup_is_ordinary_average_and_m_is_maximum_parent_value():
    search = ImplicitMinimaxPUCT(HeuristicEvaluator(), seed=1, budget=2)
    parent = ImplicitNode(game())
    left, right = (
        ImplicitNode(game(), visit_count=2, value_sum=0.4, m_value=0.2),
        ImplicitNode(game(), visit_count=1, value_sum=-0.8, m_value=0.8),
    )
    parent.children = {0: left, 1: right}
    search._recompute_m(parent)
    assert left.q_value == 0.2 and right.q_value == -0.8
    assert parent.m_value == 0.8


def test_player_change_inverts_m_but_extra_turn_does_not():
    search = ImplicitMinimaxPUCT(HeuristicEvaluator(), seed=1, budget=2)
    parent = ImplicitNode(game(0))
    same = ImplicitNode(game(0), m_value=0.5)
    other = ImplicitNode(game(1), m_value=0.5)
    assert search._transformed(parent, same, same.m_value) == 0.5
    assert search._transformed(parent, other, other.m_value) == -0.5


def test_terminal_m_is_exact_and_terminal_move_is_not_extra_turn():
    terminal = KalahGame([0] * 12, [25, 23], 0, winner=0, _over=True)
    search = ImplicitMinimaxPUCT(HeuristicEvaluator(), seed=1, budget=2)
    assert search._terminal(terminal) == 1.0
    # Rule metadata marks a terminal result separately from a genuine extra turn.
    state = {
        "player_pits": [1, 0, 0, 0, 0, 0],
        "opponent_pits": [0] * 6,
        "player_store": 23,
        "opponent_store": 24,
        "current_player": 0,
    }
    assert move_consequence_for_state(state, 0)["game_over_after_move"]
    assert not move_consequence_for_state(state, 0)["gives_extra_turn"]


def test_determinism_legality_and_heuristic_accounting_for_both_players():
    for player in (0, 1):
        first = run_implicit_minimax_puct(
            game(player), HeuristicEvaluator(), seed=274102, budget=32
        )
        second = run_implicit_minimax_puct(
            game(player), HeuristicEvaluator(), seed=274102, budget=32
        )
        assert first.__dict__ | {"runtime_seconds": 0} == second.__dict__ | {
            "runtime_seconds": 0
        }
        assert first.selected_move in game(player).possible_moves()
        assert first.evaluator_calls == 32 and first.heuristic_evaluator_calls > 0
