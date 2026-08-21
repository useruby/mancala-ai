from __future__ import annotations

import random

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.postdivergence_amplification import (
    paired_postdivergence_metrics,
    reconstruct_root_trajectory,
    validate_final_root_trajectory,
)
from ml.alphazero_lite.self_play import Evaluator, PUCT


class FixedEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        return np.asarray([0.31, 0.19, 0.17, 0.13, 0.11, 0.09]), 0.2


def _game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def test_reconstruction_matches_live_puct_summary() -> None:
    trace: list[dict] = []
    search = PUCT(FixedEvaluator(), 48, 1.25, random.Random(9), selection_trace=trace)
    search.run(_game(), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    assert validate_final_root_trajectory(
        reconstruct_root_trajectory(trace), search.root_summary()
    )


def test_postdivergence_metrics_use_zero_for_unvisited_q() -> None:
    base = {
        "simulation_index": 1,
        "selection_path": [{"chosen_move": 0}],
        "backed_up_value": 0.2,
        "selected_leaf_state_hash": "a",
        "terminal_leaf": False,
    }
    other = {
        **base,
        "selection_path": [{"chosen_move": 1}],
        "backed_up_value": -0.2,
        "selected_leaf_state_hash": "b",
        "terminal_leaf": True,
    }
    trajectory = [
        {
            "actions": [0, 1],
            "q_value": {0: 0.2, 1: 0.0},
            "visit_distribution": [1, 0],
            "q_ranking": [0, 1],
            "best_q_action": 0,
            "deterministic_move": 0,
            "visit_leader": 0,
            "top1_top2_visit_margin": 1,
        }
    ]
    changed = [
        {
            **trajectory[0],
            "q_value": {0: 0.0, 1: -0.2},
            "visit_distribution": [0, 1],
            "q_ranking": [0, 1],
            "deterministic_move": 1,
            "visit_leader": 1,
            "top1_top2_visit_margin": 1,
        }
    ]
    result = paired_postdivergence_metrics([base], [other], trajectory, changed, 1)
    assert result["early_metrics"]["q_divergence_auc_32"] == 0.4
    assert result["early_metrics"]["backup_gap_auc_32"] == 0.4
