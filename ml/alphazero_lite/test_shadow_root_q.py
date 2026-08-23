from __future__ import annotations

import random

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, PUCT
from ml.alphazero_lite.shadow_root_q import run_shadow_root_q_search


class FixedEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        priors = np.zeros(6, dtype=np.float32)
        for move in game.possible_moves():
            priors[move] = 1.0 / len(game.possible_moves())
        return priors, 0.25


def _game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [2, 2, 0, 0, 0, 0],
            "opponent_pits": [2, 2, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def test_self_shadow_is_an_exact_identity_search() -> None:
    evaluator = FixedEvaluator()
    ordinary = PUCT(
        evaluator, 24, 1.25, random.Random(8), root_policy_mode="deterministic"
    )
    ordinary_visits, ordinary_root = ordinary.run(
        _game(), dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    shadow_visits, shadow_root, telemetry = run_shadow_root_q_search(
        _game(),
        main_evaluator=evaluator,
        shadow_evaluator=evaluator,
        simulations=24,
        c_puct=1.25,
        seed=8,
        root_policy_mode="deterministic",
    )
    assert np.array_equal(shadow_visits, ordinary_visits)
    assert shadow_root.visit_count == ordinary_root.visit_count
    assert shadow_root.value_sum == ordinary_root.value_sum
    assert telemetry["no_future_information"]
    assert telemetry["shadow_pre_simulation_snapshots"] == 24
