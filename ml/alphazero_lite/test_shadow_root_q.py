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


def test_shadow_q_weight_endpoints_and_interpolation_preserve_contracts() -> None:
    evaluator = FixedEvaluator()
    ordinary = PUCT(
        evaluator, 24, 1.25, random.Random(8), root_policy_mode="deterministic"
    )
    ordinary_visits, ordinary_root = ordinary.run(
        _game(), dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        visits, root, telemetry = run_shadow_root_q_search(
            _game(),
            main_evaluator=evaluator,
            shadow_evaluator=evaluator,
            simulations=24,
            c_puct=1.25,
            seed=8,
            root_policy_mode="deterministic",
            shadow_q_weight=weight,
            record_selection_q_direction=True,
        )
        assert np.array_equal(visits, ordinary_visits)
        assert root.visit_count == ordinary_root.visit_count
        for use in telemetry["selection_q_direction_telemetry"]:
            candidate_q = float(use["candidate_q"])
            parent_q = float(use["parent_q"])
            assert float(use["blended_q"]) == (
                (1.0 - weight) * candidate_q + weight * parent_q
            )
            if weight == 0.0:
                assert candidate_q == float(use["blended_q"])
            if weight == 1.0:
                assert parent_q == float(use["blended_q"])


def test_shadow_q_weight_zero_is_an_exact_ordinary_search_identity() -> None:
    evaluator = FixedEvaluator()
    ordinary = PUCT(evaluator, 24, 1.25, random.Random(8))
    ordinary_visits, ordinary_root = ordinary.run(
        _game(), dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    visits, root, telemetry = run_shadow_root_q_search(
        _game(),
        main_evaluator=evaluator,
        shadow_evaluator=FixedEvaluator(),
        simulations=24,
        c_puct=1.25,
        seed=8,
        shadow_q_weight=0.0,
        record_selection_q_direction=True,
    )
    assert np.array_equal(visits, ordinary_visits)
    assert root.value_sum == ordinary_root.value_sum
    assert all(
        use["candidate_q"] == use["blended_q"]
        for use in telemetry["selection_q_direction_telemetry"]
    )


def test_shadow_q_weight_one_matches_existing_exact_shadow_behavior() -> None:
    evaluator = FixedEvaluator()
    default_visits, default_root, _default_telemetry = run_shadow_root_q_search(
        _game(),
        main_evaluator=evaluator,
        shadow_evaluator=FixedEvaluator(),
        simulations=24,
        c_puct=1.25,
        seed=8,
    )
    visits, root, telemetry = run_shadow_root_q_search(
        _game(),
        main_evaluator=evaluator,
        shadow_evaluator=FixedEvaluator(),
        simulations=24,
        c_puct=1.25,
        seed=8,
        shadow_q_weight=1.0,
        record_selection_q_direction=True,
    )
    assert np.array_equal(visits, default_visits)
    assert root.value_sum == default_root.value_sum
    assert all(
        use["parent_q"] == use["blended_q"]
        for use in telemetry["selection_q_direction_telemetry"]
    )


def test_shadow_q_override_is_root_only_and_leaves_fpu_unchanged() -> None:
    _visits, _root, telemetry = run_shadow_root_q_search(
        _game(),
        main_evaluator=FixedEvaluator(),
        shadow_evaluator=FixedEvaluator(),
        simulations=64,
        c_puct=1.25,
        seed=8,
        shadow_q_weight=0.5,
        record_selection_trace=True,
    )
    root_hash = telemetry["selection_trace"][0]["selection_path"][0]["state_hash"]
    for trace in telemetry["selection_trace"]:
        for node in trace["selection_path"]:
            if node["state_hash"] != root_hash:
                assert not any(
                    child["selection_q_overridden"] for child in node["children"]
                )
            for child in node["children"]:
                if child["used_fpu"]:
                    assert not child["selection_q_overridden"]
