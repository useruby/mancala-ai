import random

import numpy as np

from ml.alphazero_lite import run_pr248_prospective_fixed_target_budget as prospective
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, encode_state, standard_start_state


class UniformEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        priors = np.zeros(6, dtype=np.float32)
        legal = game.possible_moves()
        priors[legal] = 1.0 / len(legal)
        return priors, 0.0


def test_matched_target_rng_preserves_noisy_root_prior(monkeypatch):
    monkeypatch.setattr(prospective, "_EVALUATOR", UniformEvaluator())
    state = random.Random(46).getstate()
    priors = []
    for simulations in (384, 768, 1024):
        rng = random.Random()
        rng.setstate(state)
        _policy, _root, prior = prospective.search(
            KalahGame.from_state(standard_start_state()),
            rng,
            root=None,
            reuse=False,
            simulations=simulations,
            noisy=True,
        )
        priors.append(prior)
    assert all(np.array_equal(priors[0], prior) for prior in priors[1:])


def test_target_lane_contract_is_exactly_preregistered():
    assert prospective.LANES == ("reused", "fresh384", "fresh768", "fresh1024")
    assert prospective.SEED == 46
    assert prospective.BASE == 384


def test_budget_movement_reports_the_preregistered_categories():
    rows = [
        {
            "legal_moves": [0, 1],
            "move_index": 0,
            "action_sampling_noise_enabled": True,
            "state": encode_state(standard_start_state(), input_encoding="kalah_v3"),
        }
    ]
    telemetry = [
        {
            "inherited_child_visit_mass": 0,
            "policies": {
                "reused": [1, 0, 0, 0, 0, 0],
                "fresh384": [1, 0, 0, 0, 0, 0],
                "fresh768": [0, 1, 0, 0, 0, 0],
                "fresh1024": [1, 0, 0, 0, 0, 0],
            },
        }
    ]

    diagnostics = prospective.target_diagnostics(rows, telemetry)

    assert diagnostics["search_budget_movement"]["768_only_flip"]["frequency"] == 1.0
