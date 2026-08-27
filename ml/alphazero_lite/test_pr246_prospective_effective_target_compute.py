import random

import numpy as np

from ml.alphazero_lite import (
    run_pr246_prospective_effective_target_compute as prospective,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, standard_start_state


class UniformEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        priors = np.zeros(6, dtype=np.float32)
        legal = game.possible_moves()
        priors[legal] = 1.0 / len(legal)
        return priors, 0.0


def test_matched_target_rngs_receive_identical_noisy_root_priors(monkeypatch):
    monkeypatch.setattr(prospective, "_EVALUATOR", UniformEvaluator())
    monkeypatch.setattr(prospective, "_PLY", 0)
    state = random.Random(45).getstate()
    priors = []
    for simulations in (384, 384, 511):
        rng = random.Random()
        rng.setstate(state)
        _policy, _root, prior = prospective.search(
            KalahGame.from_state(standard_start_state()),
            rng,
            root=None,
            reuse_subtree=False,
            simulations=simulations,
            noisy=True,
        )
        priors.append(prior)
    assert np.array_equal(priors[0], priors[1])
    assert np.array_equal(priors[0], priors[2])
