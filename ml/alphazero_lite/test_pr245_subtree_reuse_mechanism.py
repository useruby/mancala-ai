import random

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_pr245_subtree_reuse_mechanism import search
from ml.alphazero_lite.self_play import Evaluator, standard_start_state


class UniformEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        priors = np.zeros(6, dtype=np.float32)
        priors[game.possible_moves()] = 1.0 / len(game.possible_moves())
        return priors, 0.0


def test_matched_rng_searches_receive_identical_noisy_root_priors():
    row = {
        "teacher_search_profile": {
            "c_puct": 1.25,
            "search_options": {
                "fpu_mode": "zero",
                "normalize_values": False,
                "root_policy_mode": "visit_count",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
            },
        },
        "action_sampling_noise_enabled": True,
        "dirichlet_alpha": 0.3,
        "target_dirichlet_epsilon": 0.3,
        "legal_moves": list(range(6)),
        "move_index": 0,
        "policy_target_mode": "default",
    }
    rng_state = random.Random(44).getstate()
    priors = []
    for simulations in (1, 1, 5):
        rng = random.Random()
        rng.setstate(rng_state)
        _policy, _root, prior = search(
            UniformEvaluator(),
            KalahGame.from_state(standard_start_state()),
            row,
            rng,
            root=None,
            reuse_subtree=False,
            simulations=simulations,
        )
        priors.append(prior)
    assert np.array_equal(priors[0], priors[1])
    assert np.array_equal(priors[0], priors[2])
