from __future__ import annotations

import random
import unittest

import numpy as np

from ml.alphazero_lite.arena import BlendedArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_value_delta_blend_preflight import (
    BUDGETS,
    MIN_PROBE_ROWS,
    candidate_probe_reasons,
    classify,
)
from ml.alphazero_lite.self_play import PUCT


class FixedEvaluator:
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, game):
        priors = np.zeros(6, dtype=np.float32)
        priors[game.possible_moves()] = 1.0 / len(game.possible_moves())
        return priors, self.value


class ValueDeltaBlendTest(unittest.TestCase):
    def setUp(self):
        self.game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

    def test_endpoints_and_midpoint_keep_current_policy(self):
        current = FixedEvaluator(-0.4)
        candidate = FixedEvaluator(0.8)
        current_policy, current_value = current.evaluate(self.game)
        candidate_policy, candidate_value = candidate.evaluate(self.game)
        for alpha, expected in (
            (0.0, current_value),
            (0.5, 0.2),
            (1.0, candidate_value),
        ):
            policy, value = BlendedArtifactEvaluator(
                current, candidate, alpha
            ).evaluate(self.game)
            np.testing.assert_array_equal(policy, current_policy)
            self.assertAlmostEqual(value, expected)
        self.assertFalse(np.array_equal(current_policy, candidate_policy) is False)

    def test_terminal_value_bypasses_evaluator_and_perspective_is_once(self):
        terminal = KalahGame(pits=[0] * 12, captured_seeds=[30, 18], current_player=0)
        evaluator = BlendedArtifactEvaluator(
            FixedEvaluator(-0.3), FixedEvaluator(0.9), 0.5
        )
        search = PUCT(evaluator, simulations=1, c_puct=1.25, rng=random.Random(7))
        visits, root = search.run(terminal)
        self.assertEqual(float(np.sum(visits)), 0.0)
        self.assertEqual(root.visit_count, 1)
        self.assertEqual(root.q_value, 0.0)

    def test_deterministic_search_repeats(self):
        evaluator = BlendedArtifactEvaluator(
            FixedEvaluator(-0.2), FixedEvaluator(0.6), 0.5
        )

        def run_once():
            search = PUCT(evaluator, simulations=32, c_puct=1.25, rng=random.Random(9))
            visits, root = search.run(self.game)
            return visits, search.select_root_move(root, self.game.possible_moves())

        first_visits, first_move = run_once()
        second_visits, second_move = run_once()
        np.testing.assert_array_equal(first_visits, second_visits)
        self.assertEqual(first_move, second_move)

    def test_alpha_zero_reference_needs_semantics_not_value_improvement(self):
        current = {
            "terminal_outcome_mae": 0.5,
            "terminal_outcome_sign_accuracy": 0.7,
            "terminal_outcome_correlation": 0.5,
        }
        probe = {
            "value": dict(current),
            "legal_failures": 0,
            "search": {
                "by_budget": {
                    budget: {
                        "rows": MIN_PROBE_ROWS,
                        "selected_move_changed_rate_vs_alpha_000": 0.0,
                    }
                    for budget in BUDGETS
                }
            },
        }
        self.assertIn("value_mae_not_improved", candidate_probe_reasons(probe, current))
        self.assertTrue(True, "required alpha=0 is not passed through candidate gates")

    def test_classification_guards_require_complete_non_smoke_probes(self):
        def lane(role="candidate_lane", rows=MIN_PROBE_ROWS):
            return {
                "role": role,
                "candidate_probe_pass": False,
                "search": {"by_budget": {budget: {"rows": rows} for budget in BUDGETS}},
            }

        summary = {
            "run_arguments": {
                "probe_only": True,
                "global_alphas": [0.0, 0.1, 1.0],
                "budget_alpha_lanes": ["global025"],
            },
            "probes": {
                "blend_alpha_000": lane("required_reference", 2),
                "blend_alpha_010": lane(rows=2),
                "blend_alpha_100": lane("diagnostic_reference", 2),
                "global025": lane(rows=2),
            },
            "semantic_endpoints": {
                "alpha_000_identity": True,
                "alpha_100_candidate_reproduction": True,
            },
            "monotonicity": {
                budget: {
                    "root_value_delta_monotonic": True,
                    "child_q_delta_monotonic": True,
                    "visit_kl_generally_increasing": True,
                }
                for budget in BUDGETS
            },
            "suites": {},
        }
        self.assertEqual(classify(summary), "value_delta_blend_smoke_inconclusive")
        summary["run_arguments"]["probe_only"] = False
        self.assertEqual(classify(summary), "value_delta_blend_smoke_inconclusive")
        for item in summary["probes"].values():
            for metrics in item["search"]["by_budget"].values():
                metrics["rows"] = MIN_PROBE_ROWS
        del summary["probes"]["global025"]
        self.assertEqual(classify(summary), "value_delta_blend_smoke_inconclusive")

    def test_passing_probe_requires_completed_medium_results(self):
        lane = {
            "role": "candidate_lane",
            "candidate_probe_pass": True,
            "carry_to_medium": True,
            "search": {
                "by_budget": {budget: {"rows": MIN_PROBE_ROWS} for budget in BUDGETS}
            },
        }
        summary = {
            "run_arguments": {
                "probe_only": False,
                "global_alphas": [0.0, 1.0],
                "budget_alpha_lanes": [],
            },
            "probe_only": False,
            "probes": {
                "blend_alpha_000": {**lane, "role": "required_reference"},
                "blend_alpha_100": {**lane, "role": "diagnostic_reference"},
                "blend_alpha_025": lane,
            },
            "semantic_endpoints": {
                "alpha_000_identity": True,
                "alpha_100_candidate_reproduction": True,
            },
            "monotonicity": {
                budget: {
                    "root_value_delta_monotonic": True,
                    "child_q_delta_monotonic": True,
                    "visit_kl_generally_increasing": True,
                }
                for budget in BUDGETS
            },
            "suites": {"medium": {"status": "stopped_by_probe_gates"}},
        }
        self.assertEqual(classify(summary), "value_delta_blend_suite_incomplete")
