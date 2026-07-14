import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_joint_heads_arena_failure_attribution import (
    LANES,
    aggregate,
    classify,
    composition_hash,
    play_unit,
    replay_coverage,
)


class _Evaluator:
    def __init__(self, policy, value):
        self.policy = np.asarray(policy, dtype=np.float32)
        self.value = value

    def evaluate(self, _game):
        return self.policy.copy(), self.value


class JointHeadsArenaFailureAttributionTests(unittest.TestCase):
    def setUp(self):
        self.game = KalahGame(pits=[4] * 12, captured_seeds=[0, 0], current_player=0)
        self.current = _Evaluator([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], 0.25)
        self.candidate = _Evaluator([0.4, 0.3, 0.2, 0.1, 0.0, 0.0], -0.5)

    def test_compositions_select_requested_outputs_and_preserve_masks(self):
        outputs = {}
        for name, (policy_source, value_source) in LANES.items():
            evaluator = arena.ComposedArtifactEvaluator(
                self.current,
                self.candidate,
                policy_source=policy_source,
                value_source=value_source,
            )
            outputs[name] = evaluator.evaluate(self.game)
        np.testing.assert_array_equal(
            outputs["current_policy_current_value"][0], self.current.policy
        )
        self.assertEqual(outputs["current_policy_current_value"][1], self.current.value)
        np.testing.assert_array_equal(
            outputs["candidate_policy_candidate_value"][0], self.candidate.policy
        )
        self.assertEqual(
            outputs["candidate_policy_candidate_value"][1], self.candidate.value
        )
        np.testing.assert_array_equal(
            outputs["candidate_policy_current_value"][0], self.candidate.policy
        )
        self.assertEqual(
            outputs["candidate_policy_current_value"][1], self.current.value
        )
        np.testing.assert_array_equal(
            outputs["current_policy_candidate_value"][0], self.current.policy
        )
        self.assertEqual(
            outputs["current_policy_candidate_value"][1], self.candidate.value
        )

    def test_terminal_values_are_not_composed(self):
        terminal = KalahGame(pits=[0] * 12, captured_seeds=[24, 24], current_player=0)
        evaluator = arena.ComposedArtifactEvaluator(
            self.current,
            self.candidate,
            policy_source="candidate",
            value_source="current",
        )
        policy, value = evaluator.evaluate(terminal)
        np.testing.assert_array_equal(policy, np.zeros(6, dtype=np.float32))
        self.assertEqual(value, 0.0)

    def test_profile_hash_includes_both_artifacts_and_lane(self):
        first = composition_hash(
            "current-a", "candidate-a", "candidate_policy_current_value"
        )
        self.assertNotEqual(
            first,
            composition_hash(
                "current-b", "candidate-a", "candidate_policy_current_value"
            ),
        )
        self.assertNotEqual(
            first,
            composition_hash(
                "current-a", "candidate-b", "candidate_policy_current_value"
            ),
        )
        self.assertNotEqual(
            first,
            composition_hash(
                "current-a", "candidate-a", "current_policy_candidate_value"
            ),
        )

    def test_raw_state_replay_overlap_never_compares_encoded_hashes(self):
        state = self.game.to_state()
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "replay.jsonl"
            replay.write_text(
                json.dumps({"state": [0.0], "raw_state": state, "game_index": 7}) + "\n"
            )
            coverage, rows = replay_coverage(
                [
                    {
                        "state": state,
                        "state_hash": __import__("hashlib")
                        .sha256(
                            json.dumps(
                                state, sort_keys=True, separators=(",", ":")
                            ).encode()
                        )
                        .hexdigest(),
                        "acting_player": 0,
                        "phase": "opening",
                        "legal_moves": self.game.possible_moves(),
                    }
                ],
                replay,
                {"train_source_row_indexes": [0], "validation_source_row_indexes": []},
                1,
            )
        self.assertEqual(coverage["exact_training_overlap_rate"], 1.0)
        self.assertFalse(coverage["encoded_hashes_compared"])
        self.assertEqual(rows[0]["train_replay_game_ids"], [7])

    def test_trace_row_preserves_budget_and_composition_context(self):
        evaluators = {lane: object() for lane in LANES}
        state = KalahGame(
            pits=[1] + [0] * 11, captured_seeds=[23, 24], current_player=0
        ).to_state()

        def fake_search(_evaluator, raw_state, _sims, _seed, _cpuct):
            legal = KalahGame.from_state(raw_state).possible_moves()
            return {"selected_move": legal[0], "visits": [1.0] * 6}

        with (
            patch(
                "ml.alphazero_lite.run_joint_heads_arena_failure_attribution.lane_evaluators",
                return_value=evaluators,
            ),
            patch(
                "ml.alphazero_lite.run_joint_heads_arena_failure_attribution.search",
                side_effect=fake_search,
            ),
        ):
            _game, rows = play_unit(
                {
                    "current": ".",
                    "candidate": ".",
                    "opening": {"state": state, "id": "o"},
                    "opening_index": 0,
                    "lane": "candidate_policy_current_value",
                    "challenger_player": 0,
                    "challenger_sims": 768,
                    "current_sims": 768,
                    "budget_pair": "768:768",
                    "c_puct": 0.90,
                    "seed": 42,
                }
            )
        self.assertTrue(rows)
        row = rows[0]
        self.assertEqual(row["budget_pair"], "768:768")
        self.assertEqual(row["effective_c_puct"], 0.90)
        self.assertEqual(
            row["composition_trajectory_source"], "candidate_policy_current_value"
        )
        self.assertTrue(row["search_profile_hash"])

    def test_classification_consumes_distribution_shift_coverage(self):
        metrics = {
            lane: {"384:256": {"raw_ds": value}}
            for lane, value in {
                "current_policy_current_value": 0.0,
                "candidate_policy_current_value": -0.04,
                "current_policy_candidate_value": 0.01,
                "candidate_policy_candidate_value": -0.1,
            }.items()
        }
        forced = {
            mode: {
                component: {
                    "bootstrap_95_ci": [-0.2, -0.1],
                    "unique_changed_states": 64,
                }
                for component in ("policy", "value", "joint")
            }
            for mode in ("neutral", "context_matched")
        }
        coverage = {
            "exact_training_overlap_rate": 0.01,
            "exact_validation_overlap_rate": 0.0,
            "covered_harm_count": 64,
            "uncovered_harm_count": 64,
            "overlapping_materially_better": True,
            "distance_harm_concentrated": False,
        }
        result = classify(metrics, forced, coverage, [{}])
        self.assertEqual(result["classification"], "replay_to_arena_distribution_shift")

    def test_classification_detects_replay_target_harm_on_covered_states(self):
        metrics = {
            lane: {"384:256": {"raw_ds": value}}
            for lane, value in {
                "current_policy_current_value": 0.0,
                "candidate_policy_current_value": -0.04,
                "current_policy_candidate_value": 0.01,
                "candidate_policy_candidate_value": -0.1,
            }.items()
        }
        forced = {
            mode: {
                component: {"bootstrap_95_ci": [-0.2, -0.1], "unique_changed_states": 1}
                for component in ("policy", "value", "joint")
            }
            for mode in ("neutral", "context_matched")
        }
        coverage = {
            "exact_training_overlap_rate": 0.3,
            "exact_validation_overlap_rate": 0.0,
            "covered_harm_count": 1,
            "uncovered_harm_count": 0,
            "covered_harmful_changes": True,
            "covered_forced_ci_upper": -0.01,
        }
        result = classify(metrics, forced, coverage, [{}])
        self.assertEqual(result["classification"], "replay_targets_bad_on_arena_states")

    def test_missing_stages_are_incomplete_and_aggregate_is_deterministic(self):
        self.assertEqual(
            classify({}, {}, {}, [])["classification"],
            "attribution_experiment_incomplete",
        )
        self.assertEqual(
            aggregate([-0.5, 0.0, 0.5], 42), aggregate([-0.5, 0.0, 0.5], 42)
        )


if __name__ == "__main__":
    unittest.main()
