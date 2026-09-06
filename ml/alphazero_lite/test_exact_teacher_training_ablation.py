"""Tests for the exact-vs-MCTS controlled training ablation."""

import json
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.run_exact_teacher_training_ablation import (
    aggregate_lane,
    build_holdout_contrast,
    parse_lane_list,
    parse_seed_list,
    score_holdout,
    verify_holdout_disjoint,
    verify_identical_states,
)


def _lane_row(source_id: str, canonical: str, state: list[float]) -> dict:
    return {
        "source_id": source_id,
        "canonical_state": canonical,
        "state": list(state),
        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "value": 1.0,
    }


def _holdout_row(source_id: str, state: list[float], optimal: list[int]) -> dict:
    return {
        "source_id": source_id,
        "state": list(state),
        "exact_optimal_actions": list(optimal),
        "exact_value_training": 1.0,
        "stones_bucket": "8-16",
    }


class AblationInputValidationTest(unittest.TestCase):
    def test_identical_states_pass(self) -> None:
        state = [0.1] * 27
        left = [_lane_row("a", "k1", state), _lane_row("b", "k2", state)]
        right = [_lane_row("a", "k1", state), _lane_row("b", "k2", state)]
        verify_identical_states(left, right)

    def test_source_id_drift_rejected(self) -> None:
        state = [0.1] * 27
        left = [_lane_row("a", "k1", state)]
        right = [_lane_row("b", "k1", state)]
        with self.assertRaises(ValueError):
            verify_identical_states(left, right)

    def test_encoded_state_drift_rejected(self) -> None:
        left = [_lane_row("a", "k1", [0.1] * 27)]
        right = [_lane_row("a", "k1", [0.2] * 27)]
        with self.assertRaises(ValueError):
            verify_identical_states(left, right)

    def test_row_count_mismatch_rejected(self) -> None:
        state = [0.1] * 27
        with self.assertRaises(ValueError):
            verify_identical_states(
                [_lane_row("a", "k1", state)], [_lane_row("a", "k1", state)] * 2
            )

    def test_holdout_overlap_rejected(self) -> None:
        state = [0.1] * 27
        train = [_lane_row("a", "k1", state)]
        holdout = [_lane_row("b", "k1", state)]
        with self.assertRaises(ValueError):
            verify_holdout_disjoint(train, holdout)

    def test_holdout_empty_rejected(self) -> None:
        state = [0.1] * 27
        with self.assertRaises(ValueError):
            verify_holdout_disjoint([_lane_row("a", "k1", state)], [])

    def test_holdout_disjoint_pass(self) -> None:
        state = [0.1] * 27
        verify_holdout_disjoint(
            [_lane_row("a", "k1", state)], [_lane_row("b", "k2", state)]
        )


class HoldoutScoringStructureTest(unittest.TestCase):
    def test_score_holdout_schema_and_bounds(self) -> None:
        from ml.alphazero_lite.self_play import encode_state

        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        encoded = encode_state(state, input_encoding="kalah_v3")
        rows = [
            _holdout_row("h1", encoded, [2]),
            _holdout_row("h2", encoded, [0, 1]),
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "init.npz"
            from ml.alphazero_lite.train import (
                PolicyValueNet,
                checkpoint_from_state_dict,
            )

            model = PolicyValueNet((8, 2), "mlp_v1", 27)
            arrays = {
                k: np.asarray(v)
                for k, v in checkpoint_from_state_dict(model.state_dict()).items()
            }
            np.savez(checkpoint, **arrays)
            summary = score_holdout(checkpoint, rows, input_encoding="kalah_v3")
        self.assertEqual(2, summary["n"])
        self.assertIn(summary["top_in_exact_set"], (0, 1, 2))
        self.assertGreaterEqual(summary["top_in_exact_set_rate"], 0.0)
        self.assertLessEqual(summary["top_in_exact_set_rate"], 1.0)
        self.assertGreaterEqual(summary["value_mae"], 0.0)
        self.assertLessEqual(summary["value_mae"], 2.0)
        self.assertEqual(1, summary["single_n"])
        self.assertEqual(1, summary["multi_n"])
        self.assertIn("8-16", summary["by_bucket"])

    def test_frozen_production_holdout_rows_scoreable(self) -> None:
        holdout_path = Path("/tmp/azlite_exact_teacher_production/holdout.jsonl")
        if not holdout_path.is_file():
            self.skipTest("production holdout not present")
        rows = [json.loads(line) for line in holdout_path.open()]
        self.assertEqual(2000, len(rows))
        for row in rows[:5]:
            self.assertTrue(row["exact_optimal_actions"])
            self.assertIn(row["exact_value_training"], (-1.0, 0.0, 1.0))
            self.assertEqual(27, len(row["state"]))


class LaneAndSeedParsingTest(unittest.TestCase):
    def test_parse_lane_list_defaults(self) -> None:
        self.assertEqual(["exact", "mcts"], parse_lane_list("exact,mcts"))
        self.assertEqual(
            ["exact", "mcts", "blend"], parse_lane_list("exact,mcts,blend")
        )

    def test_parse_lane_list_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            parse_lane_list("exact,nope")

    def test_parse_lane_list_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            parse_lane_list("exact,exact")

    def test_parse_seed_list(self) -> None:
        self.assertEqual([42, 43, 44], parse_seed_list("42,43,44"))
        with self.assertRaises(ValueError):
            parse_seed_list("42,42")

    def test_aggregate_lane_means(self) -> None:
        seeds = {
            "seed42": {
                "holdout": {
                    "top_in_exact_set_rate": 0.5,
                    "value_mae": 0.8,
                    "policy_cross_entropy_vs_uniform_optimal": 1.4,
                }
            },
            "seed43": {
                "holdout": {
                    "top_in_exact_set_rate": 0.6,
                    "value_mae": 0.7,
                    "policy_cross_entropy_vs_uniform_optimal": 1.3,
                }
            },
        }
        agg = aggregate_lane(seeds)
        self.assertEqual(2, agg["n_seeds"])
        self.assertAlmostEqual(0.55, agg["top_in_exact_set_rate_mean"])
        self.assertAlmostEqual(0.75, agg["value_mae_mean"])

    def test_build_holdout_contrast_three_lanes(self) -> None:
        def lane(rate: float, mae: float) -> dict:
            return {
                "holdout": {
                    "top_in_exact_set_rate": rate,
                    "value_mae": mae,
                },
                "aggregate": {
                    "top_in_exact_set_rate_mean": rate,
                    "value_mae_mean": mae,
                },
            }

        contrast = build_holdout_contrast(
            {
                "exact": lane(0.49, 0.83),
                "mcts": lane(0.48, 0.77),
                "blend": lane(0.485, 0.78),
            }
        )
        self.assertAlmostEqual(0.01, contrast["top_in_set_rate_diff_exact_minus_mcts"])
        self.assertLess(contrast["value_mae_diff_blend_minus_exact"], 0.0)
        self.assertGreater(contrast["value_mae_diff_blend_minus_mcts"], 0.0)


class BlendLabelBuilderTest(unittest.TestCase):
    def test_blend_takes_exact_policy_and_mcts_value(self) -> None:
        from ml.alphazero_lite.run_exact_ablation_blend_labels import build_blend_rows

        state = [0.1] * 27
        exact_rows = [
            {
                "source_id": "a",
                "canonical_state": "k1",
                "state": list(state),
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 1.0,
                "policy_target_mode": "default",
                "player": 0,
                "move_index": 3,
                "legal_moves": [0, 1],
                "teacher": "native_hybrid_exact",
                "exact_optimal_actions": [0],
                "exact_root_margin": 4,
                "exact_value_training": 1.0,
            }
        ]
        mcts_rows = [
            {
                "source_id": "a",
                "canonical_state": "k1",
                "state": list(state),
                "policy": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                "value": 0.25,
                "value_target_mode": "default",
                "teacher": "classic_mcts_1200",
                "teacher_simulations": 1200,
            }
        ]
        blended = build_blend_rows(exact_rows, mcts_rows)
        self.assertEqual(1, len(blended))
        self.assertEqual([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], blended[0]["policy"])
        self.assertEqual(0.25, blended[0]["value"])
        self.assertEqual("exact_policy_mcts_value_blend", blended[0]["teacher"])

    def test_blend_rejects_state_drift(self) -> None:
        from ml.alphazero_lite.run_exact_ablation_blend_labels import build_blend_rows

        exact_rows = [
            {
                "source_id": "a",
                "canonical_state": "k1",
                "state": [0.1] * 27,
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 1.0,
                "player": 0,
                "move_index": 0,
                "legal_moves": [0],
            }
        ]
        mcts_rows = [
            {
                "source_id": "a",
                "canonical_state": "k1",
                "state": [0.2] * 27,
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.5,
                "player": 0,
                "move_index": 0,
                "legal_moves": [0],
            }
        ]
        with self.assertRaises(ValueError):
            build_blend_rows(exact_rows, mcts_rows)

    def test_frozen_blend_file_validates(self) -> None:
        blend_path = Path("/tmp/azlite_exact_ablation/blend_train.jsonl")
        if not blend_path.is_file():
            self.skipTest("blend file not present")
        from ml.alphazero_lite.train import load_jsonl

        x, _, _ = load_jsonl(
            blend_path, policy_target_mode="default", value_target_mode="default"
        )
        self.assertEqual((8000, 27), (x.shape[0], x.shape[1]))


if __name__ == "__main__":
    unittest.main()
