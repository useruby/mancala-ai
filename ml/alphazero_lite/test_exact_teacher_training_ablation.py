"""Tests for the exact-vs-MCTS controlled training ablation."""

import json
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.run_exact_teacher_training_ablation import (
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


if __name__ == "__main__":
    unittest.main()
