"""Mathematical coverage for the frozen high-stone checkpoint selector."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
    high_stone_validation_metrics,
    phase_mask_for_encoded_states,
    select_epoch_by_score,
)
from ml.alphazero_lite.self_play import encode_state


class CheckpointPhaseSelectionTest(unittest.TestCase):
    def _metrics(
        self,
        *,
        values: list[float],
        targets: list[float],
        weights: list[float],
        mask: list[bool],
    ) -> dict[str, float]:
        return high_stone_validation_metrics(
            logits=torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]),
            value_predictions=torch.tensor(values).reshape(-1, 1),
            policy_targets=torch.tensor([[1.0, 0.0]] * 3),
            value_targets=torch.tensor(targets).reshape(-1, 1),
            legal_mask=torch.ones((3, 2)),
            policy_loss_weights=torch.tensor(weights),
            phase_mask=torch.tensor(mask),
            value_loss_weight=0.3,
            value_loss="huber",
            huber_delta=1.0,
            policy_cross_entropy=train.compute_policy_cross_entropy,
            weighted_policy_loss=train.weighted_policy_loss,
            value_loss_vector=train.compute_value_loss_vector,
        )

    def test_active_pits_exclude_stores_and_phase_is_strictly_over_32(self) -> None:
        state = {
            "player_pits": [6, 6, 6, 6, 6, 2],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 20,
            "opponent_store": 28,
            "current_player": 0,
        }
        encoded = np.asarray([encode_state(state, input_encoding="kalah_v3")])

        self.assertEqual(32, active_pit_stones_from_encoded_state(encoded[0]))
        self.assertEqual([False], phase_mask_for_encoded_states(encoded).tolist())

    def test_phase_score_uses_normal_weighted_policy_huber_objective(self) -> None:
        metrics = self._metrics(
            values=[2.0, 0.0, 0.0],
            targets=[0.0, 0.0, 0.0],
            weights=[1.0, 3.0, 0.0],
            mask=[True, True, False],
        )

        # Policy CE is weighted (0.693... and 0.126...) and value Huber(2)=1.5.
        expected_policy = (np.log(2.0) + 3.0 * np.log1p(np.exp(-2.0))) / 4.0
        self.assertAlmostEqual(expected_policy, metrics["phase_validation_policy_loss"])
        self.assertAlmostEqual(0.75, metrics["phase_validation_value_loss"])
        self.assertAlmostEqual(
            expected_policy + 0.3 * 0.75, metrics["phase_validation_score"]
        )
        self.assertAlmostEqual(1.0, metrics["phase_validation_value_mae"])

    def test_non_phase_rows_do_not_affect_score_and_multiplicity_does(self) -> None:
        baseline = self._metrics(
            values=[0.0, 0.0, 100.0],
            targets=[0.0, 0.0, 0.0],
            weights=[1.0, 1.0, 1.0],
            mask=[True, True, False],
        )
        repeated = self._metrics(
            values=[0.0, 0.0, 100.0],
            targets=[0.0, 0.0, 0.0],
            weights=[1.0, 3.0, 1.0],
            mask=[True, True, False],
        )

        self.assertNotEqual(
            baseline["phase_validation_score"], repeated["phase_validation_score"]
        )
        self.assertEqual(2, baseline["phase_validation_count"])

    def test_empty_phase_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_phase_validation_empty"):
            self._metrics(
                values=[0.0, 0.0, 0.0],
                targets=[0.0, 0.0, 0.0],
                weights=[1.0, 1.0, 1.0],
                mask=[False, False, False],
            )

    def test_selects_minimum_epoch_and_earliest_tie(self) -> None:
        self.assertEqual(2, select_epoch_by_score([0.8, 0.7, 0.9]))
        self.assertEqual(1, select_epoch_by_score([0.7, 0.7]))
