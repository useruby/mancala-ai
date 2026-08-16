"""Focused invariants for the optimizer-aware PR #191 audit."""

from __future__ import annotations

import unittest

import torch

from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (
    bootstrap_summary,
    classify,
    cosine,
    snapshot_steps,
    trunk_group,
    vector_metrics,
)


class OptimizerAwareTrunkDynamicsAuditTest(unittest.TestCase):
    def test_snapshot_steps_use_real_batch_boundaries(self) -> None:
        self.assertEqual([0, 5, 12, 23, 35, 46], snapshot_steps(46))
        self.assertEqual([0, 1], snapshot_steps(1))

    def test_trunk_group_only_includes_shared_parameters(self) -> None:
        self.assertEqual("input_layer", trunk_group("input_layer.weight"))
        self.assertEqual("residual_block_2", trunk_group("residual_layers.2.1.bias"))
        self.assertIsNone(trunk_group("policy_head.weight"))

    def test_vector_metrics_distinguishes_opposed_updates(self) -> None:
        metrics = vector_metrics(
            {
                "policy": torch.tensor([1.0, 0.0]),
                "value": torch.tensor([-1.0, 0.0]),
                "joint": torch.tensor([0.0, 1.0]),
            }
        )
        self.assertAlmostEqual(-1.0, metrics["cosine_policy_value"])
        self.assertAlmostEqual(0.0, metrics["cosine_policy_joint"])
        self.assertAlmostEqual(0.0, cosine(torch.tensor([0.0]), torch.tensor([0.0])))

    def test_bootstrap_summary_is_stable(self) -> None:
        samples = [{"policy_norm": 1.0}, {"policy_norm": 3.0}]
        self.assertEqual(
            bootstrap_summary(samples, seed=191), bootstrap_summary(samples, seed=191)
        )
        self.assertEqual(
            2.0, bootstrap_summary(samples, seed=191)["policy_norm"]["mean"]
        )

    def test_conflict_requires_crossing_initial_ci_and_consecutive_snapshots(
        self,
    ) -> None:
        audit = {
            "raw_gradients": {
                "global": {
                    "policy_norm": {"median": 4.0},
                    "value_norm": {"median": 1.0},
                    "cosine_policy_value": {"lower_95": -0.1, "upper_95": 0.1},
                }
            },
            "virtual_updates": {
                "global": {
                    "policy_norm": {"median": 1.0},
                    "value_norm": {"median": 1.0},
                    "cosine_policy_joint": {"mean": 1.0},
                }
            },
        }
        negative = {
            **audit,
            "raw_gradients": {
                "global": {
                    **audit["raw_gradients"]["global"],
                    "cosine_policy_value": {"lower_95": -0.4, "upper_95": -0.1},
                }
            },
        }
        summary = {
            "snapshot_steps": [0, 5, 12],
            "snapshots": {
                "0": {"audit": audit},
                "5": {"audit": negative},
                "12": {"audit": negative},
            },
            "movement": {
                "5": {
                    "global": {
                        "relative_l2": 1.0,
                        "average_virtual_cosines": {"policy": 1.0, "value": 0.0},
                    }
                }
            },
        }
        self.assertIn(
            "gradient_conflict_emerges_during_training",
            classify(summary)["classifications"],
        )


if __name__ == "__main__":
    unittest.main()
