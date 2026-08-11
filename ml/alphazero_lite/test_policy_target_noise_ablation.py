"""Focused invariants for the paired policy-target-noise ablation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ml.alphazero_lite.run_policy_target_noise_ablation import (
    consistency_audit,
    materialize_paired_views,
    select_probe_states,
)


def state(index: int) -> dict:
    return {
        "player_pits": [4 + (index % 2), 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": index,
        "opponent_store": 0,
        "current_player": index % 2,
    }


def teacher(policy: list[float]) -> dict:
    return {
        "policy": policy,
        "legal_moves": [0, 1, 2],
        "top_move": max(range(3), key=lambda move: policy[move]),
        "entropy": 0.5,
        "visit_margin": policy[0] - policy[1],
    }


class PolicyTargetNoiseAblationTest(unittest.TestCase):
    def test_probe_selection_is_deterministic_domain_balanced_and_unique(self) -> None:
        # The selector's fixed 384-per-domain contract is intentionally tested
        # with a sufficiently large source population.
        pilot = [
            {
                "raw_state": state(index),
                "state": [0.0] * 27,
                "policy": [1, 0, 0, 0, 0, 0],
            }
            for index in range(500)
        ]
        evaluation = [
            {"state": state(index + 1000), "policy": [1, 0, 0, 0, 0, 0]}
            for index in range(500)
        ]
        first, manifest = select_probe_states(pilot, evaluation, seed=42)
        second, second_manifest = select_probe_states(pilot, evaluation, seed=42)
        self.assertEqual(768, len(first))
        self.assertEqual(manifest, second_manifest)
        self.assertEqual(
            [row["state_hash"] for row in first], [row["state_hash"] for row in second]
        )
        self.assertEqual(768, len({row["state_hash"] for row in first}))
        self.assertEqual(
            {"pr176_standard_start_pilot": 384, "evaluation_opening_diagnostic": 384},
            manifest["domain_counts"],
        )

    def test_consistency_gate_requires_all_prespecified_signals(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "phase": "opening",
                    "player": index % 2,
                    "source_domain": "pilot",
                    "noisy_n384": teacher([0.45, 0.5, 0.05, 0, 0, 0]),
                    "denoised_d384": teacher([0.85, 0.1, 0.05, 0, 0, 0]),
                    "reference_d1200": teacher([0.9, 0.08, 0.02, 0, 0, 0]),
                }
            )
        report = consistency_audit(rows)
        self.assertTrue(report["passes"])
        self.assertGreaterEqual(report["js_improvement_fraction"], 0.15)
        self.assertGreaterEqual(
            report["denoised_d384"]["top1_agreement"],
            report["noisy_n384"]["top1_agreement"] + 0.05,
        )

    def test_training_views_differ_only_in_policy(self) -> None:
        rows = [
            {
                "state": [0.0] * 27,
                "game_index": 0,
                "winner": 1,
                "value": 1.0,
                "legal_moves": [0, 1],
                "policy_target_noisy": [0.8, 0.2, 0, 0, 0, 0],
                "policy_target_denoised": [0.2, 0.8, 0, 0, 0, 0],
                "trajectory_hash": "trajectory",
            }
        ]
        with TemporaryDirectory() as directory:
            noisy_path, denoised_path, audit = materialize_paired_views(
                rows, workdir=Path(directory)
            )
            self.assertTrue(all(audit["pairing_invariants"].values()))
            self.assertNotEqual(
                audit["noisy_policy_target_sha256"],
                audit["denoised_policy_target_sha256"],
            )
            self.assertIn('"policy_target_noise_mode": "noisy"', noisy_path.read_text())
            self.assertIn(
                '"policy_target_noise_mode": "denoised"', denoised_path.read_text()
            )


if __name__ == "__main__":
    unittest.main()
