"""Focused contracts for the hard-arena semantic-equivalence audit."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.arena_command_contract import normalize_arena_command
from ml.alphazero_lite.run_hard_arena_semantic_calibration import (
    INCUMBENT_SHA,
    artifact_identity,
    build_result,
    production_commands,
)


class HardArenaSemanticCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = build_result()

    def test_default_paths_resolve_to_the_same_incumbent_artifact(self) -> None:
        identity = self.result["opponent_identity"]
        self.assertTrue(identity["default_path_equality"])
        self.assertTrue(identity["pr332_path_equality"])
        self.assertTrue(identity["artifact_equality"])
        self.assertEqual(INCUMBENT_SHA, identity["incumbent_weights_sha256"])

    def test_normalized_gate_commands_are_semantically_equal(self) -> None:
        prefilter, hard = production_commands()
        left = normalize_arena_command(prefilter, artifact_identity=artifact_identity)
        right = normalize_arena_command(hard, artifact_identity=artifact_identity)
        self.assertEqual(left, right)
        self.assertEqual(384, left["challenger_simulations"])
        self.assertEqual(256, left["current_simulations"])
        self.assertEqual(42, left["seed"])
        self.assertEqual(0.55, left["min_score"])

    def test_normalization_only_ignores_outputs(self) -> None:
        command, _unused = production_commands()
        base = normalize_arena_command(command, artifact_identity=artifact_identity)
        for flag, value in (
            ("--current-simulations", "384"),
            ("--seed", "43"),
            ("--random-opening-plies", "1"),
            ("--min-score", "0.56"),
            (
                "--current",
                ".tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1",
            ),
            ("--fpu-mode", "parent"),
        ):
            changed = [*command, flag, value]
            self.assertNotEqual(
                base,
                normalize_arena_command(changed, artifact_identity=artifact_identity),
            )
        changed_output = [*command[:-1], "elsewhere.json"]
        self.assertEqual(
            base,
            normalize_arena_command(
                changed_output, artifact_identity=artifact_identity
            ),
        )

    def test_frozen_artifact_ledgers_and_profiles_match(self) -> None:
        comparison = self.result["artifact_comparison"]
        self.assertTrue(comparison["equal"])
        for key in (
            "suite_sha256",
            "search_profile_hash",
            "seed_identity_ledger_sha256",
            "search_configuration_ledger_sha256",
            "search_outcome_ledger_sha256",
        ):
            self.assertEqual(comparison["pr328"][key], comparison["pr332"][key])

    def test_buckets_are_trajectory_telemetry_not_starting_suite(self) -> None:
        buckets = self.result["hard_suite_buckets"]
        self.assertEqual("trajectory_phase_telemetry", buckets["meaning"])
        self.assertFalse(buckets["starting_state_suite"])
        self.assertEqual(60, buckets["reported"]["opening"]["games"])
        self.assertEqual(60, buckets["reported"]["late"]["games"])

    def test_transfer_avoids_duplicate_calibration(self) -> None:
        self.assertTrue(self.result["pr330_pr331_calibration_transfer_valid"])
        self.assertFalse(self.result["fallback_calibration"]["eligible_to_execute"])
        self.assertFalse(self.result["fallback_calibration"]["executed"])
        self.assertFalse(self.result["new_games_required"])
        self.assertFalse(self.result["prefilter_hard_evidence_independent"])
        self.assertFalse(self.result["training_invoked"])
        self.assertFalse(self.result["promotion"]["performed"])
        self.assertFalse(self.result["production_gate_modified"])
        self.assertEqual(
            "hard_arena_duplicate_of_legacy_prefilter_calibration_transfers",
            self.result["classification"],
        )
        rows = self.result["inherited_hard_calibration"]
        self.assertEqual(9, len(rows))
        self.assertTrue(
            all(
                row["data_source"] == "inherited_from_semantically_identical_evaluator"
                for row in rows
            )
        )
        self.assertEqual(
            7, self.result["known_positive_canonical_positive_evidence_count"]
        )
        self.assertTrue(self.result["identity_asymmetric_bias"])
        self.assertTrue(self.result["equal_budget_identity_symmetry"])


if __name__ == "__main__":
    unittest.main()
