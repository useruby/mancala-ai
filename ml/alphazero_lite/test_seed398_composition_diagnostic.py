"""Focused fail-closed tests for the frozen seed398 composition diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed398_composition_diagnostic import (
    TREATMENTS,
    SEED455_ARTIFACT,
    actual_consumed_identities,
    build_complete_exclusion_proof,
    cached_evidence_mode,
    require_hash,
    validate_command_source_selection,
    validate_treatment_evidence,
)


class CompositionDiagnosticTests(unittest.TestCase):
    def test_recomputed_complete_exclusion_union_is_at_least_known_base(self) -> None:
        proof = build_complete_exclusion_proof()
        self.assertEqual(proof["known_base_count"], 94_370)
        self.assertEqual(proof["excluded_state_count"], 94_370)
        self.assertEqual(proof["additional_consumed_state_count_beyond_known_base"], 0)

    def test_treatments_bind_expected_independent_sources(self) -> None:
        self.assertEqual(
            TREATMENTS["SS"], {"policy_source": "seed455", "value_source": "seed455"}
        )
        self.assertEqual(
            TREATMENTS["FS"],
            {"policy_source": "original_o0_e4", "value_source": "seed455"},
        )
        self.assertEqual(
            TREATMENTS["SF"],
            {"policy_source": "seed455", "value_source": "original_o0_e4"},
        )
        self.assertEqual(
            TREATMENTS["FF"],
            {"policy_source": "original_o0_e4", "value_source": "original_o0_e4"},
        )

    def test_composed_evaluator_skips_networks_at_terminal_positions(self) -> None:
        current = Mock()
        candidate = Mock()
        evaluator = arena.ComposedArtifactEvaluator(
            current, candidate, policy_source="current", value_source="candidate"
        )
        terminal = KalahGame.from_state(
            {
                "player_pits": [0] * 6,
                "opponent_pits": [0] * 6,
                "player_store": 24,
                "opponent_store": 24,
                "current_player": 0,
            }
        )
        policy, value = evaluator.evaluate(terminal)
        np.testing.assert_array_equal(policy, np.zeros(6, dtype=np.float32))
        self.assertEqual(value, 0.0)
        current.evaluate.assert_not_called()
        candidate.evaluate.assert_not_called()

    def test_changed_component_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "artifact_identity_mismatch"):
            require_hash("changed", "registered", "E4 model")

    def test_partial_and_unbound_recovery_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "partial_or_unbound"):
            cached_evidence_mode(True, False, {"state": "running"})
        with self.assertRaisesRegex(ValueError, "partial_or_unbound"):
            cached_evidence_mode(True, True, None)
        self.assertEqual(
            cached_evidence_mode(True, True, {"state": "running"}), "recover"
        )

    def test_incorrect_component_command_identity_is_rejected(self) -> None:
        treatment = {
            "policy_artifact": "e4-artifact",
            "value_artifact": "seed455-artifact",
        }
        command = [
            "arena.py",
            "--challenger-policy-artifact",
            "wrong-policy",
            "--challenger-value-artifact",
            "seed455-artifact",
            "--current-policy-artifact",
            str(SEED455_ARTIFACT),
            "--current-value-artifact",
            str(SEED455_ARTIFACT),
        ]
        with self.assertRaisesRegex(ValueError, "command_component_source_mismatch"):
            validate_command_source_selection(command, treatment)

    def test_malformed_composition_report_is_rejected(self) -> None:
        registration = {
            "treatments": {
                "FS": {
                    "policy_artifact": "e4",
                    "value_artifact": "seed455",
                    "policy_source": "original_o0_e4",
                    "value_source": "seed455",
                }
            },
            "evaluation": {"suite": {"sha256": "suite"}},
        }
        with self.assertRaisesRegex(ValueError, "report_composition_identity_mismatch"):
            validate_treatment_evidence(
                "FS",
                {"notes": {}},
                [],
                [],
                registration,
                {"native_runtime_contract": {}},
            )

    def test_observed_consumed_state_hash_is_not_replaced_by_suite_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            row = {
                "opening_prefix_moves": [0, 1],
                "opening_applied_prefix_length": 2,
                "opening_state_hash": "incorrect-observed-hash",
                "opening_contract": "arena_player_relative_v2",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "consumed_state_hash_mismatch"):
                actual_consumed_identities(path)

    def test_run_refuses_to_launch_without_frozen_registration(self) -> None:
        with patch(
            "ml.alphazero_lite.seed398_composition_diagnostic.read_json",
            side_effect=FileNotFoundError,
        ):
            with patch(
                "ml.alphazero_lite.seed398_composition_diagnostic.subprocess.run"
            ) as launch:
                from ml.alphazero_lite.seed398_composition_diagnostic import run

                with self.assertRaises(FileNotFoundError):
                    run()
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
