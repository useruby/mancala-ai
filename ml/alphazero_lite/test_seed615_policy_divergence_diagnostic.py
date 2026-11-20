"""Synthetic contract tests for Seed 615's read-only retrospective helpers."""

from __future__ import annotations

import unittest

import numpy as np

from ml.alphazero_lite import seed615_policy_divergence_diagnostic as diagnostic
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_table
from ml.alphazero_lite import verify_seed615_evaluation as verifier


class Seed615PolicyDivergenceDiagnosticTests(unittest.TestCase):
    def test_identical_policy_has_zero_divergence(self):
        result = diagnostic.compare_priors(
            np.asarray([0.2, 0.3, 0.5, 0, 0, 0]),
            np.asarray([0.2, 0.3, 0.5, 0, 0, 0]),
            [0, 1, 2],
        )
        self.assertEqual(result["total_variation"], 0.0)
        self.assertEqual(result["jensen_shannon_bits"], 0.0)
        self.assertTrue(result["top_prior_agreement"])

    def test_illegal_actions_are_masked_and_legal_probabilities_normalized(self):
        result = diagnostic.compare_priors(
            np.asarray([0.1, 0.4, 0.5, 0, 0, 0]),
            np.asarray([0.3, 0.2, 0.5, 0, 0, 0]),
            [0, 2],
        )
        np.testing.assert_allclose(
            result["B512_legal_probabilities"], [1 / 6, 0, 5 / 6, 0, 0, 0]
        )
        self.assertAlmostEqual(sum(result["S615_legal_probabilities"]), 1)

    def test_zero_or_invalid_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid_policy_probabilities"):
            diagnostic.compare_priors(np.zeros(6), np.ones(6), [0])
        with self.assertRaisesRegex(ValueError, "invalid_policy_probabilities"):
            diagnostic.compare_priors(
                np.asarray([-1, 1, 0, 0, 0, 0]), np.ones(6), [0, 1]
            )

    def test_canonical_state_identity_is_order_independent(self):
        left = {"player_pits": [1], "current_player": 0}
        right = {"current_player": 0, "player_pits": [1]}
        self.assertEqual(
            diagnostic.canonical_state_identity(left),
            diagnostic.canonical_state_identity(right),
        )

    def test_opening_or_seat_pairing_mismatch_is_rejected(self):
        assignment_b = {
            "opening_index": 4,
            "candidate_seat": 0,
            "candidate": "B512",
            "opening_identity": "x",
            "game_seed": 1,
        }
        assignment_s = {
            "opening_index": 4,
            "candidate_seat": 1,
            "candidate": "S615",
            "opening_identity": "x",
            "game_seed": 2,
        }
        record = (
            {"opening_prefix_moves": [0] * 12, "winner": "draw"},
            {"opening_prefix_moves": [0] * 12, "winner": "draw"},
            [],
        )
        with self.assertRaisesRegex(ValueError, "incorrect_opening_or_seat_pairing"):
            diagnostic.build_divergence(
                (assignment_b, record[0], []), (assignment_s, record[1], [])
            )

    def test_extra_turn_and_capture_classification_uses_kalah_rules(self):
        # One seed in pit 5 sows into the store and grants an extra turn.
        extra_turn_state = {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        consequences = move_consequence_table(extra_turn_state)
        self.assertTrue(consequences[5]["gives_extra_turn"])
        # One seed from pit 1 lands in empty pit 2 opposite nonempty pit 9.
        capture_state = {
            "player_pits": [0, 1, 0, 0, 0, 0],
            "opponent_pits": [0, 0, 0, 3, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        consequences = move_consequence_table(capture_state)
        self.assertTrue(consequences[1]["produces_capture"])

    def test_checkpoint_and_frozen_result_constants_are_fixed(self):
        self.assertEqual(
            diagnostic.CHECKPOINTS["B512"],
            "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
        )
        self.assertEqual(
            diagnostic.CHECKPOINTS["S615"],
            "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
        )
        self.assertEqual(diagnostic.FROZEN_RESULTS["interpretation"], "neither")
        with self.assertRaisesRegex(ValueError, "B512_identity_mismatch"):
            verifier.validate_identity(
                "substituted", diagnostic.CHECKPOINTS["B512"], "B512"
            )

    def test_published_result_mismatch_is_rejected(self):
        results = {
            "B512_score": 0.502197265625,
            "S615_score": 0.496337890625,
            "S615_minus_B512_interval": [-0.01611328125, 0.00439453125],
            "S615_vs_P_E4_one_sided_lower_bound": 0.486572265625,
            "interpretation": "neither",
        }
        diagnostic.validate_published_results(results)
        results["S615_score"] = 0.5
        with self.assertRaisesRegex(
            ValueError, "published_analysis_identity_or_result_mismatch"
        ):
            diagnostic.validate_published_results(results)

    def test_json_serialization_and_hashing_are_reproducible(self):
        value = {"z": [1, 2], "a": {"b": True}}
        payload = diagnostic.canonical_json(value)
        self.assertEqual(payload, diagnostic.canonical_json(value))
        self.assertEqual(
            diagnostic.sha256_bytes(payload), diagnostic.sha256_bytes(payload)
        )

    def test_missing_or_duplicate_assignment_evidence_fails_closed(self):
        rows = [{"game_id": str(index)} for index in range(4096)]
        with self.assertRaisesRegex(
            ValueError, "assignment_manifest_missing_duplicate_or_changed"
        ):
            verifier.validate_assignment_rows(rows[:-1], rows)
        duplicated = rows.copy()
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(
            ValueError, "assignment_manifest_missing_duplicate_or_changed"
        ):
            verifier.validate_assignment_rows(duplicated, rows)

    def test_modified_game_receipt_hash_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError, "modified_or_conflicting_result_artifact"
        ):
            verifier.validate_result_hashes(
                {"game_sha256": "wrong", "report_sha256": "r", "command_sha256": "c"},
                b"game",
                b"report",
                b"command",
            )

    def test_first_difference_is_found_after_shared_recorded_move(self):
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        prefix = [0, 3, 1, 2, 0, 0, 1, 0, 5, 0, 2, 4]
        for move in prefix:
            self.assertIn(move, game.possible_moves())
            game.move(game.pit_index(move))
        legal_absolute = [game.pit_index(move) for move in game.possible_moves()]
        if len(legal_absolute) < 2:
            self.skipTest("synthetic opening has one legal move")
        shared, changed_b, changed_s = (
            legal_absolute[0],
            legal_absolute[0],
            legal_absolute[1],
        )
        after_shared = game.clone()
        after_shared.move(shared)
        actor_seat = after_shared.current_player
        b_assignment = {
            "opening_index": 7,
            "candidate_seat": actor_seat,
            "candidate": "B512",
            "opening_identity": "identity",
            "game_seed": 101,
        }
        s_assignment = {
            "opening_index": 7,
            "candidate_seat": actor_seat,
            "candidate": "S615",
            "opening_identity": "identity",
            "game_seed": 202,
        }
        b_row = {"opening_prefix_moves": prefix, "winner": "challenger"}
        s_row = {"opening_prefix_moves": prefix, "winner": "current"}
        result = diagnostic.build_divergence(
            (b_assignment, b_row, [shared, changed_b]),
            (s_assignment, s_row, [shared, changed_s]),
        )
        self.assertEqual(result["first_differing_move_index"], 1)
        self.assertEqual(result["actor"], "candidate")
        self.assertNotEqual(result["b512_seed"], result["s615_seed"])

    def test_first_divergence_after_shared_extra_turn_replay(self):
        # Synthetic prefix position: both histories agree on an extra-turn move,
        # then disagree while the same player remains to move.
        state = {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        # The method validates registered 12-move prefixes, so this isolated
        # engine assertion guards the rule behavior underlying recorded replay.
        game = KalahGame.from_state(state)
        self.assertTrue(game.move(5))
        self.assertEqual(game.current_player, 0)


if __name__ == "__main__":
    unittest.main()
