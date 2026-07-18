import unittest

from ml.alphazero_lite.evaluation_seed_contract import seed_identity_ledger_record
from ml.alphazero_lite.run_canonical_runtime_profile_revalidation import (
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
    PROFILE_D,
    PROFILES,
    bootstrap_ci,
    opening_ds,
    paired_delta,
    profile_definition,
)


class CanonicalRuntimeProfileRevalidationTests(unittest.TestCase):
    def test_profiles_are_exact_promoted_two_by_two(self):
        self.assertEqual(0.10, PROFILES[PROFILE_A]["tactical_root_bias"])
        self.assertEqual({}, PROFILES[PROFILE_A]["schedule"])
        self.assertEqual(0.00, PROFILES[PROFILE_B]["tactical_root_bias"])
        self.assertEqual({}, PROFILES[PROFILE_B]["schedule"])
        self.assertEqual(0.10, PROFILES[PROFILE_C]["tactical_root_bias"])
        self.assertEqual({"768:768": 0.90}, PROFILES[PROFILE_C]["schedule"])
        self.assertEqual(0.00, PROFILES[PROFILE_D]["tactical_root_bias"])
        self.assertEqual({"768:768": 0.90}, PROFILES[PROFILE_D]["schedule"])

    def test_profile_hash_is_runtime_configuration_specific(self):
        self.assertNotEqual(
            profile_definition(PROFILE_A)["hash"], profile_definition(PROFILE_D)["hash"]
        )

    def test_seed_identity_excludes_artifact_and_outcome_details(self):
        context = {
            "base_seed": 42,
            "suite_sha256": "suite",
            "budget_pair": "384:256",
            "opening_index": 0,
            "opening_state_hash": "opening",
            "challenger_player": 0,
            "game_within_opening": 3,
            "ply": 2,
            "canonical_current_state_hash": "state",
            "acting_role": "challenger",
            "simulations": 384,
            "effective_c_puct": 1.25,
        }
        first = seed_identity_ledger_record(**context)
        second = seed_identity_ledger_record(**context)
        self.assertEqual(first, second)
        self.assertEqual(3, first["game_within_opening"])
        self.assertNotIn("selected_move", first)
        self.assertNotIn("artifact_hash", first)
        self.assertNotIn("cache_hit", first)

    def test_paired_delta_is_clustered_by_opening(self):
        left = [
            {
                "game_index": 0,
                "challenger_player": 0,
                "winner": "challenger",
                "margin": 1,
                "game_length": 1,
                "trajectory": "a",
            },
            {
                "game_index": 1,
                "challenger_player": 1,
                "winner": "current",
                "margin": -1,
                "game_length": 1,
                "trajectory": "b",
            },
        ]
        right = [{**entry, "winner": "draw", "margin": 0} for entry in left]
        result = paired_delta(left, right, seed=42)
        self.assertEqual(1, len(result["paired_per_opening_delta"]))
        self.assertEqual(1, result["positive_openings"])
        self.assertEqual(0, result["negative_openings"])
        self.assertEqual(1.0, opening_ds(left)[0])

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(
            bootstrap_ci([0.0, 1.0, -1.0], seed=42),
            bootstrap_ci([0.0, 1.0, -1.0], seed=42),
        )


if __name__ == "__main__":
    unittest.main()
