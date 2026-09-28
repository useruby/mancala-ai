import unittest

from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.seed455_vs_seed461_fresh_selfplay_distribution import (
    active_bucket,
    aggregate_targets,
    diversity,
    ply_bucket,
    simpson_effective,
    source_seed,
)


class Seed455VsSeed461FreshDistributionTest(unittest.TestCase):
    def test_canonical_identity_retains_player_perspective(self) -> None:
        state = {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        self.assertNotEqual(
            canonical_state_key(state),
            canonical_state_key({**state, "current_player": 1}),
        )

    def test_pre_registered_buckets_and_primary_region_boundary(self) -> None:
        self.assertEqual(">40", active_bucket(41))
        self.assertEqual("33-40", active_bucket(33))
        self.assertEqual("<=16", active_bucket(16))
        self.assertEqual("0-4", ply_bucket(4))
        self.assertEqual("5-8", ply_bucket(5))
        self.assertEqual(">12", ply_bucket(13))

    def test_simpson_and_duplicate_concentration_are_deterministic(self) -> None:
        self.assertAlmostEqual(1.8, simpson_effective([2, 1]))
        rows = [{"_key": "a"}, {"_key": "a"}, {"_key": "b"}]
        self.assertEqual(2, diversity(rows)["unique_states"])
        self.assertEqual(2 / 3, diversity(rows)["largest_state_frequency"])

    def test_source_seed_assignment_matches_self_play_pooling(self) -> None:
        self.assertEqual(454, source_seed(0, (454, 455, 456)))
        self.assertEqual(455, source_seed(1, (454, 455, 456)))
        self.assertEqual(454, source_seed(3, (454, 455, 456)))

    def test_shared_target_aggregation_means_duplicate_rows(self) -> None:
        rows = [
            {
                "_key": "state",
                "_active": 40,
                "move_index": 2,
                "policy": [1, 0, 0, 0, 0, 0],
                "value": 1,
            },
            {
                "_key": "state",
                "_active": 40,
                "move_index": 2,
                "policy": [0, 1, 0, 0, 0, 0],
                "value": -1,
            },
            {
                "_key": "late",
                "_active": 40,
                "move_index": 13,
                "policy": [1, 0, 0, 0, 0, 0],
                "value": 1,
            },
            {
                "_key": "end",
                "_active": 16,
                "move_index": 2,
                "policy": [1, 0, 0, 0, 0, 0],
                "value": 1,
            },
        ]
        summary = aggregate_targets(rows)
        self.assertEqual(2, summary["state"]["count"])
        self.assertEqual([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], summary["state"]["policy"])
        self.assertEqual(0.0, summary["state"]["value"])
        self.assertEqual({"state"}, set(summary))


if __name__ == "__main__":
    unittest.main()
