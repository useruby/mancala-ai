"""Negative regression tests for full-trajectory historical coverage replay."""

from __future__ import annotations

import unittest

from ml.alphazero_lite.verify_policy_search_historical_coverage_attempt_0002 import (
    ReplayBoard,
    assign_decision_sources,
    pair_occurrences,
    role_for_player,
    unique_source_index,
)


def occurrence(candidate: str, ply: int, identity: str = "state") -> dict:
    return {
        "canonical_identity": identity,
        "candidate": candidate,
        "candidate_seat": 0,
        "opening_id": "opening-1",
        "opening_identity": "opening-identity",
        "opening_index": 1,
        "game_id": f"{candidate}-{ply}",
        "game_seed": ply + 10,
        "ply": ply,
        "actor_role": "candidate",
        "recorded_action": ply % 6,
        "decision_source": "mcts",
        "has_both_raw_policies": True,
        "has_exact_action_margins": True,
    }


class HistoricalCoverageAttemptTwoTests(unittest.TestCase):
    def test_duplicate_seed633_canonical_identity_is_rejected(self) -> None:
        rows = [
            {"canonical_identity": "same-state"},
            {"canonical_identity": "same-state"},
        ]
        with self.assertRaisesRegex(ValueError, "policy_duplicate_identity"):
            unique_source_index(rows, label="policy")

    def test_incomplete_candidate_intersection_does_not_make_a_pair(self) -> None:
        pairs = pair_occurrences([occurrence("B512", 4)])
        self.assertEqual([], pairs)

    def test_duplicate_root_occurrences_pair_one_to_one_without_cartesian_join(
        self,
    ) -> None:
        b512 = [occurrence("B512", 3), occurrence("B512", 8)]
        s615 = [occurrence("S615", 5), occurrence("S615", 9), occurrence("S615", 13)]
        pairs = pair_occurrences(b512 + s615)
        self.assertEqual(2, len(pairs))
        self.assertEqual([0, 1], [pair["occurrence_ordinal"] for pair in pairs])
        self.assertEqual(
            {"B512-3", "B512-8"}, {pair["B512"]["game_id"] for pair in pairs}
        )

    def test_opponent_turn_is_not_attributed_to_candidate_checkpoint(self) -> None:
        self.assertEqual(
            "candidate", role_for_player(current_player=0, candidate_seat=0)
        )
        self.assertEqual(
            "opponent", role_for_player(current_player=1, candidate_seat=0)
        )

    def test_opponent_occurrence_cannot_pair_as_candidate_search_action(self) -> None:
        opponent = occurrence("B512", 3)
        opponent["actor_role"] = "opponent"
        self.assertEqual([], pair_occurrences([opponent, occurrence("S615", 3)]))

    def test_exact_root_eligibility_without_handoff_is_unknown_not_mcts(self) -> None:
        rows = [
            {
                "ply": 7,
                "actor_role": "candidate",
                "exact_root_eligible": True,
                "decision_source": "unknown",
            }
        ]
        assign_decision_sources(rows, [7], handoff_count=0, first_handoff=None)
        self.assertEqual("unknown", rows[0]["decision_source"])

    def test_recorded_solver_handoff_is_not_classified_as_mcts(self) -> None:
        rows = [
            {
                "ply": 7,
                "actor_role": "candidate",
                "exact_root_eligible": True,
                "decision_source": "unknown",
            }
        ]
        assign_decision_sources(rows, [7], handoff_count=1, first_handoff=7)
        self.assertEqual("exact_root_solver", rows[0]["decision_source"])

    def test_incomplete_handoff_telemetry_identifies_only_reported_first_solver_action(
        self,
    ) -> None:
        rows = [
            {
                "ply": ply,
                "actor_role": "candidate",
                "exact_root_eligible": True,
                "decision_source": "unknown",
            }
            for ply in (7, 9)
        ]
        assign_decision_sources(rows, [7, 9], handoff_count=1, first_handoff=7)
        self.assertEqual(
            ["exact_root_solver", "unknown"],
            [row["decision_source"] for row in rows],
        )

    def test_illegal_player_perspective_action_is_rejected(self) -> None:
        board = ReplayBoard()
        with self.assertRaisesRegex(ValueError, "illegal_action_owner_or_empty_pit"):
            board.play_absolute(6)


if __name__ == "__main__":
    unittest.main()
