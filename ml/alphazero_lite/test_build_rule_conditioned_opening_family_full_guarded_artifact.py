from __future__ import annotations

import unittest


class BuildRuleConditionedOpeningFamilyFullGuardedArtifactTest(unittest.TestCase):
    def test_build_rows_merges_opening_family_and_rule_collision_guards(self) -> None:
        from ml.alphazero_lite import (
            build_rule_conditioned_opening_family_full_guarded_artifact as module,
        )

        opening_family_rows = [
            {
                "teacher_selected_move": 4,
                "source_runs": [{"id": "capture_available-005"}],
                "replay_role": "opening_capture_no_extra_turn_reference",
                "reference_move_extra_turn_available": False,
            },
            {
                "teacher_selected_move": 2,
                "source_runs": [{"id": "capture_available-006"}],
                "replay_role": "opening_capture_extra_turn_reference",
                "reference_move_extra_turn_available": True,
            },
        ]
        rule_collision_guard_rows = [
            {
                "teacher_selected_move": 4,
                "source_runs": [{"id": "capture_available-002"}],
                "replay_role": "rule_collision_no_extra_turn_reference_guard",
                "reference_move_extra_turn_available": False,
            },
            {
                "teacher_selected_move": 1,
                "source_runs": [{"id": "capture_available-003"}],
                "replay_role": "rule_collision_extra_turn_reference_guard",
                "reference_move_extra_turn_available": True,
            },
        ]

        rows, summary = module.build_rows(
            opening_family_rows=opening_family_rows,
            rule_collision_guard_rows=rule_collision_guard_rows,
        )

        self.assertEqual(4, len(rows))
        self.assertEqual(2, summary["rule_collision_guard_count"])
        self.assertEqual(
            ["capture_available-005", "capture_available-006"],
            summary["tracked_opening_row_ids"],
        )
        self.assertEqual(
            ["capture_available-002", "capture_available-003"],
            summary["rule_collision_guard_row_ids"],
        )
        self.assertEqual(
            {
                "opening_capture_extra_turn_reference": 1,
                "opening_capture_no_extra_turn_reference": 1,
                "rule_collision_extra_turn_reference_guard": 1,
                "rule_collision_no_extra_turn_reference_guard": 1,
            },
            summary["replay_role_counts"],
        )


if __name__ == "__main__":
    unittest.main()
