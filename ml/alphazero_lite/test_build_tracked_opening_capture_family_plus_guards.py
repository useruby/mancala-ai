from __future__ import annotations

import unittest


class BuildTrackedOpeningCaptureFamilyPlusGuardsTest(unittest.TestCase):
    def test_build_rows_replaces_old_opening_rows_with_tracked_rows(self) -> None:
        from ml.alphazero_lite import build_tracked_opening_capture_family_plus_guards as module

        tracked_rows = [
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
        guard_rows = [
            {"replay_role": "capture_protection", "teacher_selected_move": 1},
            {"replay_role": "opening_capture_family", "teacher_selected_move": 3},
            {"replay_role": "nearby_preservation", "teacher_selected_move": 0},
        ]

        rows, summary = module.build_rows(tracked_rows=tracked_rows, guard_rows=guard_rows)

        self.assertEqual(4, len(rows))
        self.assertEqual(1, summary["replay_role_counts"]["opening_capture_extra_turn_reference"])
        self.assertEqual(1, summary["replay_role_counts"]["opening_capture_no_extra_turn_reference"])
        self.assertEqual(["capture_available-005", "capture_available-006"], summary["tracked_row_ids"])
        self.assertEqual({0: 1, 1: 1, 2: 1, 4: 1}, summary["teacher_selected_move_distribution"])
        self.assertEqual(
            {
                "opening_capture_extra_turn_reference": 1,
                "opening_capture_no_extra_turn_reference": 1,
            },
            summary["tracked_replay_role_counts"],
        )
        self.assertEqual(
            {
                "opening_capture_extra_turn_reference": ["capture_available-006"],
                "opening_capture_no_extra_turn_reference": ["capture_available-005"],
            },
            summary["row_ids_by_replay_role"],
        )
        self.assertEqual(
            {
                "opening_capture_extra_turn_reference": {2: 1},
                "opening_capture_no_extra_turn_reference": {4: 1},
            },
            summary["teacher_selected_move_distribution_by_replay_role"],
        )
        self.assertEqual(
            {"capture_available-005": False, "capture_available-006": True},
            summary["reference_extra_turn_by_row"],
        )
        self.assertEqual(2, summary["guard_row_count"])


if __name__ == "__main__":
    unittest.main()
