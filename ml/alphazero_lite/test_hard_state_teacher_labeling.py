import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ml.alphazero_lite import hard_state_teacher_labeling as labeling
from ml.alphazero_lite import train


class HardStateTeacherLabelingTest(unittest.TestCase):
    def sample_mined_row(self, **overrides):
        row = {
            "canonical_state": "4,4,4,4,4,4|4,4,4,4,4,4|0|0|0",
            "state": {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            "legal_moves": [0, 1, 2, 3, 4, 5],
            "priority_score": 13.5,
            "selection_reasons": ["large_value_error", "student_teacher_disagreement"],
            "source_artifacts": ["/tmp/mined.jsonl"],
        }
        row.update(overrides)
        return row

    def test_select_top_ranked_rows_preserves_existing_artifact_order(self):
        rows = [
            self.sample_mined_row(canonical_state="state-b", priority_score=20.0),
            self.sample_mined_row(canonical_state="state-a", priority_score=19.0),
            self.sample_mined_row(canonical_state="state-c", priority_score=18.0),
        ]

        selected = labeling.select_top_ranked_rows(rows, top_n=2)

        self.assertEqual(["state-b", "state-a"], [row["canonical_state"] for row in selected])

    def test_select_top_ranked_rows_uses_normalized_top_n_for_slicing(self):
        rows = [
            self.sample_mined_row(canonical_state="state-b", priority_score=20.0),
            self.sample_mined_row(canonical_state="state-a", priority_score=19.0),
            self.sample_mined_row(canonical_state="state-c", priority_score=18.0),
        ]

        selected = labeling.select_top_ranked_rows(rows, top_n="2")

        self.assertEqual(["state-b", "state-a"], [row["canonical_state"] for row in selected])

    def test_select_top_ranked_rows_rejects_invalid_top_n_types(self):
        invalid_top_n_values = [
            (True, "top_n must be an integer"),
            (2.5, "top_n must be an integer"),
            ("two", "top_n must be an integer"),
        ]

        rows = [self.sample_mined_row()]

        for top_n, expected_error in invalid_top_n_values:
            with self.subTest(top_n=top_n):
                with self.assertRaisesRegex(ValueError, expected_error):
                    labeling.select_top_ranked_rows(rows, top_n=top_n)

    def test_build_dual_budget_rows_emits_canonical_and_stronger_profiles(self):
        mined_row = self.sample_mined_row()
        canonical_result = {
            "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "value": 0.25,
        }
        stronger_result = {
            "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "value": 0.75,
        }

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[canonical_result, stronger_result],
        ):
            rows = labeling.build_dual_budget_rows(
                [mined_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        self.assertEqual(2, len(rows))
        self.assertEqual(["canonical", "stronger"], [row["teacher_profile"] for row in rows])
        self.assertEqual([128, 512], [row["teacher_budget"] for row in rows])
        self.assertEqual(rows[0]["budget_pair_id"], rows[1]["budget_pair_id"])
        self.assertEqual("default", rows[0]["policy_target_mode"])
        self.assertEqual("default", rows[0]["value_target_mode"])
        self.assertEqual(["large_value_error", "student_teacher_disagreement"], rows[0]["selection_reasons"])
        self.assertEqual(13.5, rows[0]["source_priority_score"])
        self.assertEqual([0, 1, 2, 3, 4, 5], rows[0]["legal_moves"])

    def test_build_dual_budget_rows_accepts_rows_without_source_artifacts(self):
        mined_row = self.sample_mined_row()
        del mined_row["source_artifacts"]

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ):
            rows = labeling.build_dual_budget_rows(
                [mined_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        self.assertEqual(
            [
                "direct_input:4,4,4,4,4,4|4,4,4,4,4,4|0|0|0:rank1",
                "direct_input:4,4,4,4,4,4|4,4,4,4,4,4|0|0|0:rank1",
            ],
            [row["source_artifact"] for row in rows],
        )
        self.assertEqual(["large_value_error", "student_teacher_disagreement"], rows[0]["selection_reasons"])

    def test_build_dual_budget_rows_prefers_singular_source_artifact(self):
        mined_row = self.sample_mined_row(
            source_artifact="/tmp/labeling-input.jsonl",
            source_artifacts=["/tmp/mined.jsonl"],
        )

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ):
            rows = labeling.build_dual_budget_rows(
                [mined_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        self.assertEqual(["/tmp/labeling-input.jsonl", "/tmp/labeling-input.jsonl"], [row["source_artifact"] for row in rows])

    def test_build_dual_budget_rows_falls_back_to_plural_source_artifacts(self):
        mined_row = self.sample_mined_row()

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ):
            rows = labeling.build_dual_budget_rows(
                [mined_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        self.assertEqual(["/tmp/mined.jsonl", "/tmp/mined.jsonl"], [row["source_artifact"] for row in rows])

    def test_build_dual_budget_rows_uses_stable_seed_not_source_rank(self):
        first_row = self.sample_mined_row(
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 1,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        second_row = self.sample_mined_row(
            canonical_state="state-b",
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 2,
                "opponent_store": 0,
                "current_player": 0,
            },
        )

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.1},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.2},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.3},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.4},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.5},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.6},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.7},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.8},
            ],
        ) as run_teacher_label:
            labeling.build_dual_budget_rows(
                [first_row, second_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )
            seeds_by_state_in_original_order = {
                call.args[0]["player_store"]: [
                    current_call.kwargs["seed"]
                    for current_call in run_teacher_label.call_args_list[index * 2 : (index * 2) + 2]
                ]
                for index, call in enumerate(run_teacher_label.call_args_list[::2])
            }

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.1},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.2},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.3},
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.4},
            ],
        ) as reordered_run_teacher_label:
            labeling.build_dual_budget_rows(
                [second_row, first_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )
            seeds_by_state_in_reordered_run = {
                call.args[0]["player_store"]: [
                    current_call.kwargs["seed"]
                    for current_call in reordered_run_teacher_label.call_args_list[index * 2 : (index * 2) + 2]
                ]
                for index, call in enumerate(reordered_run_teacher_label.call_args_list[::2])
            }

        self.assertEqual(
            seeds_by_state_in_original_order[1],
            seeds_by_state_in_reordered_run[1],
        )
        self.assertEqual(
            seeds_by_state_in_original_order[2],
            seeds_by_state_in_reordered_run[2],
        )

    def test_derive_value_from_selected_move_win_rate_requires_child_stat_for_selected_move(self):
        summary = {
            "selected_move": 4,
            "child_stats": [
                {"move": 0, "win_rate": 0.25},
                {"move": 1, "win_rate": 0.5},
            ],
        }

        with self.assertRaisesRegex(ValueError, "selected_move 4 missing from child_stats"):
            labeling.derive_value_from_selected_move_win_rate(summary)

    def test_run_teacher_label_emits_one_hot_policy_for_single_legal_move(self):
        label = labeling.run_teacher_label(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            teacher_budget=128,
            teacher_mode="classic_mcts",
            seed=41,
        )

        self.assertEqual([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], label["policy"])
        self.assertEqual(0.0, label["value"])

    def test_load_hard_state_rows_rejects_missing_required_fields(self):
        row = self.sample_mined_row()
        del row["legal_moves"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "hard_states.jsonl"
            dataset_path.write_text(f"{labeling.json.dumps(row)}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required field 'legal_moves'"):
                labeling.load_hard_state_rows(dataset_path)

    def test_load_hard_state_rows_rejects_invalid_required_field_types(self):
        row = self.sample_mined_row(selection_reasons="not-a-list")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "hard_states.jsonl"
            dataset_path.write_text(f"{labeling.json.dumps(row)}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "selection_reasons must be a list"):
                labeling.load_hard_state_rows(dataset_path)

    def test_validate_hard_state_row_rejects_non_string_source_artifacts_entries(self):
        invalid_source_artifacts = [
            ["/tmp/mined.jsonl", 7],
            ["/tmp/mined.jsonl", None],
            ["/tmp/mined.jsonl", False],
        ]

        for source_artifacts in invalid_source_artifacts:
            with self.subTest(source_artifacts=source_artifacts):
                with self.assertRaisesRegex(ValueError, r"source_artifacts\[1\] must be a string"):
                    labeling.validate_hard_state_row(
                        self.sample_mined_row(source_artifacts=source_artifacts),
                        source="row",
                    )

    def test_validate_hard_state_row_rejects_malformed_selection_reasons_entries(self):
        invalid_selection_reasons = [
            (["large_value_error", 7], r"selection_reasons\[1\] must be a non-empty string"),
            (["large_value_error", None], r"selection_reasons\[1\] must be a non-empty string"),
            (["large_value_error", False], r"selection_reasons\[1\] must be a non-empty string"),
            (["large_value_error", ""], r"selection_reasons\[1\] must be a non-empty string"),
        ]

        for selection_reasons, expected_error in invalid_selection_reasons:
            with self.subTest(selection_reasons=selection_reasons):
                with self.assertRaisesRegex(ValueError, expected_error):
                    labeling.validate_hard_state_row(
                        self.sample_mined_row(selection_reasons=selection_reasons),
                        source="row",
                    )

    def test_validate_hard_state_row_rejects_malformed_state_shape(self):
        row = self.sample_mined_row(
            state={
                "player_pits": [4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        with self.assertRaisesRegex(ValueError, "state player_pits must be a list of 6 integers"):
            labeling.validate_hard_state_row(row, source="row")

    def test_validate_hard_state_row_rejects_boolean_state_values(self):
        invalid_rows = [
            (
                self.sample_mined_row(
                    state={
                        "player_pits": [4, 4, True, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 0,
                    }
                ),
                "state player_pits must be a list of 6 integers",
            ),
            (
                self.sample_mined_row(
                    state={
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": True,
                        "opponent_store": 0,
                        "current_player": 0,
                    }
                ),
                "state player_store must be an integer",
            ),
            (
                self.sample_mined_row(
                    state={
                        "player_pits": [4, 4, 4, 4, 4, 4],
                        "opponent_pits": [4, 4, 4, 4, 4, 4],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": False,
                    }
                ),
                "state current_player must be an integer 0 or 1",
            ),
        ]

        for row, expected_error in invalid_rows:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    labeling.validate_hard_state_row(row, source="row")

    def test_validate_hard_state_row_rejects_float_current_player(self):
        row = self.sample_mined_row(
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0.0,
            }
        )

        with self.assertRaisesRegex(ValueError, "state current_player must be an integer 0 or 1"):
            labeling.validate_hard_state_row(row, source="row")

    def test_validate_hard_state_row_rejects_invalid_legal_moves(self):
        invalid_legal_moves = [
            ([0, 1, "2"], r"legal_moves\[2\] must be an integer"),
            ([0, True], r"legal_moves\[1\] must be an integer"),
            ([0, 6], r"legal_moves\[1\] must be between 0 and 5"),
            ([0, 0], "legal_moves must not contain duplicates"),
            ([], "legal_moves must not be empty"),
        ]

        for legal_moves, expected_error in invalid_legal_moves:
            with self.subTest(legal_moves=legal_moves):
                with self.assertRaisesRegex(ValueError, expected_error):
                    labeling.validate_hard_state_row(
                        self.sample_mined_row(legal_moves=legal_moves),
                        source="row",
                    )

    def test_validate_hard_state_row_rejects_legal_moves_that_do_not_match_state(self):
        row = self.sample_mined_row(
            state={
                "player_pits": [0, 4, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            legal_moves=[0, 1, 4],
        )

        with self.assertRaisesRegex(ValueError, r"legal_moves must match state-derived legal moves \[1, 4\]"):
            labeling.validate_hard_state_row(row, source="row")

    def test_validate_hard_state_row_rejects_non_finite_priority_scores(self):
        invalid_priority_scores = [float("nan"), float("inf"), float("-inf")]

        for priority_score in invalid_priority_scores:
            with self.subTest(priority_score=priority_score):
                with self.assertRaisesRegex(ValueError, "priority_score must be finite"):
                    labeling.validate_hard_state_row(
                        self.sample_mined_row(priority_score=priority_score),
                        source="row",
                    )

    def test_load_hard_state_rows_annotates_dataset_path_as_source_artifact(self):
        row = self.sample_mined_row()
        del row["source_artifacts"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "hard_states.jsonl"
            dataset_path.write_text(f"{labeling.json.dumps(row)}\n", encoding="utf-8")

            loaded_rows = labeling.load_hard_state_rows(dataset_path)

        self.assertEqual(str(dataset_path), loaded_rows[0]["source_artifact"])

    def test_load_hard_state_rows_preserves_prior_source_artifact_as_provenance(self):
        row = self.sample_mined_row(source_artifact="/tmp/original-labeling-input.jsonl")
        del row["source_artifacts"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "reloaded-hard_states.jsonl"
            dataset_path.write_text(f"{labeling.json.dumps(row)}\n", encoding="utf-8")

            loaded_rows = labeling.load_hard_state_rows(dataset_path)

        self.assertEqual(str(dataset_path), loaded_rows[0]["source_artifact"])
        self.assertEqual(["/tmp/original-labeling-input.jsonl"], loaded_rows[0]["source_artifacts"])

    def test_build_dual_budget_rows_uses_loaded_dataset_path_for_source_artifact(self):
        row = self.sample_mined_row()
        del row["source_artifacts"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "hard_states.jsonl"
            dataset_path.write_text(f"{labeling.json.dumps(row)}\n", encoding="utf-8")
            loaded_rows = labeling.load_hard_state_rows(dataset_path)

            with mock.patch.object(
                labeling,
                "run_teacher_label",
                side_effect=[
                    {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                    {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
                ],
            ):
                labeled_rows = labeling.build_dual_budget_rows(
                    loaded_rows,
                    canonical_budget=128,
                    stronger_budget=512,
                    teacher_mode="classic_mcts",
                    input_encoding="kalah_v3",
                    seed=41,
                )

        self.assertEqual([str(dataset_path), str(dataset_path)], [row["source_artifact"] for row in labeled_rows])

    def test_build_dual_budget_rows_rejects_non_positive_canonical_budget(self):
        with self.assertRaisesRegex(ValueError, "canonical_budget must be >= 1"):
            labeling.build_dual_budget_rows(
                [self.sample_mined_row()],
                canonical_budget=0,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

    def test_build_dual_budget_rows_rejects_non_positive_stronger_budget(self):
        with self.assertRaisesRegex(ValueError, "stronger_budget must be >= 1"):
            labeling.build_dual_budget_rows(
                [self.sample_mined_row()],
                canonical_budget=128,
                stronger_budget=-1,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

    def test_build_dual_budget_rows_uses_stable_seed_for_same_loaded_row_across_filenames(self):
        row = self.sample_mined_row()

        with tempfile.TemporaryDirectory() as tmpdir:
            first_dataset_path = Path(tmpdir) / "first-hard_states.jsonl"
            second_dataset_path = Path(tmpdir) / "second-hard_states.jsonl"
            payload = f"{labeling.json.dumps(row)}\n"
            first_dataset_path.write_text(payload, encoding="utf-8")
            second_dataset_path.write_text(payload, encoding="utf-8")

            first_loaded_rows = labeling.load_hard_state_rows(first_dataset_path)
            second_loaded_rows = labeling.load_hard_state_rows(second_dataset_path)

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ) as first_run_teacher_label:
            labeling.build_dual_budget_rows(
                first_loaded_rows,
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ) as second_run_teacher_label:
            labeling.build_dual_budget_rows(
                second_loaded_rows,
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        self.assertEqual(
            [call.kwargs["seed"] for call in first_run_teacher_label.call_args_list],
            [call.kwargs["seed"] for call in second_run_teacher_label.call_args_list],
        )

    def test_build_dual_budget_rows_uses_order_invariant_seed_identity_for_source_artifacts(self):
        first_row = self.sample_mined_row(source_artifacts=["b.json", "a.json"])
        second_row = self.sample_mined_row(source_artifacts=["a.json", "b.json"])

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ) as first_run_teacher_label:
            labeling.build_dual_budget_rows(
                [first_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ) as second_run_teacher_label:
            labeling.build_dual_budget_rows(
                [second_row],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        self.assertEqual(
            [call.kwargs["seed"] for call in first_run_teacher_label.call_args_list],
            [call.kwargs["seed"] for call in second_run_teacher_label.call_args_list],
        )

    def test_build_dual_budget_rows_copies_mutable_lists_per_profile(self):
        with mock.patch.object(
            labeling,
            "run_teacher_label",
            side_effect=[
                {"policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "value": 0.25},
                {"policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "value": 0.75},
            ],
        ):
            rows = labeling.build_dual_budget_rows(
                [self.sample_mined_row()],
                canonical_budget=128,
                stronger_budget=512,
                teacher_mode="classic_mcts",
                input_encoding="kalah_v3",
                seed=41,
            )

        rows[0]["legal_moves"].append(99)
        rows[0]["selection_reasons"].append("mutated")

        self.assertEqual([0, 1, 2, 3, 4, 5], rows[1]["legal_moves"])
        self.assertEqual(
            ["large_value_error", "student_teacher_disagreement"],
            rows[1]["selection_reasons"],
        )

    def test_label_hard_state_subset_cli_writes_dual_budget_rows(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-hard-state-subset-") as tmp:
            mined_path = Path(tmp) / "mined.jsonl"
            out_path = Path(tmp) / "labeled.jsonl"
            mined_row = self.sample_mined_row()
            mined_path.write_text(json.dumps(mined_row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/label_hard_state_subset.py",
                    "--mined-jsonl",
                    str(mined_path),
                    "--out",
                    str(out_path),
                    "--top-n",
                    "1",
                    "--canonical-budget",
                    "32",
                    "--stronger-budget",
                    "64",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual({"out": str(out_path), "rows": 2}, json.loads(result.stdout.strip()))
            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(2, len(rows))
            self.assertEqual({"canonical", "stronger"}, {row["teacher_profile"] for row in rows})

    def test_cli_output_loads_through_train_load_jsonl(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-hard-state-train-") as tmp:
            mined_path = Path(tmp) / "mined.jsonl"
            out_path = Path(tmp) / "labeled.jsonl"
            mined_row = self.sample_mined_row()
            mined_path.write_text(json.dumps(mined_row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/label_hard_state_subset.py",
                    "--mined-jsonl",
                    str(mined_path),
                    "--out",
                    str(out_path),
                    "--top-n",
                    "1",
                    "--canonical-budget",
                    "32",
                    "--stronger-budget",
                    "64",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual({"out": str(out_path), "rows": 2}, json.loads(result.stdout.strip()))

            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            x, p, v = train.load_jsonl(out_path)
            expected_state_width = len(rows[0]["state"])

            self.assertEqual((2, expected_state_width), x.shape)
            self.assertEqual((2, 6), p.shape)
            self.assertEqual((2, 1), v.shape)

    def test_run_teacher_label_rejects_invalid_teacher_budget(self):
        for teacher_budget, expected_message in [
            (0, r"teacher_budget must be >= 1"),
            (-1, r"teacher_budget must be >= 1"),
            (True, r"teacher_budget must be an integer"),
            (1.5, r"teacher_budget must be an integer"),
        ]:
            with self.subTest(teacher_budget=teacher_budget):
                with self.assertRaisesRegex(ValueError, expected_message):
                    labeling.run_teacher_label(
                        self.sample_mined_row()["state"],
                        teacher_budget=teacher_budget,
                        teacher_mode="classic_mcts",
                        seed=123,
                    )

    def test_build_comparison_report_measures_top1_disagreement_and_value_delta(self):
        rows = [
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "canonical",
                "teacher_budget": 32,
                "source_rank": 1,
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.2,
            },
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "stronger",
                "teacher_budget": 64,
                "source_rank": 1,
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.8,
            },
        ]

        report = labeling.build_comparison_report(rows)

        self.assertEqual(1, report["pair_count"])
        self.assertEqual(1.0, report["top1_disagreement_rate"])
        self.assertEqual(1.0, report["average_policy_divergence"])
        self.assertEqual(1.0, report["maximum_policy_divergence"])
        self.assertEqual(1, len(report["largest_disagreements"]))
        self.assertAlmostEqual(0.6, report["average_absolute_value_delta"])

    def test_build_comparison_report_handles_identical_labels_without_disagreement_or_delta(self):
        rows = [
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "canonical",
                "teacher_budget": 32,
                "source_rank": 1,
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.2,
            },
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "stronger",
                "teacher_budget": 64,
                "source_rank": 1,
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.2,
            },
        ]

        report = labeling.build_comparison_report(rows)

        self.assertEqual(1, report["pair_count"])
        self.assertEqual(0.0, report["top1_disagreement_rate"])
        self.assertEqual(0.0, report["average_policy_divergence"])
        self.assertEqual(0.0, report["maximum_policy_divergence"])
        self.assertEqual(0.0, report["average_absolute_value_delta"])
        self.assertEqual(
            [
                {
                    "budget_pair_id": "pair-a",
                    "canonical_state": "state-a",
                    "source_rank": 1,
                    "canonical_top_move": 1,
                    "stronger_top_move": 1,
                    "policy_divergence": 0.0,
                    "value_delta": 0.0,
                }
            ],
            report["largest_disagreements"],
        )

    def test_build_issue264_report_recommends_remain_selective_when_arena_fails_threshold(self):
        label_report = {
            "pair_count": 3,
            "top1_disagreement_rate": 0.333333,
            "average_policy_divergence": 0.25,
            "maximum_policy_divergence": 0.5,
            "average_absolute_value_delta": 0.1,
            "largest_disagreements": [],
        }
        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 5,
            "losses": 4,
            "draws": 1,
            "score": 0.55,
            "promotion_decision": {"passed": False},
            "hard_suite_buckets": {
                "opening": {"games": 2, "score": None},
                "midgame": {"games": 3, "score": None},
                "late": {"games": 5, "score": None},
            },
        }

        report = labeling.build_issue264_report(
            experiment={"top_n": 3, "seed": 41},
            label_report=label_report,
            arena_report=arena_report,
            baseline_checkpoint="/tmp/base.npz",
            challenger_checkpoint="/tmp/challenger.npz",
            challenger_artifact_dir="/tmp/challenger_artifact",
            min_score=0.6,
        )

        self.assertEqual("remain_selective", report["recommendation"]["recommendation"])
        self.assertIn("arena", report["recommendation"]["rationale"])

    def test_build_issue264_report_recommends_standard_step_when_arena_passes_and_hard_suite_present(self):
        label_report = {
            "pair_count": 3,
            "top1_disagreement_rate": 0.666667,
            "average_policy_divergence": 0.4,
            "maximum_policy_divergence": 0.8,
            "average_absolute_value_delta": 0.2,
            "largest_disagreements": [],
        }
        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 7,
            "losses": 2,
            "draws": 1,
            "score": 0.75,
            "promotion_decision": {"passed": True},
            "hard_suite_buckets": {
                "opening": {"games": 2, "score": None},
                "midgame": {"games": 3, "score": None},
                "late": {"games": 5, "score": None},
            },
        }

        report = labeling.build_issue264_report(
            experiment={"top_n": 3, "seed": 41},
            label_report=label_report,
            arena_report=arena_report,
            baseline_checkpoint="/tmp/base.npz",
            challenger_checkpoint="/tmp/challenger.npz",
            challenger_artifact_dir="/tmp/challenger_artifact",
            min_score=0.6,
        )

        self.assertEqual("promote_to_standard_step", report["recommendation"]["recommendation"])
        self.assertEqual(arena_report["hard_suite_buckets"], report["arena"]["hard_suite_buckets"])
        self.assertIn("hard-suite coverage", report["recommendation"]["rationale"])

    def test_build_issue264_report_rejects_mismatched_arena_promotion_decision(self):
        label_report = {
            "pair_count": 3,
            "top1_disagreement_rate": 0.666667,
            "average_policy_divergence": 0.4,
            "maximum_policy_divergence": 0.8,
            "average_absolute_value_delta": 0.2,
            "largest_disagreements": [],
        }
        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 7,
            "losses": 2,
            "draws": 1,
            "score": 0.75,
            "promotion_decision": {"passed": False},
            "hard_suite_buckets": {
                "opening": {"games": 2, "score": 0.5},
                "midgame": {"games": 3, "score": 0.67},
                "late": {"games": 5, "score": 1.0},
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            r"arena promotion_decision\.passed must match the score threshold outcome",
        ):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report=label_report,
                arena_report=arena_report,
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_non_boolean_promotion_decision_passed(self):
        invalid_passed_values = [1, 0, "true", None]

        for passed in invalid_passed_values:
            with self.subTest(passed=passed):
                with self.assertRaisesRegex(
                    ValueError,
                    r"arena promotion_decision\.passed must be a boolean",
                ):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report={
                            "schema": "arena_v1",
                            "games_played": 2,
                            "wins": 1,
                            "losses": 1,
                            "draws": 0,
                            "score": 0.75,
                            "promotion_decision": {"passed": passed},
                            "hard_suite_buckets": {
                                "opening": {"games": 2, "score": 0.75},
                            },
                        },
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=0.6,
                    )

    def test_build_issue264_report_rejects_missing_promotion_decision(self):
        with self.assertRaisesRegex(ValueError, r"arena promotion_decision must be an object"):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 2,
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "score": 0.5,
                    "hard_suite_buckets": {
                        "opening": {"games": 1, "score": None},
                        "midgame": {"games": 1, "score": None},
                        "late": {"games": 0, "score": None},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_non_dict_promotion_decision(self):
        for promotion_decision in (None, False, "passed", [True]):
            with self.subTest(promotion_decision=promotion_decision):
                with self.assertRaisesRegex(ValueError, r"arena promotion_decision must be an object"):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report={
                            "schema": "arena_v1",
                            "games_played": 2,
                            "wins": 1,
                            "losses": 1,
                            "draws": 0,
                            "score": 0.5,
                            "promotion_decision": promotion_decision,
                            "hard_suite_buckets": {
                                "opening": {"games": 1, "score": None},
                                "midgame": {"games": 1, "score": None},
                                "late": {"games": 0, "score": None},
                            },
                        },
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=0.6,
                    )

    def test_build_issue264_report_rejects_invalid_arena_score(self):
        invalid_scores = ["0.75", float("nan"), float("inf"), float("-inf"), -0.01, 1.01]

        for score in invalid_scores:
            with self.subTest(score=score):
                with self.assertRaisesRegex(
                    ValueError,
                    r"arena score must be a finite numeric value between 0\.0 and 1\.0",
                ):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report={
                            "schema": "arena_v1",
                            "games_played": 2,
                            "wins": 1,
                            "losses": 1,
                            "draws": 0,
                            "score": score,
                            "promotion_decision": {"passed": True},
                            "hard_suite_buckets": {
                                "opening": {"games": 2, "score": 0.75},
                            },
                        },
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=0.6,
                    )

    def test_build_issue264_report_rejects_invalid_min_score(self):
        invalid_min_scores = ["0.6", float("nan"), float("inf"), float("-inf"), -0.01, 1.01]

        for min_score in invalid_min_scores:
            with self.subTest(min_score=min_score):
                with self.assertRaisesRegex(
                    ValueError,
                    r"min_score must be a finite numeric value between 0\.0 and 1\.0",
                ):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report={
                            "schema": "arena_v1",
                            "games_played": 2,
                            "wins": 1,
                            "losses": 1,
                            "draws": 0,
                            "score": 0.5,
                            "promotion_decision": {"passed": False},
                            "hard_suite_buckets": {
                                "opening": {"games": 1, "score": None},
                                "midgame": {"games": 1, "score": None},
                                "late": {"games": 0, "score": None},
                            },
                        },
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=min_score,
                    )

    def test_build_issue264_report_rejects_arena_score_that_does_not_match_record(self):
        with self.assertRaisesRegex(
            ValueError,
            r"arena score must match wins/losses/draws-derived score",
        ):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 10,
                    "wins": 7,
                    "losses": 2,
                    "draws": 1,
                    "score": 0.7,
                    "promotion_decision": {"passed": True},
                    "hard_suite_buckets": {
                        "opening": {"games": 2, "score": 0.5},
                        "midgame": {"games": 3, "score": 0.67},
                        "late": {"games": 5, "score": 1.0},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_invalid_arena_count_fields(self):
        invalid_cases = [
            ("games_played", True),
            ("games_played", "10"),
            ("games_played", -1),
            ("wins", False),
            ("wins", "7"),
            ("wins", -1),
            ("losses", True),
            ("losses", "2"),
            ("losses", -1),
            ("draws", False),
            ("draws", "1"),
            ("draws", -1),
        ]

        for field_name, value in invalid_cases:
            with self.subTest(field_name=field_name, value=value):
                arena_report = {
                    "schema": "arena_v1",
                    "games_played": 10,
                    "wins": 7,
                    "losses": 2,
                    "draws": 1,
                    "score": 0.75,
                    "promotion_decision": {"passed": True},
                    "hard_suite_buckets": {
                        "opening": {"games": 2, "score": 0.5},
                        "midgame": {"games": 3, "score": 0.67},
                        "late": {"games": 1, "score": 1.0},
                    },
                }
                arena_report[field_name] = value

                with self.assertRaisesRegex(
                    ValueError,
                    rf"arena {field_name} must be a non-negative integer",
                ):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report=arena_report,
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=0.6,
                    )

    def test_build_issue264_report_rejects_zero_game_arena_report(self):
        with self.assertRaisesRegex(ValueError, r"arena games_played must be >= 1"):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "score": 0.0,
                    "promotion_decision": {"passed": False},
                    "hard_suite_buckets": {
                        "opening": {"games": 0, "score": None},
                        "midgame": {"games": 0, "score": None},
                        "late": {"games": 0, "score": None},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_missing_required_hard_suite_bucket_keys(self):
        with self.assertRaisesRegex(
            ValueError,
            r"arena hard_suite_buckets must include opening, midgame, and late buckets",
        ):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 2,
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "score": 0.5,
                    "promotion_decision": {"passed": False},
                    "hard_suite_buckets": {
                        "opening": {"games": 2, "score": 0.5},
                        "midgame": {"games": 3, "score": 0.67},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_hard_suite_bucket_games_total_mismatch(self):
        with self.assertRaisesRegex(
            ValueError,
            r"arena hard_suite_buckets games must sum to games_played",
        ):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 10,
                    "wins": 7,
                    "losses": 2,
                    "draws": 1,
                    "score": 0.75,
                    "promotion_decision": {"passed": True},
                    "hard_suite_buckets": {
                        "opening": {"games": 2, "score": 0.5},
                        "midgame": {"games": 3, "score": 0.67},
                        "late": {"games": 4, "score": 1.0},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_missing_hard_suite_buckets(self):
        with self.assertRaisesRegex(ValueError, "arena report must include hard_suite_buckets"):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 2,
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "score": 0.5,
                    "promotion_decision": {"passed": False},
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_malformed_hard_suite_bucket_payload(self):
        with self.assertRaisesRegex(
            ValueError,
            r"arena hard_suite_buckets\['opening'\] must include integer games and numeric-or-null score",
        ):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 2,
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "score": 0.75,
                    "promotion_decision": {"passed": True},
                    "hard_suite_buckets": {
                        "opening": {"games": "2", "score": 1.0},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_recommends_standard_step_when_arena_passes_with_descriptive_hard_suite_scores(self):
        label_report = {
            "pair_count": 3,
            "top1_disagreement_rate": 0.666667,
            "average_policy_divergence": 0.4,
            "maximum_policy_divergence": 0.8,
            "average_absolute_value_delta": 0.2,
            "largest_disagreements": [],
        }
        arena_report = {
            "schema": "arena_v1",
            "games_played": 10,
            "wins": 7,
            "losses": 2,
            "draws": 1,
            "score": 0.75,
            "promotion_decision": {"passed": True},
            "hard_suite_buckets": {
                "opening": {"games": 1, "score": None},
                "midgame": {"games": 3, "score": None},
                "late": {"games": 6, "score": None},
            },
        }

        report = labeling.build_issue264_report(
            experiment={"top_n": 3, "seed": 41},
            label_report=label_report,
            arena_report=arena_report,
            baseline_checkpoint="/tmp/base.npz",
            challenger_checkpoint="/tmp/challenger.npz",
            challenger_artifact_dir="/tmp/challenger_artifact",
            min_score=0.6,
        )

        self.assertEqual("promote_to_standard_step", report["recommendation"]["recommendation"])
        self.assertIn("hard-suite coverage", report["recommendation"]["rationale"])

    def test_build_issue264_report_rejects_negative_hard_suite_bucket_games(self):
        with self.assertRaisesRegex(
            ValueError,
            r"arena hard_suite_buckets\['opening'\] must include integer games and numeric-or-null score",
        ):
            labeling.build_issue264_report(
                experiment={"top_n": 3, "seed": 41},
                label_report={
                    "pair_count": 1,
                    "top1_disagreement_rate": 0.0,
                    "average_policy_divergence": 0.0,
                    "maximum_policy_divergence": 0.0,
                    "average_absolute_value_delta": 0.0,
                    "largest_disagreements": [],
                },
                arena_report={
                    "schema": "arena_v1",
                    "games_played": 2,
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "score": 0.75,
                    "promotion_decision": {"passed": True},
                    "hard_suite_buckets": {
                        "opening": {"games": -1, "score": 0.75},
                    },
                },
                baseline_checkpoint="/tmp/base.npz",
                challenger_checkpoint="/tmp/challenger.npz",
                challenger_artifact_dir="/tmp/challenger_artifact",
                min_score=0.6,
            )

    def test_build_issue264_report_rejects_non_finite_hard_suite_bucket_score(self):
        invalid_scores = [float("nan"), float("inf"), float("-inf")]

        for score in invalid_scores:
            with self.subTest(score=score):
                with self.assertRaisesRegex(
                    ValueError,
                    r"arena hard_suite_buckets\['opening'\] must include integer games and numeric-or-null score",
                ):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report={
                            "schema": "arena_v1",
                            "games_played": 2,
                            "wins": 1,
                            "losses": 1,
                            "draws": 0,
                            "score": 0.75,
                            "promotion_decision": {"passed": True},
                            "hard_suite_buckets": {
                                "opening": {"games": 2, "score": score},
                            },
                        },
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=0.6,
                    )

    def test_build_issue264_report_rejects_out_of_range_hard_suite_bucket_score(self):
        for score in (-0.01, 1.01):
            with self.subTest(score=score):
                with self.assertRaisesRegex(
                    ValueError,
                    r"arena hard_suite_buckets\['opening'\] must include integer games and numeric-or-null score",
                ):
                    labeling.build_issue264_report(
                        experiment={"top_n": 3, "seed": 41},
                        label_report={
                            "pair_count": 1,
                            "top1_disagreement_rate": 0.0,
                            "average_policy_divergence": 0.0,
                            "maximum_policy_divergence": 0.0,
                            "average_absolute_value_delta": 0.0,
                            "largest_disagreements": [],
                        },
                        arena_report={
                            "schema": "arena_v1",
                            "games_played": 2,
                            "wins": 1,
                            "losses": 1,
                            "draws": 0,
                            "score": 0.75,
                            "promotion_decision": {"passed": True},
                            "hard_suite_buckets": {
                                "opening": {"games": 2, "score": score},
                            },
                        },
                        baseline_checkpoint="/tmp/base.npz",
                        challenger_checkpoint="/tmp/challenger.npz",
                        challenger_artifact_dir="/tmp/challenger_artifact",
                        min_score=0.6,
                    )

    def test_pair_budget_rows_rejects_missing_stronger_partner(self):
        rows = [
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "canonical",
                "teacher_budget": 32,
                "source_rank": 1,
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.2,
            }
        ]

        with self.assertRaisesRegex(ValueError, "must contain exactly one canonical row and one stronger row"):
            labeling.pair_budget_rows(rows)

    def test_pair_budget_rows_rejects_duplicate_canonical_rows(self):
        rows = [
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "canonical",
                "teacher_budget": 32,
                "source_rank": 1,
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.2,
            },
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "canonical",
                "teacher_budget": 32,
                "source_rank": 1,
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.4,
            },
            {
                "budget_pair_id": "pair-a",
                "canonical_state": "state-a",
                "teacher_profile": "stronger",
                "teacher_budget": 64,
                "source_rank": 1,
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.8,
            },
        ]

        with self.assertRaisesRegex(ValueError, "must contain exactly one canonical row and one stronger row"):
            labeling.pair_budget_rows(rows)

    def test_validate_labeled_row_rejects_negative_policy_probability(self):
        row = {
            "budget_pair_id": "pair-a",
            "canonical_state": "state-a",
            "teacher_profile": "canonical",
            "teacher_budget": 32,
            "source_rank": 1,
            "policy": [1.0, -0.1, 0.1, 0.0, 0.0, 0.0],
            "value": 0.2,
        }

        with self.assertRaisesRegex(ValueError, r"policy\[1\] must be >= 0"):
            labeling.validate_labeled_row(row, source="row")

    def test_validate_labeled_row_rejects_policy_that_does_not_sum_to_one(self):
        row = {
            "budget_pair_id": "pair-a",
            "canonical_state": "state-a",
            "teacher_profile": "canonical",
            "teacher_budget": 32,
            "source_rank": 1,
            "policy": [0.5, 0.25, 0.25, 0.1, 0.0, 0.0],
            "value": 0.2,
        }

        with self.assertRaisesRegex(ValueError, "policy must sum to 1.0"):
            labeling.validate_labeled_row(row, source="row")

    def test_validate_labeled_row_rejects_value_outside_training_range(self):
        row = {
            "budget_pair_id": "pair-a",
            "canonical_state": "state-a",
            "teacher_profile": "canonical",
            "teacher_budget": 32,
            "source_rank": 1,
            "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "value": 1.5,
        }

        with self.assertRaisesRegex(ValueError, "value must be between -1.0 and 1.0"):
            labeling.validate_labeled_row(row, source="row")

    def test_compare_hard_state_teacher_labels_cli_writes_report(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-hard-state-report-") as tmp:
            dataset_path = Path(tmp) / "labeled.jsonl"
            report_path = Path(tmp) / "report.json"
            rows = [
                {
                    "budget_pair_id": "pair-a",
                    "canonical_state": "state-a",
                    "teacher_profile": "canonical",
                    "teacher_budget": 32,
                    "source_rank": 1,
                    "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 0.2,
                },
                {
                    "budget_pair_id": "pair-a",
                    "canonical_state": "state-a",
                    "teacher_profile": "stronger",
                    "teacher_budget": 64,
                    "source_rank": 1,
                    "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 0.8,
                },
            ]
            dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/compare_hard_state_teacher_labels.py",
                    "--input-jsonl",
                    str(dataset_path),
                    "--out-report",
                    str(report_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual({"out_report": str(report_path), "pair_count": 1}, json.loads(result.stdout.strip()))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(1, report["pair_count"])
            self.assertEqual(1.0, report["top1_disagreement_rate"])

    def test_compare_hard_state_teacher_labels_cli_rejects_malformed_labeled_rows(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-hard-state-report-") as tmp:
            dataset_path = Path(tmp) / "labeled.jsonl"
            report_path = Path(tmp) / "report.json"
            rows = [
                {
                    "canonical_state": "state-a",
                    "teacher_profile": "canonical",
                    "teacher_budget": 32,
                    "source_rank": 1,
                    "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 0.2,
                },
                {
                    "budget_pair_id": "pair-a",
                    "canonical_state": "state-a",
                    "teacher_profile": "stronger",
                    "teacher_budget": 64,
                    "source_rank": 1,
                    "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 0.8,
                },
            ]
            dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/compare_hard_state_teacher_labels.py",
                    "--input-jsonl",
                    str(dataset_path),
                    "--out-report",
                    str(report_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertIn("missing required field 'budget_pair_id'", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
