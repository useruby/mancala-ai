import contextlib
import io
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class BuildTacticalBalancedReplayTest(unittest.TestCase):
    def regression_state(self):
        return {
            "player_pits": [1, 8, 0, 1, 1, 0],
            "opponent_pits": [1, 3, 5, 0, 0, 1],
            "player_store": 20,
            "opponent_store": 7,
            "current_player": 1,
        }

    def contradictory_state(self):
        return {
            "player_pits": [1, 6, 6, 0, 6, 5],
            "opponent_pits": [5, 5, 1, 5, 5, 0],
            "player_store": 1,
            "opponent_store": 2,
            "current_player": 1,
        }

    def diversity_state_gain_4_5_pit_4(self):
        return {
            "player_pits": [0, 5, 3, 1, 0, 0],
            "opponent_pits": [3, 3, 0, 2, 3, 2],
            "player_store": 15,
            "opponent_store": 21,
            "current_player": 0,
        }

    def diversity_state_gain_2_3_pit_3(self):
        return {
            "player_pits": [3, 6, 1, 8, 3, 2],
            "opponent_pits": [4, 2, 1, 0, 2, 4],
            "player_store": 19,
            "opponent_store": 23,
            "current_player": 1,
        }

    def diversity_state_gain_6_plus_pit_3(self):
        return {
            "player_pits": [0, 1, 6, 5, 8, 2],
            "opponent_pits": [0, 2, 7, 0, 2, 1],
            "player_store": 7,
            "opponent_store": 24,
            "current_player": 1,
        }

    def regression_fixture(self):
        return [
            {
                "id": "missed_capture_f67bd4k0_move_28",
                "state": self.regression_state(),
                "expected_move": 1,
                "acceptable_moves": [1],
                "move_number": 28,
            }
        ]

    def capture_row(self, canonical_state, *, raw_state, policy, priority, teacher_selected_move):
        return {
            "canonical_state": canonical_state,
            "raw_state": raw_state,
            "state": [0.0] * 27,
            "side_to_move": raw_state["current_player"],
            "legal_moves": [move for move, seeds in enumerate(raw_state["player_pits"]) if seeds > 0],
            "bucket": "capture_available",
            "bucket_group": "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.5,
            "priority_score": priority,
            "policy": policy,
            "teacher_selected_move": teacher_selected_move,
        }

    def nearby_row(self, bucket, index, *, priority):
        return {
            "canonical_state": f"{bucket}-{index}",
            "state": [0.0] * 27,
            "side_to_move": 0,
            "legal_moves": [0],
            "bucket": bucket,
            "bucket_group": "preservation" if bucket == "starvation_pressure" else "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.25,
            "priority_score": priority,
            "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

    def test_default_balanced_replay_source_points_to_validation_input(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        self.assertEqual(
            Path(__file__).resolve().parents[2] / "ml/alphazero_lite/tactical_balanced_replay_source.jsonl",
            module.DEFAULT_BALANCED_REPLAY_SOURCE,
        )

    def test_build_balanced_replay_dataset_uses_keyword_only_planned_signature(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        self.assertEqual(
            "(*, regression_positions_path: 'Path', tactical_replay_path: 'Path', out_path: 'Path')",
            str(inspect.signature(module.build_balanced_replay_dataset)),
        )

    def test_build_balanced_replay_dataset_rejects_missing_balanced_replay_source(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "regressions.json"
            out_path = tmp_path / "balanced.jsonl"
            missing_source = tmp_path / "missing.jsonl"

            regression_path.write_text(
                json.dumps([
                    {
                        "id": "x",
                        "state": {
                            "player_pits": [1, 8, 0, 1, 1, 0],
                            "opponent_pits": [1, 3, 5, 0, 0, 1],
                            "player_store": 20,
                            "opponent_store": 7,
                            "current_player": 1,
                        },
                        "expected_move": 1,
                    }
                ]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "balanced replay source not found"):
                module.build_balanced_replay_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=missing_source,
                    out_path=out_path,
                )

    def test_build_balanced_replay_dataset_rejects_source_without_required_bucket_counts(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "regressions.json"
            source_path = tmp_path / "source.jsonl"
            out_path = tmp_path / "balanced.jsonl"

            regression_path.write_text(json.dumps(self.regression_fixture()), encoding="utf-8")
            source_path.write_text(
                "\n".join(
                    json.dumps(self.nearby_row("high_imbalance", index, priority=10 - index))
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "starvation_pressure=0"):
                module.build_balanced_replay_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=source_path,
                    out_path=out_path,
                )

    def test_build_balanced_replay_dataset_uses_cli_planned_signature(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        self.assertTrue(callable(module.parse_args))
        self.assertTrue(callable(module.main))

    def test_default_balanced_source_supports_minimum_two_non_protection_capture_shapes(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        regression_position = self.regression_fixture()[0]
        source_rows = module._load_jsonl(module.DEFAULT_BALANCED_REPLAY_SOURCE)
        protection_rows = module._select_capture_protection_rows(regression_position, source_rows)
        preservation_rows = module._select_capture_preservation_rows(source_rows, protection_rows)

        self.assertGreaterEqual(len(preservation_rows), 2)

    def test_build_balanced_replay_dataset_writes_role_labeled_two_two_six_composition(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "regressions.json"
            source_path = tmp_path / "source.jsonl"
            out_path = tmp_path / "balanced.jsonl"

            regression_path.write_text(json.dumps(self.regression_fixture()), encoding="utf-8")

            support = self.capture_row(
                json.dumps(self.contradictory_state(), separators=(",", ":"), sort_keys=True),
                raw_state=self.contradictory_state(),
                policy=[0.06637731194496155, 0.3550935685634613, 0.0543360710144043, 0.3244529366493225, 0.19974012672901154, 0.0],
                priority=20.0,
                teacher_selected_move=1,
            )
            preservation_rows = [
                self.capture_row(
                    json.dumps(self.diversity_state_gain_4_5_pit_4(), separators=(",", ":"), sort_keys=True),
                    raw_state=self.diversity_state_gain_4_5_pit_4(),
                    policy=[0.5639695819300191, 0.21796454090650363, 0.6994649712461508, 0.7668980983562408, 0.16778914336780226, 0.6072474938909317],
                    priority=10.0,
                    teacher_selected_move=3,
                ),
                self.capture_row(
                    json.dumps(self.diversity_state_gain_2_3_pit_3(), separators=(",", ":"), sort_keys=True),
                    raw_state=self.diversity_state_gain_2_3_pit_3(),
                    policy=[0.4391534486737374, 0.4686979702761619, 0.30409119054476674, 0.4025085191628056, 0.2722461931442631, 0.539981562598014],
                    priority=9.0,
                    teacher_selected_move=5,
                ),
            ]
            nearby_rows = []
            for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure"):
                nearby_rows.extend(self.nearby_row(bucket, index, priority=10 - index) for index in range(3))

            source_rows = [support, *preservation_rows, *nearby_rows]
            source_path.write_text("\n".join(json.dumps(row) for row in source_rows) + "\n", encoding="utf-8")

            with patch.object(module, "teacher_label_regression_row") as teacher_labeler:
                teacher_labeler.return_value = {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": self.regression_state(),
                    "legal_moves": [0, 1, 2, 5],
                    "policy": [0.0001, 0.9993, 0.0001, 0.0, 0.0, 0.0005],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                }

                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    rows = module.build_balanced_replay_dataset(
                        regression_positions_path=regression_path,
                        tactical_replay_path=source_path,
                        out_path=out_path,
                    )

            self.assertEqual(10, len(rows))
            self.assertEqual(rows, [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()])
            self.assertEqual(
                ["capture_protection"] * 2 + ["capture_preservation"] * 2 + ["nearby_preservation"] * 6,
                [row["replay_role"] for row in rows],
            )
            self.assertEqual(
                ["high_imbalance", "high_imbalance", "high_value_swing", "high_value_swing", "starvation_pressure", "starvation_pressure"],
                [row["bucket"] for row in rows if row["replay_role"] == "nearby_preservation"],
            )
            self.assertIn("supports only 2 distinct capture-preservation shapes", stderr.getvalue())

    def test_build_balanced_replay_dataset_requires_at_least_two_distinct_capture_preservation_shapes(self):
        from ml.alphazero_lite import build_tactical_balanced_replay as module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "regressions.json"
            source_path = tmp_path / "source.jsonl"
            out_path = tmp_path / "balanced.jsonl"

            regression_path.write_text(json.dumps(self.regression_fixture()), encoding="utf-8")

            source_rows = [
                self.capture_row(
                    json.dumps(self.contradictory_state(), separators=(",", ":"), sort_keys=True),
                    raw_state=self.contradictory_state(),
                    policy=[0.06637731194496155, 0.3550935685634613, 0.0543360710144043, 0.3244529366493225, 0.19974012672901154, 0.0],
                    priority=20.0,
                    teacher_selected_move=1,
                ),
                self.capture_row(
                    json.dumps(self.diversity_state_gain_4_5_pit_4(), separators=(",", ":"), sort_keys=True),
                    raw_state=self.diversity_state_gain_4_5_pit_4(),
                    policy=[0.5639695819300191, 0.21796454090650363, 0.6994649712461508, 0.7668980983562408, 0.16778914336780226, 0.6072474938909317],
                    priority=10.0,
                    teacher_selected_move=3,
                ),
                self.capture_row(
                    "duplicate-capture-shape-row",
                    raw_state=self.diversity_state_gain_4_5_pit_4(),
                    policy=[0.5639695819300191, 0.21796454090650363, 0.6994649712461508, 0.7668980983562408, 0.16778914336780226, 0.6072474938909317],
                    priority=9.0,
                    teacher_selected_move=3,
                ),
            ]
            for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure"):
                source_rows.extend(self.nearby_row(bucket, index, priority=10 - index) for index in range(2))
            source_path.write_text("\n".join(json.dumps(row) for row in source_rows) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "capture_preservation distinct shapes below minimum: 1"):
                module.build_balanced_replay_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=source_path,
                    out_path=out_path,
                )
