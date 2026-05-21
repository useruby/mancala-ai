import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.alphazero_lite import build_tactical_capture_protection as module


class BuildTacticalCaptureProtectionTest(unittest.TestCase):
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

    def rejected_support_state(self):
        return {
            "player_pits": [1, 6, 0, 7, 6, 5],
            "opponent_pits": [5, 5, 0, 5, 5, 0],
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

    def contradictory_policy(self):
        return [
            0.06637731194496155,
            0.3550935685634613,
            0.0543360710144043,
            0.3244529366493225,
            0.19974012672901154,
            0.0,
        ]

    def regression_policy(self):
        return [0.0001, 0.9993, 0.0001, 0.0, 0.0, 0.0005]

    def rejected_support_policy(self):
        return [0.0, 0.0, 0.1, 0.8, 0.1, 0.0]

    def diversity_policy_gain_4_5_pit_4(self):
        return [
            0.5639695819300191,
            0.21796454090650363,
            0.6994649712461508,
            0.7668980983562408,
            0.16778914336780226,
            0.6072474938909317,
        ]

    def diversity_policy_gain_2_3_pit_3(self):
        return [
            0.4391534486737374,
            0.4686979702761619,
            0.30409119054476674,
            0.4025085191628056,
            0.2722461931442631,
            0.539981562598014,
        ]

    def diversity_policy_gain_6_plus_pit_3(self):
        return [
            0.3621505825434207,
            0.9494671118799197,
            0.03238532397966698,
            0.1534381633927081,
            0.44992256871479397,
            0.3709645075501008,
        ]

    def capture_candidate(
        self,
        canonical_state,
        *,
        raw_state,
        policy,
        priority,
        motif_score,
        teacher_selected_move=None,
    ):
        return {
            "canonical_state": canonical_state,
            "state": [0.0] * 27,
            "raw_state": raw_state,
            "bucket": "capture_available",
            "bucket_group": "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.5,
            "priority_score": priority,
            "motif_support_score": motif_score,
            "policy": policy,
            "teacher_selected_move": teacher_selected_move,
        }

    def test_extract_regression_motif_signature_returns_expected_known_values(self):
        signature = module.extract_regression_motif_signature(
            self.regression_state(), expected_move=1
        )

        self.assertEqual(1, signature["target_capture_move"])
        self.assertEqual(5, signature["tempting_extra_turn_move"])
        self.assertEqual([0, 1, 2, 5], signature["legal_moves"])
        self.assertEqual(9, signature["capture_store_gain"])
        self.assertEqual(1, signature["extra_turn_store_gain"])
        self.assertEqual(4, signature["capture_landing_pit"])
        self.assertFalse(signature["capture_lands_in_store"])
        self.assertEqual(1, signature["capture_opposite_pit"])
        self.assertIsNone(signature["extra_turn_landing_pit"])
        self.assertTrue(signature["extra_turn_lands_in_store"])
        self.assertTrue(signature["teacher_prefers_capture"])

    def test_choose_candidate_conflict_moves_uses_deterministic_tie_breakers(self):
        row = {
            "raw_state": self.contradictory_state(),
            "policy": [
                0.06637731194496155,
                0.3550935685634613,
                0.0543360710144043,
                0.3244529366493225,
                0.19974012672901154,
                0.0,
            ],
        }

        conflict = module.choose_candidate_conflict_moves(row)

        self.assertEqual(0, conflict["capture_move"])
        self.assertEqual(1, conflict["extra_turn_move"])

    def test_capture_shape_key_uses_store_and_gain_buckets(self):
        row = {
            "raw_state": self.regression_state(),
            "policy": [0.0001, 0.9993, 0.0001, 0.0, 0.0, 0.0005],
        }

        key = module.capture_shape_key(row)

        self.assertEqual(
            (
                ("pit", 4),
                1,
                "gain_6_plus",
                "extra_turn",
                ("store",),
            ),
            key,
        )

    def test_capture_shape_key_reuses_existing_conflict_move_selection(self):
        row = {
            "raw_state": self.contradictory_state(),
            "policy": [
                0.06637731194496155,
                0.3550935685634613,
                0.0543360710144043,
                0.3244529366493225,
                0.19974012672901154,
                0.0,
            ],
        }

        conflict = module.choose_candidate_conflict_moves(row)
        key = module.capture_shape_key(row)

        self.assertEqual({"capture_move": 0, "extra_turn_move": 1}, conflict)
        self.assertEqual("extra_turn", key[3])

    def test_legal_move_features_normalizes_non_store_landings_to_zero_through_five(
        self,
    ):
        features = module.legal_move_features(self.regression_state())
        non_store_landings = [
            feature["landing_pit"]
            for feature in features
            if not feature["lands_in_store"]
        ]

        self.assertTrue(non_store_landings)
        self.assertTrue(all(0 <= landing <= 5 for landing in non_store_landings))

    def test_score_candidate_support_row_marks_old_contradictory_row_motif_protective(
        self,
    ):
        signature = module.extract_regression_motif_signature(
            self.regression_state(), expected_move=1
        )
        row = {
            "canonical_state": json.dumps(
                self.contradictory_state(), separators=(",", ":"), sort_keys=True
            ),
            "raw_state": self.contradictory_state(),
            "policy": [
                0.06637731194496155,
                0.3550935685634613,
                0.0543360710144043,
                0.3244529366493225,
                0.19974012672901154,
                0.0,
            ],
            "teacher_selected_move": 1,
        }

        scored = module.score_candidate_support_row(row, signature)

        self.assertEqual(13, scored["score"])
        self.assertGreaterEqual(
            scored["score"], module.MOTIF_PROTECTIVE_SCORE_THRESHOLD
        )
        self.assertTrue(scored["motif_protective"])
        self.assertEqual(0, scored["capture_move"])
        self.assertEqual(1, scored["extra_turn_move"])

    def test_score_candidate_support_row_rejects_failed_support_row(self):
        signature = module.extract_regression_motif_signature(
            self.regression_state(), expected_move=1
        )
        row = {
            "canonical_state": json.dumps(
                self.rejected_support_state(), separators=(",", ":"), sort_keys=True
            ),
            "raw_state": self.rejected_support_state(),
            "policy": [0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
            "teacher_selected_move": 3,
        }

        scored = module.score_candidate_support_row(row, signature)

        self.assertEqual(7, scored["score"])
        self.assertLess(scored["score"], module.MOTIF_PROTECTIVE_SCORE_THRESHOLD)
        self.assertFalse(scored["motif_protective"])
        self.assertEqual(0, scored["capture_move"])
        self.assertEqual(1, scored["extra_turn_move"])

    def test_teacher_label_regression_row_proves_capture_available_tactical_bucket_contract(
        self,
    ):
        regression_position = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "test/fixtures/ai/superhuman_regression_positions.json"
            ).read_text(encoding="utf-8")
        )[0]

        row = module.teacher_label_regression_row(
            regression_position["state"],
            source_id=regression_position["id"],
            move_number=regression_position["move_number"],
        )

        self.assertEqual("capture_available", row.get("bucket"))
        self.assertEqual("tactical", row.get("bucket_group"))

    def test_real_dataset_contains_exactly_five_capture_rows_with_support_row_second(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        out_path = repo_root / "ml/alphazero_lite/tactical_capture_protection.jsonl"

        rows = module.load_jsonl(out_path)

        self.assertEqual(5, len(rows))
        self.assertEqual(
            "missed_capture_f67bd4k0_move_28", rows[0]["capture_protection_source"]
        )
        self.assertEqual("capture_available", rows[0]["bucket"])
        copied_rows = [
            {
                **row,
                "raw_state": row.get("raw_state")
                or module.raw_state_from_canonical_state(row["canonical_state"]),
            }
            for row in rows[1:]
        ]
        self.assertEqual(4, len(copied_rows))
        shape_keys = [module.capture_shape_key(row) for row in copied_rows]
        self.assertEqual(
            [
                (("pit", 4), 1, "gain_2_3", "extra_turn", ("store",)),
                (("pit", 5), 0, "gain_2_3", "extra_turn", ("store",)),
                (("pit", 5), 0, "gain_2_3", "extra_turn", ("store",)),
                (("pit", 5), 0, "gain_2_3", "extra_turn", ("store",)),
            ],
            shape_keys,
        )

    def test_build_dataset_rejects_empty_regression_fixture(self):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text("[]", encoding="utf-8")
            tactical_path.write_text(
                json.dumps(
                    {
                        "canonical_state": "capture-row",
                        "state": [0.0] * 27,
                        "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 0.5,
                        "bucket": "capture_available",
                        "bucket_group": "tactical",
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "regression fixture must contain at least one position"
            ):
                module.build_capture_protection_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=tactical_path,
                    out_path=out_path,
                    teacher_labeler=lambda raw_state: None,
                )

    def test_build_dataset_rejects_invalid_copied_tactical_row_contract(self):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            tactical_path.write_text(
                json.dumps(
                    {
                        "canonical_state": "capture-row",
                        "state": [0.0] * 27,
                        "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 0.5,
                        "bucket": "capture_available",
                        "bucket_group": "preservation",
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "bucket_group must equal tactical"):
                module.build_capture_protection_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=tactical_path,
                    out_path=out_path,
                    teacher_labeler=lambda raw_state: {
                        "canonical_state": "capture-regression-state",
                        "state": [0.0] * 27,
                        "raw_state": raw_state,
                        "legal_moves": [0, 1, 3, 4],
                        "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                        "value": 0.75,
                        "teacher_selected_move": 1,
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                        "bucket": "capture_available",
                        "bucket_group": "tactical",
                    },
                )

    def test_build_dataset_rejects_regression_row_bucket_drift(self):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            tactical_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "regression row bucket must equal capture_available"
            ):
                module.build_capture_protection_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=tactical_path,
                    out_path=out_path,
                    teacher_labeler=lambda raw_state: {
                        "canonical_state": "capture-regression-state",
                        "state": [0.0] * 27,
                        "raw_state": raw_state,
                        "legal_moves": [0, 1, 3, 4],
                        "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                        "value": 0.75,
                        "teacher_selected_move": 1,
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                        "bucket": "high_imbalance",
                        "bucket_group": "tactical",
                    },
                )

    def test_build_dataset_writes_exact_regression_row_with_capture_bucket_and_move_one_peak(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            tactical_path.write_text("", encoding="utf-8")

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": raw_state,
                    "legal_moves": [0, 1, 3, 4],
                    "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                },
            )

            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(1, len(rows))
            self.assertEqual("capture-regression-state", rows[0]["canonical_state"])
            self.assertEqual("capture_available", rows[0]["bucket"])
            self.assertEqual("tactical", rows[0]["bucket_group"])
            self.assertEqual(1, max(range(6), key=lambda idx: rows[0]["policy"][idx]))

    def test_build_dataset_selects_top_four_capture_rows_deterministically(self):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            z_low_state = {
                "player_pits": [2, 8, 7, 4, 3, 1],
                "opponent_pits": [5, 2, 7, 4, 2, 0],
                "player_store": 23,
                "opponent_store": 10,
                "current_player": 1,
            }
            a_high_state = self.contradictory_state()
            b_high_state = self.regression_state()
            c_mid_state = self.diversity_state_gain_4_5_pit_4()
            d_mid_state = self.diversity_state_gain_2_3_pit_3()
            rows = [
                {
                    "canonical_state": json.dumps(
                        z_low_state, separators=(",", ":"), sort_keys=True
                    ),
                    "raw_state": z_low_state,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.2,
                    "priority_score": 1.0,
                    "policy": [
                        0.5677812224209393,
                        0.7572827502575612,
                        0.17549491383846472,
                        0.8561465042734577,
                        0.8970427617839751,
                        0.8269898252763096,
                    ],
                    "teacher_selected_move": 4,
                },
                {
                    "canonical_state": json.dumps(
                        a_high_state, separators=(",", ":"), sort_keys=True
                    ),
                    "raw_state": a_high_state,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.2,
                    "priority_score": 10.0,
                    "policy": self.contradictory_policy(),
                    "teacher_selected_move": 1,
                },
                {
                    "canonical_state": json.dumps(
                        b_high_state, separators=(",", ":"), sort_keys=True
                    ),
                    "raw_state": b_high_state,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.2,
                    "priority_score": 10.0,
                    "policy": self.regression_policy(),
                    "teacher_selected_move": 5,
                },
                {
                    "canonical_state": json.dumps(
                        c_mid_state, separators=(",", ":"), sort_keys=True
                    ),
                    "raw_state": c_mid_state,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.2,
                    "priority_score": 8.0,
                    "policy": self.diversity_policy_gain_4_5_pit_4(),
                    "teacher_selected_move": 1,
                },
                {
                    "canonical_state": json.dumps(
                        d_mid_state, separators=(",", ":"), sort_keys=True
                    ),
                    "raw_state": d_mid_state,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.2,
                    "priority_score": 7.0,
                    "policy": self.diversity_policy_gain_2_3_pit_3(),
                },
                {
                    "canonical_state": "skip-other-bucket",
                    "bucket": "sparse_endgame",
                    "bucket_group": "tactical",
                    "priority_score": 999.0,
                    "policy": [0, 1, 0, 0, 0, 0],
                },
            ]
            tactical_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": raw_state,
                    "legal_moves": [0, 1, 3, 4],
                    "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                },
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [
                    "capture-regression-state",
                    json.dumps(a_high_state, separators=(",", ":"), sort_keys=True),
                    json.dumps(b_high_state, separators=(",", ":"), sort_keys=True),
                    json.dumps(c_mid_state, separators=(",", ":"), sort_keys=True),
                    json.dumps(d_mid_state, separators=(",", ":"), sort_keys=True),
                ],
                [row["canonical_state"] for row in built_rows],
            )

    def test_build_dataset_rejects_teacher_label_when_selected_move_is_not_one(self):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            tactical_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "teacher_selected_move must equal expected_move for capture protection row",
            ):
                module.build_capture_protection_dataset(
                    regression_positions_path=regression_path,
                    tactical_replay_path=tactical_path,
                    out_path=out_path,
                    teacher_labeler=lambda raw_state: {
                        "canonical_state": "capture-regression-state",
                        "state": [0.0] * 27,
                        "raw_state": raw_state,
                        "legal_moves": [0, 1, 3, 4],
                        "policy": [0.1, 0.1, 0.0, 0.7, 0.1, 0.0],
                        "value": 0.75,
                        "teacher_selected_move": 3,
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                    },
                )

    def test_build_dataset_skips_single_copied_capture_row_when_policy_peak_disagrees_with_forensic_reference(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            tactical_path.write_text(
                json.dumps(
                    {
                        "canonical_state": json.dumps(
                            {
                                "player_pits": [1, 6, 6, 0, 6, 5],
                                "opponent_pits": [5, 5, 1, 5, 5, 0],
                                "player_store": 1,
                                "opponent_store": 2,
                                "current_player": 1,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "state": [0.0] * 27,
                        "bucket": "capture_available",
                        "bucket_group": "tactical",
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                        "value": 0.5,
                        "reference_move": 3,
                        "priority_score": 10.0,
                        "policy": [0.0, 0.7, 0.0, 0.2, 0.1, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": raw_state,
                    "legal_moves": [0, 1, 3, 4],
                    "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                },
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                ["capture-regression-state"],
                [row["canonical_state"] for row in built_rows],
            )

    def test_build_dataset_uses_forensic_input_reference_moves_when_tactical_rows_omit_them(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            forensic_path = tmp_path / "mining_forensic_input.json"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            canonical_state = json.dumps(
                {
                    "player_pits": [1, 6, 6, 0, 6, 5],
                    "opponent_pits": [5, 5, 1, 5, 5, 0],
                    "player_store": 1,
                    "opponent_store": 2,
                    "current_player": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            tactical_path.write_text(
                json.dumps(
                    {
                        "canonical_state": canonical_state,
                        "state": [0.0] * 27,
                        "bucket": "capture_available",
                        "bucket_group": "tactical",
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                        "value": 0.5,
                        "priority_score": 10.0,
                        "policy": [0.0, 0.7, 0.0, 0.2, 0.1, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            forensic_path.write_text(
                json.dumps(
                    {
                        "systems": {
                            "challenger": {
                                "rows": [
                                    {
                                        "canonical_state": canonical_state,
                                        "bucket": "capture_available",
                                        "reference_move": 3,
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": raw_state,
                    "legal_moves": [0, 1, 3, 4],
                    "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                },
                forensic_suite_path=forensic_path,
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                ["capture-regression-state"],
                [row["canonical_state"] for row in built_rows],
            )

    def test_build_dataset_matches_forensic_reference_moves_from_raw_state_when_canonical_state_is_absent(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            forensic_path = tmp_path / "mining_forensic_input.json"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            raw_state = {
                "player_pits": [1, 6, 6, 0, 6, 5],
                "opponent_pits": [5, 5, 1, 5, 5, 0],
                "player_store": 1,
                "opponent_store": 2,
                "current_player": 1,
            }
            canonical_state = json.dumps(
                raw_state, separators=(",", ":"), sort_keys=True
            )

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            tactical_path.write_text(
                json.dumps(
                    {
                        "canonical_state": canonical_state,
                        "state": [0.0] * 27,
                        "bucket": "capture_available",
                        "bucket_group": "tactical",
                        "input_encoding": "kalah_v3",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                        "value": 0.5,
                        "priority_score": 10.0,
                        "policy": [0.0, 0.7, 0.0, 0.2, 0.1, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            forensic_path.write_text(
                json.dumps(
                    {
                        "systems": {
                            "challenger": {
                                "rows": [
                                    {
                                        "state": raw_state,
                                        "bucket": "capture_available",
                                        "reference_move": 3,
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": raw_state,
                    "legal_moves": [0, 1, 3, 4],
                    "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                },
                forensic_suite_path=forensic_path,
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                ["capture-regression-state"],
                [row["canonical_state"] for row in built_rows],
            )

    def test_build_dataset_skips_contradictory_capture_rows_and_keeps_next_valid_candidates(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            forensic_path = tmp_path / "mining_forensic_input.json"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": {
                                "player_pits": [1, 8, 0, 1, 1, 0],
                                "opponent_pits": [1, 3, 5, 0, 0, 1],
                                "player_store": 20,
                                "opponent_store": 7,
                                "current_player": 1,
                            },
                            "expected_move": 1,
                            "acceptable_moves": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            def state_key(player_pits, opponent_pits, current_player):
                return json.dumps(
                    {
                        "player_pits": player_pits,
                        "opponent_pits": opponent_pits,
                        "player_store": 1,
                        "opponent_store": 2,
                        "current_player": current_player,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )

            contradictory = state_key([1, 6, 6, 0, 6, 5], [5, 5, 1, 5, 5, 0], 1)
            valid_a = state_key([1, 6, 6, 6, 5, 0], [5, 5, 1, 5, 5, 0], 1)
            valid_b = json.dumps(
                self.diversity_state_gain_6_plus_pit_3(),
                separators=(",", ":"),
                sort_keys=True,
            )
            valid_c = json.dumps(
                self.diversity_state_gain_4_5_pit_4(),
                separators=(",", ":"),
                sort_keys=True,
            )
            valid_d = state_key([1, 0, 7, 6, 6, 5], [5, 4, 4, 4, 4, 0], 1)

            tactical_rows = [
                {
                    "canonical_state": contradictory,
                    "raw_state": json.loads(contradictory),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 10.0,
                    "policy": [0.0, 0.7, 0.0, 0.2, 0.1, 0.0],
                },
                {
                    "canonical_state": valid_a,
                    "raw_state": json.loads(valid_a),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 9.0,
                    "policy": [0.0, 0.1, 0.0, 0.8, 0.1, 0.0],
                },
                {
                    "canonical_state": valid_b,
                    "raw_state": self.diversity_state_gain_6_plus_pit_3(),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 8.0,
                    "policy": self.diversity_policy_gain_6_plus_pit_3(),
                },
                {
                    "canonical_state": valid_c,
                    "raw_state": self.diversity_state_gain_4_5_pit_4(),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 7.0,
                    "policy": self.diversity_policy_gain_4_5_pit_4(),
                    "teacher_selected_move": 1,
                },
                {
                    "canonical_state": valid_d,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 6.0,
                    "policy": [0.0, 0.0, 0.8, 0.1, 0.1, 0.0],
                },
            ]
            tactical_path.write_text(
                "\n".join(json.dumps(row) for row in tactical_rows) + "\n",
                encoding="utf-8",
            )

            forensic_rows = [
                {
                    "state": json.loads(contradictory),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_a),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_b),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_c),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_d),
                    "bucket": "capture_available",
                    "reference_move": 2,
                },
            ]
            forensic_path.write_text(
                json.dumps({"systems": {"challenger": {"rows": forensic_rows}}}),
                encoding="utf-8",
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: {
                    "canonical_state": "capture-regression-state",
                    "state": [0.0] * 27,
                    "raw_state": raw_state,
                    "legal_moves": [0, 1, 3, 4],
                    "policy": [0.1, 0.7, 0.0, 0.1, 0.1, 0.0],
                    "value": 0.75,
                    "teacher_selected_move": 1,
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                },
                forensic_suite_path=forensic_path,
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                ["capture-regression-state", valid_c, valid_a, valid_d],
                [row["canonical_state"] for row in built_rows],
            )

    def test_build_dataset_allows_only_motif_protective_candidates_and_contradiction_override(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            forensic_path = tmp_path / "mining_forensic_input.json"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": self.regression_state(),
                            "expected_move": 1,
                            "acceptable_moves": [1],
                            "move_number": 28,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            contradictory = json.dumps(
                self.contradictory_state(), separators=(",", ":"), sort_keys=True
            )
            rejected_support = json.dumps(
                self.rejected_support_state(), separators=(",", ":"), sort_keys=True
            )
            valid_a = json.dumps(
                {
                    "player_pits": [1, 6, 6, 6, 5, 0],
                    "opponent_pits": [5, 5, 1, 5, 5, 0],
                    "player_store": 1,
                    "opponent_store": 2,
                    "current_player": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            valid_b = json.dumps(
                self.diversity_state_gain_6_plus_pit_3(),
                separators=(",", ":"),
                sort_keys=True,
            )
            valid_c = json.dumps(
                {
                    "player_pits": [1, 0, 7, 6, 6, 5],
                    "opponent_pits": [5, 4, 4, 4, 4, 0],
                    "player_store": 1,
                    "opponent_store": 2,
                    "current_player": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            )

            tactical_rows = [
                {
                    "canonical_state": contradictory,
                    "raw_state": self.contradictory_state(),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 10.0,
                    "policy": [
                        0.06637731194496155,
                        0.3550935685634613,
                        0.0543360710144043,
                        0.3244529366493225,
                        0.19974012672901154,
                        0.0,
                    ],
                    "teacher_selected_move": 1,
                },
                {
                    "canonical_state": rejected_support,
                    "raw_state": self.rejected_support_state(),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 9.5,
                    "policy": [0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
                    "teacher_selected_move": 3,
                },
                {
                    "canonical_state": valid_a,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 9.0,
                    "policy": [0.0, 0.1, 0.0, 0.8, 0.1, 0.0],
                },
                {
                    "canonical_state": valid_b,
                    "raw_state": self.diversity_state_gain_6_plus_pit_3(),
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 8.0,
                    "policy": self.diversity_policy_gain_6_plus_pit_3(),
                },
                {
                    "canonical_state": valid_c,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 7.0,
                    "policy": [0.0, 0.0, 0.8, 0.1, 0.1, 0.0],
                },
            ]
            tactical_path.write_text(
                "\n".join(json.dumps(row) for row in tactical_rows) + "\n",
                encoding="utf-8",
            )

            forensic_rows = [
                {
                    "state": self.contradictory_state(),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": self.rejected_support_state(),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_a),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_b),
                    "bucket": "capture_available",
                    "reference_move": 1,
                },
                {
                    "state": json.loads(valid_c),
                    "bucket": "capture_available",
                    "reference_move": 2,
                },
            ]
            forensic_path.write_text(
                json.dumps({"systems": {"challenger": {"rows": forensic_rows}}}),
                encoding="utf-8",
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: module.teacher_label_regression_row(
                    raw_state,
                    source_id="missed_capture_f67bd4k0_move_28",
                    move_number=28,
                ),
                forensic_suite_path=forensic_path,
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            canonical_states = [row["canonical_state"] for row in built_rows]

            self.assertIn(contradictory, canonical_states)
            self.assertNotIn(rejected_support, canonical_states)
            self.assertEqual(5, len(built_rows))

    def test_build_dataset_targets_five_rows_with_support_row_second(self):
        rows = [
            self.capture_candidate(
                "shape-a-top",
                raw_state=self.contradictory_state(),
                policy=self.contradictory_policy(),
                priority=10.0,
                motif_score=15,
                teacher_selected_move=1,
            ),
            self.capture_candidate(
                "shape-a-clone",
                raw_state=self.rejected_support_state(),
                policy=self.rejected_support_policy(),
                priority=9.5,
                motif_score=14,
                teacher_selected_move=3,
            ),
            self.capture_candidate(
                "shape-b",
                raw_state=self.regression_state(),
                policy=self.regression_policy(),
                priority=9.0,
                motif_score=14,
                teacher_selected_move=1,
            ),
            self.capture_candidate(
                "shape-c",
                raw_state=self.diversity_state_gain_4_5_pit_4(),
                policy=self.diversity_policy_gain_4_5_pit_4(),
                priority=8.0,
                motif_score=13,
            ),
            self.capture_candidate(
                "shape-d",
                raw_state=self.diversity_state_gain_2_3_pit_3(),
                policy=self.diversity_policy_gain_2_3_pit_3(),
                priority=7.0,
                motif_score=12,
            ),
        ]
        support_scores = {
            row["canonical_state"]: {
                "score": row["motif_support_score"],
                "motif_protective": True,
            }
            for row in rows
        }

        with patch.object(
            module,
            "score_candidate_support_row",
            side_effect=lambda row, signature: support_scores[row["canonical_state"]],
        ):
            built_rows = module.select_capture_rows(
                rows, limit=4, regression_signature=self.regression_state()
            )

        self.assertEqual(
            ["shape-a-top", "shape-b", "shape-c", "shape-d"],
            [row["canonical_state"] for row in built_rows],
        )

    def test_select_capture_rows_reuses_shape_only_after_distinct_shapes_exhausted(
        self,
    ):
        rows = [
            self.capture_candidate(
                "shape-a-top",
                raw_state=self.contradictory_state(),
                policy=self.contradictory_policy(),
                priority=10.0,
                motif_score=15,
                teacher_selected_move=1,
            ),
            self.capture_candidate(
                "shape-a-clone",
                raw_state=self.rejected_support_state(),
                policy=self.rejected_support_policy(),
                priority=9.8,
                motif_score=14,
                teacher_selected_move=3,
            ),
            self.capture_candidate(
                "shape-b",
                raw_state=self.regression_state(),
                policy=self.regression_policy(),
                priority=9.0,
                motif_score=14,
                teacher_selected_move=1,
            ),
        ]
        support_scores = {
            row["canonical_state"]: {
                "score": row["motif_support_score"],
                "motif_protective": True,
            }
            for row in rows
        }

        with patch.object(
            module,
            "score_candidate_support_row",
            side_effect=lambda row, signature: support_scores[row["canonical_state"]],
        ):
            built_rows = module.select_capture_rows(
                rows, limit=4, regression_signature=self.regression_state()
            )

        self.assertEqual(
            ["shape-a-top", "shape-b", "shape-a-clone"],
            [row["canonical_state"] for row in built_rows],
        )

    def test_select_capture_rows_skips_candidate_without_real_shape_key(self):
        rows = [
            self.capture_candidate(
                "shape-a-top",
                raw_state=self.contradictory_state(),
                policy=self.contradictory_policy(),
                priority=10.0,
                motif_score=15,
                teacher_selected_move=1,
            ),
            {
                "canonical_state": "invalid-shape-row",
                "state": [0.0] * 27,
                "bucket": "capture_available",
                "bucket_group": "tactical",
                "input_encoding": "kalah_v3",
                "policy_target_mode": "sharpened",
                "value_target_mode": "sharpened",
                "value": 0.5,
                "priority_score": 9.0,
                "motif_support_score": 14,
                "policy": [0.0, 0.7, 0.0, 0.2, 0.1, 0.0],
            },
        ]
        support_scores = {
            row["canonical_state"]: {
                "score": row["motif_support_score"],
                "motif_protective": True,
            }
            for row in rows
        }

        with patch.object(
            module,
            "score_candidate_support_row",
            side_effect=lambda row, signature: support_scores[row["canonical_state"]],
        ):
            built_rows = module.select_capture_rows(
                rows, limit=4, regression_signature=self.regression_state()
            )

        self.assertEqual(
            ["shape-a-top"], [row["canonical_state"] for row in built_rows]
        )

    def test_build_dataset_reconstructs_raw_state_from_canonical_state_for_motif_filtering(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="capture-protection-") as tmp:
            tmp_path = Path(tmp)
            regression_path = tmp_path / "superhuman_regression_positions.json"
            tactical_path = tmp_path / "tactical_replay_train.jsonl"
            forensic_path = tmp_path / "mining_forensic_input.json"
            out_path = tmp_path / "tactical_capture_protection.jsonl"

            regression_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "missed_capture_f67bd4k0_move_28",
                            "state": self.regression_state(),
                            "expected_move": 1,
                            "acceptable_moves": [1],
                            "move_number": 28,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            contradictory = json.dumps(
                self.contradictory_state(), separators=(",", ":"), sort_keys=True
            )
            rejected_support = json.dumps(
                self.rejected_support_state(), separators=(",", ":"), sort_keys=True
            )
            valid_a = json.dumps(
                {
                    "player_pits": [1, 6, 6, 6, 5, 0],
                    "opponent_pits": [5, 5, 1, 5, 5, 0],
                    "player_store": 1,
                    "opponent_store": 2,
                    "current_player": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            valid_b = json.dumps(
                self.diversity_state_gain_6_plus_pit_3(),
                separators=(",", ":"),
                sort_keys=True,
            )
            valid_c = json.dumps(
                {
                    "player_pits": [1, 0, 7, 6, 6, 5],
                    "opponent_pits": [5, 4, 4, 4, 4, 0],
                    "player_store": 1,
                    "opponent_store": 2,
                    "current_player": 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            )

            tactical_rows = [
                {
                    "canonical_state": contradictory,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 10.0,
                    "policy": [
                        0.06637731194496155,
                        0.3550935685634613,
                        0.0543360710144043,
                        0.3244529366493225,
                        0.19974012672901154,
                        0.0,
                    ],
                    "teacher_selected_move": 1,
                },
                {
                    "canonical_state": rejected_support,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 9.5,
                    "policy": [0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
                    "teacher_selected_move": 3,
                },
                {
                    "canonical_state": valid_a,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 9.0,
                    "policy": [0.0, 0.1, 0.0, 0.8, 0.1, 0.0],
                },
                {
                    "canonical_state": valid_b,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 8.0,
                    "policy": self.diversity_policy_gain_6_plus_pit_3(),
                },
                {
                    "canonical_state": valid_c,
                    "state": [0.0] * 27,
                    "bucket": "capture_available",
                    "bucket_group": "tactical",
                    "input_encoding": "kalah_v3",
                    "policy_target_mode": "sharpened",
                    "value_target_mode": "sharpened",
                    "value": 0.5,
                    "priority_score": 7.0,
                    "policy": [0.0, 0.0, 0.8, 0.1, 0.1, 0.0],
                },
            ]
            tactical_path.write_text(
                "\n".join(json.dumps(row) for row in tactical_rows) + "\n",
                encoding="utf-8",
            )

            forensic_rows = [
                {
                    "state": self.contradictory_state(),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": self.rejected_support_state(),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_a),
                    "bucket": "capture_available",
                    "reference_move": 3,
                },
                {
                    "state": json.loads(valid_b),
                    "bucket": "capture_available",
                    "reference_move": 1,
                },
                {
                    "state": json.loads(valid_c),
                    "bucket": "capture_available",
                    "reference_move": 2,
                },
            ]
            forensic_path.write_text(
                json.dumps({"systems": {"challenger": {"rows": forensic_rows}}}),
                encoding="utf-8",
            )

            module.build_capture_protection_dataset(
                regression_positions_path=regression_path,
                tactical_replay_path=tactical_path,
                out_path=out_path,
                teacher_labeler=lambda raw_state: module.teacher_label_regression_row(
                    raw_state,
                    source_id="missed_capture_f67bd4k0_move_28",
                    move_number=28,
                ),
                forensic_suite_path=forensic_path,
            )

            built_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            canonical_states = [row["canonical_state"] for row in built_rows]

            self.assertIn(contradictory, canonical_states)
            self.assertNotIn(rejected_support, canonical_states)
            self.assertEqual(5, len(built_rows))

    def test_select_capture_rows_falls_back_to_skipfix_when_no_contradiction_clears_threshold(
        self,
    ):
        regression_signature = module.extract_regression_motif_signature(
            self.regression_state(), expected_move=1
        )
        rows = [
            {
                "canonical_state": json.dumps(
                    self.rejected_support_state(), separators=(",", ":"), sort_keys=True
                ),
                "raw_state": self.rejected_support_state(),
                "state": [0.0] * 27,
                "bucket": "capture_available",
                "bucket_group": "tactical",
                "input_encoding": "kalah_v3",
                "policy_target_mode": "sharpened",
                "value_target_mode": "sharpened",
                "value": 0.5,
                "priority_score": 9.5,
                "policy": [0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
                "teacher_selected_move": 3,
            }
        ]
        selected = module.select_capture_rows(
            rows,
            forensic_reference_moves={rows[0]["canonical_state"]: 3},
            regression_signature=regression_signature,
        )

        self.assertEqual([], selected)

    def test_score_candidate_support_row_rejects_row_without_extra_turn(self):
        signature = module.extract_regression_motif_signature(
            self.regression_state(), expected_move=1
        )
        row = {
            "raw_state": {
                "player_pits": [5, 0, 5, 5, 5, 0],
                "opponent_pits": [1, 0, 7, 7, 6, 5],
                "player_store": 1,
                "opponent_store": 1,
                "current_player": 0,
            },
            "policy": [0.9, 0.0, 0.05, 0.03, 0.02, 0.0],
        }

        with self.assertRaisesRegex(
            ValueError, "candidate row must expose both capture and extra-turn moves"
        ):
            module.score_candidate_support_row(row, signature)


if __name__ == "__main__":
    unittest.main()
