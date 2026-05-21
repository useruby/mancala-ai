import json
from pathlib import Path
import tempfile
import unittest


class BuildTacticalOpeningCaptureFamilyReplayTest(unittest.TestCase):
    def test_default_opening_capture_family_source_stays_within_repo(self):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        repo_root = Path(__file__).resolve().parents[2]
        source_path = module.DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE.resolve()

        self.assertEqual(
            repo_root / "ml/alphazero_lite/tactical_balanced_replay_source.jsonl",
            source_path,
        )
        self.assertTrue(source_path.is_relative_to(repo_root))

    def capture_row(
        self,
        canonical_state,
        *,
        teacher_selected_move=3,
        legal_moves=None,
        priority=1.0,
        bucket="capture_available",
        ply=None,
        move_number=None,
        source_runs=None,
    ):
        row = {
            "canonical_state": canonical_state,
            "state": [0.0] * 27,
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4] if legal_moves is None else legal_moves,
            "bucket": bucket,
            "bucket_group": "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.25,
            "priority_score": priority,
            "policy": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "teacher_selected_move": teacher_selected_move,
        }
        if ply is not None:
            row["ply"] = ply
        if move_number is not None:
            row["move_number"] = move_number
        if source_runs is not None:
            row["source_runs"] = source_runs
        return row

    def nearby_row(self, bucket, index, *, priority):
        return {
            "canonical_state": f"{bucket}-{index}",
            "state": [0.0] * 27,
            "side_to_move": 0,
            "legal_moves": [0],
            "bucket": bucket,
            "bucket_group": "preservation"
            if bucket == "starvation_pressure"
            else "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.25,
            "priority_score": priority,
            "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }

    def preservation_capture_row(self, canonical_state, *, priority, variant):
        rows = {
            "gain_4_5": {
                "raw_state": {
                    "player_pits": [5, 5, 1, 0, 6, 6],
                    "opponent_pits": [5, 5, 4, 4, 4, 0],
                    "player_store": 2,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "legal_moves": [0, 1, 2, 4, 5],
                "policy": [
                    0.0013778912834823132,
                    0.24870026111602783,
                    0.059389978647232056,
                    0.0,
                    0.4638570547103882,
                    0.22667483985424042,
                ],
                "teacher_selected_move": 4,
            },
            "gain_2_3": {
                "raw_state": {
                    "player_pits": [4, 4, 4, 4, 0, 5],
                    "opponent_pits": [0, 1, 6, 6, 6, 6],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "legal_moves": [0, 1, 2, 3, 5],
                "policy": [
                    0.01769360341131687,
                    0.03087548352777958,
                    0.44671961665153503,
                    0.3265913426876068,
                    0.0,
                    0.17811991274356842,
                ],
                "teacher_selected_move": 2,
            },
        }
        selected = rows[variant]
        return {
            "canonical_state": canonical_state,
            "state": [0.0] * 27,
            "raw_state": selected["raw_state"],
            "side_to_move": 0,
            "legal_moves": selected["legal_moves"],
            "bucket": "capture_available",
            "bucket_group": "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.5,
            "priority_score": priority,
            "policy": selected["policy"],
            "teacher_selected_move": selected["teacher_selected_move"],
        }

    def test_select_opening_capture_family_rows_requires_teacher_move_three_and_opening_phase(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        source_rows = [
            self.capture_row("valid-ply", priority=9.0, ply=4),
            self.capture_row(
                "wrong-teacher", priority=8.0, ply=4, teacher_selected_move=2
            ),
            self.capture_row("late-ply", priority=7.0, ply=5),
            self.capture_row(
                "wrong-legal-moves", priority=6.0, ply=4, legal_moves=[0, 1, 2, 3]
            ),
            self.capture_row("valid-move-number", priority=5.0, move_number=4),
            self.capture_row(
                "valid-provenance",
                priority=4.0,
                source_runs=[{"kind": "mined_position", "id": "capture_available-019"}],
            ),
            self.capture_row(
                "wrong-provenance",
                priority=3.0,
                source_runs=[{"kind": "mined_position", "id": "capture_available-018"}],
            ),
        ]

        rows = module._select_opening_capture_family_rows(source_rows)

        self.assertEqual(
            ["valid-ply", "valid-move-number", "valid-provenance"],
            [row["canonical_state"] for row in rows],
        )

    def test_select_opening_capture_family_rows_keeps_highest_priority_row_per_family_provenance(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        source_rows = [
            self.capture_row(
                "family-017-high",
                priority=9.0,
                source_runs=[{"kind": "mined_position", "id": "capture_available-017"}],
            ),
            self.capture_row(
                "family-017-low",
                priority=3.0,
                source_runs=[{"kind": "mined_position", "id": "capture_available-017"}],
            ),
            self.capture_row(
                "family-019",
                priority=8.0,
                source_runs=[{"kind": "mined_position", "id": "capture_available-019"}],
            ),
            self.capture_row(
                "family-024",
                priority=7.0,
                source_runs=[{"kind": "mined_position", "id": "capture_available-024"}],
            ),
        ]

        rows = module._select_opening_capture_family_rows(source_rows)

        self.assertEqual(
            ["family-017-high", "family-019", "family-024"],
            [row["canonical_state"] for row in rows],
        )

    def test_build_opening_capture_family_replay_dataset_writes_explicit_roles(self):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.jsonl"
            out_path = tmp_path / "tactical_opening_capture_family_replay.jsonl"

            source_rows = [
                self.capture_row("opening-family-1", priority=9.0, ply=4),
                self.capture_row(
                    "opening-family-2",
                    priority=8.0,
                    source_runs=[
                        {"kind": "mined_position", "id": "capture_available-024"}
                    ],
                ),
                self.preservation_capture_row(
                    "capture-preservation-1", priority=7.0, variant="gain_4_5"
                ),
                self.preservation_capture_row(
                    "capture-preservation-2", priority=6.0, variant="gain_2_3"
                ),
            ]
            for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure"):
                source_rows.extend(
                    self.nearby_row(bucket, index, priority=10 - index)
                    for index in range(2)
                )

            source_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )

            rows = module.build_opening_capture_family_replay_dataset(
                tactical_replay_path=source_path,
                out_path=out_path,
            )

            self.assertEqual(11, len(rows))
            self.assertEqual(
                rows,
                [
                    json.loads(line)
                    for line in out_path.read_text(encoding="utf-8").splitlines()
                ],
            )
            self.assertEqual(
                ["capture_protection"]
                + ["capture_preservation"] * 2
                + ["opening_capture_family"] * 2
                + ["nearby_preservation"] * 6,
                [row["replay_role"] for row in rows],
            )
            self.assertEqual(
                [
                    "high_imbalance",
                    "high_imbalance",
                    "high_value_swing",
                    "high_value_swing",
                    "starvation_pressure",
                    "starvation_pressure",
                ],
                [
                    row["bucket"]
                    for row in rows
                    if row["replay_role"] == "nearby_preservation"
                ],
            )

    def test_default_opening_capture_family_source_builds_opening_rows_and_sanitizes_artifacts(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tactical_opening_capture_family_replay.jsonl"

            rows = module.build_opening_capture_family_replay_dataset(
                tactical_replay_path=module.DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE,
                out_path=out_path,
            )

        opening_rows = [
            row for row in rows if row["replay_role"] == "opening_capture_family"
        ]
        self.assertGreaterEqual(len(opening_rows), 1)
        for row in opening_rows:
            for artifact in row.get("source_artifacts", []):
                self.assertFalse(Path(artifact).is_absolute(), msg=artifact)

    def test_default_opening_capture_family_source_retains_capture_protection_anchor(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tactical_opening_capture_family_replay.jsonl"

            rows = module.build_opening_capture_family_replay_dataset(
                tactical_replay_path=module.DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE,
                out_path=out_path,
            )

        protection_rows = [
            row for row in rows if row["replay_role"] == "capture_protection"
        ]
        self.assertEqual(1, len(protection_rows))
        self.assertEqual(
            "missed_capture_f67bd4k0_move_28",
            protection_rows[0]["source_runs"][0]["id"],
        )

    def test_default_opening_capture_family_source_limits_family_rows_to_three_distinct_provenance_ids(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tactical_opening_capture_family_replay.jsonl"

            rows = module.build_opening_capture_family_replay_dataset(
                tactical_replay_path=module.DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE,
                out_path=out_path,
            )

        family_rows = [
            row for row in rows if row["replay_role"] == "opening_capture_family"
        ]
        self.assertEqual(3, len(family_rows))
        self.assertEqual(
            ["capture_available-017", "capture_available-019", "capture_available-024"],
            sorted(row["source_runs"][0]["id"] for row in family_rows),
        )

    def test_default_opening_capture_family_source_retains_two_capture_preservation_rows(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_opening_capture_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "tactical_opening_capture_family_replay.jsonl"

            rows = module.build_opening_capture_family_replay_dataset(
                tactical_replay_path=module.DEFAULT_OPENING_CAPTURE_FAMILY_REPLAY_SOURCE,
                out_path=out_path,
            )

        preservation_rows = [
            row for row in rows if row["replay_role"] == "capture_preservation"
        ]
        self.assertEqual(2, len(preservation_rows))
