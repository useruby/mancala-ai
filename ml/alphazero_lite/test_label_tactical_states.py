import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import train as train_module
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite import label_tactical_states
from ml.alphazero_lite.input_encodings import feature_count_for
from ml.alphazero_lite.kalah_rules import KalahGame


class _FakeChild:
    def __init__(self, visits, wins):
        self.visits = visits
        self.wins = wins


class _FakeRoot:
    def __init__(self, children, visits, wins):
        self.children = children
        self.visits = visits
        self.wins = wins


class _FakeMCTS:
    def __init__(self, game, *, simulations, seed=None):
        self.game = game
        self.simulations = simulations
        self.seed = seed

    def search_root(self):
        legal_moves = self.game.possible_moves()
        wins = 3.5 if self.simulations >= 18 else 0.0
        children = {
            move: _FakeChild(visits=(index + 1) * 3, wins=float(index + 1))
            for index, move in enumerate(legal_moves)
        }
        total_visits = sum(child.visits for child in children.values())
        return _FakeRoot(children=children, visits=max(total_visits, 1), wins=wins)


class _MissingLegalMoveFakeMCTS(_FakeMCTS):
    def search_root(self):
        root = super().search_root()
        legal_moves = self.game.possible_moves()
        if legal_moves:
            root.children.pop(legal_moves[-1], None)
        return root


class LabelTacticalStatesTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def test_argument_parser_requires_target_modes_and_input_encoding(self):
        parser = label_tactical_states.build_argument_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--input",
                    "in.jsonl",
                    "--policy-simulations",
                    "4",
                    "--value-simulations",
                    "5",
                    "--seed",
                    "11",
                    "--out-labeled",
                    "labeled.jsonl",
                    "--out-tactical-train",
                    "tactical.jsonl",
                    "--out-preservation-train",
                    "preservation.jsonl",
                ]
            )

    def test_labeled_rows_emit_raw_state_and_encoded_state_with_supported_target_modes(
        self,
    ):
        raw_state = self._early_extra_turn_state()
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "move_index": 9,
                "selection_reasons": ["student_teacher_disagreement"],
                "source_artifacts": ["forensics.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 12.5,
            }
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS", _FakeMCTS
        ):
            for policy_mode in train_module.SUPPORTED_POLICY_TARGET_MODES:
                for value_mode in train_module.SUPPORTED_VALUE_TARGET_MODES:
                    labeled_rows = label_tactical_states.label_rows(
                        mined_rows,
                        policy_simulations=12,
                        value_simulations=18,
                        seed=7,
                        policy_target_mode=policy_mode,
                        value_target_mode=value_mode,
                        input_encoding="kalah_v3",
                    )

                    self.assertEqual(1, len(labeled_rows))
                    labeled = labeled_rows[0]
                    self.assertEqual(
                        canonical_state_key(raw_state), labeled["canonical_state"]
                    )
                    self.assertEqual(raw_state, labeled["raw_state"])
                    self.assertEqual(
                        raw_state["current_player"], labeled["side_to_move"]
                    )
                    self.assertEqual(
                        KalahGame.from_state(raw_state).possible_moves(),
                        labeled["legal_moves"],
                    )
                    self.assertEqual(
                        feature_count_for("kalah_v3"), len(labeled["state"])
                    )
                    self.assertEqual("early_extra_turn", labeled["bucket"])
                    self.assertEqual("tactical", labeled["bucket_group"])
                    self.assertEqual(policy_mode, labeled["policy_target_mode"])
                    self.assertEqual(value_mode, labeled["value_target_mode"])
                    self.assertEqual("kalah_v3", labeled["input_encoding"])
                    self.assertEqual(
                        ["student_teacher_disagreement"], labeled["selection_reasons"]
                    )
                    self.assertEqual(["forensics.json"], labeled["source_artifacts"])
                    self.assertEqual(
                        [{"kind": "forensic_suite"}], labeled["source_runs"]
                    )
                    self.assertEqual(12.5, labeled["priority_score"])
                    self.assertEqual(12, labeled["teacher_policy_simulations"])
                    self.assertEqual(18, labeled["teacher_value_simulations"])
                    self.assertEqual(7, labeled["teacher_seed"])
                    self.assertEqual(7, labeled["teacher_policy_seed"])
                    self.assertEqual(10007, labeled["teacher_value_seed"])
                    self.assertIn("teacher_selected_move", labeled)
                    self.assertIn("teacher_child_stats", labeled)
                    self.assertNotEqual(0.0, labeled["value"])

    def test_label_rows_preserves_teacher_seed_contract_by_row_index(self):
        raw_state = self._early_extra_turn_state()
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "move_index": 3,
                "selection_reasons": ["student_teacher_disagreement"],
                "source_artifacts": ["forensics-a.json"],
                "source_runs": [{"kind": "forensic_suite", "run": "a"}],
                "priority_score": 10.0,
            },
            {
                "canonical_state": canonical_state_key(self._opening_state()),
                "state": self._opening_state(),
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 4, 5],
                "move_index": 4,
                "selection_reasons": ["large_value_error"],
                "source_artifacts": ["forensics-b.json"],
                "source_runs": [{"kind": "forensic_suite", "run": "b"}],
                "priority_score": 11.0,
            },
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS", _FakeMCTS
        ):
            labeled_rows = label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=101,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        self.assertEqual(101, labeled_rows[0]["teacher_seed"])
        self.assertEqual(102, labeled_rows[1]["teacher_seed"])
        self.assertEqual(101, labeled_rows[0]["teacher_policy_seed"])
        self.assertEqual(102, labeled_rows[1]["teacher_policy_seed"])
        self.assertEqual(10101, labeled_rows[0]["teacher_value_seed"])
        self.assertEqual(10102, labeled_rows[1]["teacher_value_seed"])

    def test_label_rows_rejects_mined_row_consistency_mismatches(self):
        raw_state = self._early_extra_turn_state()
        base_row = {
            "canonical_state": canonical_state_key(raw_state),
            "state": raw_state,
            "side_to_move": raw_state["current_player"],
            "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
            "move_index": 9,
            "selection_reasons": ["student_teacher_disagreement"],
            "source_artifacts": ["forensics.json"],
            "source_runs": [{"kind": "forensic_suite"}],
            "priority_score": 12.5,
        }

        bad_side = dict(base_row, side_to_move=0)
        with self.assertRaisesRegex(
            ValueError, "side_to_move must match state.current_player"
        ):
            label_tactical_states.label_rows(
                [bad_side],
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        bad_legal_moves = dict(base_row, legal_moves=[0, 1])
        with self.assertRaisesRegex(
            ValueError, "legal_moves must match state legal moves"
        ):
            label_tactical_states.label_rows(
                [bad_legal_moves],
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        bad_canonical = dict(base_row, canonical_state="wrong")
        with self.assertRaisesRegex(ValueError, "canonical_state does not match state"):
            label_tactical_states.label_rows(
                [bad_canonical],
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

    def test_label_rows_rejects_rows_without_ply_metadata(self):
        raw_state = self._opening_state()
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "selection_reasons": ["large_value_error"],
                "source_artifacts": ["forensics.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 8.0,
            }
        ]

        with self.assertRaisesRegex(ValueError, "row must include move_index or ply"):
            label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

    def test_label_rows_accepts_mined_rows_with_ply_field(self):
        raw_state = self._opening_state()
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "ply": 8,
                "selection_reasons": ["large_value_error"],
                "source_artifacts": ["forensics.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 8.0,
            }
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS", _FakeMCTS
        ):
            labeled_rows = label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        self.assertEqual(1, len(labeled_rows))
        self.assertEqual("opening_plies_1_8", labeled_rows[0]["bucket"])

    def test_label_rows_skips_unclassifiable_forensic_states(self):
        unclassifiable_state = {
            "player_pits": [0, 2, 1, 8, 0, 7],
            "opponent_pits": [0, 9, 6, 6, 0, 5],
            "player_store": 3,
            "opponent_store": 1,
            "current_player": 0,
        }
        mined_rows = [
            {
                "canonical_state": canonical_state_key(unclassifiable_state),
                "state": unclassifiable_state,
                "side_to_move": unclassifiable_state["current_player"],
                "legal_moves": KalahGame.from_state(
                    unclassifiable_state
                ).possible_moves(),
                "ply": 9,
                "selection_reasons": ["student_teacher_disagreement"],
                "source_artifacts": ["forensics.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 10.0,
            }
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS", _FakeMCTS
        ):
            labeled_rows = label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        self.assertEqual([], labeled_rows)

    def test_label_rows_skips_unclassifiable_rows_without_disrupting_neighboring_rows(
        self,
    ):
        valid_state = self._opening_state()
        unclassifiable_state = {
            "player_pits": [0, 2, 1, 8, 0, 7],
            "opponent_pits": [0, 9, 6, 6, 0, 5],
            "player_store": 3,
            "opponent_store": 1,
            "current_player": 0,
        }
        mined_rows = [
            {
                "canonical_state": canonical_state_key(valid_state),
                "state": valid_state,
                "side_to_move": valid_state["current_player"],
                "legal_moves": KalahGame.from_state(valid_state).possible_moves(),
                "ply": 8,
                "selection_reasons": ["large_value_error"],
                "source_artifacts": ["forensics-a.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 8.0,
            },
            {
                "canonical_state": canonical_state_key(unclassifiable_state),
                "state": unclassifiable_state,
                "side_to_move": unclassifiable_state["current_player"],
                "legal_moves": KalahGame.from_state(
                    unclassifiable_state
                ).possible_moves(),
                "ply": 9,
                "selection_reasons": ["student_teacher_disagreement"],
                "source_artifacts": ["forensics-b.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 10.0,
            },
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS", _FakeMCTS
        ):
            labeled_rows = label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        self.assertEqual(1, len(labeled_rows))
        self.assertEqual(
            canonical_state_key(valid_state), labeled_rows[0]["canonical_state"]
        )
        self.assertEqual(7, labeled_rows[0]["teacher_seed"])

    def test_label_rows_rejects_missing_teacher_coverage_for_legal_move(self):
        raw_state = self._early_extra_turn_state()
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "move_index": 9,
                "selection_reasons": ["student_teacher_disagreement"],
                "source_artifacts": ["forensics.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 12.5,
            }
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS",
            _MissingLegalMoveFakeMCTS,
        ):
            with self.assertRaisesRegex(
                ValueError, "teacher child_stats missing legal moves"
            ):
                label_tactical_states.label_rows(
                    mined_rows,
                    policy_simulations=12,
                    value_simulations=18,
                    seed=7,
                    policy_target_mode="default",
                    value_target_mode="default",
                    input_encoding="kalah_v1",
                )

    def test_label_rows_rejects_invalid_target_modes(self):
        raw_state = self._early_extra_turn_state()
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "move_index": 9,
            }
        ]

        with self.assertRaisesRegex(ValueError, "unsupported policy_target_mode"):
            label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="bogus",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )

        with self.assertRaisesRegex(ValueError, "unsupported value_target_mode"):
            label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=12,
                value_simulations=18,
                seed=7,
                policy_target_mode="default",
                value_target_mode="bogus",
                input_encoding="kalah_v1",
            )

    def test_policy_sums_to_one_and_assigns_zero_to_illegal_moves(self):
        raw_state = {
            "player_pits": [0, 1, 0, 0, 0, 1],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        mined_rows = [
            {
                "canonical_state": canonical_state_key(raw_state),
                "state": raw_state,
                "side_to_move": raw_state["current_player"],
                "legal_moves": KalahGame.from_state(raw_state).possible_moves(),
                "move_index": 9,
                "selection_reasons": ["student_teacher_disagreement"],
                "source_artifacts": ["forensics.json"],
                "source_runs": [{"kind": "forensic_suite"}],
                "priority_score": 12.5,
            }
        ]

        with mock.patch(
            "ml.alphazero_lite.label_tactical_states.classic_mcts.MCTS", _FakeMCTS
        ):
            labeled = label_tactical_states.label_rows(
                mined_rows,
                policy_simulations=10,
                value_simulations=10,
                seed=3,
                policy_target_mode="sharpened",
                value_target_mode="default",
                input_encoding="kalah_v1",
            )[0]

        legal_moves = set(KalahGame.from_state(raw_state).possible_moves())
        self.assertAlmostEqual(1.0, sum(labeled["policy"]), places=6)
        for move, probability in enumerate(labeled["policy"]):
            if move not in legal_moves:
                self.assertEqual(0.0, probability)

    def test_split_train_rows_separates_tactical_vs_preservation_rows(self):
        labeled_rows = [
            {
                "state": [0.1] * 15,
                "raw_state": self._early_extra_turn_state(),
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 0.5,
                "bucket": "early_extra_turn",
                "bucket_group": "tactical",
                "policy_target_mode": "default",
                "value_target_mode": "default",
            },
            {
                "state": [0.2] * 15,
                "raw_state": self._opening_state(),
                "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "value": -0.25,
                "bucket": "opening_plies_1_8",
                "bucket_group": "preservation",
                "policy_target_mode": "default",
                "value_target_mode": "default",
            },
        ]

        tactical_rows, preservation_rows = label_tactical_states.split_train_rows(
            labeled_rows
        )

        self.assertEqual(["early_extra_turn"], [row["bucket"] for row in tactical_rows])
        self.assertEqual(
            ["opening_plies_1_8"], [row["bucket"] for row in preservation_rows]
        )
        self.assertNotIn("raw_state", tactical_rows[0])
        self.assertNotIn("raw_state", preservation_rows[0])

    def test_cli_writes_labeled_and_train_outputs(self):
        with tempfile.TemporaryDirectory(prefix="azlite-label-tactical-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "mined.jsonl"
            labeled_path = tmp_path / "labeled.jsonl"
            tactical_path = tmp_path / "tactical.jsonl"
            preservation_path = tmp_path / "preservation.jsonl"

            self._write_jsonl(
                input_path,
                [
                    {
                        "canonical_state": canonical_state_key(
                            self._early_extra_turn_state()
                        ),
                        "state": self._early_extra_turn_state(),
                        "side_to_move": self._early_extra_turn_state()[
                            "current_player"
                        ],
                        "legal_moves": KalahGame.from_state(
                            self._early_extra_turn_state()
                        ).possible_moves(),
                        "move_index": 9,
                        "selection_reasons": ["student_teacher_disagreement"],
                        "source_artifacts": ["forensics-a.json"],
                        "source_runs": [{"kind": "forensic_suite"}],
                        "priority_score": 12.5,
                    },
                    {
                        "canonical_state": canonical_state_key(self._opening_state()),
                        "state": self._opening_state(),
                        "side_to_move": 0,
                        "legal_moves": KalahGame.from_state(
                            self._opening_state()
                        ).possible_moves(),
                        "move_index": 4,
                        "selection_reasons": ["large_value_error"],
                        "source_artifacts": ["forensics-b.json"],
                        "source_runs": [{"kind": "forensic_suite"}],
                        "priority_score": 9.0,
                    },
                ],
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/label_tactical_states.py",
                    "--input",
                    str(input_path),
                    "--policy-simulations",
                    "24",
                    "--value-simulations",
                    "24",
                    "--seed",
                    "11",
                    "--policy-target-mode",
                    "default",
                    "--value-target-mode",
                    "default",
                    "--input-encoding",
                    "kalah_v1",
                    "--out-labeled",
                    str(labeled_path),
                    "--out-tactical-train",
                    str(tactical_path),
                    "--out-preservation-train",
                    str(preservation_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)

            labeled_rows = self._read_jsonl(labeled_path)
            tactical_rows = self._read_jsonl(tactical_path)
            preservation_rows = self._read_jsonl(preservation_path)

            self.assertEqual(2, len(labeled_rows))
            self.assertEqual(1, len(tactical_rows))
            self.assertEqual(1, len(preservation_rows))
            self.assertIn("raw_state", labeled_rows[0])
            self.assertNotIn("raw_state", tactical_rows[0])
            self.assertNotIn("raw_state", preservation_rows[0])

            train_module.load_jsonl(tactical_path)
            train_module.load_jsonl(preservation_path)

    def test_cli_returns_error_for_malformed_jsonl(self):
        with tempfile.TemporaryDirectory(prefix="azlite-label-tactical-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "mined.jsonl"
            labeled_path = tmp_path / "labeled.jsonl"
            tactical_path = tmp_path / "tactical.jsonl"
            preservation_path = tmp_path / "preservation.jsonl"
            input_path.write_text("{not json}\n", encoding="utf-8")

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/label_tactical_states.py",
                    "--input",
                    str(input_path),
                    "--policy-simulations",
                    "24",
                    "--value-simulations",
                    "24",
                    "--seed",
                    "11",
                    "--policy-target-mode",
                    "default",
                    "--value-target-mode",
                    "default",
                    "--input-encoding",
                    "kalah_v1",
                    "--out-labeled",
                    str(labeled_path),
                    "--out-tactical-train",
                    str(tactical_path),
                    "--out-preservation-train",
                    str(preservation_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertNotEqual("", result.stderr.strip())
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_returns_error_for_missing_required_row_fields_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="azlite-label-tactical-") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "mined.jsonl"
            labeled_path = tmp_path / "labeled.jsonl"
            tactical_path = tmp_path / "tactical.jsonl"
            preservation_path = tmp_path / "preservation.jsonl"
            self._write_jsonl(input_path, [{"state": self._opening_state()}])

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/label_tactical_states.py",
                    "--input",
                    str(input_path),
                    "--policy-simulations",
                    "24",
                    "--value-simulations",
                    "24",
                    "--seed",
                    "11",
                    "--policy-target-mode",
                    "default",
                    "--value-target-mode",
                    "default",
                    "--input-encoding",
                    "kalah_v1",
                    "--out-labeled",
                    str(labeled_path),
                    "--out-tactical-train",
                    str(tactical_path),
                    "--out-preservation-train",
                    str(preservation_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertNotEqual("", result.stderr.strip())
            self.assertNotIn("Traceback", result.stderr)

    def _opening_state(self):
        return {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

    def _early_extra_turn_state(self):
        return {
            "player_pits": [1, 1, 1, 1, 2, 4],
            "opponent_pits": [1, 1, 1, 1, 2, 1],
            "player_store": 10,
            "opponent_store": 10,
            "current_player": 1,
        }

    def _write_jsonl(self, path, rows):
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )

    def _read_jsonl(self, path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


if __name__ == "__main__":
    unittest.main()
