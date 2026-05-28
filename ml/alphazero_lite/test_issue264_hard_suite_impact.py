import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import run_issue264_hard_suite_impact as impact


class Issue264HardSuiteImpactCliTest(unittest.TestCase):
    def _flag_value(self, command, flag):
        return command[command.index(flag) + 1]

    def test_main_uses_shared_default_workers_when_flag_omitted(self):
        with tempfile.TemporaryDirectory(prefix="issue264-impact-") as tmp:
            tmp_path = Path(tmp)
            mined_path = tmp_path / "mined.jsonl"
            init_checkpoint = tmp_path / "baseline.npz"
            current_artifact = tmp_path / "current_artifact"
            out_dir = tmp_path / "out"
            mined_path.write_text(
                json.dumps(
                    {
                        "canonical_state": "state-a",
                        "state": {
                            "player_pits": [4, 4, 4, 4, 4, 4],
                            "opponent_pits": [4, 4, 4, 4, 4, 4],
                            "player_store": 0,
                            "opponent_store": 0,
                            "current_player": 0,
                        },
                        "legal_moves": [0, 1, 2, 3, 4, 5],
                        "priority_score": 13.5,
                        "selection_reasons": ["large_value_error"],
                        "source_artifacts": ["/tmp/mined.jsonl"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            init_checkpoint.write_bytes(b"baseline")
            current_artifact.mkdir()

            with mock.patch.object(impact, "run_command") as run_command:

                def side_effect(command, *, cwd):
                    if command[1].endswith("train.py"):
                        (out_dir / "challenger.npz").write_bytes(b"challenger")
                        return "saved checkpoint to challenger.npz\n"
                    if command[1].endswith("export_artifact.py"):
                        artifact_dir = out_dir / "challenger_artifact"
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        (artifact_dir / "weights.json").write_text(
                            "{}", encoding="utf-8"
                        )
                        (artifact_dir / "metadata.json").write_text(
                            "{}", encoding="utf-8"
                        )
                        return "exported artifact to challenger_artifact\n"
                    if command[1].endswith("arena.py"):
                        arena_report = {
                            "schema": "arena_v1",
                            "games_played": 6,
                            "wins": 4,
                            "losses": 1,
                            "draws": 1,
                            "score": 0.75,
                            "promotion_decision": {"passed": True},
                            "hard_suite_buckets": {
                                "opening": {"games": 2, "score": 0.5},
                                "midgame": {"games": 2, "score": 1.0},
                                "late": {"games": 2, "score": None},
                            },
                        }
                        (out_dir / "arena_report.json").write_text(
                            json.dumps(arena_report), encoding="utf-8"
                        )
                        return "wrote arena report to arena_report.json\nscore=0.7500 passed=True\n"
                    raise AssertionError(command)

                run_command.side_effect = side_effect

                original_cwd = Path.cwd()
                os.chdir(tmp_path)
                try:
                    exit_code = impact.main(
                        [
                            "--mined-jsonl",
                            "mined.jsonl",
                            "--out-dir",
                            "out",
                            "--init-checkpoint",
                            "baseline.npz",
                            "--current-artifact",
                            "current_artifact",
                            "--top-n",
                            "1",
                            "--canonical-budget",
                            "32",
                            "--stronger-budget",
                            "64",
                            "--epochs",
                            "1",
                            "--batch-size",
                            "8",
                            "--games",
                            "6",
                            "--challenger-simulations",
                            "32",
                            "--current-simulations",
                            "32",
                            "--min-score",
                            "0.6",
                        ]
                    )
                finally:
                    os.chdir(original_cwd)

            self.assertEqual(0, exit_code)
            arena_command = run_command.call_args_list[2].args[0]
            self.assertEqual("24", self._flag_value(arena_command, "--workers"))

    def test_main_runs_train_export_and_arena_and_writes_combined_report(self):
        with tempfile.TemporaryDirectory(prefix="issue264-impact-") as tmp:
            tmp_path = Path(tmp)
            mined_path = tmp_path / "mined.jsonl"
            init_checkpoint = tmp_path / "baseline.npz"
            current_artifact = tmp_path / "current_artifact"
            out_dir = tmp_path / "out"
            mined_path.write_text(
                json.dumps(
                    {
                        "canonical_state": "state-a",
                        "state": {
                            "player_pits": [4, 4, 4, 4, 4, 4],
                            "opponent_pits": [4, 4, 4, 4, 4, 4],
                            "player_store": 0,
                            "opponent_store": 0,
                            "current_player": 0,
                        },
                        "legal_moves": [0, 1, 2, 3, 4, 5],
                        "priority_score": 13.5,
                        "selection_reasons": ["large_value_error"],
                        "source_artifacts": ["/tmp/mined.jsonl"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            init_checkpoint.write_bytes(b"baseline")
            current_artifact.mkdir()

            with mock.patch.object(impact, "run_command") as run_command:

                def side_effect(command, *, cwd):
                    if command[1].endswith("train.py"):
                        (out_dir / "challenger.npz").write_bytes(b"challenger")
                        return "saved checkpoint to challenger.npz\n"
                    if command[1].endswith("export_artifact.py"):
                        artifact_dir = out_dir / "challenger_artifact"
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        (artifact_dir / "weights.json").write_text(
                            "{}", encoding="utf-8"
                        )
                        (artifact_dir / "metadata.json").write_text(
                            "{}", encoding="utf-8"
                        )
                        return "exported artifact to challenger_artifact\n"
                    if command[1].endswith("arena.py"):
                        arena_report = {
                            "schema": "arena_v1",
                            "games_played": 6,
                            "wins": 4,
                            "losses": 1,
                            "draws": 1,
                            "score": 0.75,
                            "promotion_decision": {"passed": True},
                            "hard_suite_buckets": {
                                "opening": {"games": 2, "score": 0.5},
                                "midgame": {"games": 2, "score": 1.0},
                                "late": {"games": 2, "score": None},
                            },
                        }
                        (out_dir / "arena_report.json").write_text(
                            json.dumps(arena_report), encoding="utf-8"
                        )
                        return "wrote arena report to arena_report.json\nscore=0.7500 passed=True\n"
                    raise AssertionError(command)

                run_command.side_effect = side_effect

                original_cwd = Path.cwd()
                os.chdir(tmp_path)
                try:
                    exit_code = impact.main(
                        [
                            "--mined-jsonl",
                            "mined.jsonl",
                            "--out-dir",
                            "out",
                            "--init-checkpoint",
                            "baseline.npz",
                            "--current-artifact",
                            "current_artifact",
                            "--top-n",
                            "1",
                            "--canonical-budget",
                            "32",
                            "--stronger-budget",
                            "64",
                            "--epochs",
                            "1",
                            "--batch-size",
                            "8",
                            "--games",
                            "6",
                            "--challenger-simulations",
                            "32",
                            "--current-simulations",
                            "32",
                            "--workers",
                            "1",
                            "--min-score",
                            "0.6",
                        ]
                    )
                finally:
                    os.chdir(original_cwd)

            self.assertEqual(0, exit_code)
            report = json.loads(
                (out_dir / "issue264_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual("issue264_hard_suite_impact_v1", report["schema"])
            self.assertEqual(
                "promote_to_standard_step", report["recommendation"]["recommendation"]
            )
            self.assertEqual(0.75, report["arena"]["score"])
            self.assertEqual(1, report["label_report"]["pair_count"])
            self.assertEqual(3, run_command.call_count)
            self.assertTrue(Path(report["experiment"]["mined_jsonl"]).is_absolute())
            self.assertEqual(
                str(mined_path.resolve()), report["experiment"]["mined_jsonl"]
            )
            self.assertEqual(
                str(init_checkpoint.resolve()),
                report["training"]["baseline_checkpoint"],
            )
            self.assertEqual(
                str((out_dir / "challenger.npz").resolve()),
                report["training"]["challenger_checkpoint"],
            )
            self.assertEqual(
                str((out_dir / "challenger_artifact").resolve()),
                report["training"]["challenger_artifact_dir"],
            )

            repo_root = Path(__file__).resolve().parents[2]
            expected_python = impact.resolve_python_executable(repo_root)
            train_command = run_command.call_args_list[0].args[0]
            train_cwd = run_command.call_args_list[0].kwargs["cwd"]
            self.assertEqual(repo_root.resolve(), train_cwd)
            self.assertEqual(expected_python, train_command[0])
            self.assertEqual(
                str((repo_root / "ml/alphazero_lite/train.py").resolve()),
                train_command[1],
            )
            self.assertEqual(
                str((out_dir / "stronger_only.jsonl").resolve()),
                self._flag_value(train_command, "--data"),
            )
            self.assertEqual(
                str((out_dir / "challenger.npz").resolve()),
                self._flag_value(train_command, "--out"),
            )
            self.assertEqual(
                str(init_checkpoint.resolve()),
                self._flag_value(train_command, "--init-checkpoint"),
            )

            export_command = run_command.call_args_list[1].args[0]
            export_cwd = run_command.call_args_list[1].kwargs["cwd"]
            self.assertEqual(repo_root.resolve(), export_cwd)
            self.assertEqual(expected_python, export_command[0])
            self.assertEqual(
                str((repo_root / "ml/alphazero_lite/export_artifact.py").resolve()),
                export_command[1],
            )
            self.assertEqual(
                str((out_dir / "challenger.npz").resolve()),
                self._flag_value(export_command, "--checkpoint"),
            )
            self.assertEqual(
                str((out_dir / "challenger_artifact").resolve()),
                self._flag_value(export_command, "--out-dir"),
            )

            arena_command = run_command.call_args_list[2].args[0]
            arena_cwd = run_command.call_args_list[2].kwargs["cwd"]
            self.assertEqual(repo_root.resolve(), arena_cwd)
            self.assertEqual(expected_python, arena_command[0])
            self.assertEqual(
                str((repo_root / "ml/alphazero_lite/arena.py").resolve()),
                arena_command[1],
            )
            self.assertEqual(
                str((out_dir / "challenger_artifact").resolve()),
                self._flag_value(arena_command, "--challenger"),
            )
            self.assertEqual(
                str(current_artifact.resolve()),
                self._flag_value(arena_command, "--current"),
            )
            self.assertEqual(
                str((out_dir / "arena_report.json").resolve()),
                self._flag_value(arena_command, "--out"),
            )

    def test_main_fails_when_arena_report_omits_hard_suite_buckets(self):
        with tempfile.TemporaryDirectory(prefix="issue264-impact-") as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "out"
            out_dir.mkdir()

            (out_dir / "label_report.json").write_text(
                json.dumps(
                    {
                        "pair_count": 1,
                        "top1_disagreement_rate": 0.0,
                        "average_policy_divergence": 0.0,
                        "maximum_policy_divergence": 0.0,
                        "average_absolute_value_delta": 0.0,
                        "largest_disagreements": [],
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 2,
                        "wins": 1,
                        "losses": 1,
                        "draws": 0,
                        "score": 0.5,
                        "promotion_decision": {"passed": False},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "arena report must include hard_suite_buckets"
            ):
                impact.build_final_report_from_paths(
                    experiment={"top_n": 1, "seed": 42},
                    label_report_path=out_dir / "label_report.json",
                    arena_report_path=out_dir / "arena_report.json",
                    baseline_checkpoint=tmp_path / "baseline.npz",
                    challenger_checkpoint=out_dir / "challenger.npz",
                    challenger_artifact_dir=out_dir / "challenger_artifact",
                    min_score=0.6,
                )

    def test_resolve_python_executable_prefers_repo_local_venv(self):
        with tempfile.TemporaryDirectory(prefix="issue264-impact-python-") as tmp:
            repo_root = Path(tmp)
            system_python = repo_root / "python3"
            system_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            venv_python = repo_root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(system_python)

            self.assertEqual(
                str(venv_python), impact.resolve_python_executable(repo_root)
            )

    def test_resolve_python_executable_falls_back_to_current_interpreter(self):
        with tempfile.TemporaryDirectory(prefix="issue264-impact-python-") as tmp:
            repo_root = Path(tmp)

            self.assertEqual(
                str(Path(sys.executable).resolve()),
                impact.resolve_python_executable(repo_root),
            )
