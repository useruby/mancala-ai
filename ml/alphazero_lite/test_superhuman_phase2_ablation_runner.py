import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


class SuperhumanPhase2AblationRunnerTest(unittest.TestCase):
    def test_dry_run_prints_phase2_ablation_contract(self):
        repo_root = Path(__file__).resolve().parents[2]
        expected_configs = {
            "replay_balanced": {
                "config": "aggressive_v3_superhuman_phase2_ablation_replay_balanced.json",
                "run_id": "aggressive-v3-superhuman-ablation-replay-balanced",
            },
            "selfplay_only": {
                "config": "aggressive_v3_superhuman_phase2_ablation_selfplay_only.json",
                "run_id": "aggressive-v3-superhuman-ablation-selfplay-only",
            },
            "phase1_arch": {
                "config": "aggressive_v3_superhuman_phase2_ablation_phase1_arch.json",
                "run_id": "aggressive-v3-superhuman-ablation-phase1-arch",
            },
        }

        with tempfile.TemporaryDirectory(prefix="azlite-phase2-ablation-") as tmp:
            parent_artifact = Path(tmp) / "aggressive-v3-superhuman-iter1"
            parent_artifact.mkdir()

            for variant, expected in expected_configs.items():
                result = subprocess.run(
                    [
                        str(
                            repo_root / "script/ai/run_local_superhuman_phase2_ablation"
                        ),
                        "--variant",
                        variant,
                        "--parent-artifact",
                        str(parent_artifact),
                        "--dry-run",
                    ],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=os.environ.copy(),
                )

                self.assertEqual(0, result.returncode, msg=result.stderr)
                report = json.loads(result.stdout)
                runtime_config_path = Path(report["config_path"])
                runtime_config = json.loads(
                    runtime_config_path.read_text(encoding="utf-8")
                )
                self.assertTrue(
                    all(
                        step["command"][0] == report["pipeline_command"][0]
                        for step in runtime_config["steps"]
                        if step.get("command")
                    )
                )

                self.assertEqual(variant, report["variant"])
                self.assertTrue(report["config_path"].endswith("pipeline_config.json"))
                self.assertTrue(Path(report["pipeline_command"][0]).exists())
                self.assertIn(
                    "ml/alphazero_lite/pipeline.py", report["pipeline_command"]
                )
                self.assertTrue(
                    report["pipeline_command"][
                        report["pipeline_command"].index("--config") + 1
                    ].endswith("pipeline_config.json")
                )
                self.assertIn(
                    f"tmp/local_superhuman_phase2_ablation/{variant}/pipeline_config.json",
                    report["gate_command"][
                        report["gate_command"].index("--config-path") + 1
                    ],
                )
                self.assertEqual(
                    Path(report["pipeline_command"][0]), Path(report["gate_command"][0])
                )
                self.assertEqual(
                    report["pipeline_command"][0], report["gate_command"][0]
                )
                self.assertEqual(
                    str(repo_root / "script/ai/compare_superhuman_regressions"),
                    report["current_comparison_command"][0],
                )
                self.assertEqual(
                    str(repo_root / "script/ai/compare_superhuman_regressions"),
                    report["parent_comparison_command"][0],
                )
                self.assertIn(
                    "script/ai/local_promotion_gate", report["gate_command"][1]
                )
                self.assertNotEqual(
                    report["pipeline_command"][0],
                    report["current_comparison_command"][0],
                )
                self.assertNotEqual(
                    report["pipeline_command"][0],
                    report["parent_comparison_command"][0],
                )
                self.assertEqual(expected["run_id"], runtime_config["run_id"])

    def test_non_dry_run_executes_pipeline_gate_and_both_comparisons(self):
        source_repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-phase2-ablation-live-") as tmp:
            repo_root = Path(tmp) / "repo"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "model-artifact/current").mkdir(parents=True)
            (
                repo_root / "tmp/local_results_partial/aggressive-v3-superhuman-iter1"
            ).mkdir(parents=True)

            runner_path = repo_root / "script/ai/run_local_superhuman_phase2_ablation"
            runner_path.write_text(
                (
                    source_repo_root / "script/ai/run_local_superhuman_phase2_ablation"
                ).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            runner_path.chmod(0o755)

            config = {
                "run_id": "aggressive-v3-superhuman-ablation-replay-balanced",
                "start_iteration": 2,
                "iterations": 1,
                "versions_dir": "/tmp/placeholder",
                "current_path": "/tmp/placeholder-parent",
                "steps": [],
            }
            (
                repo_root
                / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2_ablation_replay_balanced.json"
            ).write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            (repo_root / "ml/alphazero_lite/pipeline.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "config_path = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])\n"
                "config = json.loads(config_path.read_text(encoding='utf-8'))\n"
                "iteration = int(config.get('start_iteration', 1))\n"
                "parent_replay = pathlib.Path(config['versions_dir']) / f\"{config['run_id']}-iter{iteration - 1}\" / 'self_play.jsonl'\n"
                "assert parent_replay.exists(), parent_replay\n"
                "candidate = pathlib.Path(config['versions_dir']) / f\"{config['run_id']}-iter{iteration}\"\n"
                "candidate.mkdir(parents=True, exist_ok=True)\n"
                "(candidate / 'checkpoint.npz').write_bytes(b'')\n"
                "print('pipeline_scaffold_complete')\n",
                encoding="utf-8",
            )

            (repo_root / "script/ai/local_promotion_gate").write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "candidate = pathlib.Path(sys.argv[sys.argv.index('--candidate-path') + 1])\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                "config_path = sys.argv[sys.argv.index('--config-path') + 1]\n"
                "out.write_text(json.dumps({'candidate_path': str(candidate), 'config_path': config_path, 'passed': True}), encoding='utf-8')\n"
                "print(json.dumps({'report_path': str(out)}))\n",
                encoding="utf-8",
            )
            (repo_root / "script/ai/local_promotion_gate").chmod(0o755)

            (repo_root / "script/ai/compare_superhuman_regressions").write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "baseline = sys.argv[sys.argv.index('--baseline-artifact') + 1]\n"
                "candidate = sys.argv[sys.argv.index('--candidate-artifact') + 1]\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
                "out.write_text(json.dumps({'baseline_artifact_path': baseline, 'candidate_artifact_path': candidate, 'comparisons': []}), encoding='utf-8')\n"
                "print(json.dumps({'out': str(out)}))\n",
                encoding="utf-8",
            )
            (repo_root / "script/ai/compare_superhuman_regressions").chmod(0o755)

            (
                repo_root
                / "tmp/local_results_partial/aggressive-v3-superhuman-iter1/self_play.jsonl"
            ).write_text(
                '{"game": 1}\n',
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(runner_path),
                    "--variant",
                    "replay_balanced",
                    "--parent-artifact",
                    str(
                        repo_root
                        / "tmp/local_results_partial/aggressive-v3-superhuman-iter1"
                    ),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(result.stdout)

            self.assertTrue(Path(report["candidate_path"]).exists())
            self.assertTrue(Path(report["gate_report_path"]).exists())
            self.assertTrue(Path(report["current_comparison_path"]).exists())
            self.assertTrue(Path(report["parent_comparison_path"]).exists())
            self.assertTrue(report["config_path"].endswith("pipeline_config.json"))

            gate_report = json.loads(
                Path(report["gate_report_path"]).read_text(encoding="utf-8")
            )
            current_comparison = json.loads(
                Path(report["current_comparison_path"]).read_text(encoding="utf-8")
            )
            parent_comparison = json.loads(
                Path(report["parent_comparison_path"]).read_text(encoding="utf-8")
            )

            self.assertEqual(report["candidate_path"], gate_report["candidate_path"])
            self.assertEqual(
                report["candidate_path"], current_comparison["candidate_artifact_path"]
            )
            self.assertEqual(
                report["candidate_path"], parent_comparison["candidate_artifact_path"]
            )
            self.assertEqual(
                str(repo_root / "model-artifact/current"),
                current_comparison["baseline_artifact_path"],
            )
            self.assertEqual(
                str(
                    repo_root
                    / "tmp/local_results_partial/aggressive-v3-superhuman-iter1"
                ),
                parent_comparison["baseline_artifact_path"],
            )

    def test_dry_run_invokes_regression_comparison_via_script_path(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-phase2-ablation-ruby-") as tmp:
            parent_artifact = Path(tmp) / "aggressive-v3-superhuman-iter1"
            parent_artifact.mkdir()

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_phase2_ablation"),
                    "--variant",
                    "replay_balanced",
                    "--parent-artifact",
                    str(parent_artifact),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(
                str(repo_root / "script/ai/compare_superhuman_regressions"),
                report["current_comparison_command"][0],
            )
            self.assertEqual(
                str(repo_root / "script/ai/compare_superhuman_regressions"),
                report["parent_comparison_command"][0],
            )
            self.assertNotEqual(
                report["pipeline_command"][0], report["current_comparison_command"][0]
            )
            self.assertNotEqual(
                report["pipeline_command"][0], report["parent_comparison_command"][0]
            )

    def test_dry_run_skips_non_executable_repo_python_candidate(self):
        source_repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-phase2-ablation-python-"
        ) as tmp:
            repo_root = Path(tmp) / "repo"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / ".venv/bin").mkdir(parents=True)
            (Path(tmp) / ".venv/bin").mkdir(parents=True)

            runner_path = repo_root / "script/ai/run_local_superhuman_phase2_ablation"
            runner_path.write_text(
                (
                    source_repo_root / "script/ai/run_local_superhuman_phase2_ablation"
                ).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            runner_path.chmod(0o755)

            (
                repo_root
                / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2_ablation_replay_balanced.json"
            ).write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v3-superhuman-ablation-replay-balanced",
                        "start_iteration": 2,
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )

            parent_artifact = (
                repo_root / "tmp/local_results_partial/aggressive-v3-superhuman-iter1"
            )
            parent_artifact.mkdir(parents=True)

            repo_python = repo_root / ".venv/bin/python"
            repo_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            repo_python.chmod(stat.S_IRUSR | stat.S_IWUSR)

            fallback_python = Path(tmp) / ".venv/bin/python"
            fallback_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fallback_python.chmod(0o755)

            result = subprocess.run(
                [
                    str(runner_path),
                    "--variant",
                    "replay_balanced",
                    "--parent-artifact",
                    str(parent_artifact),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertNotEqual(str(repo_python), report["pipeline_command"][0])


if __name__ == "__main__":
    unittest.main()
