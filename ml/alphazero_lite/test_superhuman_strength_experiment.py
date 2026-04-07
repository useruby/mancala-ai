import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SuperhumanStrengthExperimentScriptTest(unittest.TestCase):
    def test_non_dry_run_keeps_stdout_machine_parseable(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            (repo_root / "model-artifact/current").mkdir(parents=True)

            wrapper_path = repo_root / "script/ai/run_local_superhuman_strength_experiment"
            wrapper_path.write_text(
                (source_repo_root / "script/ai/run_local_superhuman_strength_experiment").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            wrapper_path.chmod(0o755)

            (repo_root / "script/ai/local_promotion_gate").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "print(json.dumps({\"gate\": True}))\n",
                encoding="utf-8",
            )
            (repo_root / "script/ai/local_promotion_gate").chmod(0o755)

            pipeline_stub = (
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "config = json.load(open(sys.argv[sys.argv.index(\"--config\") + 1], encoding=\"utf-8\"))\n"
                "for step in config[\"steps\"]:\n"
                "    if step[\"name\"] == \"train\":\n"
                "        out = step[\"command\"][step[\"command\"].index(\"--out\") + 1]\n"
                "        break\n"
                "    out = None\n"
                "if out:\n"
                "    open(out.replace(\"{iter_dir}\", \".\"), \"wb\").close()\n"
                "print(\"pipeline_scaffold_complete\")\n"
            )
            (repo_root / "ml/alphazero_lite/pipeline.py").write_text(pipeline_stub, encoding="utf-8")
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").chmod(0o755)
            fixture_path = repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"
            fixture_path.write_text("[]", encoding="utf-8")
            (repo_root / "model-artifact/current/weights.json").write_text("{}", encoding="utf-8")
            phase_config = {
                "run_id": "stub",
                "steps": [
                    {"name": "self_play", "command": [".venv/bin/python", "ml/alphazero_lite/self_play.py"]},
                    {"name": "mcts_bootstrap_dataset", "command": [".venv/bin/python", "ml/alphazero_lite/generate_bootstrap_dataset.py"]},
                    {"name": "train", "command": [".venv/bin/python", "ml/alphazero_lite/train.py", "--out", "{iter_dir}/checkpoint.npz"]},
                ],
            }
            (repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json").write_text(json.dumps(phase_config), encoding="utf-8")
            (repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json").write_text(json.dumps(phase_config), encoding="utf-8")
            (repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json").write_text(json.dumps({"steps": []}), encoding="utf-8")

            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(wrapper_path),
                    "--run-id",
                    "superhuman-strength-live-test",
                    "--games-json",
                    str(fixture_path),
                    "--output-root",
                    str(output_root),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual("superhuman-strength-live-test", report["run_id"])
            self.assertIn('"gate": true', result.stderr)
            self.assertIn("pipeline_scaffold_complete", result.stderr)

    def test_dry_run_writes_lane_manifests(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "superhuman_strength"
            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "superhuman-strength-dry-run-test",
                    "--games-json",
                    str(repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"),
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(result.stdout)

            self.assertEqual("superhuman-strength-dry-run-test", report["run_id"])
            self.assertTrue(Path(report["curated_dataset_path"]).exists())
            self.assertTrue(Path(report["stage1_config_path"]).exists())
            self.assertTrue(Path(report["stage2_config_path"]).exists())
            self.assertTrue(str(report["curated_dataset_path"]).startswith(str(output_root)))

            lane_a = report["lane_a_gate_command"]
            lane_b_stage1 = report["lane_b_stage1_command"]
            lane_b_stage2 = report["lane_b_stage2_command"]
            lane_b_gate = report["lane_b_gate_command"]

            self.assertIn("script/ai/local_promotion_gate", lane_a[0])
            self.assertTrue(any("aggressive_v3_superhuman_search_control.json" in part for part in lane_a))
            self.assertIn("--min-arena-score", lane_a)
            self.assertEqual("0.0", lane_a[lane_a.index("--min-arena-score") + 1])
            self.assertIn("--hard-min-score", lane_a)
            self.assertEqual("0.0", lane_a[lane_a.index("--hard-min-score") + 1])
            self.assertIn("--skip-mcts-relative-check", lane_a)
            self.assertTrue(any("ml/alphazero_lite/pipeline.py" in part for part in lane_b_stage1))
            self.assertTrue(any("ml/alphazero_lite/pipeline.py" in part for part in lane_b_stage2))
            self.assertIn("script/ai/local_promotion_gate", lane_b_gate[0])

            stage1_config = json.loads(Path(report["stage1_config_path"]).read_text(encoding="utf-8"))
            stage2_config = json.loads(Path(report["stage2_config_path"]).read_text(encoding="utf-8"))
            stage1_self_play = next(step["command"] for step in stage1_config["steps"] if step["name"] == "self_play")
            stage1_train = next(step["command"] for step in stage1_config["steps"] if step["name"] == "train")
            stage2_train = next(step["command"] for step in stage2_config["steps"] if step["name"] == "train")

            self.assertEqual(sys.executable, stage1_self_play[0])
            self.assertEqual(sys.executable, stage1_train[0])
            self.assertEqual(sys.executable, stage2_train[0])
            self.assertIn(report["curated_dataset_path"], stage1_train[stage1_train.index("--data-files") + 1])
            self.assertIn(report["curated_dataset_path"], stage2_train[stage2_train.index("--data-files") + 1])
            self.assertIn("--init-checkpoint", stage1_train)
            self.assertTrue(stage1_train[stage1_train.index("--init-checkpoint") + 1].endswith("current_init_checkpoint.npz"))
            self.assertIn("--init-checkpoint", stage2_train)


if __name__ == "__main__":
    unittest.main()
