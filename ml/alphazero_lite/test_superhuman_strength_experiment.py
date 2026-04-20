import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SuperhumanStrengthExperimentTest(unittest.TestCase):
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
                "import json, pathlib, sys\n"
                "candidate_path = pathlib.Path(sys.argv[sys.argv.index(\"--candidate-path\") + 1])\n"
                "if not candidate_path.exists():\n"
                "    raise SystemExit(f'missing candidate path: {candidate_path}')\n"
                "print(json.dumps({\"gate\": True, \"candidate_path\": str(candidate_path)}))\n",
                encoding="utf-8",
            )
            (repo_root / "script/ai/local_promotion_gate").chmod(0o755)

            pipeline_stub = (
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "config = json.load(open(sys.argv[sys.argv.index(\"--config\") + 1], encoding=\"utf-8\"))\n"
                "run_id = config[\"run_id\"]\n"
                "versions_dir = pathlib.Path(config.get(\"versions_dir\", \"storage/ai/alphazero_lite/versions\"))\n"
                "iter_dir = versions_dir / f'{run_id}-iter1'\n"
                "iter_dir.mkdir(parents=True, exist_ok=True)\n"
                "for step in config[\"steps\"]:\n"
                "    if step[\"name\"] == \"train\":\n"
                "        out = step[\"command\"][step[\"command\"].index(\"--out\") + 1]\n"
                "        break\n"
                "    out = None\n"
                "if out:\n"
                "    pathlib.Path(out.replace(\"{iter_dir}\", str(iter_dir))).write_bytes(b'')\n"
                "print(\"pipeline_scaffold_complete\")\n"
            )
            (repo_root / "ml/alphazero_lite/pipeline.py").write_text(pipeline_stub, encoding="utf-8")
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n"
                "print(\"dataset_scaffold_complete\")\n",
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
            self.assertIn("dataset_scaffold_complete", result.stderr)
            self.assertIn('"gate": true', result.stderr)
            self.assertIn("balanced-stage2-iter1", result.stderr)
            self.assertIn("pipeline_scaffold_complete", result.stderr)
            self.assertNotIn("dataset_scaffold_complete", result.stdout)

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

    def test_direct_launcher_prefers_workspace_root_venv_in_worktree_without_env_override(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspace"
            repo_root = workspace_root / ".worktrees/feature"
            output_root = Path(tmp) / "superhuman_strength"
            marker_dir = Path(tmp) / "markers"
            bin_dir = Path(tmp) / "bin"

            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            (repo_root / "model-artifact/current").mkdir(parents=True)
            (workspace_root / ".venv/bin").mkdir(parents=True)
            marker_dir.mkdir(parents=True)
            bin_dir.mkdir(parents=True)

            wrapper_path = repo_root / "script/ai/run_local_superhuman_strength_experiment"
            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                wrapper_path,
            )
            wrapper_path.chmod(0o755)

            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            (repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json").write_text("[]", encoding="utf-8")
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

            workspace_python = workspace_root / ".venv/bin/python"
            workspace_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$0\" > \"$WORKSPACE_PYTHON_MARKER\"\n"
                f"exec {sys.executable!r} \"$@\"\n",
                encoding="utf-8",
            )
            workspace_python.chmod(0o755)

            fake_python3 = bin_dir / "python3"
            fake_python3.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'python3\\n' > \"$PYTHON3_MARKER\"\n"
                "printf 'fake python3 used\\n' >&2\n"
                "exit 88\n",
                encoding="utf-8",
            )
            fake_python3.chmod(0o755)

            env = os.environ.copy()
            env.pop("AZLITE_EXPERIMENT_PYTHON", None)
            env["WORKSPACE_PYTHON_MARKER"] = str(marker_dir / "workspace_python.txt")
            env["PYTHON3_MARKER"] = str(marker_dir / "python3.txt")
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                [
                    str(wrapper_path),
                    "--run-id",
                    "workspace-python-test",
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
            self.assertFalse(Path(env["PYTHON3_MARKER"]).exists(), msg=result.stderr)
            self.assertEqual(str(workspace_python), Path(env["WORKSPACE_PYTHON_MARKER"]).read_text(encoding="utf-8").strip())

            report = json.loads(result.stdout)
            stage1_config = json.loads(Path(report["stage1_config_path"]).read_text(encoding="utf-8"))
            stage1_self_play = next(step["command"] for step in stage1_config["steps"] if step["name"] == "self_play")
            self.assertEqual(str(workspace_python), stage1_self_play[0])

    def test_superhuman_strength_experiment_lane_b_stage1_uses_opponent_pool(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )
            current_model_path = current_model_dir / "model.npz"
            current_model_path.write_bytes(b"fixture")
            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "opponent-pool-lane-test",
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
            stage1_config = json.loads(Path(report["stage1_config_path"]).read_text(encoding="utf-8"))
            stage1_self_play = next(step["command"] for step in stage1_config["steps"] if step["name"] == "self_play")

            self.assertIn("--opponent-pool-config", stage1_self_play)
            pool_path = Path(stage1_self_play[stage1_self_play.index("--opponent-pool-config") + 1])
            self.assertEqual(str(pool_path), report["opponent_pool_path"])
            self.assertTrue(pool_path.exists())

            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            self.assertIn("checkpoints", pool)
            self.assertTrue(pool["checkpoints"])
            self.assertEqual(len(pool["checkpoints"]), len(set(pool["checkpoints"])))

    def test_superhuman_strength_experiment_accepts_explicit_opponent_checkpoint(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )
            current_model_path = current_model_dir / "model.npz"
            current_model_path.write_bytes(b"fixture")

            extra_model = repo_root / "tmp/extra/model.npz"
            extra_model.parent.mkdir(parents=True, exist_ok=True)
            extra_model.write_bytes(b"fixture")

            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "explicit-opponent-test",
                    "--games-json",
                    str(repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"),
                    "--output-root",
                    str(output_root),
                    "--opponent-checkpoint",
                    str(extra_model),
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
            pool = json.loads(Path(report["opponent_pool_path"]).read_text(encoding="utf-8"))

            self.assertIn(str(extra_model.resolve()), pool["checkpoints"])
            self.assertIn(str(extra_model.resolve()), report["opponent_checkpoint_inputs"])

    def test_superhuman_strength_experiment_adds_fallback_init_checkpoint_when_only_explicit_opponent_exists(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )

            explicit_model = repo_root / "tmp/extra/model.npz"
            explicit_model.parent.mkdir(parents=True, exist_ok=True)
            explicit_model.write_bytes(b"fixture")

            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "explicit-plus-fallback-test",
                    "--games-json",
                    str(repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"),
                    "--output-root",
                    str(output_root),
                    "--opponent-checkpoint",
                    str(explicit_model),
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
            pool = json.loads(Path(report["opponent_pool_path"]).read_text(encoding="utf-8"))
            fallback_checkpoint = output_root / "explicit-plus-fallback-test/current_init_checkpoint.npz"

            self.assertEqual(
                [str(explicit_model.resolve()), str(fallback_checkpoint.resolve())],
                pool["checkpoints"],
            )

    def test_superhuman_strength_experiment_deduplicates_explicit_and_discovered_checkpoints(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )
            current_model_path = current_model_dir / "model.npz"
            current_model_path.write_bytes(b"fixture")

            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "dedup-opponent-test",
                    "--games-json",
                    str(repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"),
                    "--output-root",
                    str(output_root),
                    "--opponent-checkpoint",
                    str(current_model_path),
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
            pool = json.loads(Path(report["opponent_pool_path"]).read_text(encoding="utf-8"))

            self.assertEqual(pool["checkpoints"].count(str(current_model_path.resolve())), 1)

    def test_superhuman_strength_experiment_deduplicates_explicit_checkpoint_inputs(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )

            explicit_model = repo_root / "tmp/extra/model.npz"
            explicit_model.parent.mkdir(parents=True, exist_ok=True)
            explicit_model.write_bytes(b"fixture")

            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "dedup-opponent-inputs-test",
                    "--games-json",
                    str(repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"),
                    "--output-root",
                    str(output_root),
                    "--opponent-checkpoint",
                    str(explicit_model),
                    "--opponent-checkpoint",
                    str(explicit_model),
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

            self.assertEqual([str(explicit_model.resolve())], report["opponent_checkpoint_inputs"])

    def test_superhuman_strength_experiment_rejects_non_model_opponent_checkpoint(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )
            current_model_path = current_model_dir / "model.npz"
            current_model_path.write_bytes(b"fixture")

            real_model = repo_root / "tmp/extra/model.npz"
            real_model.parent.mkdir(parents=True, exist_ok=True)
            real_model.write_bytes(b"fixture")
            bad_path = repo_root / "tmp/extra/checkpoint.npz"
            bad_path.symlink_to(real_model)

            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "reject-explicit-opponent-test",
                    "--games-json",
                    str(repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json"),
                    "--output-root",
                    str(output_root),
                    "--opponent-checkpoint",
                    str(bad_path),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("must point to model.npz", result.stderr)

    def test_superhuman_strength_experiment_keeps_stage2_self_play_without_opponent_pool(self):
        source_repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            output_root = Path(tmp) / "superhuman_strength"
            (repo_root / "script/ai").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/configs").mkdir(parents=True)
            (repo_root / "ml/alphazero_lite/fixtures").mkdir(parents=True)
            current_model_dir = repo_root / "model-artifact/current"
            current_model_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                source_repo_root / "script/ai/run_local_superhuman_strength_experiment",
                repo_root / "script/ai/run_local_superhuman_strength_experiment",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
                repo_root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            )
            shutil.copy2(
                source_repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
                repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json",
            )
            (repo_root / "ml/alphazero_lite/build_superhuman_strength_dataset.py").write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "out = sys.argv[sys.argv.index(\"--out\") + 1]\n"
                "open(out, \"w\", encoding=\"utf-8\").write(json.dumps({\"ok\": True}) + \"\\n\")\n",
                encoding="utf-8",
            )
            shutil.copy2(
                source_repo_root / "model-artifact/current/weights.json",
                current_model_dir / "weights.json",
            )
            (current_model_dir / "model.npz").write_bytes(b"fixture")
            env = os.environ.copy()
            env["AZLITE_EXPERIMENT_PYTHON"] = sys.executable

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_superhuman_strength_experiment"),
                    "--run-id",
                    "opponent-pool-stage2-guard-test",
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
            stage2_config = json.loads(Path(report["stage2_config_path"]).read_text(encoding="utf-8"))
            stage2_self_play = next(step["command"] for step in stage2_config["steps"] if step["name"] == "self_play")

            self.assertNotIn("--opponent-pool-config", stage2_self_play)


if __name__ == "__main__":
    unittest.main()
