import json
import importlib.machinery
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite.pipeline import load_config, render_command


class PipelineScriptTest(unittest.TestCase):
    HYBRID_TEACHER_LOCAL_CONFIG = "aggressive_v2_hybrid_teacher_local.json"
    POLICY_TARGET_LOCAL_CONFIG = "aggressive_v3_policy_target_local.json"
    SEARCH_QUALITY_LOCAL_CONFIG = "aggressive_v2_search_quality_local.json"
    SPECIALIZED_HEADS_LOCAL_CONFIG = "aggressive_v3_specialized_heads_local.json"
    SPECIALIZED_HEADS_WIDE_LOCAL_CONFIG = "aggressive_v3_specialized_heads_wide_local.json"
    TACTICAL_TEACHER_LOCAL_CONFIG = "aggressive_v2_tactical_teacher_local.json"
    TACTICAL_ENCODING_LOCAL_CONFIG = "aggressive_v3_tactical_encoding_local.json"
    VALUE_TARGET_LOCAL_CONFIG = "aggressive_v3_value_target_local.json"
    VALUE_TARGET_ALIGNED_LOCAL_CONFIG = "aggressive_v3_value_target_aligned_local.json"
    V3_CAPACITY_LOCAL_CONFIG = "aggressive_v3_capacity_local.json"
    V3_CAPACITY_LARGE_LOCAL_CONFIG = "aggressive_v3_capacity_large_local.json"
    V3_STRONGER_BOOTSTRAP_LOCAL_CONFIG = "aggressive_v3_stronger_bootstrap_local.json"
    V3_STRONGER_BOOTSTRAP_CONFIRM_LOCAL_CONFIG = "aggressive_v3_stronger_bootstrap_confirm_local.json"
    V3_STRONGER_BOOTSTRAP_CONFIRM_B_LOCAL_CONFIG = "aggressive_v3_stronger_bootstrap_confirm_b_local.json"
    V3_STRONGER_BOOTSTRAP_CONFIRM_C_LOCAL_CONFIG = "aggressive_v3_stronger_bootstrap_confirm_c_local.json"
    V3_STRONGER_BOOTSTRAP_MORE_DATA_LOCAL_CONFIG = "aggressive_v3_stronger_bootstrap_more_data_local.json"
    V3_SUPERHUMAN_PHASE1_CONFIG = "aggressive_v3_superhuman_phase1.json"
    V3_SUPERHUMAN_PHASE2_CONFIG = "aggressive_v3_superhuman_phase2.json"
    HYBRID_VALUE_TARGET_LOCAL_CONFIG = "aggressive_v3_hybrid_value_target_local.json"
    PHASE_AWARE_VALUE_TARGET_LOCAL_CONFIG = "aggressive_v3_phase_aware_value_target_local.json"
    V2_LOCAL_CONFIG_EXPECTATIONS = {
        "aggressive_v2_budget_up_local.json": {
            "run_id": "aggressive-v2-budget-up-local",
            "versions_dir": "/tmp/azlite_v2_budget_up_local_versions",
            "self_play_games": "2200",
            "self_play_simulations": "192",
            "self_play_temperature_late": "0.15",
            "bootstrap_games": "800",
            "bootstrap_simulations": "1200",
            "train_epochs": "12",
            "train_hidden_sizes": "64,2",
        },
        "aggressive_v2_search_up_local.json": {
            "run_id": "aggressive-v2-search-up-local",
            "versions_dir": "/tmp/azlite_v2_search_up_local_versions",
            "self_play_games": "1600",
            "self_play_simulations": "256",
            "self_play_temperature_late": "0.05",
            "bootstrap_games": "600",
            "bootstrap_simulations": "1200",
            "train_epochs": "10",
            "train_hidden_sizes": "64,2",
        },
        "aggressive_v2_budget_plus_small_widen_local.json": {
            "run_id": "aggressive-v2-budget-plus-small-widen-local",
            "versions_dir": "/tmp/azlite_v2_budget_plus_small_widen_local_versions",
            "self_play_games": "2200",
            "self_play_simulations": "192",
            "self_play_temperature_late": "0.15",
            "bootstrap_games": "800",
            "bootstrap_simulations": "1200",
            "train_epochs": "12",
            "train_hidden_sizes": "96,3",
        },
    }

    def find_bash_block(self, markdown: str, *, containing: tuple[str, ...]) -> str:
        commands = [block.split("```", maxsplit=1)[0] for block in markdown.split("```bash\n")[1:]]

        for needle in containing:
            for command in commands:
                if needle in command:
                    return command

        raise AssertionError(f"No bash block found containing any of: {containing}")

    def load_local_promotion_gate_module(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/local_promotion_gate"
        loader = importlib.machinery.SourceFileLoader("local_promotion_gate", str(script_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    def load_v2_local_config(self, filename: str) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        return load_config(repo_root / "ml/alphazero_lite/configs" / filename)

    def iter_json_pipeline_configs(self):
        repo_root = Path(__file__).resolve().parents[2]

        for path in sorted((repo_root / "ml/alphazero_lite/configs").glob("*.json")):
            yield path, load_config(path)

    def iter_step_commands(self, config: dict, *, command_name: str):
        for step in config.get("steps", []):
            command = step.get("command", [])
            if command_name in command:
                yield step, command

    def test_checkpoint_self_play_configs_require_evaluator_cache_size(self):
        configs_dir = Path(__file__).resolve().parents[2] / "ml/alphazero_lite/configs"
        found_checkpoint_self_play = False

        for path, config in self.iter_json_pipeline_configs():
            for step, command in self.iter_step_commands(config, command_name="ml/alphazero_lite/self_play.py"):
                with self.subTest(config=path.name, step=step["name"]):
                    config_path = str(path)

                    if "--checkpoint" in command:
                        found_checkpoint_self_play = True
                        self.assertIn(
                            "--evaluator-cache-size",
                            command,
                            msg=f"{config_path}: checkpoint self_play step must set --evaluator-cache-size",
                        )
                        cache_size_index = command.index("--evaluator-cache-size")
                        self.assertLess(
                            cache_size_index + 1,
                            len(command),
                            msg=f"{config_path}: checkpoint self_play step sets --evaluator-cache-size without a value",
                        )
                        self.assertEqual(
                            "50000",
                            command[cache_size_index + 1],
                            msg=f"{config_path}: checkpoint self_play step must set --evaluator-cache-size 50000",
                        )
                    else:
                        self.assertNotIn(
                            "--evaluator-cache-size",
                            command,
                            msg=f"{config_path}: non-checkpoint self_play step must not set --evaluator-cache-size",
                        )

        self.assertTrue(found_checkpoint_self_play, f"No checkpoint-driven self_play config found under {configs_dir}")

    def test_bootstrap_configs_gate_teacher_search_reuse_by_hybrid_teacher_mode(self):
        configs_dir = Path(__file__).resolve().parents[2] / "ml/alphazero_lite/configs"
        found_hybrid_teacher_puct_bootstrap = False

        for path, config in self.iter_json_pipeline_configs():
            for step, command in self.iter_step_commands(
                config,
                command_name="ml/alphazero_lite/generate_bootstrap_dataset.py",
            ):
                with self.subTest(config=path.name, step=step["name"]):
                    config_path = str(path)
                    has_hybrid_teacher_mode = any(
                        flag == "--position-selection-mode" and value == "hybrid_teacher"
                        for flag, value in zip(command, command[1:])
                    )
                    teacher_mode = self.command_flag_value(command, "--teacher-mode") if "--teacher-mode" in command else "puct"
                    has_hybrid_teacher_puct_semantics = has_hybrid_teacher_mode and teacher_mode == "puct"

                    if has_hybrid_teacher_puct_semantics:
                        found_hybrid_teacher_puct_bootstrap = True
                        self.assertIn(
                            "--teacher-search-reuse",
                            command,
                            msg=(
                                f"{config_path}: hybrid_teacher bootstrap step with effective "
                                "teacher-mode puct must set --teacher-search-reuse"
                            ),
                        )
                    else:
                        self.assertNotIn(
                            "--teacher-search-reuse",
                            command,
                            msg=(
                                f"{config_path}: bootstrap step without effective hybrid_teacher + puct semantics "
                                "must not set --teacher-search-reuse"
                            ),
                        )

        self.assertTrue(
            found_hybrid_teacher_puct_bootstrap,
            f"No hybrid_teacher bootstrap config with effective teacher-mode puct found under {configs_dir}",
        )

    def test_pipeline_skip_step_omits_named_step(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="pipeline-skip-step-") as tmp:
            run_id = "skip-step-smoke"
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "versions_dir": str(Path(tmp) / "versions"),
                        "seed": 42,
                        "iterations": 1,
                        "steps": [
                            {
                                "name": "first_step",
                                "command": ["python3", "-c", "from pathlib import Path; Path('{iter_dir}/first.txt').write_text('ok')"],
                            },
                            {
                                "name": "rules_parity_fuzz",
                                "command": ["python3", "-c", "raise SystemExit(9)"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "ml/alphazero_lite/pipeline.py"),
                    "--config",
                    str(config_path),
                    "--skip-step",
                    "rules_parity_fuzz",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            manifest = json.loads((Path(tmp) / "versions" / f"{run_id}-iter1" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["status"])
            self.assertEqual(["first_step"], [step["name"] for step in manifest["steps"]])

    def test_pipeline_skip_rules_parity_fuzz_also_skips_parity_gate(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="pipeline-skip-gate-") as tmp:
            run_id = "skip-parity-gate"
            config_path = Path(tmp) / "config.json"
            versions_dir = Path(tmp) / "versions"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "versions_dir": str(versions_dir),
                        "seed": 42,
                        "iterations": 1,
                        "steps": [
                            {
                                "name": "rules_parity_fuzz",
                                "command": ["python3", "-c", "from pathlib import Path; Path('{iter_dir}/parity_report.json').write_text('{}')"],
                            },
                        ],
                        "gates": {
                            "rules_parity_report": "{iter_dir}/parity_report.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "ml/alphazero_lite/pipeline.py"),
                    "--config",
                    str(config_path),
                    "--skip-step",
                    "rules_parity_fuzz",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            manifest = json.loads((versions_dir / f"{run_id}-iter1" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["status"])
            self.assertEqual([], manifest["gate_failures"])

    def test_superhuman_configs_use_current_baseline(self):
        phase1 = self.load_v2_local_config(self.V3_SUPERHUMAN_PHASE1_CONFIG)
        phase2 = self.load_v2_local_config(self.V3_SUPERHUMAN_PHASE2_CONFIG)

        self.assertEqual("aggressive-v3-superhuman", phase1["run_id"])
        self.assertEqual("aggressive-v3-superhuman", phase2["run_id"])
        self.assertEqual("model-artifact/current", phase1["current_path"])
        self.assertEqual("/tmp/azlite_v3_superhuman_versions/aggressive-v3-superhuman-iter1", phase2["current_path"])

        phase2_steps = self.config_steps_by_name(phase2)
        self.assertIn("arena_vs_hard_report", phase2_steps)
        self.assertIn("model-artifact/current", " ".join(phase2_steps["benchmark_contract"]["command"]))
        self.assertIn("model-artifact/current", " ".join(phase2_steps["arena_confirm_report"]["command"]))

    def test_superhuman_configs_tighten_training_recipe(self):
        phase1 = self.load_v2_local_config(self.V3_SUPERHUMAN_PHASE1_CONFIG)
        phase2 = self.load_v2_local_config(self.V3_SUPERHUMAN_PHASE2_CONFIG)

        phase1_steps = self.config_steps_by_name(phase1)
        phase2_steps = self.config_steps_by_name(phase2)

        phase1_self_play = int(self.command_flag_value(phase1_steps["self_play"]["command"], "--games"))
        phase1_bootstrap = int(self.command_flag_value(phase1_steps["mcts_bootstrap_dataset"]["command"], "--games"))
        phase1_epochs = int(self.command_flag_value(phase1_steps["train"]["command"], "--epochs"))

        phase2_self_play = int(self.command_flag_value(phase2_steps["self_play"]["command"], "--games"))
        phase2_epochs = int(self.command_flag_value(phase2_steps["train"]["command"], "--epochs"))
        phase2_bootstrap = int(self.command_flag_value(phase2_steps["mcts_bootstrap_dataset"]["command"], "--games"))

        self.assertGreater(phase1_bootstrap, phase1_self_play)
        self.assertGreater(phase2_bootstrap, phase2_self_play)
        self.assertGreater(phase2_bootstrap, phase1_bootstrap)
        self.assertLessEqual(phase2_self_play, phase1_self_play)
        self.assertGreater(phase2_epochs, phase1_epochs)
        self.assertEqual("kalah_v3", self.command_flag_value(phase1_steps["self_play"]["command"], "--input-encoding"))
        self.assertEqual("kalah_v3", self.command_flag_value(phase1_steps["train"]["command"], "--input-encoding"))
        self.assertEqual("residual_v3", self.command_flag_value(phase1_steps["train"]["command"], "--model-type"))
        self.assertEqual("sharpened", self.command_flag_value(phase1_steps["train"]["command"], "--policy-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(phase1_steps["train"]["command"], "--value-target-mode"))
        self.assertIn("mcts_bootstrap_dataset", phase2_steps)

    def command_flag_value(self, command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    def render_config_step(self, config: dict, step_name: str, *, iteration: int = 1) -> list[str]:
        repo_root = Path(__file__).resolve().parents[2]
        iter_dir = repo_root / "tmp" / f"{config['run_id']}-iter{iteration}"
        return render_command(
            self.config_steps_by_name(config)[step_name]["command"],
            iteration=iteration,
            iter_dir=iter_dir,
            run_id=config["run_id"],
            versions_dir=Path(config["versions_dir"]),
            current_path=config["current_path"],
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )

    def normalize_rendered_command(self, command: list[str], *, config: dict, iteration: int = 1) -> list[str]:
        repo_root = Path(__file__).resolve().parents[2]
        iter_dir = repo_root / "tmp" / f"{config['run_id']}-iter{iteration}"
        normalized: list[str] = []

        for token in command:
            normalized.append(
                token.replace(str(iter_dir), "{iter_dir}")
                .replace(config["run_id"], "{run_id}")
                .replace(str(config["versions_dir"]), "{versions_dir}")
            )

        return normalized

    def config_steps_by_name(self, config: dict) -> dict[str, dict]:
        return {step["name"]: step for step in config["steps"]}

    def bootstrap_shell(self, config: dict) -> str:
        return " ".join(self.config_steps_by_name(config)["mcts_bootstrap_dataset"]["command"])

    def test_tactical_teacher_local_config_keeps_v2_student_family_and_encoding(self):
        config = self.load_v2_local_config(self.TACTICAL_TEACHER_LOCAL_CONFIG)
        steps = self.config_steps_by_name(config)

        train_step = steps["train"]["command"]
        export_step = steps["export_artifact"]["command"]

        self.assertEqual("residual_v2", self.command_flag_value(train_step, "--model-type"))
        self.assertEqual("kalah_v2", self.command_flag_value(train_step, "--input-encoding"))
        self.assertEqual("residual_v2", self.command_flag_value(export_step, "--model-type"))
        self.assertEqual("kalah_v2", self.command_flag_value(export_step, "--input-encoding"))

    def test_tactical_teacher_local_config_uses_targeted_teacher_generation(self):
        base_config = self.load_v2_local_config("aggressive_v2.yaml")
        tactical_config = self.load_v2_local_config(self.TACTICAL_TEACHER_LOCAL_CONFIG)

        base_bootstrap = self.bootstrap_shell(base_config)
        tactical_bootstrap = self.bootstrap_shell(tactical_config)

        self.assertIn("ml/alphazero_lite/generate_bootstrap_dataset.py", tactical_bootstrap)
        self.assertIn("--position-selection-mode tactical", tactical_bootstrap)
        self.assertIn("--simulations 2400", tactical_bootstrap)
        self.assertNotEqual(base_bootstrap, tactical_bootstrap)

    def test_tactical_teacher_local_config_keeps_existing_mcts1200_evaluation_contract(self):
        base_config = self.load_v2_local_config("aggressive_v2.yaml")
        tactical_config = self.load_v2_local_config(self.TACTICAL_TEACHER_LOCAL_CONFIG)

        base_steps = self.config_steps_by_name(base_config)
        tactical_steps = self.config_steps_by_name(tactical_config)

        for step_name in (
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(base_steps[step_name], tactical_steps[step_name])

        self.assertEqual(base_config["gates"], tactical_config["gates"])

    def test_hybrid_teacher_local_config_keeps_v2_student_family_and_encoding(self):
        config = self.load_v2_local_config(self.HYBRID_TEACHER_LOCAL_CONFIG)
        steps = self.config_steps_by_name(config)

        train_step = steps["train"]["command"]
        export_step = steps["export_artifact"]["command"]

        self.assertEqual("residual_v2", self.command_flag_value(train_step, "--model-type"))
        self.assertEqual("kalah_v2", self.command_flag_value(train_step, "--input-encoding"))
        self.assertEqual("residual_v2", self.command_flag_value(export_step, "--model-type"))
        self.assertEqual("kalah_v2", self.command_flag_value(export_step, "--input-encoding"))

    def test_hybrid_teacher_local_config_uses_hybrid_teacher_generation(self):
        base_config = self.load_v2_local_config("aggressive_v2.yaml")
        hybrid_config = self.load_v2_local_config(self.HYBRID_TEACHER_LOCAL_CONFIG)

        base_bootstrap = self.bootstrap_shell(base_config)
        hybrid_bootstrap = self.bootstrap_shell(hybrid_config)

        self.assertIn("ml/alphazero_lite/generate_bootstrap_dataset.py", hybrid_bootstrap)
        self.assertIn("--position-selection-mode hybrid_teacher", hybrid_bootstrap)
        self.assertNotEqual(base_bootstrap, hybrid_bootstrap)

    def test_hybrid_teacher_local_config_keeps_existing_mcts1200_evaluation_contract(self):
        base_config = self.load_v2_local_config("aggressive_v2.yaml")
        hybrid_config = self.load_v2_local_config(self.HYBRID_TEACHER_LOCAL_CONFIG)

        base_steps = self.config_steps_by_name(base_config)
        hybrid_steps = self.config_steps_by_name(hybrid_config)

        for step_name in (
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(base_steps[step_name], hybrid_steps[step_name])

        self.assertEqual(base_config["gates"], hybrid_config["gates"])

    def run_local_promotion_gate_with_stub_reports(
        self,
        *,
        arena_report: dict,
        hard_report: dict | None = None,
        candidate_mcts_report: dict,
        current_mcts_report: dict,
        min_arena_score: float = 0.55,
        hard_min_score: float = 0.55,
        min_arena_games: int = 120,
        hard_arena_games: int = 120,
        min_mcts_games: int = 40,
        extra_args: list[str] | None = None,
    ):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate.npz"
            report_path = tmp_path / "promotion_report.json"
            arena_report_path = tmp_path / "arena_report.json"
            hard_report_path = tmp_path / "hard_report.json"
            candidate_mcts_report_path = tmp_path / "candidate_mcts_report.json"
            current_mcts_report_path = tmp_path / "current_mcts_report.json"

            candidate_path.write_text("stub", encoding="utf-8")
            arena_report_path.write_text(json.dumps(arena_report), encoding="utf-8")
            hard_payload = hard_report or {"wins": 66, "losses": 42, "draws": 12, "games_played": 120}
            hard_report_path.write_text(json.dumps(hard_payload), encoding="utf-8")
            candidate_mcts_report_path.write_text(json.dumps(candidate_mcts_report), encoding="utf-8")
            current_mcts_report_path.write_text(json.dumps(current_mcts_report), encoding="utf-8")

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/local_promotion_gate"),
                    "--candidate-path",
                    str(candidate_path),
                    "--out",
                    str(report_path),
                    "--min-arena-score",
                    str(min_arena_score),
                    "--hard-min-score",
                    str(hard_min_score),
                    "--min-arena-games",
                    str(min_arena_games),
                    "--hard-arena-games",
                    str(hard_arena_games),
                    "--min-mcts-games",
                    str(min_mcts_games),
                    "--hard-path",
                    "model-artifact/current",
                    "--stub-arena-report",
                    str(arena_report_path),
                    "--stub-hard-report",
                    str(hard_report_path),
                    "--stub-candidate-mcts-report",
                    str(candidate_mcts_report_path),
                    "--stub-current-mcts-report",
                    str(current_mcts_report_path),
                    *(extra_args or []),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
            return result, report

    def test_local_promotion_gate_fails_when_only_some_stub_reports_are_provided(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate.npz"
            report_path = tmp_path / "promotion_report.json"
            arena_report_path = tmp_path / "arena_report.json"

            candidate_path.write_text("stub", encoding="utf-8")
            arena_report_path.write_text(
                json.dumps({"wins": 66, "losses": 42, "draws": 12, "games_played": 120}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/local_promotion_gate"),
                    "--candidate-path",
                    str(candidate_path),
                    "--out",
                    str(report_path),
                    "--stub-arena-report",
                    str(arena_report_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("provide all stub report paths or none", result.stderr)
            self.assertFalse(report_path.exists())

    def test_local_promotion_gate_fails_when_stub_report_json_is_malformed(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate.npz"
            report_path = tmp_path / "promotion_report.json"
            arena_report_path = tmp_path / "arena_report.json"
            hard_report_path = tmp_path / "hard_report.json"
            candidate_mcts_report_path = tmp_path / "candidate_mcts_report.json"
            current_mcts_report_path = tmp_path / "current_mcts_report.json"

            candidate_path.write_text("stub", encoding="utf-8")
            arena_report_path.write_text("{not-json", encoding="utf-8")
            hard_report_path.write_text(json.dumps({"wins": 57, "losses": 34, "draws": 29, "games_played": 120}), encoding="utf-8")
            candidate_mcts_report_path.write_text(
                json.dumps({"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40}),
                encoding="utf-8",
            )
            current_mcts_report_path.write_text(
                json.dumps({"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/local_promotion_gate"),
                    "--candidate-path",
                    str(candidate_path),
                    "--out",
                    str(report_path),
                    "--stub-arena-report",
                    str(arena_report_path),
                    "--stub-hard-report",
                    str(hard_report_path),
                    "--stub-candidate-mcts-report",
                    str(candidate_mcts_report_path),
                    "--stub-current-mcts-report",
                    str(current_mcts_report_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("invalid JSON in stub report", result.stderr)
            self.assertIn(str(arena_report_path), result.stderr)
            self.assertFalse(report_path.exists())

    def test_dry_run_creates_iteration_manifest(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 1,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/pipeline.py",
                    "--config",
                    str(config_path),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("pipeline_dry_run_complete", result.stdout)

            iter_dir = out_dir / "aggressive-v1-iter1"
            self.assertTrue(iter_dir.exists())

            manifest_path = iter_dir / "run_manifest.json"
            self.assertTrue(manifest_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_run_manifest_v1", manifest["schema"])
            self.assertEqual("aggressive-v1", manifest["run_id"])
            self.assertEqual(1, manifest["iteration"])
            self.assertEqual("planned", manifest["status"])

    def test_run_executes_configured_steps_and_marks_manifest_completed(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            step2_file = tmp_path / "train_done.txt"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 1,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                        "steps": [
                            {
                                "name": "self_play",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok', encoding='utf-8')",
                                    "{iter_dir}/self_play_done.txt",
                                ],
                            },
                            {
                                "name": "train",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    f"from pathlib import Path; Path(r'{step2_file}').write_text('ok', encoding='utf-8')",
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/pipeline.py",
                    "--config",
                    str(config_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("pipeline_scaffold_complete", result.stdout)
            self.assertTrue((out_dir / "aggressive-v1-iter1" / "self_play_done.txt").exists())
            self.assertTrue(step2_file.exists())

            manifest_path = out_dir / "aggressive-v1-iter1" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["status"])
            self.assertEqual(2, len(manifest["steps"]))
            self.assertTrue(all(step["status"] == "completed" for step in manifest["steps"]))

    def test_run_marks_manifest_failed_when_step_fails(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 1,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                        "steps": [
                            {
                                "name": "failing_step",
                                "command": [sys.executable, "-c", "import sys; sys.exit(2)"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/pipeline.py",
                    "--config",
                    str(config_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            manifest_path = out_dir / "aggressive-v1-iter1" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("failed", manifest["status"])
            self.assertEqual("failed", manifest["steps"][0]["status"])

    def test_run_fails_when_rules_parity_gate_fails(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 1,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                        "steps": [
                            {
                                "name": "write_parity_report",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    (
                                        "import json,sys; "
                                        "path=sys.argv[1]; "
                                        "json.dump(dict(schema='kalah_parity_fuzz_v1', parity_passed=False, mismatch_count=1), open(path,'w',encoding='utf-8'))"
                                    ),
                                    "{iter_dir}/parity_report.json",
                                ],
                            }
                        ],
                        "gates": {
                            "rules_parity_report": "{iter_dir}/parity_report.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [".venv/bin/python", "ml/alphazero_lite/pipeline.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            manifest = json.loads((out_dir / "aggressive-v1-iter1" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", manifest["status"])
            self.assertTrue(any(failure["code"] == "rules_parity_failed" for failure in manifest["gate_failures"]))

    def test_run_fails_when_arena_gate_score_below_threshold(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 1,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                        "steps": [
                            {
                                "name": "write_arena_report",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    (
                                        "import json,sys; "
                                        "path=sys.argv[1]; "
                                        "json.dump(dict(schema='arena_v1', games_played=10, wins=4, losses=6, draws=0, promotion_decision=dict(passed=False)), open(path,'w',encoding='utf-8'))"
                                    ),
                                    "{iter_dir}/arena_report.json",
                                ],
                            }
                        ],
                        "gates": {
                            "arena_report": "{iter_dir}/arena_report.json",
                            "min_arena_score": 0.55,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [".venv/bin/python", "ml/alphazero_lite/pipeline.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            manifest = json.loads((out_dir / "aggressive-v1-iter1" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", manifest["status"])
            self.assertTrue(any(failure["code"] == "arena_score_below_threshold" for failure in manifest["gate_failures"]))

    def test_run_passes_when_all_configured_gates_pass(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 1,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                        "steps": [
                            {
                                "name": "write_reports",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    (
                                        "import json,sys,pathlib; "
                                        "iter_dir=pathlib.Path(sys.argv[1]); "
                                        "json.dump(dict(schema='kalah_parity_fuzz_v1', parity_passed=True, mismatch_count=0), open(iter_dir/'parity_report.json','w',encoding='utf-8')); "
                                        "json.dump(dict(schema='arena_v1', games_played=10, wins=7, losses=3, draws=0, promotion_decision=dict(passed=True)), open(iter_dir/'arena_report.json','w',encoding='utf-8')); "
                                        "json.dump(dict(schema='azlite_benchmark_v1', checks=[dict(id='az_identity', passed=True), dict(id='runtime_parity', passed=True)]), open(iter_dir/'benchmark_report.json','w',encoding='utf-8'))"
                                    ),
                                    "{iter_dir}",
                                ],
                            }
                        ],
                        "gates": {
                            "rules_parity_report": "{iter_dir}/parity_report.json",
                            "arena_report": "{iter_dir}/arena_report.json",
                            "benchmark_report": "{iter_dir}/benchmark_report.json",
                            "min_arena_score": 0.55,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [".venv/bin/python", "ml/alphazero_lite/pipeline.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            manifest = json.loads((out_dir / "aggressive-v1-iter1" / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["status"])
            self.assertEqual([], manifest.get("gate_failures", []))

    def test_run_renders_replay_data_placeholder_with_recent_iterations(self):
        with tempfile.TemporaryDirectory(prefix="azlite-pipeline-") as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            out_dir = tmp_path / "versions"

            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "aggressive-v1",
                        "seed": 42,
                        "iterations": 2,
                        "replay_window": 3,
                        "versions_dir": str(out_dir),
                        "current_path": "storage/ai/alphazero_lite/current",
                        "steps": [
                            {
                                "name": "self_play",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('row', encoding='utf-8')",
                                    "{iter_dir}/self_play.jsonl",
                                ],
                            },
                            {
                                "name": "capture_replay",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; import sys; Path(sys.argv[2]).write_text(sys.argv[1], encoding='utf-8')",
                                    "{replay_data}",
                                    "{iter_dir}/replay_capture.txt",
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [".venv/bin/python", "ml/alphazero_lite/pipeline.py", "--config", str(config_path)],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            replay_capture = (out_dir / "aggressive-v1-iter2" / "replay_capture.txt").read_text(encoding="utf-8")
            self.assertIn("aggressive-v1-iter2/self_play.jsonl", replay_capture)
            self.assertIn("aggressive-v1-iter1/self_play.jsonl", replay_capture)

    def test_aggressive_config_uses_python_arena_workers_for_prefilter_and_confirm(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        steps = {step["name"]: step["command"] for step in config["steps"]}
        iter_dir = repo_root / "tmp" / "aggressive-v1-iter1"

        self.assertNotIn("arena_prefilter_ruby_sweep", steps)
        self.assertNotIn("arena_confirm_ruby_sweep", steps)

        for step_name, out_name, games in (
            ("arena_prefilter_report", "arena_prefilter_report.json", "30"),
            ("arena_confirm_report", "arena_report.json", "120"),
        ):
            rendered = render_command(
                steps[step_name],
                iteration=1,
                iter_dir=iter_dir,
                run_id="aggressive-v1",
                versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
                current_path="storage/ai/alphazero_lite/current",
                parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
                parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
                replay_data="",
                replay_weights="",
            )

            self.assertEqual(".venv/bin/python", rendered[0])
            self.assertEqual("ml/alphazero_lite/arena.py", rendered[1])
            self.assertIn("--workers", rendered)
            self.assertIn("6", rendered)
            self.assertIn("--games", rendered)
            self.assertIn(games, rendered)
            self.assertIn("--out", rendered)
            self.assertIn(str(iter_dir / out_name), rendered)

        self.assertEqual(
            [".venv/bin/python", "ml/alphazero_lite/validate_arena_report.py"],
            steps["arena_prefilter_validate"][:2],
        )
        self.assertEqual(
            [".venv/bin/python", "ml/alphazero_lite/validate_arena_report.py"],
            steps["arena_validate"][:2],
        )

    def test_experiment_fast_config_uses_direct_python_arena_reports(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/experiment1_fast_s41.yaml")
        step_names = [step["name"] for step in config["steps"]]
        steps = {step["name"]: step["command"] for step in config["steps"]}
        iter_dir = repo_root / "tmp" / "experiment1-fast-s41-iter1"

        self.assertNotIn("arena_prefilter_ruby_sweep", step_names)
        self.assertNotIn("arena_prefilter_to_report", step_names)
        self.assertNotIn("arena_confirm_ruby_sweep", step_names)
        self.assertNotIn("arena_confirm_to_report", step_names)

        for step_name, out_name, games in (
            ("arena_prefilter_report", "arena_prefilter_report.json", "30"),
            ("arena_confirm_report", "arena_report.json", "120"),
        ):
            rendered = render_command(
                steps[step_name],
                iteration=1,
                iter_dir=iter_dir,
                run_id="experiment1-fast-s41",
                versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
                current_path="storage/ai/alphazero_lite/current",
                parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
                parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
                replay_data="",
                replay_weights="",
            )

            self.assertEqual(".venv/bin/python", rendered[0])
            self.assertEqual("ml/alphazero_lite/arena.py", rendered[1])
            self.assertIn("--games", rendered)
            self.assertIn(games, rendered)
            self.assertIn("--out", rendered)
            self.assertIn(str(iter_dir / out_name), rendered)

    def test_experiment2_dynamic_config_uses_python_mcts1200_baseline(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/experiment2_dynamic_s41.yaml")
        steps = {step["name"]: step["command"] for step in config["steps"]}
        rendered = render_command(
            steps["mcts1200_baseline_report"],
            iteration=1,
            iter_dir=repo_root / "tmp" / "exp2-dynamic-s41-iter1",
            run_id="exp2-dynamic-s41",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )

        self.assertEqual(".venv/bin/python", rendered[0])
        self.assertEqual("ml/alphazero_lite/mcts1200_baseline.py", rendered[1])
        self.assertIn("--out", rendered)
        self.assertIn(str(repo_root / "tmp" / "exp2-dynamic-s41-iter1" / "mcts1200_report.json"), rendered)

    def test_aggressive_config_uses_dedicated_mcts1200_baseline_driver(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        steps = {step["name"]: step["command"] for step in config["steps"]}
        iter_dir = repo_root / "tmp" / "aggressive-v1-iter1"

        rendered = render_command(
            steps["mcts1200_baseline_report"],
            iteration=1,
            iter_dir=iter_dir,
            run_id="aggressive-v1",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )

        rendered_command = " ".join(rendered)
        self.assertIn("ml/alphazero_lite/mcts1200_baseline.py", rendered_command)
        self.assertIn("--workers", rendered_command)
        self.assertIn(str(iter_dir / "mcts1200_report.json"), rendered)

    def test_aggressive_config_uses_baseline_relative_mcts_gate(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        step_names = [step["name"] for step in config["steps"]]
        steps = {step["name"]: step["command"] for step in config["steps"]}
        iter_dir = repo_root / "tmp" / "aggressive-v1-iter1"

        self.assertIn("current_mcts1200_baseline_report", step_names)
        self.assertLess(
            step_names.index("current_mcts1200_baseline_report"),
            step_names.index("benchmark_contract"),
        )

        current_baseline_rendered = render_command(
            steps["current_mcts1200_baseline_report"],
            iteration=1,
            iter_dir=iter_dir,
            run_id="aggressive-v1",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )

        self.assertIn("ml/alphazero_lite/mcts1200_baseline.py", current_baseline_rendered)
        self.assertIn("storage/ai/alphazero_lite/current", current_baseline_rendered)
        self.assertIn(str(iter_dir / "current_mcts1200_report.json"), current_baseline_rendered)

        rendered = render_command(
            steps["benchmark_contract"],
            iteration=1,
            iter_dir=iter_dir,
            run_id="aggressive-v1",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )

        self.assertIn("--mcts-report", rendered)
        self.assertIn(str(iter_dir / "mcts1200_report.json"), rendered)
        self.assertIn("--current-baseline-mcts-report", rendered)
        self.assertIn(str(iter_dir / "current_mcts1200_report.json"), rendered)
        self.assertNotIn("storage/ai/alphazero_lite/current/mcts1200_report.json", rendered)
        self.assertNotIn("--min-mcts-score", rendered)

    def test_aggressive_v2_config_targets_v2_encoding_and_model_family(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")
        steps = {step["name"]: step["command"] for step in config["steps"]}

        self.assertEqual("aggressive-v2", config["run_id"])
        self.assertIn("--input-encoding", steps["self_play"])
        self.assertEqual("kalah_v2", steps["self_play"][steps["self_play"].index("--input-encoding") + 1])

        train_step = steps["train"]
        self.assertEqual("residual_v2", train_step[train_step.index("--model-type") + 1])
        self.assertEqual("64,2", train_step[train_step.index("--hidden-sizes") + 1])
        self.assertEqual("kalah_v2", train_step[train_step.index("--input-encoding") + 1])

        export_step = steps["export_artifact"]
        self.assertEqual("residual_v2", export_step[export_step.index("--model-type") + 1])
        self.assertEqual("kalah_v2", export_step[export_step.index("--input-encoding") + 1])
        self.assertEqual("kalah_v1", export_step[export_step.index("--rules-version") + 1])

    def test_aggressive_v2_config_preserves_benchmark_and_promotion_shape(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        v2_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")

        self.assertEqual(base_config["gates"], v2_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in v2_config["steps"]],
        )

        base_steps = {step["name"]: step for step in base_config["steps"]}
        v2_steps = {step["name"]: step for step in v2_config["steps"]}

        for step_name in (
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(base_steps[step_name], v2_steps[step_name])

    def test_aggressive_v2_config_keeps_current_artifact_contract_compatibility(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        v2_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")
        v2_steps = {step["name"]: step["command"] for step in v2_config["steps"]}
        iter_dir = repo_root / "tmp" / "aggressive-v2-iter1"

        self.assertEqual(base_config["current_path"], v2_config["current_path"])
        self.assertEqual(base_config["versions_dir"], v2_config["versions_dir"])

        export_step = v2_steps["export_artifact"]
        self.assertEqual("kalah_v1", export_step[export_step.index("--rules-version") + 1])

        rendered_current_baseline = render_command(
            v2_steps["current_mcts1200_baseline_report"],
            iteration=1,
            iter_dir=iter_dir,
            run_id="aggressive-v2",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )
        self.assertIn("storage/ai/alphazero_lite/current", rendered_current_baseline)

        rendered_benchmark = render_command(
            v2_steps["benchmark_contract"],
            iteration=1,
            iter_dir=iter_dir,
            run_id="aggressive-v2",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )
        self.assertIn("storage/ai/alphazero_lite/current", rendered_benchmark)
        self.assertIn(str(iter_dir / "current_mcts1200_report.json"), rendered_benchmark)

    def test_v2_local_tuning_configs_preserve_aggressive_v2_contract_shape(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")
        base_steps = self.config_steps_by_name(base_config)

        for filename in self.V2_LOCAL_CONFIG_EXPECTATIONS:
            with self.subTest(config=filename):
                variant_config = self.load_v2_local_config(filename)
                variant_steps = self.config_steps_by_name(variant_config)

                self.assertEqual(set(base_config), set(variant_config))
                self.assertEqual(base_config["seed"], variant_config["seed"])
                self.assertEqual(base_config["replay_window"], variant_config["replay_window"])
                self.assertEqual(base_config["current_path"], variant_config["current_path"])
                self.assertEqual(base_config["gates"], variant_config["gates"])
                self.assertEqual(
                    [step["name"] for step in base_config["steps"]],
                    [step["name"] for step in variant_config["steps"]],
                )

                for step_name in (
                    "perspective_audit",
                    "export_artifact",
                    "rules_parity_fuzz",
                    "arena_prefilter_report",
                    "arena_prefilter_validate",
                    "arena_confirm_report",
                    "mcts1200_baseline_report",
                    "current_mcts1200_baseline_report",
                    "benchmark_contract",
                    "arena_validate",
                ):
                    self.assertEqual(base_steps[step_name], variant_steps[step_name])

    def test_v2_local_tuning_configs_only_change_intended_knobs(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")
        base_steps = self.config_steps_by_name(base_config)
        base_bootstrap = self.bootstrap_shell(base_config)
        base_self_play = base_steps["self_play"]["command"]
        base_train = base_steps["train"]["command"]

        for filename, expected in self.V2_LOCAL_CONFIG_EXPECTATIONS.items():
            with self.subTest(config=filename):
                variant_config = self.load_v2_local_config(filename)
                variant_steps = self.config_steps_by_name(variant_config)
                variant_self_play = variant_steps["self_play"]["command"]
                variant_train = variant_steps["train"]["command"]

                self.assertEqual(1, variant_config["iterations"])

                self.assertEqual(
                    expected["self_play_games"],
                    self.command_flag_value(variant_steps["self_play"]["command"], "--games"),
                )
                self.assertEqual(
                    expected["self_play_simulations"],
                    self.command_flag_value(variant_steps["self_play"]["command"], "--simulations"),
                )
                self.assertEqual(
                    expected["self_play_temperature_late"],
                    self.command_flag_value(variant_self_play, "--temperature-late"),
                )
                self.assertEqual(
                    "1",
                    self.command_flag_value(variant_self_play, "--total-iterations"),
                )

                self.assertEqual(
                    expected["train_epochs"],
                    self.command_flag_value(variant_train, "--epochs"),
                )
                self.assertEqual(
                    expected["train_hidden_sizes"],
                    self.command_flag_value(variant_train, "--hidden-sizes"),
                )
                self.assertEqual(
                    self.command_flag_value(base_train, "--model-type"),
                    self.command_flag_value(variant_train, "--model-type"),
                )
                self.assertEqual(
                    self.command_flag_value(base_train, "--input-encoding"),
                    self.command_flag_value(variant_train, "--input-encoding"),
                )

                expected_self_play = (
                    base_self_play.copy()
                    if filename == "aggressive_v2_search_up_local.json"
                    else base_self_play.copy()
                )
                expected_self_play[expected_self_play.index("--total-iterations") + 1] = "1"
                expected_self_play[expected_self_play.index("--games") + 1] = expected["self_play_games"]
                expected_self_play[expected_self_play.index("--simulations") + 1] = expected["self_play_simulations"]
                expected_self_play[expected_self_play.index("--temperature-late") + 1] = expected["self_play_temperature_late"]
                expected_self_play.extend(["--evaluator-cache-size", "50000"])
                self.assertEqual(expected_self_play, variant_self_play)

                expected_train = base_train.copy()
                expected_train[expected_train.index("--epochs") + 1] = expected["train_epochs"]
                expected_train[expected_train.index("--hidden-sizes") + 1] = expected["train_hidden_sizes"]
                self.assertEqual(expected_train, variant_train)

                bootstrap_shell = self.bootstrap_shell(variant_config)
                self.assertIn(f"--games {expected['bootstrap_games']}", bootstrap_shell)
                self.assertIn(f"--simulations {expected['bootstrap_simulations']}", bootstrap_shell)
                self.assertEqual(
                    bootstrap_shell,
                    base_bootstrap
                    .replace("--games 600", f"--games {expected['bootstrap_games']}")
                    .replace("--simulations 1200", f"--simulations {expected['bootstrap_simulations']}")
                    if filename == "aggressive_v2_search_up_local.json"
                    else base_bootstrap.replace("--games 600", f"--games {expected['bootstrap_games']}"),
                )

    def test_v2_local_tuning_configs_use_separate_local_run_ids_and_versions_dirs(self):
        run_ids = set()
        versions_dirs = set()

        for filename, expected in self.V2_LOCAL_CONFIG_EXPECTATIONS.items():
            with self.subTest(config=filename):
                config = self.load_v2_local_config(filename)
                run_ids.add(config["run_id"])
                versions_dirs.add(config["versions_dir"])

                self.assertEqual(expected["run_id"], config["run_id"])
                self.assertEqual(expected["versions_dir"], config["versions_dir"])
                self.assertTrue(config["versions_dir"].startswith("/tmp/"))

        self.assertEqual(len(self.V2_LOCAL_CONFIG_EXPECTATIONS), len(run_ids))
        self.assertEqual(len(self.V2_LOCAL_CONFIG_EXPECTATIONS), len(versions_dirs))

    def test_search_quality_local_config_renders_expected_search_flags(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")
        variant_config = self.load_v2_local_config(self.SEARCH_QUALITY_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        iter_dir = repo_root / "tmp" / "aggressive-v2-search-quality-local-iter1"

        self.assertEqual("aggressive-v2-search-quality-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v2_search_quality_local_versions", variant_config["versions_dir"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )

        self.assertEqual("parent_q", self.command_flag_value(variant_steps["self_play"]["command"], "--fpu-mode"))
        self.assertIn("--reuse-subtree", variant_steps["self_play"]["command"])
        self.assertIn("--normalize-values", variant_steps["self_play"]["command"])
        self.assertEqual(
            "deterministic",
            self.command_flag_value(variant_steps["self_play"]["command"], "--root-policy-mode"),
        )
        self.assertEqual(
            "0.1",
            self.command_flag_value(variant_steps["self_play"]["command"], "--tactical-root-bias"),
        )

        for step_name in (
            "perspective_audit",
            "mcts_bootstrap_dataset",
            "train",
            "export_artifact",
            "rules_parity_fuzz",
            "arena_prefilter_validate",
            "arena_validate",
        ):
            self.assertEqual(base_steps[step_name], variant_steps[step_name])

        for step_name in (
            "arena_prefilter_report",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
        ):
            rendered = render_command(
                variant_steps[step_name]["command"],
                iteration=1,
                iter_dir=iter_dir,
                run_id=variant_config["run_id"],
                versions_dir=Path(variant_config["versions_dir"]),
                current_path=variant_config["current_path"],
                parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
                parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
                replay_data="",
                replay_weights="",
            )
            self.assertIn("--fpu-mode", rendered)
            self.assertIn("parent_q", rendered)
            self.assertIn("--reuse-subtree", rendered)
            self.assertIn("--normalize-values", rendered)
            self.assertIn("--root-policy-mode", rendered)
            self.assertIn("deterministic", rendered)
            self.assertIn("--tactical-root-bias", rendered)
            self.assertIn("0.1", rendered)

    def test_aggressive_v3_tactical_encoding_local_config_renders_kalah_v3_commands(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v2.yaml")
        variant_config = self.load_v2_local_config(self.TACTICAL_ENCODING_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        iter_dir = repo_root / "tmp" / "aggressive-v3-tactical-encoding-local-iter1"

        self.assertEqual("aggressive-v3-tactical-encoding-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_tactical_encoding_local_versions", variant_config["versions_dir"])
        self.assertEqual(1, variant_config["iterations"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )

        bootstrap_shell = self.bootstrap_shell(variant_config)
        self.assertIn("--input-encoding kalah_v3", bootstrap_shell)

        for step_name in ("self_play", "train", "export_artifact"):
            rendered = render_command(
                variant_steps[step_name]["command"],
                iteration=1,
                iter_dir=iter_dir,
                run_id=variant_config["run_id"],
                versions_dir=Path(variant_config["versions_dir"]),
                current_path=variant_config["current_path"],
                parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
                parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
                replay_data="",
                replay_weights="",
            )
            self.assertIn("--input-encoding", rendered)
            self.assertEqual("kalah_v3", self.command_flag_value(rendered, "--input-encoding"))

        self.assertEqual(
            self.command_flag_value(base_steps["train"]["command"], "--model-type"),
            self.command_flag_value(variant_steps["train"]["command"], "--model-type"),
        )
        self.assertEqual(
            self.command_flag_value(base_steps["export_artifact"]["command"], "--model-type"),
            self.command_flag_value(variant_steps["export_artifact"]["command"], "--model-type"),
        )

        for step_name in (
            "perspective_audit",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(base_steps[step_name], variant_steps[step_name])

        self.assertNotEqual(base_steps["mcts_bootstrap_dataset"], variant_steps["mcts_bootstrap_dataset"])

    def test_aggressive_v3_self_play_omits_incompatible_parent_checkpoint(self):
        repo_root = Path(__file__).resolve().parents[2]
        variant_config = self.load_v2_local_config(self.TACTICAL_ENCODING_LOCAL_CONFIG)
        self_play_command = self.config_steps_by_name(variant_config)["self_play"]["command"]
        iter_dir = repo_root / "tmp" / "aggressive-v3-tactical-encoding-local-iter1"

        with tempfile.TemporaryDirectory(prefix="pipeline-parent-checkpoint-") as tmp:
            parent_checkpoint = Path(tmp) / "model.npz"
            parent_checkpoint.write_bytes(b"stub")

            with mock.patch("ml.alphazero_lite.pipeline.checkpoint_feature_count", return_value=10):
                rendered = render_command(
                    self_play_command,
                    iteration=1,
                    iter_dir=iter_dir,
                    run_id=variant_config["run_id"],
                    versions_dir=Path(variant_config["versions_dir"]),
                    current_path=variant_config["current_path"],
                    parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
                    parent_checkpoint=parent_checkpoint,
                    replay_data="",
                    replay_weights="",
                )

        self.assertIn("--input-encoding", rendered)
        self.assertEqual("kalah_v3", self.command_flag_value(rendered, "--input-encoding"))
        self.assertNotIn("--checkpoint", rendered)

    def test_aggressive_v3_specialized_heads_local_config_renders_residual_v3_and_kalah_v3_commands(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = self.load_v2_local_config(self.TACTICAL_ENCODING_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.SPECIALIZED_HEADS_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        iter_dir = repo_root / "tmp" / "aggressive-v3-specialized-heads-local-iter1"

        self.assertEqual("aggressive-v3-specialized-heads-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_specialized_heads_local_versions", variant_config["versions_dir"])
        self.assertEqual(1, variant_config["iterations"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )
        self.assertEqual(self.bootstrap_shell(base_config), self.bootstrap_shell(variant_config))

        self_play_rendered = render_command(
            variant_steps["self_play"]["command"],
            iteration=1,
            iter_dir=iter_dir,
            run_id=variant_config["run_id"],
            versions_dir=Path(variant_config["versions_dir"]),
            current_path=variant_config["current_path"],
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )
        self.assertEqual("kalah_v3", self.command_flag_value(self_play_rendered, "--input-encoding"))

        for step_name in ("train", "export_artifact"):
            rendered = render_command(
                variant_steps[step_name]["command"],
                iteration=1,
                iter_dir=iter_dir,
                run_id=variant_config["run_id"],
                versions_dir=Path(variant_config["versions_dir"]),
                current_path=variant_config["current_path"],
                parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
                parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
                replay_data="",
                replay_weights="",
            )
            self.assertEqual("residual_v3", self.command_flag_value(rendered, "--model-type"))
            self.assertEqual("kalah_v3", self.command_flag_value(rendered, "--input-encoding"))

        for step_name in (
            "self_play",
            "perspective_audit",
            "mcts_bootstrap_dataset",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(base_steps[step_name], variant_steps[step_name])

    def test_aggressive_v3_specialized_heads_wide_local_only_changes_hidden_sizes(self):
        base_config = self.load_v2_local_config(self.SPECIALIZED_HEADS_LOCAL_CONFIG)
        wide_config = self.load_v2_local_config(self.SPECIALIZED_HEADS_WIDE_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        wide_steps = self.config_steps_by_name(wide_config)
        base_train = self.render_config_step(base_config, "train")
        wide_train = self.render_config_step(wide_config, "train")
        base_export = self.render_config_step(base_config, "export_artifact")
        wide_export = self.render_config_step(wide_config, "export_artifact")
        base_hidden_sizes = self.command_flag_value(base_train, "--hidden-sizes")
        wide_hidden_sizes = self.command_flag_value(wide_train, "--hidden-sizes")

        self.assertEqual("aggressive-v3-specialized-heads-wide-local", wide_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_specialized_heads_wide_local_versions", wide_config["versions_dir"])
        self.assertEqual(set(base_config), set(wide_config))
        self.assertEqual(base_config["seed"], wide_config["seed"])
        self.assertEqual(base_config["iterations"], wide_config["iterations"])
        self.assertEqual(base_config["replay_window"], wide_config["replay_window"])
        self.assertEqual(base_config["current_path"], wide_config["current_path"])
        self.assertEqual(base_config["gates"], wide_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in wide_config["steps"]],
        )

        for step_name in (
            "self_play",
            "perspective_audit",
            "mcts_bootstrap_dataset",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(
                self.normalize_rendered_command(self.render_config_step(base_config, step_name), config=base_config),
                self.normalize_rendered_command(self.render_config_step(wide_config, step_name), config=wide_config),
            )

        for command in (base_train, wide_train, base_export, wide_export):
            self.assertEqual("residual_v3", self.command_flag_value(command, "--model-type"))
            self.assertEqual("kalah_v3", self.command_flag_value(command, "--input-encoding"))

        self.assertNotEqual(base_train, wide_train)
        self.assertNotEqual(base_export, wide_export)

        self.assertNotEqual(base_hidden_sizes, wide_hidden_sizes)
        self.assertEqual("96,2", wide_hidden_sizes)
        self.assertEqual(base_hidden_sizes.split(",")[1], wide_hidden_sizes.split(",")[1])

        normalized_base_train = self.normalize_rendered_command(base_train, config=base_config)
        normalized_wide_train = self.normalize_rendered_command(wide_train, config=wide_config)
        expected_train = normalized_base_train.copy()
        expected_train[expected_train.index("--hidden-sizes") + 1] = wide_hidden_sizes
        self.assertEqual(expected_train, normalized_wide_train)

        self.assertEqual(
            self.normalize_rendered_command(base_export, config=base_config),
            self.normalize_rendered_command(wide_export, config=wide_config),
        )

    def test_readme_documents_v3_specialized_heads_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_specialized_heads_local.json", readme)
        self.assertIn("aggressive-v3-specialized-heads-local-iter1", readme)
        self.assertIn("`kalah_v3`", readme)
        self.assertIn("`residual_v3`", readme)

    def test_readme_documents_v3_specialized_heads_wide_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        wide_section = readme.split("### Widened specialized-heads lane", maxsplit=1)[1]
        wide_commands = self.find_bash_block(
            wide_section,
            containing=("aggressive_v3_specialized_heads_wide_local.json",),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_specialized_heads_wide_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", wide_commands)
        self.assertIn("script/ai/local_promotion_gate", wide_commands)
        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_specialized_heads_wide_local.json", wide_commands)
        self.assertIn(
            "/tmp/azlite_v3_specialized_heads_wide_local_versions/aggressive-v3-specialized-heads-wide-local-iter1",
            wide_commands,
        )
        self.assertIn("`residual_v3`", readme)

    def test_aggressive_v3_policy_target_local_only_changes_policy_target_mode(self):
        base_config = self.load_v2_local_config(self.SPECIALIZED_HEADS_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.POLICY_TARGET_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-policy-target-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_policy_target_local_versions", variant_config["versions_dir"])
        self.assertEqual(base_config["seed"], variant_config["seed"])
        self.assertEqual(base_config["iterations"], variant_config["iterations"])
        self.assertEqual(base_config["replay_window"], variant_config["replay_window"])
        self.assertEqual(base_config["current_path"], variant_config["current_path"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )

        self.assertEqual("kalah_v3", self.command_flag_value(variant_self_play, "--input-encoding"))
        self.assertEqual("residual_v3", self.command_flag_value(variant_train, "--model-type"))
        self.assertEqual("kalah_v3", self.command_flag_value(variant_train, "--input-encoding"))
        self.assertEqual("sharpened", self.command_flag_value(variant_self_play, "--policy-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_train, "--policy-target-mode"))
        self.assertIn("--policy-target-mode sharpened", self.bootstrap_shell(variant_config))

        for step_name in (
            "perspective_audit",
            "export_artifact",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(
                self.normalize_rendered_command(self.render_config_step(base_config, step_name), config=base_config),
                self.normalize_rendered_command(self.render_config_step(variant_config, step_name), config=variant_config),
            )

        expected_self_play = self.normalize_rendered_command(
            self.render_config_step(base_config, "self_play"),
            config=base_config,
        )
        cache_index = expected_self_play.index("--evaluator-cache-size")
        expected_self_play.insert(cache_index, "--policy-target-mode")
        expected_self_play.insert(cache_index + 1, "sharpened")
        self.assertEqual(expected_self_play, self.normalize_rendered_command(variant_self_play, config=variant_config))

        expected_train = self.normalize_rendered_command(
            self.render_config_step(base_config, "train"),
            config=base_config,
        )
        expected_train.extend(["--policy-target-mode", "sharpened"])
        self.assertEqual(expected_train, self.normalize_rendered_command(variant_train, config=variant_config))

        self.assertEqual(
            self.bootstrap_shell(base_config).replace(base_config["run_id"], variant_config["run_id"]),
            self.bootstrap_shell(variant_config).replace(" --policy-target-mode sharpened", ""),
        )

    def test_readme_documents_v3_policy_target_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_policy_target_local.json", readme)
        self.assertIn("aggressive-v3-policy-target-local-iter1", readme)
        self.assertIn("`kalah_v3`", readme)
        self.assertIn("`residual_v3`", readme)
        self.assertIn("`--policy-target-mode sharpened`", readme)

    def test_aggressive_v3_value_target_local_only_changes_value_target_mode(self):
        base_config = self.load_v2_local_config(self.POLICY_TARGET_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.VALUE_TARGET_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-value-target-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_value_target_local_versions", variant_config["versions_dir"])
        self.assertEqual(base_config["seed"], variant_config["seed"])
        self.assertEqual(base_config["iterations"], variant_config["iterations"])
        self.assertEqual(base_config["replay_window"], variant_config["replay_window"])
        self.assertEqual(base_config["current_path"], variant_config["current_path"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )

        self.assertEqual("sharpened", self.command_flag_value(variant_self_play, "--policy-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_train, "--policy-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_self_play, "--value-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_train, "--value-target-mode"))
        self.assertIn("--policy-target-mode sharpened", self.bootstrap_shell(variant_config))
        self.assertIn("--value-target-mode sharpened", self.bootstrap_shell(variant_config))

        for step_name in (
            "perspective_audit",
            "export_artifact",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(
                self.normalize_rendered_command(self.render_config_step(base_config, step_name), config=base_config),
                self.normalize_rendered_command(self.render_config_step(variant_config, step_name), config=variant_config),
            )

        expected_self_play = self.normalize_rendered_command(
            self.render_config_step(base_config, "self_play"),
            config=base_config,
        )
        cache_index = expected_self_play.index("--evaluator-cache-size")
        expected_self_play.insert(cache_index, "--value-target-mode")
        expected_self_play.insert(cache_index + 1, "sharpened")
        self.assertEqual(expected_self_play, self.normalize_rendered_command(variant_self_play, config=variant_config))

        expected_train = self.normalize_rendered_command(
            self.render_config_step(base_config, "train"),
            config=base_config,
        )
        expected_train.extend(["--value-target-mode", "sharpened"])
        self.assertEqual(expected_train, self.normalize_rendered_command(variant_train, config=variant_config))

        self.assertEqual(
            self.bootstrap_shell(base_config).replace(base_config["run_id"], variant_config["run_id"]),
            self.bootstrap_shell(variant_config).replace(" --value-target-mode sharpened", ""),
        )

    def test_readme_documents_v3_value_target_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_value_target_local.json", readme)
        self.assertIn("aggressive-v3-value-target-local-iter1", readme)
        self.assertIn("`kalah_v3`", readme)
        self.assertIn("`residual_v3`", readme)
        self.assertIn("`--value-target-mode sharpened`", readme)

    def test_aggressive_v3_value_target_aligned_local_only_changes_lane_identity(self):
        base_config = self.load_v2_local_config(self.VALUE_TARGET_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.VALUE_TARGET_ALIGNED_LOCAL_CONFIG)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-value-target-aligned-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_value_target_aligned_local_versions", variant_config["versions_dir"])
        self.assertEqual("sharpened", self.command_flag_value(variant_self_play, "--value-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_train, "--value-target-mode"))
        self.assertIn("--value-target-mode sharpened", self.bootstrap_shell(variant_config))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        self.assertEqual(expected_config, variant_config)

        for step_name in (step["name"] for step in base_config["steps"]):
            self.assertEqual(
                self.normalize_rendered_command(self.render_config_step(base_config, step_name), config=base_config),
                self.normalize_rendered_command(self.render_config_step(variant_config, step_name), config=variant_config),
            )

    def test_readme_documents_v3_value_target_aligned_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        aligned_section = readme.split("### Aligned sharpened value-target lane", maxsplit=1)[1]
        aligned_commands = self.find_bash_block(
            aligned_section,
            containing=("aggressive_v3_value_target_aligned_local.json",),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_value_target_aligned_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", aligned_commands)
        self.assertIn("script/ai/local_promotion_gate", aligned_commands)
        self.assertIn("--config ml/alphazero_lite/configs/aggressive_v3_value_target_aligned_local.json", aligned_commands)
        self.assertIn(
            "--candidate-path /tmp/azlite_v3_value_target_aligned_local_versions/aggressive-v3-value-target-aligned-local-iter1",
            aligned_commands,
        )
        self.assertIn(
            "--config-path ml/alphazero_lite/configs/aggressive_v3_value_target_aligned_local.json",
            aligned_commands,
        )
        self.assertIn("aligned rerun", readme)
        self.assertIn("`--value-target-mode sharpened`", readme)

    def test_aggressive_v3_capacity_local_only_changes_lane_identity_and_hidden_sizes(self):
        base_config = self.load_v2_local_config(self.VALUE_TARGET_ALIGNED_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_CAPACITY_LOCAL_CONFIG)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-capacity-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_capacity_local_versions", variant_config["versions_dir"])
        self.assertNotIn("--checkpoint", variant_self_play)
        self.assertEqual("96,3", self.command_flag_value(variant_train, "--hidden-sizes"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_self_play = self.config_steps_by_name(expected_config)["self_play"]["command"]
        if "--checkpoint" in expected_self_play:
            checkpoint_index = expected_self_play.index("--checkpoint")
            del expected_self_play[checkpoint_index:checkpoint_index + 2]
        if "--evaluator-cache-size" in expected_self_play:
            cache_index = expected_self_play.index("--evaluator-cache-size")
            del expected_self_play[cache_index:cache_index + 2]
        expected_steps = self.config_steps_by_name(expected_config)
        expected_steps["train"]["command"][expected_steps["train"]["command"].index("--hidden-sizes") + 1] = "96,3"
        self.assertEqual(expected_config, variant_config)

    def test_readme_documents_v3_capacity_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        capacity_section = readme.split("### Capacity lane", maxsplit=1)[1]
        capacity_commands = self.find_bash_block(
            capacity_section,
            containing=("aggressive_v3_capacity_local.json",),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_capacity_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", capacity_commands)
        self.assertIn("script/ai/local_promotion_gate", capacity_commands)
        self.assertIn("--config ml/alphazero_lite/configs/aggressive_v3_capacity_local.json", capacity_commands)
        self.assertIn(
            "--candidate-path /tmp/azlite_v3_capacity_local_versions/aggressive-v3-capacity-local-iter1",
            capacity_commands,
        )
        self.assertIn(
            "--config-path ml/alphazero_lite/configs/aggressive_v3_capacity_local.json",
            capacity_commands,
        )
        self.assertIn("`96,3`", readme)

    def test_aggressive_v3_capacity_large_local_only_changes_lane_identity_and_hidden_sizes(self):
        base_config = self.load_v2_local_config(self.VALUE_TARGET_ALIGNED_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_CAPACITY_LARGE_LOCAL_CONFIG)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-capacity-large-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_capacity_large_local_versions", variant_config["versions_dir"])
        self.assertNotIn("--checkpoint", variant_self_play)
        self.assertEqual("128,3", self.command_flag_value(variant_train, "--hidden-sizes"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_self_play = self.config_steps_by_name(expected_config)["self_play"]["command"]
        if "--checkpoint" in expected_self_play:
            checkpoint_index = expected_self_play.index("--checkpoint")
            del expected_self_play[checkpoint_index:checkpoint_index + 2]
        if "--evaluator-cache-size" in expected_self_play:
            cache_index = expected_self_play.index("--evaluator-cache-size")
            del expected_self_play[cache_index:cache_index + 2]
        expected_steps = self.config_steps_by_name(expected_config)
        expected_steps["train"]["command"][expected_steps["train"]["command"].index("--hidden-sizes") + 1] = "128,3"
        self.assertEqual(expected_config, variant_config)

    def test_readme_documents_v3_capacity_large_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        capacity_section = readme.split("### Larger capacity lane", maxsplit=1)[1]
        capacity_commands = self.find_bash_block(
            capacity_section,
            containing=("aggressive_v3_capacity_large_local.json",),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_capacity_large_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", capacity_commands)
        self.assertIn("script/ai/local_promotion_gate", capacity_commands)
        self.assertIn("--config ml/alphazero_lite/configs/aggressive_v3_capacity_large_local.json", capacity_commands)
        self.assertIn(
            "--candidate-path /tmp/azlite_v3_capacity_large_local_versions/aggressive-v3-capacity-large-local-iter1",
            capacity_commands,
        )
        self.assertIn(
            "--config-path ml/alphazero_lite/configs/aggressive_v3_capacity_large_local.json",
            capacity_commands,
        )
        self.assertIn("`128,3`", readme)

    def test_aggressive_v3_stronger_bootstrap_local_only_changes_lane_identity_and_bootstrap_teacher(self):
        base_config = self.load_v2_local_config(self.V3_CAPACITY_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_LOCAL_CONFIG)
        variant_bootstrap = self.render_config_step(variant_config, "mcts_bootstrap_dataset")

        self.assertEqual("aggressive-v3-stronger-bootstrap-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_stronger_bootstrap_local_versions", variant_config["versions_dir"])
        self.assertEqual("2400", self.command_flag_value(variant_bootstrap, "--simulations"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_steps = self.config_steps_by_name(expected_config)
        bootstrap_command = expected_steps["mcts_bootstrap_dataset"]["command"]
        bootstrap_command[bootstrap_command.index("--simulations") + 1] = "2400"
        self.assertEqual(expected_config, variant_config)

    def test_readme_documents_v3_stronger_bootstrap_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        section = readme.split("### Stronger bootstrap teacher lane", maxsplit=1)[1]
        commands = self.find_bash_block(section, containing=("aggressive_v3_stronger_bootstrap_local.json",))

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", commands)
        self.assertIn("script/ai/local_promotion_gate", commands)
        self.assertIn("--config ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json", commands)
        self.assertIn(
            "--candidate-path /tmp/azlite_v3_stronger_bootstrap_local_versions/aggressive-v3-stronger-bootstrap-local-iter1",
            commands,
        )
        self.assertIn(
            "--config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json",
            commands,
        )
        self.assertIn("`2400`", readme)

    def test_aggressive_v3_stronger_bootstrap_confirm_local_only_changes_lane_identity_and_seed_schedule(self):
        base_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_CONFIRM_LOCAL_CONFIG)
        variant_self_play = self.render_config_step(variant_config, "self_play")

        self.assertEqual("aggressive-v3-stronger-bootstrap-confirm-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_stronger_bootstrap_confirm_local_versions", variant_config["versions_dir"])
        self.assertEqual(84, variant_config["seed"])
        self.assertEqual("81,82,83", self.command_flag_value(variant_self_play, "--seed-sweep"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_config["seed"] = 84
        expected_steps = self.config_steps_by_name(expected_config)
        expected_steps["self_play"]["command"][expected_steps["self_play"]["command"].index("--seed-sweep") + 1] = "81,82,83"
        for step in expected_config["steps"]:
            command = step.get("command", [])
            if "--seed" in command:
                command[command.index("--seed") + 1] = "84"
        self.assertEqual(expected_config, variant_config)

    def test_readme_documents_v3_stronger_bootstrap_confirm_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        section = readme.split("### Stronger bootstrap confirmation lane", maxsplit=1)[1]
        commands = self.find_bash_block(section, containing=("aggressive_v3_stronger_bootstrap_confirm_local.json",))

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_local.json", readme)
        self.assertIn("script/ai/local_promotion_gate", commands)
        self.assertIn("81,82,83", readme)
        self.assertIn("seed = 84", readme)

    def test_aggressive_v3_stronger_bootstrap_confirm_b_local_only_changes_lane_identity_and_seed_schedule(self):
        base_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_CONFIRM_B_LOCAL_CONFIG)
        variant_self_play = self.render_config_step(variant_config, "self_play")

        self.assertEqual("aggressive-v3-stronger-bootstrap-confirm-b-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_stronger_bootstrap_confirm_b_local_versions", variant_config["versions_dir"])
        self.assertEqual(126, variant_config["seed"])
        self.assertEqual("121,122,123", self.command_flag_value(variant_self_play, "--seed-sweep"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_config["seed"] = 126
        expected_steps = self.config_steps_by_name(expected_config)
        expected_steps["self_play"]["command"][expected_steps["self_play"]["command"].index("--seed-sweep") + 1] = "121,122,123"
        for step in expected_config["steps"]:
            command = step.get("command", [])
            if "--seed" in command:
                command[command.index("--seed") + 1] = "126"
        self.assertEqual(expected_config, variant_config)

    def test_aggressive_v3_stronger_bootstrap_confirm_c_local_only_changes_lane_identity_and_seed_schedule(self):
        base_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_CONFIRM_C_LOCAL_CONFIG)
        variant_self_play = self.render_config_step(variant_config, "self_play")

        self.assertEqual("aggressive-v3-stronger-bootstrap-confirm-c-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_stronger_bootstrap_confirm_c_local_versions", variant_config["versions_dir"])
        self.assertEqual(168, variant_config["seed"])
        self.assertEqual("161,162,163", self.command_flag_value(variant_self_play, "--seed-sweep"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_config["seed"] = 168
        expected_steps = self.config_steps_by_name(expected_config)
        expected_steps["self_play"]["command"][expected_steps["self_play"]["command"].index("--seed-sweep") + 1] = "161,162,163"
        for step in expected_config["steps"]:
            command = step.get("command", [])
            if "--seed" in command:
                command[command.index("--seed") + 1] = "168"
        self.assertEqual(expected_config, variant_config)

    def test_readme_documents_v3_stronger_bootstrap_confirmation_sweep(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        section = readme.split("### Stronger bootstrap confirmation sweep", maxsplit=1)[1]
        commands = self.find_bash_block(section, containing=("aggressive_v3_stronger_bootstrap_confirm_b_local.json",))

        self.assertIn("aggressive_v3_stronger_bootstrap_confirm_b_local.json", readme)
        self.assertIn("aggressive_v3_stronger_bootstrap_confirm_c_local.json", readme)
        self.assertIn("seed = 126", readme)
        self.assertIn("121,122,123", readme)
        self.assertIn("seed = 168", readme)
        self.assertIn("161,162,163", readme)
        self.assertIn("script/ai/local_promotion_gate", commands)
        self.assertIn(
            "/tmp/azlite_v3_stronger_bootstrap_confirm_b_local_versions/aggressive-v3-stronger-bootstrap-confirm-b-local-iter1",
            commands,
        )
        self.assertIn(
            "/tmp/azlite_v3_stronger_bootstrap_confirm_c_local_versions/aggressive-v3-stronger-bootstrap-confirm-c-local-iter1",
            commands,
        )

    def test_aggressive_v3_stronger_bootstrap_more_data_local_only_changes_lane_identity_and_bootstrap_games(self):
        base_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.V3_STRONGER_BOOTSTRAP_MORE_DATA_LOCAL_CONFIG)
        variant_bootstrap = self.render_config_step(variant_config, "mcts_bootstrap_dataset")

        self.assertEqual("aggressive-v3-stronger-bootstrap-more-data-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_stronger_bootstrap_more_data_local_versions", variant_config["versions_dir"])
        self.assertEqual("900", self.command_flag_value(variant_bootstrap, "--games"))
        self.assertEqual("2400", self.command_flag_value(variant_bootstrap, "--simulations"))

        expected_config = json.loads(json.dumps(base_config))
        expected_config["run_id"] = variant_config["run_id"]
        expected_config["versions_dir"] = variant_config["versions_dir"]
        expected_steps = self.config_steps_by_name(expected_config)
        bootstrap_command = expected_steps["mcts_bootstrap_dataset"]["command"]
        bootstrap_command[bootstrap_command.index("--games") + 1] = "900"
        self.assertEqual(expected_config, variant_config)

    def test_readme_documents_v3_stronger_bootstrap_more_data_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        section = readme.split("### Stronger bootstrap more-data lane", maxsplit=1)[1]
        commands = self.find_bash_block(section, containing=("aggressive_v3_stronger_bootstrap_more_data_local.json",))

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_more_data_local.json", readme)
        self.assertIn("script/ai/local_promotion_gate", commands)
        self.assertIn("`900`", readme)
        self.assertIn("`2400`", readme)

    def test_aggressive_v3_hybrid_value_target_local_only_changes_value_target_mode(self):
        base_config = self.load_v2_local_config(self.VALUE_TARGET_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.HYBRID_VALUE_TARGET_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-hybrid-value-target-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_hybrid_value_target_local_versions", variant_config["versions_dir"])
        self.assertEqual(base_config["seed"], variant_config["seed"])
        self.assertEqual(base_config["iterations"], variant_config["iterations"])
        self.assertEqual(base_config["replay_window"], variant_config["replay_window"])
        self.assertEqual(base_config["current_path"], variant_config["current_path"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )

        self.assertEqual("sharpened", self.command_flag_value(variant_self_play, "--policy-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_train, "--policy-target-mode"))
        self.assertEqual("hybrid", self.command_flag_value(variant_self_play, "--value-target-mode"))
        self.assertEqual("hybrid", self.command_flag_value(variant_train, "--value-target-mode"))
        self.assertIn("--policy-target-mode sharpened", self.bootstrap_shell(variant_config))
        self.assertIn("--value-target-mode hybrid", self.bootstrap_shell(variant_config))

        for step_name in (
            "perspective_audit",
            "export_artifact",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(
                self.normalize_rendered_command(self.render_config_step(base_config, step_name), config=base_config),
                self.normalize_rendered_command(self.render_config_step(variant_config, step_name), config=variant_config),
            )

        expected_self_play = self.normalize_rendered_command(
            self.render_config_step(base_config, "self_play"),
            config=base_config,
        )
        expected_self_play[expected_self_play.index("--value-target-mode") + 1] = "hybrid"
        self.assertEqual(expected_self_play, self.normalize_rendered_command(variant_self_play, config=variant_config))

        expected_train = self.normalize_rendered_command(
            self.render_config_step(base_config, "train"),
            config=base_config,
        )
        expected_train[expected_train.index("--value-target-mode") + 1] = "hybrid"
        self.assertEqual(expected_train, self.normalize_rendered_command(variant_train, config=variant_config))

        self.assertEqual(
            self.bootstrap_shell(base_config)
            .replace(base_config["run_id"], variant_config["run_id"])
            .replace("--value-target-mode sharpened", "--value-target-mode hybrid"),
            self.bootstrap_shell(variant_config),
        )

    def test_readme_documents_v3_hybrid_value_target_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        hybrid_section = readme.split("### Hybrid value-target lane", maxsplit=1)[1]
        hybrid_commands = self.find_bash_block(
            hybrid_section,
            containing=("aggressive_v3_hybrid_value_target_local.json",),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_hybrid_value_target_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", hybrid_commands)
        self.assertIn("script/ai/local_promotion_gate", hybrid_commands)
        self.assertIn("--config ml/alphazero_lite/configs/aggressive_v3_hybrid_value_target_local.json", hybrid_commands)
        self.assertIn(
            "--candidate-path /tmp/azlite_v3_hybrid_value_target_local_versions/aggressive-v3-hybrid-value-target-local-iter1",
            hybrid_commands,
        )
        self.assertIn(
            "--config-path ml/alphazero_lite/configs/aggressive_v3_hybrid_value_target_local.json",
            hybrid_commands,
        )
        self.assertIn("`--policy-target-mode sharpened`", readme)
        self.assertIn("`--value-target-mode hybrid`", readme)

    def test_aggressive_v3_phase_aware_value_target_local_only_changes_value_target_mode(self):
        base_config = self.load_v2_local_config(self.VALUE_TARGET_LOCAL_CONFIG)
        variant_config = self.load_v2_local_config(self.PHASE_AWARE_VALUE_TARGET_LOCAL_CONFIG)
        base_steps = self.config_steps_by_name(base_config)
        variant_steps = self.config_steps_by_name(variant_config)
        variant_self_play = self.render_config_step(variant_config, "self_play")
        variant_train = self.render_config_step(variant_config, "train")

        self.assertEqual("aggressive-v3-phase-aware-value-target-local", variant_config["run_id"])
        self.assertEqual("/tmp/azlite_v3_phase_aware_value_target_local_versions", variant_config["versions_dir"])
        self.assertEqual(base_config["seed"], variant_config["seed"])
        self.assertEqual(base_config["iterations"], variant_config["iterations"])
        self.assertEqual(base_config["replay_window"], variant_config["replay_window"])
        self.assertEqual(base_config["current_path"], variant_config["current_path"])
        self.assertEqual(base_config["gates"], variant_config["gates"])
        self.assertEqual(
            [step["name"] for step in base_config["steps"]],
            [step["name"] for step in variant_config["steps"]],
        )

        self.assertEqual("sharpened", self.command_flag_value(variant_self_play, "--policy-target-mode"))
        self.assertEqual("sharpened", self.command_flag_value(variant_train, "--policy-target-mode"))
        self.assertEqual("phase_aware_sharpened", self.command_flag_value(variant_self_play, "--value-target-mode"))
        self.assertEqual("phase_aware_sharpened", self.command_flag_value(variant_train, "--value-target-mode"))
        self.assertIn("--policy-target-mode sharpened", self.bootstrap_shell(variant_config))
        self.assertIn("--value-target-mode phase_aware_sharpened", self.bootstrap_shell(variant_config))

        for step_name in (
            "perspective_audit",
            "export_artifact",
            "rules_parity_fuzz",
            "arena_prefilter_report",
            "arena_prefilter_validate",
            "arena_confirm_report",
            "mcts1200_baseline_report",
            "current_mcts1200_baseline_report",
            "benchmark_contract",
            "arena_validate",
        ):
            self.assertEqual(
                self.normalize_rendered_command(self.render_config_step(base_config, step_name), config=base_config),
                self.normalize_rendered_command(self.render_config_step(variant_config, step_name), config=variant_config),
            )

        expected_self_play = self.normalize_rendered_command(
            self.render_config_step(base_config, "self_play"),
            config=base_config,
        )
        expected_self_play[expected_self_play.index("--value-target-mode") + 1] = "phase_aware_sharpened"
        self.assertEqual(expected_self_play, self.normalize_rendered_command(variant_self_play, config=variant_config))

        expected_train = self.normalize_rendered_command(
            self.render_config_step(base_config, "train"),
            config=base_config,
        )
        expected_train[expected_train.index("--value-target-mode") + 1] = "phase_aware_sharpened"
        self.assertEqual(expected_train, self.normalize_rendered_command(variant_train, config=variant_config))

        self.assertEqual(
            self.bootstrap_shell(base_config)
            .replace(base_config["run_id"], variant_config["run_id"])
            .replace("--value-target-mode sharpened", "--value-target-mode phase_aware_sharpened"),
            self.bootstrap_shell(variant_config),
        )

    def test_readme_documents_v3_phase_aware_value_target_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        phase_aware_section = readme.split("### Phase-aware value-target lane", maxsplit=1)[1]
        phase_aware_commands = self.find_bash_block(
            phase_aware_section,
            containing=("aggressive_v3_phase_aware_value_target_local.json",),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_phase_aware_value_target_local.json", readme)
        self.assertIn(".venv/bin/python ml/alphazero_lite/pipeline.py", phase_aware_commands)
        self.assertIn("script/ai/local_promotion_gate", phase_aware_commands)
        self.assertIn("--config ml/alphazero_lite/configs/aggressive_v3_phase_aware_value_target_local.json", phase_aware_commands)
        self.assertIn(
            "--candidate-path /tmp/azlite_v3_phase_aware_value_target_local_versions/aggressive-v3-phase-aware-value-target-local-iter1",
            phase_aware_commands,
        )
        self.assertIn(
            "--config-path ml/alphazero_lite/configs/aggressive_v3_phase_aware_value_target_local.json",
            phase_aware_commands,
        )
        self.assertIn("`--policy-target-mode sharpened`", readme)
        self.assertIn("`--value-target-mode phase_aware_sharpened`", readme)

    def test_local_promotion_gate_dry_run_records_optional_config_path(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate"
            report_path = tmp_path / "promotion_report.json"
            candidate_path.mkdir()

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/local_promotion_gate"),
                    "--candidate-path",
                    str(candidate_path),
                    "--config-path",
                    "ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json",
                    "--out",
                    str(report_path),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json",
                report["config_path"],
            )

    def test_readme_documents_v2_local_promotion_gate_screening_without_special_flags(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")
        local_gate_section = readme.split("## Local promotion gate", maxsplit=1)[1]
        normalized_section = local_gate_section.lower()
        v2_local_gate_command = self.find_bash_block(
            local_gate_section,
            containing=("aggressive-v2-iter1", "script/ai/local_promotion_gate"),
        )

        self.assertIn("ml/alphazero_lite/configs/aggressive_v2.yaml", local_gate_section)
        self.assertIn("dry run only confirms the planned iteration path/name", normalized_section)
        self.assertIn("real pipeline must run through export", normalized_section)
        self.assertIn("script/ai/local_promotion_gate", v2_local_gate_command)
        self.assertIn("--candidate-path storage/ai/alphazero_lite/versions/aggressive-v2-iter1", v2_local_gate_command)
        self.assertNotIn("--model-type", v2_local_gate_command)
        self.assertNotIn("--input-encoding", v2_local_gate_command)

    def test_readme_documents_search_quality_local_ablation_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")

        self.assertIn("ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json", readme)
        self.assertIn("--config-path ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json", readme)
        self.assertIn("`--fpu-mode parent_q`", readme)
        self.assertIn("`--reuse-subtree`", readme)
        self.assertIn("`--normalize-values`", readme)
        self.assertIn("`--root-policy-mode deterministic`", readme)
        self.assertIn("`--tactical-root-bias 0.1`", readme)
        self.assertIn("`mcts1200_baseline_report`", readme)
        self.assertIn("`current_mcts1200_baseline_report`", readme)

    def test_readme_documents_v3_tactical_encoding_local_lane(self):
        repo_root = Path(__file__).resolve().parents[2]
        readme = (repo_root / "ml/alphazero_lite/README.md").read_text(encoding="utf-8")

        self.assertIn("ml/alphazero_lite/configs/aggressive_v3_tactical_encoding_local.json", readme)
        self.assertIn("aggressive-v3-tactical-encoding-local-iter1", readme)
        self.assertIn("`kalah_v3`", readme)
        self.assertIn("`residual_v2`", readme)

    def test_bootstrap_300_config_renders_expected_dataset_step(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_bootstrap_300.yaml")
        step = next(step for step in config["steps"] if step["name"] == "mcts_bootstrap_dataset")
        iter_dir = repo_root / "tmp" / "aggressive-v1-iter1"

        rendered = render_command(
            step["command"],
            iteration=1,
            iter_dir=iter_dir,
            run_id="aggressive-v1",
            versions_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "versions",
            current_path="storage/ai/alphazero_lite/current",
            parent_model_dir=repo_root / "storage" / "ai" / "alphazero_lite" / "current",
            parent_checkpoint=repo_root / "storage" / "ai" / "alphazero_lite" / "current" / "checkpoint.npz",
            replay_data="",
            replay_weights="",
        )

        self.assertEqual(
            [
                ".venv/bin/python",
                "ml/alphazero_lite/generate_bootstrap_dataset.py",
                "--out",
                f"{iter_dir}/mcts_bootstrap.jsonl",
                "--games",
                "300",
                "--simulations",
                "1200",
                "--seed",
                "42",
                "--max-positions-per-game",
                "16",
                "--workers",
                "5",
                "--tree-reuse-enabled",
            ],
            rendered,
        )

    def test_bootstrap_300_config_only_changes_bootstrap_dataset_game_count(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        variant_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_bootstrap_300.yaml")

        self.assertEqual(set(base_config), set(variant_config))
        self.assertEqual(base_config["run_id"], variant_config["run_id"])
        self.assertEqual(base_config["seed"], variant_config["seed"])
        self.assertEqual(base_config["iterations"], variant_config["iterations"])
        self.assertEqual(base_config["replay_window"], variant_config["replay_window"])
        self.assertEqual(base_config["current_path"], variant_config["current_path"])
        self.assertEqual(base_config["versions_dir"], variant_config["versions_dir"])
        self.assertEqual(base_config["gates"], variant_config["gates"])

        base_steps = base_config["steps"]
        variant_steps = variant_config["steps"]

        self.assertEqual(len(base_steps), len(variant_steps))

        for base_step, variant_step in zip(base_steps, variant_steps):
            self.assertEqual(base_step["name"], variant_step["name"])

            if base_step["name"] == "mcts_bootstrap_dataset":
                continue

            self.assertEqual(base_step, variant_step)

        bootstrap_index = next(
            index for index, step in enumerate(base_steps) if step["name"] == "mcts_bootstrap_dataset"
        )
        base_bootstrap_step = base_steps[bootstrap_index]
        variant_bootstrap_step = variant_steps[bootstrap_index]
        base_bootstrap = " ".join(base_bootstrap_step["command"])
        variant_bootstrap = " ".join(variant_bootstrap_step["command"])

        self.assertEqual(base_bootstrap_step["name"], variant_bootstrap_step["name"])
        self.assertIn("--games 1200", base_bootstrap)
        self.assertIn("--games 300", variant_bootstrap)
        self.assertEqual(
            base_bootstrap.replace("--games 1200", "--games 300"),
            variant_bootstrap,
        )

    def test_stronger_bootstrap_config_uses_smaller_but_stronger_teacher(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        variant_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_bootstrap_stronger.yaml")

        self.assertEqual(set(base_config), set(variant_config))
        self.assertEqual(base_config["gates"], variant_config["gates"])

        base_steps = base_config["steps"]
        variant_steps = variant_config["steps"]
        self.assertEqual(len(base_steps), len(variant_steps))

        for base_step, variant_step in zip(base_steps, variant_steps):
            self.assertEqual(base_step["name"], variant_step["name"])

            if base_step["name"] == "mcts_bootstrap_dataset":
                continue

            self.assertEqual(base_step, variant_step)

        base_bootstrap = " ".join(next(step for step in base_steps if step["name"] == "mcts_bootstrap_dataset")["command"])
        variant_bootstrap = " ".join(next(step for step in variant_steps if step["name"] == "mcts_bootstrap_dataset")["command"])

        self.assertIn("--games 1200", base_bootstrap)
        self.assertIn("--simulations 1200", base_bootstrap)
        self.assertIn("--games 600", variant_bootstrap)
        self.assertIn("--simulations 2400", variant_bootstrap)
        self.assertEqual(
            base_bootstrap.replace("--games 1200", "--games 600").replace("--simulations 1200", "--simulations 2400"),
            variant_bootstrap,
        )

    def test_data_mix_configs_only_change_bootstrap_share(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        light_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_light.yaml")
        medium_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_medium.yaml")
        heavy_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_heavy.yaml")

        variants = [light_config, medium_config, heavy_config]
        for variant in variants:
            self.assertEqual(set(base_config), set(variant))
            self.assertEqual(base_config["gates"], variant["gates"])
            self.assertEqual(len(base_config["steps"]), len(variant["steps"]))

        base_steps = {step["name"]: step for step in base_config["steps"]}
        variant_steps = [{step["name"]: step for step in variant["steps"]} for variant in variants]

        for name, base_step in base_steps.items():
            for steps in variant_steps:
                self.assertIn(name, steps)

                if name in {"mcts_bootstrap_dataset", "train"}:
                    continue

                self.assertEqual(base_step, steps[name])

        light_bootstrap = " ".join(variant_steps[0]["mcts_bootstrap_dataset"]["command"])
        medium_bootstrap = " ".join(variant_steps[1]["mcts_bootstrap_dataset"]["command"])
        heavy_bootstrap = " ".join(variant_steps[2]["mcts_bootstrap_dataset"]["command"])

        self.assertIn("--games 300", light_bootstrap)
        self.assertIn("--games 600", medium_bootstrap)
        self.assertIn("--games 1200", heavy_bootstrap)
        self.assertIn("--simulations 1200", light_bootstrap)
        self.assertIn("--simulations 1200", medium_bootstrap)
        self.assertIn("--simulations 2400", heavy_bootstrap)

        light_train = variant_steps[0]["train"]["command"]
        medium_train = variant_steps[1]["train"]["command"]
        heavy_train = variant_steps[2]["train"]["command"]

        self.assertEqual("{replay_weights},2", light_train[light_train.index("--replay-weights") + 1])
        self.assertEqual("{replay_weights},4", medium_train[medium_train.index("--replay-weights") + 1])
        self.assertEqual("{replay_weights},6", heavy_train[heavy_train.index("--replay-weights") + 1])

    def test_staged_strict_candidate_config_adds_bootstrap_then_selfplay_training(self):
        repo_root = Path(__file__).resolve().parents[2]
        base_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1.yaml")
        candidate_config = load_config(repo_root / "ml/alphazero_lite/configs/aggressive_v1_staged_candidate.yaml")

        self.assertEqual(base_config["gates"], candidate_config["gates"])

        candidate_steps = {step["name"]: step for step in candidate_config["steps"]}
        self.assertIn("train_bootstrap_stage", candidate_steps)
        self.assertIn("train_self_play_stage", candidate_steps)

        bootstrap_train = candidate_steps["train_bootstrap_stage"]["command"]
        self_play_train = candidate_steps["train_self_play_stage"]["command"]
        benchmark_step = candidate_steps["benchmark_contract"]

        self.assertEqual("{replay_weights},8", bootstrap_train[bootstrap_train.index("--replay-weights") + 1])
        self.assertIn("--data-files", bootstrap_train)
        self.assertNotIn("--data-files", self_play_train)
        self.assertIn("--init-checkpoint", self_play_train)
        self.assertIn("{iter_dir}/bootstrap_checkpoint.npz", self_play_train[self_play_train.index("--init-checkpoint") + 1])
        self.assertEqual("{iter_dir}/mcts1200_report.json", benchmark_step["command"][benchmark_step["command"].index("--mcts-report") + 1])

    def test_local_promotion_gate_dry_run_writes_expected_report_plan(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate.npz"
            report_path = tmp_path / "promotion_report.json"
            candidate_path.write_text("stub", encoding="utf-8")

            result = subprocess.run(
                [
                    str(repo_root / "script/ai/local_promotion_gate"),
                    "--candidate-path",
                    str(candidate_path),
                    "--out",
                    str(report_path),
                    "--hard-path",
                    "model-artifact/current",
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(report_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual("azlite_local_promotion_gate_v1", report["schema"])
            self.assertEqual(str(candidate_path), report["candidate_path"])
            self.assertEqual("model-artifact/current", report["current_path"])
            self.assertEqual(str(report_path), report["report_path"])
            self.assertEqual(4, len(report["evaluations"]))
            self.assertEqual(
                [
                    "candidate_vs_current_arena",
                    "candidate_vs_hard_arena",
                    "candidate_vs_mcts1200",
                    "current_vs_mcts1200",
                ],
                [evaluation["id"] for evaluation in report["evaluations"]],
            )
            self.assertEqual(
                [
                    {
                        "id": "candidate_vs_current_arena",
                        "subject": str(candidate_path),
                        "opponent": "model-artifact/current",
                        "games": 120,
                        "report_path": str(tmp_path / "candidate_vs_current_arena.json"),
                    },
                    {
                        "id": "candidate_vs_hard_arena",
                        "subject": str(candidate_path),
                        "opponent": "model-artifact/current",
                        "games": 120,
                        "report_path": str(tmp_path / "candidate_vs_hard_arena.json"),
                    },
                    {
                        "id": "candidate_vs_mcts1200",
                        "subject": str(candidate_path),
                        "opponent": "mcts1200",
                        "games": 40,
                        "report_path": str(tmp_path / "candidate_vs_mcts1200.json"),
                    },
                    {
                        "id": "current_vs_mcts1200",
                        "subject": "model-artifact/current",
                        "opponent": "mcts1200",
                        "games": 40,
                        "report_path": str(tmp_path / "current_vs_mcts1200.json"),
                    },
                ],
                [
                    {
                        "id": evaluation["id"],
                        "subject": evaluation["subject"],
                        "opponent": evaluation["opponent"],
                        "games": evaluation["games"],
                        "report_path": evaluation["report_path"],
                    }
                    for evaluation in report["evaluations"]
                ],
            )
            self.assertTrue(all(evaluation["mode"] == "dry_run" for evaluation in report["evaluations"]))

    def test_local_promotion_gate_passes_when_reports_meet_thresholds(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 66, "losses": 42, "draws": 12, "games_played": 120},
            candidate_mcts_report={"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40},
            current_mcts_report={"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40},
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertTrue(report["passed"])
        self.assertEqual([], report["failure_reasons"])

    def test_local_promotion_gate_fails_when_arena_score_below_threshold(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 54, "losses": 60, "draws": 6, "games_played": 120},
            candidate_mcts_report={"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40},
            current_mcts_report={"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40},
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertFalse(report["passed"])
        self.assertEqual([{"code": "arena_score_below_threshold"}], report["failure_reasons"])

    def test_local_promotion_gate_fails_when_candidate_mcts_below_current(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 66, "losses": 42, "draws": 12, "games_played": 120},
            candidate_mcts_report={"az_wins": 16, "mcts_wins": 16, "draws": 8, "games": 40},
            current_mcts_report={"az_wins": 18, "mcts_wins": 14, "draws": 8, "games": 40},
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertFalse(report["passed"])
        self.assertEqual([{"code": "candidate_mcts_below_current"}], report["failure_reasons"])

    def test_local_promotion_gate_fails_when_arena_games_below_minimum(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 66, "losses": 42, "draws": 11, "games_played": 119},
            candidate_mcts_report={"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40},
            current_mcts_report={"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40},
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertFalse(report["passed"])
        self.assertEqual([{"code": "arena_games_below_minimum"}], report["failure_reasons"])

    def test_local_promotion_gate_fails_when_mcts_games_below_minimum(self):
        cases = [
            (
                "candidate",
                {"az_wins": 19, "mcts_wins": 13, "draws": 7, "games": 39},
                {"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40},
                "candidate_mcts_games_below_minimum",
            ),
            (
                "current",
                {"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40},
                {"az_wins": 17, "mcts_wins": 15, "draws": 7, "games": 39},
                "current_mcts_games_below_minimum",
            ),
        ]

        for label, candidate_mcts_report, current_mcts_report, expected_code in cases:
            with self.subTest(report=label):
                result, report = self.run_local_promotion_gate_with_stub_reports(
                    arena_report={"wins": 66, "losses": 42, "draws": 12, "games_played": 120},
                    candidate_mcts_report=candidate_mcts_report,
                    current_mcts_report=current_mcts_report,
                )

                self.assertNotEqual(0, result.returncode)
                self.assertIsNotNone(report)
                assert report is not None
                self.assertFalse(report["passed"])
                self.assertEqual([{"code": expected_code}], report["failure_reasons"])

    def test_local_promotion_gate_uses_raw_arena_score_for_threshold_decision(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 10999, "losses": 9001, "draws": 0, "games_played": 20000},
            candidate_mcts_report={"az_wins": 10020, "mcts_wins": 9980, "draws": 0, "games": 20000},
            current_mcts_report={"az_wins": 10010, "mcts_wins": 9990, "draws": 0, "games": 20000},
            min_arena_score=0.54996,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(0.55, report["arena_score"])
        self.assertFalse(report["passed"])
        self.assertEqual([{"code": "arena_score_below_threshold"}], report["failure_reasons"])

    def test_local_promotion_gate_uses_raw_mcts_scores_for_current_comparison(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 12000, "losses": 8000, "draws": 0, "games_played": 20000},
            candidate_mcts_report={"az_wins": 100005, "mcts_wins": 99995, "draws": 0, "games": 200000},
            current_mcts_report={"az_wins": 100006, "mcts_wins": 99994, "draws": 0, "games": 200000},
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(0.5, report["candidate_mcts_score"])
        self.assertEqual(0.5, report["current_mcts_score"])
        self.assertFalse(report["passed"])
        self.assertEqual([{"code": "candidate_mcts_below_current"}], report["failure_reasons"])

    def test_local_promotion_gate_passes_in_lossless_mode_with_zero_losses(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 0, "losses": 0, "draws": 400, "games_played": 400},
            candidate_mcts_report={"az_wins": 10, "mcts_wins": 20, "draws": 10, "games": 40},
            current_mcts_report={"az_wins": 30, "mcts_wins": 5, "draws": 5, "games": 40},
            min_arena_score=0.0,
            min_arena_games=400,
            extra_args=["--require-lossless", "--skip-mcts-relative-check"],
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertTrue(report["passed"])
        self.assertTrue(report["lossless_passed"])
        self.assertEqual([], report["failure_reasons"])

    def test_local_promotion_gate_fails_in_lossless_mode_when_any_loss_occurs(self):
        result, report = self.run_local_promotion_gate_with_stub_reports(
            arena_report={"wins": 399, "losses": 1, "draws": 0, "games_played": 400},
            candidate_mcts_report={"az_wins": 20, "mcts_wins": 10, "draws": 10, "games": 40},
            current_mcts_report={"az_wins": 20, "mcts_wins": 10, "draws": 10, "games": 40},
            min_arena_score=0.0,
            min_arena_games=400,
            extra_args=["--require-lossless", "--skip-mcts-relative-check"],
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertFalse(report["passed"])
        self.assertFalse(report["lossless_passed"])
        self.assertEqual([{"code": "arena_losses_above_threshold"}], report["failure_reasons"])

    def test_local_promotion_gate_runs_real_evaluations_and_writes_decision_report(self):
        module = self.load_local_promotion_gate_module()

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate"
            current_path = tmp_path / "current"
            report_path = tmp_path / "promotion_report.json"

            candidate_path.mkdir()
            current_path.mkdir()

            calls = []

            def fake_run(command, cwd=None, capture_output=None, text=None, check=None):
                calls.append({"command": command, "cwd": cwd, "capture_output": capture_output, "text": text, "check": check})
                out_path = Path(command[command.index("--out") + 1])
                if command[1].endswith("arena.py"):
                    payload = {"wins": 66, "losses": 42, "draws": 12, "games_played": 120}
                elif command[command.index("--challenger-path") + 1] == str(candidate_path):
                    payload = {"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40}
                else:
                    payload = {"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40}
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            argv = [
                str(module.__file__),
                "--candidate-path",
                str(candidate_path),
                "--current-path",
                str(current_path),
                "--out",
                str(report_path),
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                exit_code = module.main()

            self.assertEqual(0, exit_code)
            self.assertEqual(4, len(calls))
            self.assertEqual(
                [
                    str(tmp_path / "candidate_vs_current_arena.json"),
                    str(tmp_path / "candidate_vs_hard_arena.json"),
                    str(tmp_path / "candidate_vs_mcts1200.json"),
                    str(tmp_path / "current_vs_mcts1200.json"),
                ],
                [call["command"][call["command"].index("--out") + 1] for call in calls],
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(0.6, report["arena_score"])
            self.assertEqual(0.575, report["candidate_mcts_score"])
            self.assertEqual(0.525, report["current_mcts_score"])
            self.assertTrue(report["passed"])
            self.assertEqual([], report["failure_reasons"])

    def test_local_promotion_gate_forwards_search_quality_flags_from_config_to_real_commands(self):
        module = self.load_local_promotion_gate_module()

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate"
            current_path = tmp_path / "current"
            report_path = tmp_path / "promotion_report.json"

            candidate_path.mkdir()
            current_path.mkdir()

            calls = []

            def fake_run(command, cwd=None, capture_output=None, text=None, check=None):
                calls.append(command)
                out_path = Path(command[command.index("--out") + 1])
                if command[1].endswith("arena.py"):
                    payload = {"wins": 66, "losses": 42, "draws": 12, "games_played": 120}
                elif command[command.index("--challenger-path") + 1] == str(candidate_path):
                    payload = {"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40}
                else:
                    payload = {"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40}
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            argv = [
                str(module.__file__),
                "--candidate-path",
                str(candidate_path),
                "--current-path",
                str(current_path),
                "--config-path",
                "ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json",
                "--out",
                str(report_path),
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                exit_code = module.main()

            self.assertEqual(0, exit_code)
            self.assertEqual(4, len(calls))

            for command in calls:
                if command[1].endswith("arena.py") and "candidate_vs_hard_arena" not in command:
                    self.assertIn("--fpu-mode", command)
                    self.assertIn("parent_q", command)
                    self.assertIn("--reuse-subtree", command)
                    self.assertIn("--normalize-values", command)
                    self.assertIn("--root-policy-mode", command)
                    self.assertIn("deterministic", command)
                    self.assertIn("--tactical-root-bias", command)
                    self.assertIn("0.1", command)

    def test_local_promotion_gate_returns_nonzero_when_real_screening_fails(self):
        module = self.load_local_promotion_gate_module()

        with tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
            tmp_path = Path(tmp)
            candidate_path = tmp_path / "candidate"
            current_path = tmp_path / "current"
            report_path = tmp_path / "promotion_report.json"

            candidate_path.mkdir()
            current_path.mkdir()

            def fake_run(command, cwd=None, capture_output=None, text=None, check=None):
                out_path = Path(command[command.index("--out") + 1])
                if command[1].endswith("arena.py"):
                    payload = {"wins": 54, "losses": 60, "draws": 6, "games_played": 120}
                else:
                    payload = {"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40}
                out_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            argv = [
                str(module.__file__),
                "--candidate-path",
                str(candidate_path),
                "--current-path",
                str(current_path),
                "--out",
                str(report_path),
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                exit_code = module.main()

            self.assertEqual(1, exit_code)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual([{"code": "arena_score_below_threshold"}, {"code": "candidate_not_stronger_than_hard"}], report["failure_reasons"])

    def test_local_promotion_gate_surfaces_real_subprocess_failures_with_command_context(self):
        module = self.load_local_promotion_gate_module()

        cases = [
            ("arena", 1, "arena.py", "arena exploded"),
            ("candidate_mcts", 3, "mcts1200_baseline.py", "mcts exploded"),
        ]

        for label, expected_calls, failing_script, stderr_text in cases:
            with self.subTest(step=label), tempfile.TemporaryDirectory(prefix="local-promotion-gate-") as tmp:
                tmp_path = Path(tmp)
                candidate_path = tmp_path / "candidate"
                current_path = tmp_path / "current"
                report_path = tmp_path / "promotion_report.json"

                candidate_path.mkdir()
                current_path.mkdir()
                calls = []

                def fake_run(command, cwd=None, capture_output=None, text=None, check=None):
                    calls.append(command)
                    out_path = Path(command[command.index("--out") + 1])
                    script_name = Path(command[1]).name
                    if script_name == failing_script:
                        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr_text)
                    if script_name == "arena.py":
                        payload = {"wins": 66, "losses": 42, "draws": 12, "games_played": 120}
                    elif command[command.index("--challenger-path") + 1] == str(candidate_path):
                        payload = {"az_wins": 19, "mcts_wins": 13, "draws": 8, "games": 40}
                    else:
                        payload = {"az_wins": 17, "mcts_wins": 15, "draws": 8, "games": 40}
                    out_path.write_text(json.dumps(payload), encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

                argv = [
                    str(module.__file__),
                    "--candidate-path",
                    str(candidate_path),
                    "--current-path",
                    str(current_path),
                    "--out",
                    str(report_path),
                ]

                with mock.patch.object(sys, "argv", argv), mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(SystemExit) as exc:
                        module.main()

                self.assertIn(failing_script, str(exc.exception))
                self.assertIn(stderr_text, str(exc.exception))
                self.assertEqual(expected_calls, len(calls))
                self.assertFalse(report_path.exists())

    def test_local_mix_ablation_runner_dry_run_writes_reduced_local_config(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                str(repo_root / "script/ai/run_local_mix_ablation"),
                "--config",
                str(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_light.yaml"),
                "--run-id",
                "mix-light-local-test",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual("mix-light-local-test", report["run_id"])

        generated_config_path = Path(report["generated_config_path"])
        self.assertTrue(generated_config_path.exists())

        generated_config = json.loads(generated_config_path.read_text(encoding="utf-8"))
        step_names = [step["name"] for step in generated_config["steps"]]
        self.assertNotIn("arena_prefilter_ruby_sweep", step_names)
        self.assertNotIn("mcts1200_baseline_report", step_names)

        self_play_command = next(step["command"] for step in generated_config["steps"] if step["name"] == "self_play")
        bootstrap_command = next(step["command"] for step in generated_config["steps"] if step["name"] == "mcts_bootstrap_dataset")

        self.assertEqual("400", self_play_command[self_play_command.index("--games") + 1])
        self.assertEqual("60", bootstrap_command[bootstrap_command.index("--games") + 1])

    def test_local_staged_ablation_runner_dry_run_writes_two_stage_configs(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                str(repo_root / "script/ai/run_local_staged_ablation"),
                "--config",
                str(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_heavy.yaml"),
                "--run-id",
                "staged-local-test",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual("staged-local-test", report["run_id"])

        stage1_config_path = Path(report["stage1_config_path"])
        stage2_config_path = Path(report["stage2_config_path"])
        self.assertTrue(stage1_config_path.exists())
        self.assertTrue(stage2_config_path.exists())

        stage1_config = json.loads(stage1_config_path.read_text(encoding="utf-8"))
        stage2_config = json.loads(stage2_config_path.read_text(encoding="utf-8"))

        stage1_train = next(step["command"] for step in stage1_config["steps"] if step["name"] == "train")
        stage2_train = next(step["command"] for step in stage2_config["steps"] if step["name"] == "train")
        stage2_bootstrap = next(step["command"] for step in stage2_config["steps"] if step["name"] == "mcts_bootstrap_dataset")

        self.assertNotIn("--init-checkpoint", stage1_train)
        self.assertIn("--init-checkpoint", stage2_train)
        self.assertIn(f"{report['run_id']}-stage1-iter1/checkpoint.npz", stage2_train[stage2_train.index("--init-checkpoint") + 1])
        self.assertNotIn("--data-files", stage2_train)
        self.assertNotIn("--replay-weights", stage2_train)
        self.assertEqual(
            ["bash", "-lc", "printf '' > '{iter_dir}/mcts_bootstrap.jsonl'"],
            stage2_bootstrap,
        )

    def test_local_staged_ablation_runner_supports_stage2_balance_presets(self):
        repo_root = Path(__file__).resolve().parents[2]
        presets = {
            "blend_light": "{replay_weights},1",
            "blend_medium": "{replay_weights},2",
            "blend_selfplay_bias": "{replay_weights},4",
        }

        for preset, expected_replay_weights in presets.items():
            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_staged_ablation"),
                    "--config",
                    str(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_heavy.yaml"),
                    "--run-id",
                    f"{preset}-test",
                    "--stage2-preset",
                    preset,
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(result.stdout)
            stage2_config = json.loads(Path(report["stage2_config_path"]).read_text(encoding="utf-8"))
            stage2_train = next(step["command"] for step in stage2_config["steps"] if step["name"] == "train")

            self.assertIn("--data-files", stage2_train)
            self.assertIn("--replay-weights", stage2_train)
            self.assertIn(expected_replay_weights, stage2_train)
            self.assertIn(
                f"{report['run_id']}-stage1-iter1/mcts_bootstrap.jsonl",
                stage2_train[stage2_train.index("--data-files") + 1],
            )

    def test_local_staged_ablation_runner_supports_wider_student_preset(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                str(repo_root / "script/ai/run_local_staged_ablation"),
                "--config",
                str(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_heavy.yaml"),
                "--run-id",
                "wider-stage-test",
                "--model-preset",
                "wider",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        report = json.loads(result.stdout)
        stage1_config = json.loads(Path(report["stage1_config_path"]).read_text(encoding="utf-8"))
        stage2_config = json.loads(Path(report["stage2_config_path"]).read_text(encoding="utf-8"))

        stage1_train = next(step["command"] for step in stage1_config["steps"] if step["name"] == "train")
        stage2_train = next(step["command"] for step in stage2_config["steps"] if step["name"] == "train")

        self.assertEqual("256,256,128", stage1_train[stage1_train.index("--hidden-sizes") + 1])
        self.assertEqual("256,256,128", stage2_train[stage2_train.index("--hidden-sizes") + 1])
        self.assertIn("--init-checkpoint", stage2_train)
        self.assertNotIn("--data-files", stage2_train)

    def test_local_staged_ablation_runner_supports_stage2_optimization_presets(self):
        repo_root = Path(__file__).resolve().parents[2]
        presets = {
            "default": ("14", None),
            "short_ft": ("6", None),
            "low_lr": ("14", "0.0003"),
            "short_ft_low_lr": ("6", "0.0003"),
        }

        for preset, (expected_epochs, expected_lr) in presets.items():
            result = subprocess.run(
                [
                    str(repo_root / "script/ai/run_local_staged_ablation"),
                    "--config",
                    str(repo_root / "ml/alphazero_lite/configs/aggressive_v1_mix_heavy.yaml"),
                    "--run-id",
                    f"{preset}-opt-test",
                    "--stage2-optimization-preset",
                    preset,
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(result.stdout)
            stage1_config = json.loads(Path(report["stage1_config_path"]).read_text(encoding="utf-8"))
            stage2_config = json.loads(Path(report["stage2_config_path"]).read_text(encoding="utf-8"))

            stage1_train = next(step["command"] for step in stage1_config["steps"] if step["name"] == "train")
            stage2_train = next(step["command"] for step in stage2_config["steps"] if step["name"] == "train")

            self.assertEqual("14", stage1_train[stage1_train.index("--epochs") + 1])
            self.assertEqual(expected_epochs, stage2_train[stage2_train.index("--epochs") + 1])
            self.assertIn("--init-checkpoint", stage2_train)
            self.assertNotIn("--data-files", stage2_train)

            if expected_lr is None:
                self.assertNotIn("--lr", stage2_train)
            else:
                self.assertIn("--lr", stage2_train)
                self.assertEqual(expected_lr, stage2_train[stage2_train.index("--lr") + 1])


    def test_start_iteration_produces_iter2_and_iter3_dirs_with_replay_context(self):
        """Pipeline with start_iteration=2, iterations=2 must create iter2 and iter3
        directories. At iter2, replay context must include iter1 data if it exists on disk."""
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="pipeline-start-iter-") as tmp:
            tmp_path = Path(tmp)
            versions_dir = tmp_path / "versions"
            run_id = "start-iter-test"

            # Pre-populate iter1 self_play.jsonl so replay context can find it
            iter1_dir = versions_dir / f"{run_id}-iter1"
            iter1_dir.mkdir(parents=True)
            (iter1_dir / "self_play.jsonl").write_text('{"row":1}', encoding="utf-8")

            config_path = tmp_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "versions_dir": str(versions_dir),
                        "seed": 42,
                        "iterations": 2,
                        "start_iteration": 2,
                        "replay_window": 3,
                        "steps": [
                            {
                                "name": "self_play",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('row', encoding='utf-8')",
                                    "{iter_dir}/self_play.jsonl",
                                ],
                            },
                            {
                                "name": "capture_replay",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; import sys; Path(sys.argv[2]).write_text(sys.argv[1], encoding='utf-8')",
                                    "{replay_data}",
                                    "{iter_dir}/replay_capture.txt",
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "ml/alphazero_lite/pipeline.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)

            # iter2 and iter3 directories must exist, iter1 must not be recreated
            self.assertTrue((versions_dir / f"{run_id}-iter2").exists(), "iter2 dir missing")
            self.assertTrue((versions_dir / f"{run_id}-iter3").exists(), "iter3 dir missing")

            # iter2 replay context must include iter1 data
            replay_capture_iter2 = (versions_dir / f"{run_id}-iter2" / "replay_capture.txt").read_text(encoding="utf-8")
            self.assertIn(f"{run_id}-iter2/self_play.jsonl", replay_capture_iter2)
            self.assertIn(f"{run_id}-iter1/self_play.jsonl", replay_capture_iter2, "iter1 data missing from iter2 replay context")

            # iter3 replay context must include iter2 and iter1 data
            replay_capture_iter3 = (versions_dir / f"{run_id}-iter3" / "replay_capture.txt").read_text(encoding="utf-8")
            self.assertIn(f"{run_id}-iter3/self_play.jsonl", replay_capture_iter3)
            self.assertIn(f"{run_id}-iter2/self_play.jsonl", replay_capture_iter3)
            self.assertIn(f"{run_id}-iter1/self_play.jsonl", replay_capture_iter3)


    def test_skip_before_final_iteration_skips_on_non_final_and_executes_on_final(self):
        """A step with skip_before_final_iteration:true must be skipped on non-final
        iterations (manifest status=skipped, command not run) and executed normally
        on the final iteration."""
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="pipeline-skip-final-") as tmp:
            tmp_path = Path(tmp)
            versions_dir = tmp_path / "versions"
            run_id = "skip-final-test"
            sentinel_file = tmp_path / "gate_ran.txt"

            config_path = tmp_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "versions_dir": str(versions_dir),
                        "seed": 42,
                        "iterations": 2,
                        "steps": [
                            {
                                "name": "always_step",
                                "command": [sys.executable, "-c", "pass"],
                            },
                            {
                                "name": "final_only_gate",
                                "skip_before_final_iteration": True,
                                "command": [
                                    sys.executable,
                                    "-c",
                                    # Append one line per execution so we can count calls.
                                    f"from pathlib import Path; f=Path(r'{sentinel_file}'); f.write_text((f.read_text(encoding='utf-8') if f.exists() else '') + 'ran\\n', encoding='utf-8')",
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(repo_root / "ml/alphazero_lite/pipeline.py"), "--config", str(config_path)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)

            # --- iter1 manifest: gate step must be skipped ---
            iter1_manifest = json.loads(
                (versions_dir / f"{run_id}-iter1" / "run_manifest.json").read_text(encoding="utf-8")
            )
            iter1_step_names = [s["name"] for s in iter1_manifest["steps"]]
            self.assertIn("final_only_gate", iter1_step_names, "gate step must appear in iter1 manifest")
            gate_iter1 = next(s for s in iter1_manifest["steps"] if s["name"] == "final_only_gate")
            self.assertEqual("skipped", gate_iter1["status"])
            self.assertEqual("skip_before_final_iteration", gate_iter1["reason"])

            # --- iter2 manifest: gate step must be executed ---
            iter2_manifest = json.loads(
                (versions_dir / f"{run_id}-iter2" / "run_manifest.json").read_text(encoding="utf-8")
            )
            gate_iter2 = next(s for s in iter2_manifest["steps"] if s["name"] == "final_only_gate")
            self.assertEqual("completed", gate_iter2["status"])

            # sentinel must exist (command ran on iter2, the final iteration) and must contain
            # exactly one "ran" entry — proving the gate ran on iter2 only, not on iter1.
            self.assertTrue(sentinel_file.exists(), "gate command must run on the final iteration")
            self.assertEqual(
                1,
                sentinel_file.read_text(encoding="utf-8").count("ran"),
                "gate command must have run exactly once (on the final iteration only)",
            )


if __name__ == "__main__":
    unittest.main()
