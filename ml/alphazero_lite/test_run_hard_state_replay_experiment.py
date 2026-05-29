from __future__ import annotations

import unittest


class RunHardStateReplayExperimentTest(unittest.TestCase):
    def test_apply_runtime_seed_rewrites_seed_flags_and_seed_sweep(self) -> None:
        from ml.alphazero_lite import run_hard_state_replay_experiment as module

        config = {
            "seed": 42,
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--seed",
                        "42",
                        "--seed-sweep",
                        "41,42,43",
                    ],
                },
                {
                    "name": "arena_confirm_report",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/arena.py",
                        "--seed",
                        "42",
                    ],
                },
            ],
        }

        rewritten = module.apply_runtime_seed(config, seed=44)

        self.assertEqual(44, rewritten["seed"])
        self.assertEqual(
            "44",
            rewritten["steps"][0]["command"][
                rewritten["steps"][0]["command"].index("--seed") + 1
            ],
        )
        self.assertEqual(
            "43,44,45",
            rewritten["steps"][0]["command"][
                rewritten["steps"][0]["command"].index("--seed-sweep") + 1
            ],
        )
        self.assertEqual(
            "44",
            rewritten["steps"][1]["command"][
                rewritten["steps"][1]["command"].index("--seed") + 1
            ],
        )

    def test_build_runtime_config_propagates_runtime_seed(self) -> None:
        from pathlib import Path
        import tempfile

        from ml.alphazero_lite import run_hard_state_replay_experiment as module

        base_config = {
            "run_id": "demo",
            "seed": 42,
            "iterations": 1,
            "start_iteration": 1,
            "versions_dir": "/tmp/demo-versions",
            "fixed_replay_sources": [],
            "steps": [
                {
                    "name": "self_play",
                    "command": ["cmd", "--seed", "42", "--seed-sweep", "41,42,43"],
                },
                {"name": "train", "command": ["cmd", "--seed", "42"]},
                {
                    "name": "hard_state_validation",
                    "command": ["cmd", "--seed", "42"],
                },
                {
                    "name": "arena_confirm_report",
                    "command": ["cmd", "--seed", "42"],
                },
                {
                    "name": "mcts1200_baseline_report",
                    "command": ["cmd", "--seed", "42"],
                },
                {
                    "name": "benchmark_contract",
                    "command": ["cmd", "--seed", "42"],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            config, runtime_config_path = module.build_runtime_config(
                base_config=base_config,
                run_root=Path(tmp),
                stronger_train_path=Path(tmp) / "hard_state_train.jsonl",
                weight=2,
                seed=44,
                current_path="storage/ai/alphazero_lite/current",
                hard_state_validation_path="ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json",
            )

            self.assertTrue(runtime_config_path.exists())
            self.assertEqual(44, config["seed"])
            self.assertEqual(
                "43,44,45",
                config["steps"][0]["command"][
                    config["steps"][0]["command"].index("--seed-sweep") + 1
                ],
            )
            for step in config["steps"]:
                command = step["command"]
                if "--seed" in command:
                    self.assertEqual("44", command[command.index("--seed") + 1])


if __name__ == "__main__":
    unittest.main()
