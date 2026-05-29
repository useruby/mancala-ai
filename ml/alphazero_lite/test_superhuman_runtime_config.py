import unittest

from ml.alphazero_lite.superhuman_runtime_config import (
    apply_memory_speed_profile,
    apply_shared_worker_normalization,
)


class SuperhumanRuntimeConfigTest(unittest.TestCase):
    def test_apply_memory_speed_profile_keeps_equals_form_worker_override(self):
        config = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--checkpoint=parent.npz",
                        "--workers=11",
                    ],
                }
            ]
        }

        updated = apply_memory_speed_profile(
            config, memory_speed_profile="high_memory_local"
        )

        self.assertIn("--workers=11", updated["steps"][0]["command"])
        self.assertNotIn("--workers", updated["steps"][0]["command"])
        self.assertIn("--evaluator-cache-size", updated["steps"][0]["command"])
        self.assertEqual(
            "200000",
            updated["steps"][0]["command"][
                updated["steps"][0]["command"].index("--evaluator-cache-size") + 1
            ],
        )

    def test_apply_memory_speed_profile_rejects_unknown_profile(self):
        config = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--checkpoint",
                        "parent.npz",
                    ],
                }
            ]
        }

        with self.assertRaisesRegex(
            ValueError, "unsupported memory_speed_profile: high-memory-local"
        ):
            apply_memory_speed_profile(config, memory_speed_profile="high-memory-local")

    def test_apply_memory_speed_profile_keeps_explicit_worker_override(self):
        config = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--checkpoint",
                        "parent.npz",
                    ],
                }
            ]
        }

        updated = apply_shared_worker_normalization(config, workers=11)
        updated = apply_memory_speed_profile(
            updated, memory_speed_profile="high_memory_local"
        )

        self.assertEqual(
            "11",
            updated["steps"][0]["command"][
                updated["steps"][0]["command"].index("--workers") + 1
            ],
        )
        self.assertEqual(
            "200000",
            updated["steps"][0]["command"][
                updated["steps"][0]["command"].index("--evaluator-cache-size") + 1
            ],
        )

    def test_apply_memory_speed_profile_rewrites_supported_generation_steps(self):
        config = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--checkpoint",
                        "parent.npz",
                    ],
                },
                {
                    "name": "mcts_bootstrap_dataset",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/generate_bootstrap_dataset.py",
                        "--position-selection-mode",
                        "hybrid_teacher",
                    ],
                },
                {
                    "name": "train",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/train.py",
                        "--epochs",
                        "2",
                    ],
                },
            ]
        }

        updated = apply_memory_speed_profile(
            config, memory_speed_profile="high_memory_local"
        )

        self.assertEqual(
            "200000",
            updated["steps"][0]["command"][
                updated["steps"][0]["command"].index("--evaluator-cache-size") + 1
            ],
        )
        self.assertIn("--teacher-search-reuse", updated["steps"][1]["command"])
        self.assertEqual(
            [".venv/bin/python", "ml/alphazero_lite/train.py", "--epochs", "2"],
            updated["steps"][2]["command"],
        )

    def test_apply_memory_speed_profile_supports_equals_form_flags(self):
        config = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--checkpoint=parent.npz",
                        "--evaluator-cache-size=50000",
                    ],
                },
                {
                    "name": "mcts_bootstrap_dataset",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/generate_bootstrap_dataset.py",
                        "--position-selection-mode=hybrid_teacher",
                        "--teacher-mode=puct",
                    ],
                },
            ]
        }

        updated = apply_memory_speed_profile(
            config, memory_speed_profile="high_memory_local"
        )

        self.assertIn("--evaluator-cache-size=200000", updated["steps"][0]["command"])
        self.assertIn("--teacher-search-reuse", updated["steps"][1]["command"])
