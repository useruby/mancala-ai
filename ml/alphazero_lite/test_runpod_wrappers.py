import json
import subprocess
import unittest
from pathlib import Path


class RunpodWrappersTest(unittest.TestCase):
    def test_ultra_wrapper_scripts_are_removed(self):
        repo_root = Path(__file__).resolve().parents[2]

        self.assertFalse((repo_root / "script/ai/runpod_ultra_experiment").exists())
        self.assertFalse((repo_root / "script/ai/promote_ultra_candidate").exists())

    def test_runpod_superhuman_experiment_uses_lossless_superhuman_gate(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_superhuman_experiment",
                "--config-path",
                "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)
        command = plan["command"]

        self.assertIn("--current-path model-artifact/current", command)
        self.assertIn("--arena-games 400", command)
        self.assertIn("--min-arena-games 400", command)
        self.assertIn("--min-arena-score 0.55", command)
        self.assertIn("--require-lossless", command)
        self.assertIn("--max-losses 0", command)
        self.assertIn("--skip-mcts-relative-check", command)

    def test_promote_superhuman_candidate_requires_checkpoint_arg(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            ["script/ai/promote_superhuman_candidate"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("Usage: script/ai/promote_superhuman_candidate CANDIDATE_DIR", result.stderr)


if __name__ == "__main__":
    unittest.main()


class RunpodTrainingExperimentValidationTest(unittest.TestCase):
    def test_rejects_negative_promotion_max_losses(self):
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [
                "script/ai/runpod_training_experiment",
                "--config-path",
                "ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json",
                "--promotion-max-losses",
                "-1",
                "--dry-run",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("must be a non-negative integer", result.stderr)
