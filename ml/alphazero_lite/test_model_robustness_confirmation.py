import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class ModelRobustnessConfirmationTest(unittest.TestCase):
    def test_dry_run_includes_aggregate_summary_location(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-robustness-confirmation-") as tmp:
            output_root = Path(tmp)
            result = subprocess.run(
                [
                    "script/ai/model_robustness_confirmation",
                    "--dry-run",
                    "--output-root",
                    str(output_root),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        plan = json.loads(result.stdout)

        self.assertEqual(str(output_root / "aggregate_summary.json"), plan["aggregate_summary_path"])
        self.assertEqual(5, len(plan["lanes"]))


if __name__ == "__main__":
    unittest.main()
