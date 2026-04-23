import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class CheckSuperhumanRegressionsScriptTest(unittest.TestCase):
    def test_stub_mode_skips_rails_boot(self):
        with tempfile.TemporaryDirectory(prefix="azlite-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            result = subprocess.run(
                ["ruby", "script/ai/check_superhuman_regressions", "--out", str(out_path)],
                cwd=Path(__file__).resolve().parents[2],
                env={
                    **os.environ,
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_STUB": "1",
                    "BUNDLE_GEMFILE": str(tmp_path / "missing" / "Gemfile"),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual([], report["results"])


if __name__ == "__main__":
    unittest.main()
