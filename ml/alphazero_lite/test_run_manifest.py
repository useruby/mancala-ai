import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.run_manifest import build_manifest, write_manifest


class RunManifestTest(unittest.TestCase):
    def test_build_and_write_manifest(self):
        manifest = build_manifest(
            run_id="aggressive-v1",
            iteration=1,
            seed=42,
            config_path="ml/alphazero_lite/configs/aggressive_v1.yaml",
            parent_version="azlite-current",
            status="planned",
            notes={"phase": "phase_1_scaffold"},
        )

        self.assertEqual("azlite_run_manifest_v1", manifest["schema"])
        self.assertEqual("aggressive-v1", manifest["run_id"])
        self.assertEqual(1, manifest["iteration"])
        self.assertEqual("planned", manifest["status"])
        self.assertEqual("phase_1_scaffold", manifest["notes"]["phase"])

        with tempfile.TemporaryDirectory(prefix="azlite-manifest-") as tmp:
            out_path = Path(tmp) / "run_manifest.json"
            write_manifest(out_path, manifest)

            self.assertTrue(out_path.exists())
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest, loaded)


if __name__ == "__main__":
    unittest.main()
