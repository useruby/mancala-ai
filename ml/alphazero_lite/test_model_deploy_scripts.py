import subprocess
import unittest
from pathlib import Path


class ModelDeployScriptsTest(unittest.TestCase):
    def test_kamal_deploy_config_uses_accessory_file_copy(self):
        repo_root = Path(__file__).resolve().parents[2]
        deploy_text = (repo_root / "config/deploy.yml").read_text(encoding="utf-8")

        self.assertIn("accessories:", deploy_text)
        self.assertIn("model_artifact:", deploy_text)
        self.assertIn("storage/ai/alphazero_lite/current/metadata.json:/artifacts/current/metadata.json:ro", deploy_text)
        self.assertIn("storage/ai/alphazero_lite/current/weights.json:/artifacts/current/weights.json:ro", deploy_text)
        self.assertIn("storage/ai/alphazero_lite/current/arena_report.json:/artifacts/current/arena_report.json:ro", deploy_text)
        self.assertNotIn("/artifacts/superhuman", deploy_text)
        self.assertIn("- src_storage:/storage", deploy_text)
        model_deploy_text = deploy_text.split("model_deploy: >-", 1)[1].split("model_rollback:", 1)[0]
        self.assertIn("accessory exec model_artifact", model_deploy_text)
        self.assertNotIn("accessory exec model_artifact --reuse", model_deploy_text)
        self.assertNotIn("accessory reboot model_artifact &&", model_deploy_text)
        self.assertIn("rm -rf /storage/ai/alphazero_lite/superhuman_current /storage/ai/alphazero_lite/superhuman_previous", model_deploy_text)
        self.assertIn("/storage/ai/alphazero_lite/current_previous", deploy_text)

    def test_kamal_staging_config_uses_staging_volume_for_accessory(self):
        repo_root = Path(__file__).resolve().parents[2]
        deploy_text = (repo_root / "config/deploy.staging.yml").read_text(encoding="utf-8")

        self.assertIn("accessories:", deploy_text)
        self.assertIn("model_artifact:", deploy_text)
        self.assertIn("- src_storage_staging:/storage", deploy_text)
        self.assertIn("storage/ai/alphazero_lite/current/metadata.json:/artifacts/current/metadata.json:ro", deploy_text)
        self.assertNotIn("/artifacts/superhuman", deploy_text)
        model_deploy_text = deploy_text.split("model_deploy: >-", 1)[1].split("model_rollback:", 1)[0]
        self.assertIn("rm -rf /storage/ai/alphazero_lite/superhuman_current /storage/ai/alphazero_lite/superhuman_previous", model_deploy_text)


if __name__ == "__main__":
    unittest.main()
