import subprocess
import unittest
from pathlib import Path


class ModelDeployScriptsTest(unittest.TestCase):
    def test_kamal_deploy_config_uses_accessory_artifact_image(self):
        repo_root = Path(__file__).resolve().parents[2]
        deploy_text = (repo_root / "config/deploy.yml").read_text(encoding="utf-8")

        self.assertIn("accessories:", deploy_text)
        self.assertIn("model_artifact:", deploy_text)
        self.assertIn("ghcr.io/useruby/mancala-model-artifact:", deploy_text)
        self.assertIn("MODEL_ARTIFACT_TAG", deploy_text)
        self.assertNotIn("/artifacts/current/metadata.json:ro", deploy_text)
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
        self.assertIn("ghcr.io/useruby/mancala-model-artifact:", deploy_text)
        self.assertIn("MODEL_ARTIFACT_TAG", deploy_text)
        self.assertNotIn("/artifacts/current/metadata.json:ro", deploy_text)
        self.assertNotIn("/artifacts/superhuman", deploy_text)
        model_deploy_text = deploy_text.split("model_deploy: >-", 1)[1].split("model_rollback:", 1)[0]
        self.assertIn("rm -rf /storage/ai/alphazero_lite/superhuman_current /storage/ai/alphazero_lite/superhuman_previous", model_deploy_text)

    def test_model_artifact_build_script_publishes_versioned_and_latest_tags(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_text = (repo_root / "script/ai/build_model_artifact_image").read_text(encoding="utf-8")

        self.assertIn("MODEL_ARTIFACT_REPO", script_text)
        self.assertIn("MODEL_ARTIFACT_TAG", script_text)
        self.assertIn("docker build -t", script_text)
        self.assertIn("docker push \"$IMAGE_VERSION\"", script_text)
        self.assertIn("docker push \"$IMAGE_LATEST\"", script_text)


if __name__ == "__main__":
    unittest.main()
