import unittest
from pathlib import Path


class ModelDeployScriptsTest(unittest.TestCase):
    def test_kamal_deploy_config_uses_plain_app_deploy_for_model_updates(self):
        repo_root = Path(__file__).resolve().parents[2]
        deploy_text = (repo_root / "config/deploy.yml").read_text(encoding="utf-8")

        self.assertNotIn("model_deploy:", deploy_text)
        self.assertNotIn("model_rollback:", deploy_text)
        self.assertNotIn("model_artifact:", deploy_text)
        self.assertNotIn("MODEL_ARTIFACT_TAG", deploy_text)

    def test_kamal_staging_config_uses_plain_app_deploy_for_model_updates(self):
        repo_root = Path(__file__).resolve().parents[2]
        deploy_text = (repo_root / "config/deploy.staging.yml").read_text(encoding="utf-8")

        self.assertNotIn("model_deploy:", deploy_text)
        self.assertNotIn("model_rollback:", deploy_text)
        self.assertNotIn("model_artifact:", deploy_text)
        self.assertNotIn("MODEL_ARTIFACT_TAG", deploy_text)

    def test_model_artifact_files_are_checked_in(self):
        repo_root = Path(__file__).resolve().parents[2]
        self.assertTrue((repo_root / "model-artifact/current/metadata.json").exists())
        self.assertTrue((repo_root / "model-artifact/current/weights.json").exists())
        self.assertTrue((repo_root / "model-artifact/current/arena_report.json").exists())


if __name__ == "__main__":
    unittest.main()
