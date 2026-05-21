import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import promotion_backlog


class PromotionBacklogManifestTest(unittest.TestCase):
    @staticmethod
    def worktree_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def test_initialize_manifest_records_candidate_and_defaults(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )

            self.assertEqual("azlite_hard_bot_backlog_v1", manifest["schema"])
            self.assertEqual("candidate-iter3", manifest["campaign_id"])
            self.assertEqual(
                str(candidate_path), manifest["candidate"]["artifact_path"]
            )
            self.assertEqual({}, manifest["artifacts"])
            self.assertEqual({}, manifest["steps"])
            self.assertEqual({"runs": []}, manifest["multi_seed_confirmation"])
            self.assertEqual("needs_runs", manifest["readiness"]["state"])

    def test_record_step_result_updates_manifest_fields(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )

            updated = promotion_backlog.record_step_result(
                manifest_path=manifest_path,
                step_name="promotion_gate",
                status="completed",
                command=[
                    "script/ai/local_promotion_gate",
                    "--candidate-path",
                    str(candidate_path),
                ],
                outputs={
                    "promotion_report": str(tmp_path / "local_promotion_gate.json")
                },
                failure_summary=None,
            )

            self.assertEqual("completed", updated["steps"]["promotion_gate"]["status"])
            self.assertEqual(
                [
                    "script/ai/local_promotion_gate",
                    "--candidate-path",
                    str(candidate_path),
                ],
                updated["steps"]["promotion_gate"]["command"],
            )
            self.assertEqual(
                str(tmp_path / "local_promotion_gate.json"),
                updated["steps"]["promotion_gate"]["outputs"]["promotion_report"],
            )

    def test_evaluate_readiness_returns_needs_runs_for_missing_reports(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )
            persisted = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual("needs_runs", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertCountEqual(
                promotion_backlog.required_artifact_keys(),
                readiness["missing_evidence"],
            )
            self.assertEqual(readiness, persisted["readiness"])

    def test_evaluate_readiness_returns_not_ready_when_forensic_gate_fails(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {
                        "forensic_quality": {
                            "passed": False,
                            "failure_reasons": [{"code": "policy_regressed"}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("not_ready", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertEqual([], readiness["missing_evidence"])
            self.assertIn("forensic_gate_failed", readiness["failure_reasons"])

    def test_evaluate_readiness_accepts_real_forensic_suite_report_when_challenger_beats_current(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {
                                "overall": {
                                    "positions": 16,
                                    "top1_agreement": 0.42,
                                    "average_regret": 0.11,
                                    "value_calibration_mae": 0.23,
                                }
                            },
                            "challenger": {
                                "overall": {
                                    "positions": 16,
                                    "top1_agreement": 0.48,
                                    "average_regret": 0.09,
                                    "value_calibration_mae": 0.2,
                                }
                            },
                        },
                        "buckets": {},
                    }
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("promotion_candidate", readiness["state"])
            self.assertTrue(readiness["passed"])
            self.assertEqual([], readiness["failure_reasons"])

    def test_evaluate_readiness_rejects_real_forensic_suite_report_when_challenger_regresses(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "systems": {
                            "current": {
                                "overall": {
                                    "positions": 16,
                                    "top1_agreement": 0.42,
                                    "average_regret": 0.11,
                                    "value_calibration_mae": 0.23,
                                }
                            },
                            "challenger": {
                                "overall": {
                                    "positions": 16,
                                    "top1_agreement": 0.39,
                                    "average_regret": 0.12,
                                    "value_calibration_mae": 0.25,
                                }
                            },
                        },
                        "buckets": {},
                    }
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("not_ready", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn("forensic_gate_failed", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_not_ready_when_runtime_failure_reason_present(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps(
                    {
                        "passed": False,
                        "failure_reasons": [
                            {"code": "arena_move_time_mean_above_threshold"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("not_ready", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertEqual([], readiness["missing_evidence"])
            self.assertIn("promotion_gate_failed", readiness["failure_reasons"])
            self.assertIn("runtime_failed", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_promotion_gate_step_incomplete(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "running"},
                "forensic_suite": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertEqual([], readiness["missing_evidence"])
            self.assertIn("promotion_gate_incomplete", readiness["failure_reasons"])
            self.assertNotIn("promotion_gate_failed", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_forensic_step_incomplete(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "running"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertEqual([], readiness["missing_evidence"])
            self.assertIn("forensic_gate_incomplete", readiness["failure_reasons"])
            self.assertNotIn("forensic_gate_failed", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_promotion_report_file_missing(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            missing_promotion_report = tmp_path / "missing_promotion_gate.json"
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(missing_promotion_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn("promotion_gate_report_missing", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_forensic_report_is_malformed(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text("{not valid json", encoding="utf-8")
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn("forensic_report_malformed", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_multi_seed_summary_disagrees_with_manifest(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": False}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn(
                "multi_seed_summary_inconsistent", readiness["failure_reasons"]
            )

    def test_evaluate_readiness_returns_needs_investigation_when_promotion_report_shape_is_empty_object(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(json.dumps({}), encoding="utf-8")
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn(
                "promotion_gate_report_unusable", readiness["failure_reasons"]
            )

    def test_evaluate_readiness_returns_needs_investigation_when_forensic_report_shape_is_empty_object(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(json.dumps({}), encoding="utf-8")
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn("forensic_report_unusable", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_forensic_quality_is_not_mapping(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps({"forensic_quality": None}), encoding="utf-8"
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn("forensic_report_unusable", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_needs_investigation_when_promotion_failure_reasons_shape_is_invalid(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": False, "failure_reasons": "oops"}),
                encoding="utf-8",
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn(
                "promotion_gate_report_unusable", readiness["failure_reasons"]
            )

    def test_evaluate_readiness_returns_needs_investigation_when_multi_seed_step_incomplete(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "running"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                    {"seed": 43, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertIn("multi_seed_incomplete", readiness["failure_reasons"])
            self.assertEqual(
                1, readiness["failure_reasons"].count("multi_seed_incomplete")
            )

    def test_evaluate_readiness_deduplicates_multi_seed_incomplete_reason(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "running"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [{"seed": 41, "passed": True}, {"seed": 42, "passed": True}],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertEqual(
                1, readiness["failure_reasons"].count("multi_seed_incomplete")
            )

    def test_evaluate_readiness_returns_needs_investigation_for_incomplete_seed_evidence(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [
                    {"seed": 41, "passed": True},
                    {"seed": 42, "passed": True},
                ],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual("needs_investigation", readiness["state"])
            self.assertFalse(readiness["passed"])
            self.assertEqual([], readiness["missing_evidence"])
            self.assertIn("multi_seed_incomplete", readiness["failure_reasons"])

    def test_evaluate_readiness_returns_promotion_candidate_when_all_gates_pass(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )
            persisted = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual("promotion_candidate", readiness["state"])
            self.assertTrue(readiness["passed"])
            self.assertEqual([], readiness["failure_reasons"])
            self.assertEqual([], readiness["missing_evidence"])
            self.assertEqual(readiness, persisted["readiness"])

    def test_report_script_prints_readiness_summary_json(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            manifest["artifacts"] = {
                "promotion_gate_report": str(promotion_gate_report),
                "forensic_report": str(forensic_report),
                "multi_seed_summary": str(multi_seed_summary),
            }
            manifest["steps"] = {
                "promotion_gate": {"status": "completed"},
                "forensic_suite": {"status": "completed"},
                "multi_seed_confirmation": {"status": "completed"},
            }
            manifest["multi_seed_confirmation"] = {
                "runs": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                "summary": {"status": "passed"},
            }
            promotion_backlog.write_manifest(manifest_path, manifest)

            self.assertTrue(sys.executable)
            script_path = (
                self.worktree_root() / "script/ai/report_hard_bot_promotion_readiness"
            )

            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("promotion_candidate", payload["state"])
            self.assertTrue(payload["passed"])
            self.assertEqual([], payload["failure_reasons"])
            self.assertEqual([], payload["missing_evidence"])

    def test_run_backlog_dry_run_initializes_manifest_and_emits_stage_command(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            config_path = tmp_path / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )

            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--config-path",
                    str(config_path),
                    "--parent-artifact-path",
                    str(parent_artifact_path),
                    "--stage",
                    "promotion_gate",
                    "--dry-run",
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual("promotion_gate", payload["stage"])
            self.assertEqual(str(manifest_path), payload["manifest_path"])
            self.assertEqual(
                promotion_backlog.stage_command(
                    stage="promotion_gate", manifest=manifest
                ),
                payload["command"],
            )
            self.assertEqual("candidate-iter3", manifest["campaign_id"])
            self.assertEqual(
                str(candidate_path), manifest["candidate"]["artifact_path"]
            )
            self.assertEqual(str(config_path), manifest["candidate"]["config_path"])
            self.assertEqual(
                str(parent_artifact_path),
                manifest["candidate"]["parent_artifact_path"],
            )

    def test_stage_command_forensic_suite_uses_canonical_forensic_report_naming(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=parent_artifact_path,
            )

            command = promotion_backlog.stage_command(
                stage="forensic_suite", manifest=manifest
            )

            self.assertEqual("ml/alphazero_lite/run_forensic_suite.py", command[1])
            self.assertIn("--out", command)
            self.assertEqual(
                str(tmp_path / "forensic_report.json"),
                command[command.index("--out") + 1],
            )

    def test_stage_command_forensic_suite_uses_shared_workspace_venv_in_worktree(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            workspace_root = tmp_path / "workspace"
            repo_root = workspace_root / ".worktrees" / "feature"
            repo_root.mkdir(parents=True)
            shared_python = workspace_root / ".venv/bin/python"
            shared_python.parent.mkdir(parents=True)
            shared_python.write_text("#!/bin/sh\n", encoding="utf-8")
            shared_python.chmod(shared_python.stat().st_mode | stat.S_IXUSR)

            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()
            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=parent_artifact_path,
            )

            with (
                mock.patch.object(
                    promotion_backlog, "repo_root", return_value=repo_root
                ),
                mock.patch.object(
                    promotion_backlog.sys, "executable", "/usr/bin/python3"
                ),
            ):
                command = promotion_backlog.stage_command(
                    stage="forensic_suite", manifest=manifest
                )

            self.assertEqual(str(shared_python), command[0])

    def test_stage_command_forensic_suite_falls_back_to_sys_executable_outside_worktree(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "sandbox" / "repo"
            repo_root.mkdir(parents=True)
            shared_python = tmp_path / ".venv/bin/python"
            shared_python.parent.mkdir(parents=True)
            shared_python.write_text("#!/bin/sh\n", encoding="utf-8")
            shared_python.chmod(shared_python.stat().st_mode | stat.S_IXUSR)

            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()
            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=parent_artifact_path,
            )

            with (
                mock.patch.object(
                    promotion_backlog, "repo_root", return_value=repo_root
                ),
                mock.patch.object(
                    promotion_backlog.sys, "executable", "/usr/bin/python3"
                ),
            ):
                command = promotion_backlog.stage_command(
                    stage="forensic_suite", manifest=manifest
                )

            self.assertEqual("/usr/bin/python3", command[0])

    def test_stage_command_forensic_suite_accepts_executable_without_owner_execute_bit(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "sandbox" / "repo"
            repo_root.mkdir(parents=True)
            repo_python = repo_root / ".venv/bin/python"
            repo_python.parent.mkdir(parents=True)
            repo_python.write_text("#!/bin/sh\n", encoding="utf-8")
            repo_python.chmod(stat.S_IRUSR | stat.S_IWUSR)

            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()
            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=parent_artifact_path,
            )

            with (
                mock.patch.object(
                    promotion_backlog, "repo_root", return_value=repo_root
                ),
                mock.patch.object(promotion_backlog.os, "access", return_value=True),
                mock.patch.object(
                    promotion_backlog.sys, "executable", "/usr/bin/python3"
                ),
            ):
                command = promotion_backlog.stage_command(
                    stage="forensic_suite", manifest=manifest
                )

            self.assertEqual(str(repo_python), command[0])

    def test_stage_command_multi_seed_confirmation_uses_local_robustness_confirmation(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            config_path = tmp_path / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()

            manifest = promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=config_path,
                parent_artifact_path=parent_artifact_path,
            )

            command = promotion_backlog.stage_command(
                stage="multi_seed_confirmation", manifest=manifest
            )

            self.assertEqual(promotion_backlog.python_executable(), command[0])
            self.assertEqual("script/ai/model_robustness_confirmation", command[1])
            self.assertEqual(
                str(config_path), command[command.index("--base-config") + 1]
            )
            self.assertEqual(
                str(parent_artifact_path),
                command[command.index("--parent-artifact") + 1],
            )
            self.assertEqual(
                str(candidate_path), command[command.index("--current-path") + 1]
            )
            self.assertEqual(
                str(tmp_path / "robustness_confirmation"),
                command[command.index("--output-root") + 1],
            )

    def test_run_backlog_existing_manifest_rejects_conflicting_candidate_inputs(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            config_path = tmp_path / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            parent_artifact_path = tmp_path / "parent"
            parent_artifact_path.mkdir()

            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=config_path,
                parent_artifact_path=parent_artifact_path,
            )

            conflicting_candidate_path = tmp_path / "different-candidate"
            conflicting_candidate_path.mkdir()
            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )

            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--candidate-path",
                    str(conflicting_candidate_path),
                    "--stage",
                    "promotion_gate",
                    "--dry-run",
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "conflict with existing manifest candidate.artifact_path", result.stderr
            )

    def test_run_backlog_existing_manifest_rejects_conflicting_campaign_id(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()

            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )

            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "different-campaign",
                    "--stage",
                    "promotion_gate",
                    "--dry-run",
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("conflict with existing manifest campaign_id", result.stderr)

    def test_run_backlog_existing_manifest_refreshes_embedded_manifest_path(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            original_manifest_path = tmp_path / "campaign-original.json"
            reused_manifest_path = tmp_path / "campaign-reused.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()

            promotion_backlog.initialize_manifest(
                manifest_path=original_manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            reused_manifest_path.write_text(
                original_manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(reused_manifest_path),
                    "--stage",
                    "readiness_report",
                    "--dry-run",
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            payload = json.loads(result.stdout)
            manifest = promotion_backlog.load_manifest(reused_manifest_path)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(str(reused_manifest_path), manifest["manifest_path"])
            self.assertEqual(str(reused_manifest_path), payload["manifest_path"])
            self.assertEqual(str(reused_manifest_path), payload["command"][-1])

    def test_run_backlog_executes_stage_and_records_completed_step(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            stub_report = tmp_path / "local_promotion_gate.json"
            stub_report.write_text(
                json.dumps({"passed": True, "candidate_path": str(candidate_path)}),
                encoding="utf-8",
            )

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "promotion_gate",
                    "--stub-promotion-report",
                    str(stub_report),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual("completed", manifest["steps"]["promotion_gate"]["status"])
            self.assertEqual(
                str(stub_report), manifest["artifacts"]["promotion_gate_report"]
            )

    def test_run_backlog_executes_forensic_stage_and_records_completed_step(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            stub_report = tmp_path / "candidate_forensic_suite.json"
            stub_report.write_text(
                json.dumps(
                    {
                        "schema": "azlite_forensic_suite_v1",
                        "forensic_quality": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "forensic_suite",
                    "--stub-forensic-report",
                    str(stub_report),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual("completed", manifest["steps"]["forensic_suite"]["status"])
            self.assertEqual(str(stub_report), manifest["artifacts"]["forensic_report"])

    def test_run_backlog_executes_multi_seed_stage_and_records_completed_step(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            stub_summary = tmp_path / "aggregate_summary.json"
            stub_summary.write_text(json.dumps({"passed": True}), encoding="utf-8")

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "multi_seed_confirmation",
                    "--stub-multi-seed-summary",
                    str(stub_summary),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(
                "completed", manifest["steps"]["multi_seed_confirmation"]["status"]
            )
            self.assertEqual(
                str(stub_summary), manifest["artifacts"]["multi_seed_summary"]
            )
            self.assertEqual(
                [41, 42, 43],
                [run["seed"] for run in manifest["multi_seed_confirmation"]["runs"]],
            )

    def test_run_backlog_rejects_stub_directory_paths(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            stub_dir = tmp_path / "stub-dir"
            stub_dir.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "forensic_suite",
                    "--stub-forensic-report",
                    str(stub_dir),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("stub artifact is not a file", result.stderr)
            self.assertNotIn("forensic_suite", manifest["steps"])
            self.assertEqual({}, manifest["artifacts"])

    def test_load_manifest_normalizes_pre_task5_artifact_keys(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            promotion_gate_report = tmp_path / "promotion_gate.json"
            forensic_report = tmp_path / "forensic_report.json"
            multi_seed_summary = tmp_path / "multi_seed_summary.json"
            promotion_gate_report.write_text(
                json.dumps({"passed": True, "failure_reasons": []}), encoding="utf-8"
            )
            forensic_report.write_text(
                json.dumps(
                    {"forensic_quality": {"passed": True, "failure_reasons": []}}
                ),
                encoding="utf-8",
            )
            multi_seed_summary.write_text(
                json.dumps({"passed": True}), encoding="utf-8"
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_hard_bot_backlog_v1",
                        "manifest_path": str(manifest_path),
                        "campaign_id": "candidate-iter3",
                        "candidate": {
                            "artifact_path": str(tmp_path / "candidate"),
                            "config_path": None,
                            "parent_artifact_path": None,
                        },
                        "artifacts": {
                            "promotion_gate": str(promotion_gate_report),
                            "forensic_report": str(forensic_report),
                            "multi_seed_summary": str(multi_seed_summary),
                        },
                        "steps": {
                            "promotion_gate": {"status": "completed"},
                            "forensic_suite": {"status": "completed"},
                            "multi_seed_confirmation": {"status": "completed"},
                        },
                        "multi_seed_confirmation": {
                            "runs": [{"seed": 1}, {"seed": 2}, {"seed": 3}],
                            "summary": {"status": "passed"},
                        },
                        "readiness": {
                            "state": "needs_runs",
                            "passed": False,
                            "failure_reasons": [],
                            "missing_evidence": [],
                            "stale_artifacts": [],
                            "last_evaluated_at": None,
                        },
                        "created_at": "2026-05-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            manifest = promotion_backlog.load_manifest(manifest_path)
            readiness = promotion_backlog.evaluate_readiness(
                manifest_path=manifest_path
            )

            self.assertEqual(
                str(tmp_path / "promotion_gate.json"),
                manifest["artifacts"]["promotion_gate_report"],
            )
            self.assertEqual("promotion_candidate", readiness["state"])

    def test_run_backlog_multi_seed_stub_records_failed_summary(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            stub_summary = tmp_path / "aggregate_summary.json"
            stub_summary.write_text(json.dumps({"passed": False}), encoding="utf-8")

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "multi_seed_confirmation",
                    "--stub-multi-seed-summary",
                    str(stub_summary),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(
                "failed", manifest["multi_seed_confirmation"]["summary"]["status"]
            )
            self.assertEqual(
                [False, False, False],
                [run["passed"] for run in manifest["multi_seed_confirmation"]["runs"]],
            )

    def test_run_backlog_missing_stub_report_fails_before_recording_completed_step(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            missing_report = tmp_path / "missing_promotion_gate.json"

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "promotion_gate",
                    "--stub-promotion-report",
                    str(missing_report),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("stub artifact does not exist", result.stderr)
            self.assertNotIn("promotion_gate", manifest["steps"])
            self.assertEqual({}, manifest["artifacts"])

    def test_run_backlog_multi_seed_stub_requires_boolean_passed(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )
            stub_summary = tmp_path / "aggregate_summary.json"
            stub_summary.write_text(json.dumps({"passed": "false"}), encoding="utf-8")

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--campaign-id",
                    "candidate-iter3",
                    "--candidate-path",
                    str(candidate_path),
                    "--stage",
                    "multi_seed_confirmation",
                    "--stub-multi-seed-summary",
                    str(stub_summary),
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            manifest = promotion_backlog.load_manifest(manifest_path)

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "stub multi-seed summary must contain boolean 'passed'", result.stderr
            )
            self.assertNotIn("multi_seed_confirmation", manifest["steps"])
            self.assertEqual({}, manifest["artifacts"])
            self.assertEqual({"runs": []}, manifest["multi_seed_confirmation"])

    def test_run_backlog_unsupported_non_dry_run_stage_reports_supported_modes(self):
        with tempfile.TemporaryDirectory(prefix="azlite-backlog-") as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "campaign.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()
            promotion_backlog.initialize_manifest(
                manifest_path=manifest_path,
                campaign_id="candidate-iter3",
                candidate_path=candidate_path,
                config_path=None,
                parent_artifact_path=None,
            )

            script_path = (
                self.worktree_root() / "script/ai/run_hard_bot_promotion_backlog"
            )
            result = subprocess.run(
                [
                    str(script_path),
                    "--manifest-path",
                    str(manifest_path),
                    "--stage",
                    "readiness_report",
                ],
                cwd=self.worktree_root(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "non-dry-run execution for stage 'readiness_report' is not supported",
                result.stderr,
            )
            self.assertIn("Supported modes are: --dry-run", result.stderr)


if __name__ == "__main__":
    unittest.main()
