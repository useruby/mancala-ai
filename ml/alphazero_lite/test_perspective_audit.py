import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PerspectiveAuditTest(unittest.TestCase):
    def test_audit_passes_for_consistent_rows(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play.jsonl"
            out_path = tmp_path / "audit.json"

            row = {
                "state": [
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    4 / 48,
                    0.0,
                    0.0,
                    0.0,
                ],
                "policy": [1 / 6] * 6,
                "value": 0.0,
                "player": 0,
                "winner": None,
            }
            data_path.write_text("\n".join([json.dumps(row) for _ in range(8)]) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])

    def test_audit_fails_for_inconsistent_value_label(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play_bad.jsonl"
            out_path = tmp_path / "audit_bad.json"

            bad_row = {
                "state": [0.0] * 14 + [1.0],
                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "value": 1.0,
                "player": 1,
                "winner": 0,
            }
            data_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(audit["passed"])

    def test_audit_handles_legal_policy_for_current_player_one(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play_p1.jsonl"
            out_path = tmp_path / "audit_p1.json"

            # current_player=1 means legal moves come from second pit block.
            row = {
                "state": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1 / 48,
                    0.0,
                    1 / 48,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "policy": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                "value": 1.0,
                "player": 1,
                "winner": 1,
            }
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])

    def test_audit_accepts_sharpened_value_targets_when_perspective_sign_matches(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play_sharpened.jsonl"
            out_path = tmp_path / "audit_sharpened.json"

            rows = [
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 0.0],
                    "policy": [1 / 6] * 6,
                    "value": 0.64,
                    "player": 0,
                    "winner": 0,
                    "value_target_mode": "sharpened",
                },
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 1.0],
                    "policy": [1 / 6] * 6,
                    "value": -0.64,
                    "player": 1,
                    "winner": 0,
                    "value_target_mode": "sharpened",
                },
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])

    def test_audit_accepts_aligned_bootstrap_like_sharpened_rows_for_sign_and_zero_behavior(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "bootstrap_like_sharpened.jsonl"
            out_path = tmp_path / "audit_bootstrap_like_sharpened.json"

            rows = [
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 0.0],
                    "policy": [1 / 6] * 6,
                    "value": 0.27,
                    "player": 0,
                    "winner": 0,
                    "source": "bootstrap",
                    "move_index": 11,
                    "value_target_mode": "sharpened",
                },
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 1.0],
                    "policy": [1 / 6] * 6,
                    "value": 0.0,
                    "player": 1,
                    "winner": None,
                    "source": "bootstrap",
                    "move_index": 12,
                    "value_target_mode": "sharpened",
                },
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])

    def test_audit_clearly_rejects_aligned_bootstrap_like_signed_row_without_winner(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "bootstrap_like_missing_winner.jsonl"
            out_path = tmp_path / "audit_bootstrap_like_missing_winner.json"

            row = {
                "state": [4 / 48] * 12 + [0.0, 0.0, 0.0],
                "policy": [1 / 6] * 6,
                "value": 0.27,
                "player": 0,
                "winner": None,
                "source": "bootstrap",
                "move_index": 11,
                "value_target_mode": "sharpened",
            }
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(audit["passed"])
            self.assertEqual("signed_value_requires_winner", audit["errors"][0]["code"])

    def test_audit_accepts_phase_aware_value_targets_when_perspective_sign_matches(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play_phase_aware.jsonl"
            out_path = tmp_path / "audit_phase_aware.json"

            rows = [
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 0.0],
                    "policy": [1 / 6] * 6,
                    "value": 0.78,
                    "player": 0,
                    "winner": 0,
                    "value_target_mode": "phase_aware_sharpened",
                },
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 1.0],
                    "policy": [1 / 6] * 6,
                    "value": -0.91,
                    "player": 1,
                    "winner": 0,
                    "value_target_mode": "phase_aware_sharpened",
                },
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])

    def test_audit_accepts_hybrid_value_targets_when_values_stay_bounded_and_perspective_matches(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play_hybrid.jsonl"
            out_path = tmp_path / "audit_hybrid.json"

            rows = [
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 0.0],
                    "policy": [1 / 6] * 6,
                    "value": 0.25,
                    "player": 0,
                    "winner": 0,
                    "value_target_mode": "hybrid",
                },
                {
                    "state": [4 / 48] * 12 + [0.0, 0.0, 1.0],
                    "policy": [1 / 6] * 6,
                    "value": -0.25,
                    "player": 1,
                    "winner": 0,
                    "value_target_mode": "hybrid",
                },
            ]
            data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"])

    def test_audit_rejects_hybrid_value_targets_with_wrong_perspective_sign(self):
        with tempfile.TemporaryDirectory(prefix="azlite-audit-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "self_play_hybrid_wrong_sign.jsonl"
            out_path = tmp_path / "audit_hybrid_wrong_sign.json"

            row = {
                "state": [4 / 48] * 12 + [0.0, 0.0, 0.0],
                "policy": [1 / 6] * 6,
                "value": -0.25,
                "player": 0,
                "winner": 0,
                "value_target_mode": "hybrid",
            }
            data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/perspective_audit.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            audit = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertFalse(audit["passed"])
            self.assertEqual("value_perspective", audit["errors"][0]["code"])


if __name__ == "__main__":
    unittest.main()
