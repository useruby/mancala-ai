import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.input_encodings import (
    BASE_FEATURE_ORDER,
    KALAH_V3_EXTRA_FEATURE_ORDER,
    feature_count_for,
)


class ExportArtifactScriptTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def write_synthetic_residual_v3_checkpoint(
        self, path: Path, *, include_value_head: bool = True
    ) -> dict[str, np.ndarray]:
        feature_count = feature_count_for("kalah_v3")
        checkpoint = {
            "w_input": np.arange(feature_count * 4, dtype=np.float32).reshape(
                feature_count, 4
            ),
            "b_input": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "w_residual_1_1": np.arange(16, dtype=np.float32).reshape(4, 4) + 10.0,
            "b_residual_1_1": np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
            "w_residual_1_2": np.arange(16, dtype=np.float32).reshape(4, 4) + 20.0,
            "b_residual_1_2": np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
            "w_residual_2_1": np.arange(16, dtype=np.float32).reshape(4, 4) + 30.0,
            "b_residual_2_1": np.array([3.0, 3.1, 3.2, 3.3], dtype=np.float32),
            "w_residual_2_2": np.arange(16, dtype=np.float32).reshape(4, 4) + 40.0,
            "b_residual_2_2": np.array([4.0, 4.1, 4.2, 4.3], dtype=np.float32),
            "w_policy_hidden": np.arange(20, dtype=np.float32).reshape(4, 5) + 50.0,
            "b_policy_hidden": np.array([5.0, 5.1, 5.2, 5.3, 5.4], dtype=np.float32),
            "w_policy": np.arange(30, dtype=np.float32).reshape(5, 6) + 60.0,
            "b_policy": np.array([6.0, 6.1, 6.2, 6.3, 6.4, 6.5], dtype=np.float32),
            "w_value": np.arange(3, dtype=np.float32).reshape(3, 1) + 70.0,
            "b_value": np.array([7.0], dtype=np.float32),
        }
        if include_value_head:
            checkpoint["w_value_hidden"] = (
                np.arange(12, dtype=np.float32).reshape(4, 3) + 80.0
            )
            checkpoint["b_value_hidden"] = np.array([8.0, 8.1, 8.2], dtype=np.float32)

        np.savez(path, **checkpoint)
        return checkpoint

    def write_checkpoint_with_stray_specialized_heads(self, path: Path) -> None:
        np.savez(
            path,
            w_input=np.zeros((15, 4), dtype=np.float32),
            b_input=np.zeros((4,), dtype=np.float32),
            w_residual_1_1=np.zeros((4, 4), dtype=np.float32),
            b_residual_1_1=np.zeros((4,), dtype=np.float32),
            w_residual_1_2=np.zeros((4, 4), dtype=np.float32),
            b_residual_1_2=np.zeros((4,), dtype=np.float32),
            w_policy_hidden=np.zeros((4, 5), dtype=np.float32),
            b_policy_hidden=np.zeros((5,), dtype=np.float32),
            w_value_hidden=np.zeros((4, 3), dtype=np.float32),
            b_value_hidden=np.zeros((3,), dtype=np.float32),
            w_policy=np.zeros((4, 6), dtype=np.float32),
            b_policy=np.zeros((6,), dtype=np.float32),
            w_value=np.zeros((4, 1), dtype=np.float32),
            b_value=np.zeros((1,), dtype=np.float32),
        )

    def test_export_includes_rules_version_and_model_type(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text(
                "\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8"
            )

            train = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--model-type",
                    "mlp_deep",
                    "--hidden-sizes",
                    "128,128,64",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export",
                    "--model-type",
                    "mlp_deep",
                    "--rules-version",
                    "kalah_v1",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, export.returncode, msg=export.stderr)

            metadata = json.loads(
                (out_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("kalah_v1", metadata["rules_version"])
            self.assertEqual("mlp_deep", metadata["architecture"]["model_type"])
            self.assertEqual([128, 128, 64], metadata["architecture"]["hidden_sizes"])

    def test_export_includes_residual_v2_architecture_and_input_encoding(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text(
                "\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8"
            )

            train = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--model-type",
                    "residual_v2",
                    "--hidden-sizes",
                    "64,2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v2",
                    "--model-type",
                    "residual_v2",
                    "--rules-version",
                    "kalah_v1",
                    "--input-encoding",
                    "kalah_v2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, export.returncode, msg=export.stderr)

            metadata = json.loads(
                (out_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("kalah_v2", metadata["input_encoding"])
            self.assertEqual("residual_v2", metadata["architecture"]["model_type"])
            self.assertEqual("residual_policy_value", metadata["architecture"]["type"])
            self.assertEqual(64, metadata["architecture"]["trunk_size"])
            self.assertEqual(2, metadata["architecture"]["residual_block_count"])

    def test_export_residual_v3_artifact_preserves_specialized_head_contract(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            expected_weights = self.write_synthetic_residual_v3_checkpoint(
                checkpoint_path
            )

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v3-model-type",
                    "--model-type",
                    "residual_v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, export.returncode, msg=export.stderr)

            metadata = json.loads(
                (out_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("residual_v3", metadata["architecture"]["model_type"])
            self.assertEqual("residual_policy_value", metadata["architecture"]["type"])
            self.assertEqual(4, metadata["architecture"]["trunk_size"])
            self.assertEqual(2, metadata["architecture"]["residual_block_count"])
            self.assertEqual(7, metadata["architecture"]["hidden_layer_count"])
            self.assertEqual(5, metadata["architecture"]["policy_hidden_size"])
            self.assertEqual(3, metadata["architecture"]["value_hidden_size"])

            weights = json.loads((out_dir / "weights.json").read_text(encoding="utf-8"))
            self.assertEqual(
                expected_weights["w_policy_hidden"].tolist(), weights["w_policy_hidden"]
            )
            self.assertEqual(
                expected_weights["b_policy_hidden"].tolist(), weights["b_policy_hidden"]
            )
            self.assertEqual(
                expected_weights["w_value_hidden"].tolist(), weights["w_value_hidden"]
            )
            self.assertEqual(
                expected_weights["b_value_hidden"].tolist(), weights["b_value_hidden"]
            )
            self.assertEqual(expected_weights["w_policy"].tolist(), weights["w_policy"])
            self.assertEqual(expected_weights["w_value"].tolist(), weights["w_value"])

    def test_export_rejects_residual_v3_checkpoint_without_both_specialized_heads(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            self.write_synthetic_residual_v3_checkpoint(
                checkpoint_path, include_value_head=False
            )

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v3-missing-head",
                    "--model-type",
                    "residual_v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, export.returncode)
            self.assertIn("residual_v3", export.stderr)
            self.assertIn("value", export.stderr)

    def test_export_rejects_residual_v3_checkpoint_with_shape_mismatched_specialized_head(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            checkpoint = self.write_synthetic_residual_v3_checkpoint(checkpoint_path)
            checkpoint["w_value_hidden"] = np.zeros((4, 4), dtype=np.float32)
            np.savez(checkpoint_path, **checkpoint)

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v3-bad-shape",
                    "--model-type",
                    "residual_v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, export.returncode)
            self.assertIn("w_value_hidden", export.stderr)

    def test_export_rejects_residual_v2_checkpoint_with_specialized_heads(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            self.write_checkpoint_with_stray_specialized_heads(checkpoint_path)

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v2-stray-heads",
                    "--model-type",
                    "residual_v2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, export.returncode)
            self.assertIn("residual_v2", export.stderr)
            self.assertIn("specialized", export.stderr)

    def test_export_rejects_unknown_model_type(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            np.savez(
                checkpoint_path,
                w1=np.zeros((15, 64), dtype=np.float32),
                b1=np.zeros((64,), dtype=np.float32),
                w2=np.zeros((64, 64), dtype=np.float32),
                b2=np.zeros((64,), dtype=np.float32),
                w_policy=np.zeros((64, 6), dtype=np.float32),
                b_policy=np.zeros((6,), dtype=np.float32),
                w_value=np.zeros((64, 1), dtype=np.float32),
                b_value=np.zeros((1,), dtype=np.float32),
            )

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-invalid-model-type",
                    "--model-type",
                    "unknown_v9",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, export.returncode)
            self.assertIn("model-type", export.stderr)

    def test_export_rejects_checkpoint_when_kalah_v3_feature_count_does_not_match(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text(
                "\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8"
            )

            train = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(0, export.returncode)
            self.assertIn("feature_count", export.stderr)

    def test_export_accepts_kalah_v3_checkpoint_with_placeholder_feature_width(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            feature_count = feature_count_for("kalah_v3")
            np.savez(
                checkpoint_path,
                w1=np.zeros((feature_count, 64), dtype=np.float32),
                b1=np.zeros((64,), dtype=np.float32),
                w2=np.zeros((64, 64), dtype=np.float32),
                b2=np.zeros((64,), dtype=np.float32),
                w_policy=np.zeros((64, 6), dtype=np.float32),
                b_policy=np.zeros((6,), dtype=np.float32),
                w_value=np.zeros((64, 1), dtype=np.float32),
                b_value=np.zeros((1,), dtype=np.float32),
            )

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, export.returncode, msg=export.stderr)

            metadata = json.loads(
                (out_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("kalah_v3", metadata["input_encoding"])
            self.assertEqual(feature_count_for("kalah_v3"), metadata["feature_count"])
            self.assertGreater(metadata["feature_count"], 15)
            self.assertEqual(metadata["feature_count"], len(metadata["feature_order"]))

    def test_export_kalah_v3_metadata_uses_real_tactical_feature_names(self):
        with tempfile.TemporaryDirectory(prefix="azlite-export-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"
            out_dir = tmp_path / "artifact"

            feature_count = feature_count_for("kalah_v3")
            np.savez(
                checkpoint_path,
                w1=np.zeros((feature_count, 64), dtype=np.float32),
                b1=np.zeros((64,), dtype=np.float32),
                w2=np.zeros((64, 64), dtype=np.float32),
                b2=np.zeros((64,), dtype=np.float32),
                w_policy=np.zeros((64, 6), dtype=np.float32),
                b_policy=np.zeros((6,), dtype=np.float32),
                w_value=np.zeros((64, 1), dtype=np.float32),
                b_value=np.zeros((1,), dtype=np.float32),
            )

            export = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/export_artifact.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--out-dir",
                    str(out_dir),
                    "--version",
                    "azlite-test-export-v3",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, export.returncode, msg=export.stderr)

            metadata = json.loads(
                (out_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [*BASE_FEATURE_ORDER, *KALAH_V3_EXTRA_FEATURE_ORDER],
                metadata["feature_order"],
            )
            self.assertNotIn("reserved_v3_feature_0", metadata["feature_order"])


if __name__ == "__main__":
    unittest.main()
