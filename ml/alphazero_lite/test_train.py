import json
import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.alphazero_lite import self_play as self_play_module
from ml.alphazero_lite import train as train_module


class TrainScriptTest(unittest.TestCase):
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

    def test_argument_parser_accepts_hybrid_value_target_mode(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            ["--out", "checkpoint.npz", "--value-target-mode", "hybrid"]
        )

        self.assertEqual("hybrid", args.value_target_mode)

    def test_argument_parser_accepts_phase_aware_value_target_mode(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            ["--out", "checkpoint.npz", "--value-target-mode", "phase_aware_sharpened"]
        )

        self.assertEqual("phase_aware_sharpened", args.value_target_mode)

    def test_argument_parser_accepts_sharpened_value_target_mode(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            ["--out", "checkpoint.npz", "--value-target-mode", "sharpened"]
        )

        self.assertEqual("sharpened", args.value_target_mode)

    def test_argument_parser_accepts_sharpened_policy_target_mode(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            ["--out", "checkpoint.npz", "--policy-target-mode", "sharpened"]
        )

        self.assertEqual("sharpened", args.policy_target_mode)

    def test_input_encodings_module_exposes_shared_contract(self):
        input_encodings = importlib.import_module("ml.alphazero_lite.input_encodings")

        self.assertEqual("kalah_v1", input_encodings.DEFAULT_INPUT_ENCODING)
        self.assertEqual(15, input_encodings.feature_count_for("kalah_v2"))
        self.assertEqual(27, input_encodings.feature_count_for("kalah_v3"))

    def test_validate_input_features_accepts_supported_selected_encoding(self):
        features = np.ones((4, 15), dtype=np.float32)

        train_module.validate_input_features(features, input_encoding="kalah_v2")

    def test_validate_input_features_accepts_kalah_v3_feature_width(self):
        features = np.ones((4, 27), dtype=np.float32)

        train_module.validate_input_features(features, input_encoding="kalah_v3")

    def test_cli_reports_selected_encoding_when_training_succeeds(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=64)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--input-encoding",
                    "kalah_v2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("input_encoding=kalah_v2", result.stdout)

    def test_cli_reports_selected_policy_target_mode_when_training_succeeds(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=64, policy_target_mode="sharpened")

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--policy-target-mode",
                    "sharpened",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("policy_target_mode=sharpened", result.stdout)

    def test_cli_reports_selected_value_target_mode_when_training_succeeds(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=64, value_target_mode="sharpened")

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--value-target-mode",
                    "sharpened",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("value_target_mode=sharpened", result.stdout)

    def test_load_jsonl_accepts_correctly_tagged_sharpened_value_target_rows(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "sharpened-value-target.jsonl"

            self._write_dataset(data_path, rows=3, value_target_mode="sharpened")

            x, p_target, v_target = train_module.load_jsonl(
                data_path, value_target_mode="sharpened"
            )

            self.assertEqual((3, 15), x.shape)
            self.assertEqual((3, 6), p_target.shape)
            self.assertEqual((3, 1), v_target.shape)

    def test_load_jsonl_accepts_aligned_bootstrap_row_with_real_search_value_and_strict_metadata(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "aligned-bootstrap-value-target.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 1, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 22,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
                        "value": 0.27,
                        "source": "bootstrap",
                        "player": 0,
                        "winner": 0,
                        "move_index": 11,
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                    }
                ],
            )

            x, p_target, v_target = train_module.load_jsonl(
                data_path,
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
            )

            self.assertEqual((1, 15), x.shape)
            self.assertEqual((1, 6), p_target.shape)
            self.assertEqual((1, 1), v_target.shape)
            np.testing.assert_allclose(
                p_target[0],
                np.array([0.3, 0.0, 0.0, 0.0, 0.7, 0.0], dtype=np.float32),
                atol=1e-6,
            )
            self.assertAlmostEqual(0.27, float(v_target[0, 0]), places=6)

    def test_load_jsonl_accepts_correctly_tagged_phase_aware_value_target_rows(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "phase-aware-value-target.jsonl"

            self._write_dataset(
                data_path, rows=3, value_target_mode="phase_aware_sharpened"
            )

            x, p_target, v_target = train_module.load_jsonl(
                data_path, value_target_mode="phase_aware_sharpened"
            )

            self.assertEqual((3, 15), x.shape)
            self.assertEqual((3, 6), p_target.shape)
            self.assertEqual((3, 1), v_target.shape)

    def test_load_jsonl_accepts_correctly_tagged_hybrid_value_target_rows(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "hybrid-value-target.jsonl"

            self._write_dataset(data_path, rows=3, value_target_mode="hybrid")

            x, p_target, v_target = train_module.load_jsonl(
                data_path, value_target_mode="hybrid"
            )

            self.assertEqual((3, 15), x.shape)
            self.assertEqual((3, 6), p_target.shape)
            self.assertEqual((3, 1), v_target.shape)

    def test_cli_reports_selected_encoding_for_feature_width_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [0.1] * 14,
                        "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                        "value": 1.0,
                    }
                ],
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--input-encoding",
                    "kalah_v2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("feature_count must be 15 for kalah_v2", result.stderr)

    def test_cli_trains_with_epochs_and_batch_size_and_writes_contract_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=128)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "2",
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

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())

            checkpoint = np.load(out_path)
            self.assertEqual((15, 64), checkpoint["w1"].shape)
            self.assertEqual((64,), checkpoint["b1"].shape)
            self.assertEqual((64, 64), checkpoint["w2"].shape)
            self.assertEqual((64,), checkpoint["b2"].shape)
            self.assertEqual((64, 6), checkpoint["w_policy"].shape)
            self.assertEqual((6,), checkpoint["b_policy"].shape)
            self.assertEqual((64, 1), checkpoint["w_value"].shape)
            self.assertEqual((1,), checkpoint["b_value"].shape)

    def _write_dataset(
        self,
        path: Path,
        rows: int,
        policy_target_mode: str | None = None,
        value_target_mode: str | None = None,
    ):
        with path.open("w", encoding="utf-8") as handle:
            for _ in range(rows):
                payload = {
                    "state": [0.1] * 15,
                    "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                    "value": 1.0,
                }
                if policy_target_mode is not None:
                    payload["policy_target_mode"] = policy_target_mode
                if value_target_mode is not None:
                    payload["value_target_mode"] = value_target_mode
                handle.write(json.dumps(payload) + "\n")

    def _write_rows(self, path: Path, rows: list[dict[str, object]]):
        with path.open("w", encoding="utf-8") as handle:
            for payload in rows:
                handle.write(json.dumps(payload) + "\n")

    def _generate_self_play_dataset(
        self,
        path: Path,
        *,
        policy_target_mode: str,
        value_target_mode: str = "default",
        games: int,
        simulations: int,
    ):
        result = subprocess.run(
            [
                self.executable_python(),
                "ml/alphazero_lite/self_play.py",
                "--out",
                str(path),
                "--games",
                str(games),
                "--seed",
                "17",
                "--simulations",
                str(simulations),
                "--workers",
                "1",
                "--temperature-threshold",
                "4",
                "--policy-target-mode",
                policy_target_mode,
                "--value-target-mode",
                value_target_mode,
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        self.assertTrue(path.exists())

    def test_cli_accepts_tuning_flags_and_reports_validation(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=256)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "64",
                    "--device",
                    "cpu",
                    "--value-loss-weight",
                    "0.5",
                    "--val-split",
                    "0.2",
                    "--grad-clip",
                    "1.0",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("best_val_loss=", result.stdout)
            self.assertTrue(out_path.exists())

    def test_cli_supports_wider_model_huber_and_topk_checkpoints(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"
            topk_dir = tmp_path / "topk"

            self._write_dataset(data_path, rows=512)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "3",
                    "--batch-size",
                    "64",
                    "--device",
                    "cpu",
                    "--hidden-sizes",
                    "128,128",
                    "--value-loss",
                    "huber",
                    "--save-top-k",
                    "2",
                    "--top-k-dir",
                    str(topk_dir),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())
            self.assertIn("saved_top_k=2", result.stdout)

            checkpoint = np.load(out_path)
            self.assertEqual((15, 128), checkpoint["w1"].shape)
            self.assertEqual((128,), checkpoint["b1"].shape)
            self.assertEqual((128, 128), checkpoint["w2"].shape)
            self.assertEqual((128,), checkpoint["b2"].shape)
            self.assertEqual((128, 6), checkpoint["w_policy"].shape)
            self.assertEqual((6,), checkpoint["b_policy"].shape)
            self.assertEqual((128, 1), checkpoint["w_value"].shape)
            self.assertEqual((1,), checkpoint["b_value"].shape)

            top1 = topk_dir / "checkpoint.top1.npz"
            top2 = topk_dir / "checkpoint.top2.npz"
            self.assertTrue(top1.exists())
            self.assertTrue(top2.exists())

    def test_cli_supports_deep_model_type_and_hidden_layer_export_keys(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=512)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "64",
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

            self.assertEqual(0, result.returncode, msg=result.stderr)
            checkpoint = np.load(out_path)

            self.assertIn("w_hidden_1", checkpoint.files)
            self.assertIn("b_hidden_1", checkpoint.files)
            self.assertIn("w_hidden_2", checkpoint.files)
            self.assertIn("b_hidden_2", checkpoint.files)
            self.assertIn("w_hidden_3", checkpoint.files)
            self.assertIn("b_hidden_3", checkpoint.files)
            self.assertIn("w_policy", checkpoint.files)
            self.assertIn("b_policy", checkpoint.files)
            self.assertIn("w_value", checkpoint.files)
            self.assertIn("b_value", checkpoint.files)

    def test_cli_supports_residual_v2_model_type_and_checkpoint_keys(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=256)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "64",
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

            self.assertEqual(0, result.returncode, msg=result.stderr)
            checkpoint = np.load(out_path)

            self.assertIn("w_input", checkpoint.files)
            self.assertIn("b_input", checkpoint.files)
            self.assertIn("w_residual_1_1", checkpoint.files)
            self.assertIn("b_residual_1_1", checkpoint.files)
            self.assertIn("w_residual_1_2", checkpoint.files)
            self.assertIn("b_residual_1_2", checkpoint.files)
            self.assertIn("w_residual_2_1", checkpoint.files)
            self.assertIn("b_residual_2_1", checkpoint.files)
            self.assertIn("w_residual_2_2", checkpoint.files)
            self.assertIn("b_residual_2_2", checkpoint.files)
            self.assertEqual((15, 64), checkpoint["w_input"].shape)
            self.assertEqual((64,), checkpoint["b_input"].shape)
            self.assertEqual((64, 64), checkpoint["w_residual_1_1"].shape)
            self.assertEqual((64,), checkpoint["b_residual_1_1"].shape)
            self.assertEqual((64, 64), checkpoint["w_residual_2_2"].shape)
            self.assertEqual((64,), checkpoint["b_residual_2_2"].shape)
            self.assertEqual((64, 6), checkpoint["w_policy"].shape)
            self.assertEqual((6,), checkpoint["b_policy"].shape)
            self.assertEqual((64, 1), checkpoint["w_value"].shape)
            self.assertEqual((1,), checkpoint["b_value"].shape)

    def test_argument_parser_accepts_residual_v3_model_type_choice(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            ["--out", "checkpoint.npz", "--model-type", "residual_v3"]
        )

        self.assertEqual("residual_v3", args.model_type)

    def test_cli_trains_with_residual_v3_contract_dispatch(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=128)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--model-type",
                    "residual_v3",
                    "--hidden-sizes",
                    "64,2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())
            self.assertIn("model_type=residual_v3", result.stdout)

    def test_residual_v3_outputs_policy_and_value_shapes(self):
        model = train_module.PolicyValueNet(
            hidden_sizes=(64, 2), model_type="residual_v3", input_size=27
        )

        policy, value = model(torch.zeros((2, 27), dtype=torch.float32))

        self.assertEqual((2, 6), tuple(policy.shape))
        self.assertEqual((2, 1), tuple(value.shape))

    def test_residual_v3_uses_distinct_policy_and_value_paths(self):
        model = train_module.PolicyValueNet(
            hidden_sizes=(64, 2), model_type="residual_v3", input_size=27
        )

        self.assertIsInstance(model.policy_hidden_layer, torch.nn.Linear)
        self.assertIsInstance(model.value_hidden_layer, torch.nn.Linear)
        self.assertIsNot(model.policy_hidden_layer, model.value_hidden_layer)
        self.assertEqual(64, model.policy_hidden_layer.in_features)
        self.assertEqual(64, model.policy_hidden_layer.out_features)
        self.assertEqual(64, model.value_hidden_layer.in_features)
        self.assertEqual(32, model.value_hidden_layer.out_features)
        self.assertEqual(64, model.policy_head.in_features)
        self.assertEqual(32, model.value_head.in_features)

    def test_residual_v3_checkpoint_round_trip_preserves_head_specific_layers(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "checkpoint.npz"

            model = train_module.PolicyValueNet(
                hidden_sizes=(64, 2), model_type="residual_v3", input_size=27
            )
            for index, parameter in enumerate(model.parameters(), start=1):
                torch.nn.init.constant_(parameter, index / 10.0)

            np.savez(checkpoint_path, **train_module.checkpoint_from_model(model))

            restored_model = train_module.PolicyValueNet(
                hidden_sizes=(64, 2), model_type="residual_v3", input_size=27
            )
            train_module.load_checkpoint_into_model(restored_model, checkpoint_path)

            checkpoint = np.load(checkpoint_path)
            self.assertIn("w_policy_hidden", checkpoint.files)
            self.assertIn("b_policy_hidden", checkpoint.files)
            self.assertIn("w_value_hidden", checkpoint.files)
            self.assertIn("b_value_hidden", checkpoint.files)

            for key, parameter in model.state_dict().items():
                np.testing.assert_allclose(
                    parameter.detach().cpu().numpy(),
                    restored_model.state_dict()[key].detach().cpu().numpy(),
                    rtol=1e-7,
                    atol=1e-7,
                )

    def test_cli_supports_replay_data_files_with_weights(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path1 = tmp_path / "data_1.jsonl"
            data_path2 = tmp_path / "data_2.jsonl"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path1, rows=64)
            self._write_dataset(data_path2, rows=64)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data-files",
                    f"{data_path1},{data_path2}",
                    "--replay-weights",
                    "1,2",
                    "--out",
                    str(out_path),
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

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())

    def test_train_can_initialize_from_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            init_checkpoint_path = tmp_path / "init_checkpoint.npz"
            out_path = tmp_path / "checkpoint.npz"

            self._write_dataset(data_path, rows=64)

            model = train_module.PolicyValueNet(
                hidden_sizes=(64, 64), model_type="mlp_v1", input_size=15
            )
            for index, parameter in enumerate(model.parameters(), start=1):
                torch.nn.init.constant_(parameter, index / 10.0)
            np.savez(init_checkpoint_path, **train_module.checkpoint_from_model(model))

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(out_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                    "--lr",
                    "0",
                    "--init-checkpoint",
                    str(init_checkpoint_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)

            init_checkpoint = np.load(init_checkpoint_path)
            checkpoint = np.load(out_path)

            for key in [
                "w1",
                "b1",
                "w2",
                "b2",
                "w_policy",
                "b_policy",
                "w_value",
                "b_value",
            ]:
                np.testing.assert_allclose(
                    init_checkpoint[key], checkpoint[key], rtol=1e-7, atol=1e-7
                )

    def test_replay_loader_returns_expanded_sample_indexes_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path1 = tmp_path / "data_1.jsonl"
            data_path2 = tmp_path / "data_2.jsonl"

            rows1 = [
                {
                    "state": [1.0] * 15,
                    "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 1.0,
                },
                {
                    "state": [2.0] * 15,
                    "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    "value": -1.0,
                },
            ]
            rows2 = [
                {
                    "state": [3.0] * 15,
                    "policy": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "value": 0.5,
                },
            ]
            self._write_rows(data_path1, rows1)
            self._write_rows(data_path2, rows2)

            x, p_target, v_target, sample_indexes = train_module.load_jsonl_replay(
                [data_path1, data_path2],
                [1, 3],
            )

            self.assertEqual((3, 15), x.shape)
            self.assertEqual((3, 6), p_target.shape)
            self.assertEqual((3, 1), v_target.shape)
            np.testing.assert_array_equal(
                np.array([0, 1, 2, 2, 2], dtype=np.int64), sample_indexes
            )
            np.testing.assert_array_equal(
                np.array([1.0, 2.0, 3.0], dtype=np.float32), x[:, 0]
            )

    def test_load_jsonl_replay_preserves_legal_sharpened_targets_from_self_play_and_bootstrap(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            self_play_path = tmp_path / "self_play.jsonl"
            bootstrap_path = tmp_path / "bootstrap.jsonl"

            self._generate_self_play_dataset(
                self_play_path,
                policy_target_mode="sharpened",
                games=1,
                simulations=8,
            )
            self._write_rows(
                bootstrap_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 1, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 22,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
                        "value": -1.0,
                        "source": "bootstrap",
                        "policy_target_mode": "sharpened",
                    }
                ],
            )

            x, p_target, v_target, replay_indexes = train_module.load_jsonl_replay(
                [self_play_path, bootstrap_path],
                [1, 2],
                policy_target_mode="sharpened",
            )

            self.assertGreaterEqual(x.shape[0], 2)
            self.assertEqual(15, x.shape[1])
            self.assertEqual((x.shape[0], 6), p_target.shape)
            self.assertEqual((x.shape[0], 1), v_target.shape)
            np.testing.assert_allclose(
                np.sum(p_target, axis=1),
                np.ones(x.shape[0], dtype=np.float32),
                atol=1e-6,
            )
            self.assertTrue(np.all(p_target >= 0.0))
            for state, policy in zip(x, p_target):
                legal_moves = train_module.derive_legal_moves_from_encoded_state(state)
                self.assertIsNotNone(legal_moves)
                self.assertAlmostEqual(
                    1.0, float(np.sum(policy[legal_moves])), places=6
                )
                self.assertTrue(
                    all(
                        policy[move] == 0.0
                        for move in range(6)
                        if move not in legal_moves
                    )
                )
            bootstrap_index = x.shape[0] - 1
            np.testing.assert_allclose(
                p_target[bootstrap_index],
                np.array([0.3, 0.0, 0.0, 0.0, 0.7, 0.0], dtype=np.float32),
                atol=1e-6,
            )
            np.testing.assert_array_equal(
                np.array([bootstrap_index, bootstrap_index], dtype=np.int64),
                replay_indexes[-2:],
            )

    def test_load_jsonl_replay_accepts_sharpened_value_targets_from_both_sources(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            self_play_path = tmp_path / "self_play_value.jsonl"
            bootstrap_path = tmp_path / "bootstrap_value.jsonl"

            self._generate_self_play_dataset(
                self_play_path,
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
                games=1,
                simulations=8,
            )
            self._write_rows(
                bootstrap_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 1, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 22,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
                        "value": 0.64,
                        "source": "bootstrap",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                    }
                ],
            )

            _, _, values, replay_indexes = train_module.load_jsonl_replay(
                [self_play_path, bootstrap_path],
                [1, 2],
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
            )

            self.assertTrue(np.all(values <= 1.0))
            self.assertTrue(np.all(values >= -1.0))
            bootstrap_index = values.shape[0] - 1
            self.assertAlmostEqual(0.64, float(values[bootstrap_index, 0]), places=6)
            np.testing.assert_array_equal(
                np.array([bootstrap_index, bootstrap_index], dtype=np.int64),
                replay_indexes[-2:],
            )

    def test_load_jsonl_replay_accepts_aligned_bootstrap_rows_alongside_self_play_under_same_mode(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            self_play_path = tmp_path / "self_play_aligned.jsonl"
            bootstrap_path = tmp_path / "bootstrap_aligned.jsonl"

            self._generate_self_play_dataset(
                self_play_path,
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
                games=1,
                simulations=8,
            )
            self._write_rows(
                bootstrap_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 1, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 22,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
                        "value": 0.27,
                        "source": "bootstrap",
                        "player": 0,
                        "winner": 0,
                        "move_index": 11,
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "sharpened",
                    }
                ],
            )

            _, _, values, replay_indexes = train_module.load_jsonl_replay(
                [self_play_path, bootstrap_path],
                [1, 2],
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
            )

            bootstrap_index = values.shape[0] - 1
            self.assertAlmostEqual(0.27, float(values[bootstrap_index, 0]), places=6)
            np.testing.assert_array_equal(
                np.array([bootstrap_index, bootstrap_index], dtype=np.int64),
                replay_indexes[-2:],
            )

    def test_load_jsonl_replay_accepts_phase_aware_value_targets_from_both_sources(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            self_play_path = tmp_path / "self_play_phase_aware_value.jsonl"
            bootstrap_path = tmp_path / "bootstrap_phase_aware_value.jsonl"

            self._generate_self_play_dataset(
                self_play_path,
                policy_target_mode="sharpened",
                value_target_mode="phase_aware_sharpened",
                games=1,
                simulations=8,
            )
            self._write_rows(
                bootstrap_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 1, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 22,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
                        "value": 0.64,
                        "source": "bootstrap",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "phase_aware_sharpened",
                    }
                ],
            )

            _, _, values, replay_indexes = train_module.load_jsonl_replay(
                [self_play_path, bootstrap_path],
                [1, 2],
                policy_target_mode="sharpened",
                value_target_mode="phase_aware_sharpened",
            )

            self.assertTrue(np.all(values <= 1.0))
            self.assertTrue(np.all(values >= -1.0))
            bootstrap_index = values.shape[0] - 1
            self.assertAlmostEqual(0.64, float(values[bootstrap_index, 0]), places=6)
            np.testing.assert_array_equal(
                np.array([bootstrap_index, bootstrap_index], dtype=np.int64),
                replay_indexes[-2:],
            )

    def test_load_jsonl_replay_accepts_hybrid_value_targets_from_both_sources(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            self_play_path = tmp_path / "self_play_hybrid_value.jsonl"
            bootstrap_path = tmp_path / "bootstrap_hybrid_value.jsonl"

            self._generate_self_play_dataset(
                self_play_path,
                policy_target_mode="sharpened",
                value_target_mode="hybrid",
                games=1,
                simulations=8,
            )
            self._write_rows(
                bootstrap_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 1, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 22,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.3, 0.0, 0.0, 0.0, 0.7, 0.0],
                        "value": 0.2,
                        "source": "bootstrap",
                        "policy_target_mode": "sharpened",
                        "value_target_mode": "hybrid",
                    }
                ],
            )

            _, _, values, replay_indexes = train_module.load_jsonl_replay(
                [self_play_path, bootstrap_path],
                [1, 2],
                policy_target_mode="sharpened",
                value_target_mode="hybrid",
            )

            self.assertTrue(np.all(values <= 1.0))
            self.assertTrue(np.all(values >= -1.0))
            bootstrap_index = values.shape[0] - 1
            self.assertAlmostEqual(0.2, float(values[bootstrap_index, 0]), places=6)
            np.testing.assert_array_equal(
                np.array([bootstrap_index, bootstrap_index], dtype=np.int64),
                replay_indexes[-2:],
            )

    def test_load_jsonl_replay_rejects_missing_sharpened_policy_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "missing-metadata.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.0,
                        "source": "bootstrap",
                    }
                ],
            )

            with self.assertRaisesRegex(
                ValueError, "must declare policy_target_mode=sharpened"
            ):
                train_module.load_jsonl_replay(
                    [data_path], policy_target_mode="sharpened"
                )

    def test_load_jsonl_uses_policy_target_mode_when_actual_mode_is_absent(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "legacy-policy-mode.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.0,
                        "policy_target_mode": "sharpened",
                    }
                ],
            )

            x, p_target, v_target = train_module.load_jsonl(
                data_path, policy_target_mode="sharpened"
            )

            self.assertEqual((1, 15), x.shape)
            self.assertEqual((1, 6), p_target.shape)
            self.assertEqual((1, 1), v_target.shape)

    def test_load_jsonl_rejects_rows_when_policy_target_actual_mode_differs_from_request(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "actual-policy-mode-mismatch.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.0,
                        "policy_target_mode": "sharpened",
                        "policy_target_actual_mode": "default",
                    }
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "policy_target_mode=default does not match requested sharpened",
            ):
                train_module.load_jsonl(data_path, policy_target_mode="sharpened")

    def test_load_jsonl_rejects_missing_sharpened_value_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "missing-value-metadata.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.0,
                        "source": "bootstrap",
                        "policy_target_mode": "default",
                    }
                ],
            )

            with self.assertRaisesRegex(
                ValueError, "must declare value_target_mode=sharpened"
            ):
                train_module.load_jsonl(data_path, value_target_mode="sharpened")

    def test_load_jsonl_rejects_missing_phase_aware_value_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "missing-phase-aware-value-metadata.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.0,
                        "source": "bootstrap",
                        "policy_target_mode": "default",
                    }
                ],
            )

            with self.assertRaisesRegex(
                ValueError, "must declare value_target_mode=phase_aware_sharpened"
            ):
                train_module.load_jsonl(
                    data_path, value_target_mode="phase_aware_sharpened"
                )

    def test_load_jsonl_rejects_missing_hybrid_value_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "missing-hybrid-value-metadata.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 0.2,
                        "source": "bootstrap",
                        "policy_target_mode": "default",
                    }
                ],
            )

            with self.assertRaisesRegex(
                ValueError, "must declare value_target_mode=hybrid"
            ):
                train_module.load_jsonl(data_path, value_target_mode="hybrid")

    def test_load_jsonl_replay_rejects_mismatched_value_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "mismatched-value-metadata.jsonl"

            self._write_dataset(data_path, rows=2, value_target_mode="default")

            with self.assertRaisesRegex(
                ValueError,
                "value_target_mode=default does not match requested sharpened",
            ):
                train_module.load_jsonl_replay(
                    [data_path], value_target_mode="sharpened"
                )

    def test_load_jsonl_replay_rejects_mismatched_phase_aware_value_target_metadata(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "mismatched-phase-aware-value-metadata.jsonl"

            self._write_dataset(data_path, rows=2, value_target_mode="sharpened")

            with self.assertRaisesRegex(
                ValueError,
                "value_target_mode=sharpened does not match requested phase_aware_sharpened",
            ):
                train_module.load_jsonl_replay(
                    [data_path], value_target_mode="phase_aware_sharpened"
                )

    def test_load_jsonl_replay_rejects_mismatched_hybrid_value_target_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "mismatched-hybrid-value-metadata.jsonl"

            self._write_dataset(data_path, rows=2, value_target_mode="sharpened")

            with self.assertRaisesRegex(
                ValueError,
                "value_target_mode=sharpened does not match requested hybrid",
            ):
                train_module.load_jsonl_replay([data_path], value_target_mode="hybrid")

    def test_load_jsonl_rejects_out_of_range_value_targets(self):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "out-of-range-value.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": [1.0] * 15,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.2,
                        "policy_target_mode": "default",
                        "value_target_mode": "sharpened",
                    }
                ],
            )

            with self.assertRaisesRegex(
                ValueError, r"value must stay within \[-1.0, 1.0\]"
            ):
                train_module.load_jsonl(data_path, value_target_mode="sharpened")

    def test_load_jsonl_replay_rejects_positive_probability_on_illegal_move_when_state_is_decodable(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-train-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "invalid.jsonl"

            self._write_rows(
                data_path,
                [
                    {
                        "state": self_play_module.encode_state(
                            {
                                "player_pits": [1, 0, 0, 0, 0, 0],
                                "opponent_pits": [0, 0, 0, 0, 0, 0],
                                "player_store": 24,
                                "opponent_store": 23,
                                "current_player": 0,
                            }
                        ),
                        "policy": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                        "value": 1.0,
                        "source": "bootstrap",
                        "policy_target_mode": "sharpened",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "illegal moves"):
                train_module.load_jsonl_replay(
                    [data_path], policy_target_mode="sharpened"
                )

    def test_train_replay_indexes_match_duplicated_sgd_behavior_for_small_batches(self):
        x_compact = np.array([[1.0] * 15, [2.0] * 15], dtype=np.float32)
        p_compact = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        v_compact = np.array([[1.0], [-1.0]], dtype=np.float32)
        replay_indexes = np.array([0, 1, 1, 1], dtype=np.int64)

        x_duplicated = np.array(
            [[1.0] * 15, [2.0] * 15, [2.0] * 15, [2.0] * 15], dtype=np.float32
        )
        p_duplicated = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        v_duplicated = np.array([[1.0], [-1.0], [-1.0], [-1.0]], dtype=np.float32)

        indexed_model = train_module.PolicyValueNet(
            hidden_sizes=(8, 8), model_type="mlp_v1", input_size=15
        )
        duplicated_model = train_module.PolicyValueNet(
            hidden_sizes=(8, 8), model_type="mlp_v1", input_size=15
        )
        duplicated_model.load_state_dict(indexed_model.state_dict())

        train_module.set_seed(7)
        train_module.train(
            indexed_model,
            x_compact,
            p_compact,
            v_compact,
            replay_indexes=replay_indexes,
            epochs=1,
            batch_size=2,
            lr=0.01,
            device=torch.device("cpu"),
            value_loss_weight=0.5,
            value_loss="mse",
            huber_delta=1.0,
            val_split=0.0,
            grad_clip=None,
            save_top_k=0,
        )

        train_module.set_seed(7)
        train_module.train(
            duplicated_model,
            x_duplicated,
            p_duplicated,
            v_duplicated,
            epochs=1,
            batch_size=2,
            lr=0.01,
            device=torch.device("cpu"),
            value_loss_weight=0.5,
            value_loss="mse",
            huber_delta=1.0,
            val_split=0.0,
            grad_clip=None,
            save_top_k=0,
        )

        for key, indexed_param in indexed_model.state_dict().items():
            np.testing.assert_allclose(
                indexed_param.detach().cpu().numpy(),
                duplicated_model.state_dict()[key].detach().cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_split_replay_positions_by_source_row_keeps_source_rows_disjoint(self):
        replay_indexes = np.array([0, 1, 1, 1], dtype=np.int64)

        train_module.set_seed(11)
        train_positions, val_positions = (
            train_module.split_replay_positions_by_source_row(
                replay_indexes, val_split=0.5
            )
        )

        train_source_rows = set(replay_indexes[train_positions].tolist())
        val_source_rows = set(replay_indexes[val_positions].tolist())

        self.assertTrue(train_source_rows)
        self.assertTrue(val_source_rows)
        self.assertTrue(train_source_rows.isdisjoint(val_source_rows))
        self.assertEqual({0, 1}, train_source_rows | val_source_rows)

    def test_split_replay_positions_by_source_row_is_seed_deterministic(self):
        replay_indexes = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)

        train_module.set_seed(23)
        first_train, first_val = train_module.split_replay_positions_by_source_row(
            replay_indexes, val_split=0.5
        )

        train_module.set_seed(23)
        second_train, second_val = train_module.split_replay_positions_by_source_row(
            replay_indexes, val_split=0.5
        )

        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_val, second_val)

    def test_split_replay_positions_by_source_row_keeps_nonzero_val_split_nonempty(
        self,
    ):
        replay_indexes = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)

        train_module.set_seed(31)
        train_positions, val_positions = (
            train_module.split_replay_positions_by_source_row(
                replay_indexes, val_split=0.1
            )
        )

        self.assertGreater(train_positions.shape[0], 0)
        self.assertGreater(val_positions.shape[0], 0)

    def test_split_replay_positions_by_source_row_zero_val_split_stays_empty(self):
        replay_indexes = np.array([0, 0, 1, 1], dtype=np.int64)

        train_module.set_seed(31)
        train_positions, val_positions = (
            train_module.split_replay_positions_by_source_row(
                replay_indexes, val_split=0.0
            )
        )

        self.assertGreater(train_positions.shape[0], 0)
        self.assertEqual(0, val_positions.shape[0])

    def test_argument_parser_accepts_behavior_anchor_training_args(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            [
                "--out",
                "checkpoint.npz",
                "--behavior-anchor-files",
                "anchors.jsonl",
                "--behavior-loss-weight",
                "4",
            ]
        )

        self.assertEqual("anchors.jsonl", args.behavior_anchor_files)
        self.assertEqual(4.0, args.behavior_loss_weight)

    def test_argument_parser_accepts_pairwise_training_args(self):
        parser = train_module.build_argument_parser()

        args = parser.parse_args(
            [
                "--out",
                "checkpoint.npz",
                "--pairwise-target-files",
                "pairwise.jsonl",
                "--pairwise-loss-weight",
                "1.5",
                "--pairwise-margin",
                "0.1",
            ]
        )

        self.assertEqual("pairwise.jsonl", args.pairwise_target_files)
        self.assertEqual(1.5, args.pairwise_loss_weight)
        self.assertEqual(0.1, args.pairwise_margin)

    def test_compute_pairwise_ranking_loss_matches_softplus_margin_form(self):
        logits = torch.tensor([[0.0, 0.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)
        preferred = torch.tensor([1], dtype=torch.int64)
        baseline = torch.tensor([0], dtype=torch.int64)

        loss = train_module.compute_pairwise_ranking_loss(
            logits,
            preferred,
            baseline,
            margin=0.1,
        )

        self.assertAlmostEqual(
            float(np.log1p(np.exp(0.1))), float(loss.item()), places=6
        )

    def test_train_one_epoch_reports_pairwise_loss_without_supervised_rows(self):
        model = train_module.PolicyValueNet(
            hidden_sizes=(8, 8), model_type="mlp_v1", input_size=15
        )
        for parameter in model.parameters():
            parameter.data.zero_()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        empty_x = np.zeros((0, 15), dtype=np.float32)
        empty_p = np.zeros((0, 6), dtype=np.float32)
        empty_v = np.zeros((0, 1), dtype=np.float32)
        pairwise_x = np.zeros((1, 15), dtype=np.float32)
        pairwise_preferred = np.array([1], dtype=np.int64)
        pairwise_baseline = np.array([0], dtype=np.int64)
        pairwise_replay_indexes = np.array([0], dtype=np.int64)

        metrics = train_module.train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=empty_x,
            compact_p=empty_p,
            compact_v=empty_v,
            replay_indexes=np.zeros((0,), dtype=np.int64),
            batch_size=1,
            device=torch.device("cpu"),
            value_loss_weight=0.0,
            value_loss="mse",
            huber_delta=1.0,
            grad_clip=None,
            pairwise_x=pairwise_x,
            pairwise_preferred_moves=pairwise_preferred,
            pairwise_baseline_moves=pairwise_baseline,
            pairwise_replay_indexes=pairwise_replay_indexes,
            pairwise_loss_weight=1.0,
            pairwise_margin=0.1,
        )

        expected_pairwise = float(np.log1p(np.exp(0.1)))
        self.assertEqual(0.0, metrics["policy_loss"])
        self.assertAlmostEqual(expected_pairwise, metrics["pairwise_loss"], places=6)
        self.assertAlmostEqual(expected_pairwise, metrics["total_loss"], places=6)

    def test_train_one_epoch_reports_behavior_anchor_loss(self):
        model = train_module.PolicyValueNet(
            hidden_sizes=(8, 8), model_type="mlp_v1", input_size=15
        )
        for parameter in model.parameters():
            parameter.data.zero_()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        x = np.zeros((1, 15), dtype=np.float32)
        p = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        v = np.zeros((1, 1), dtype=np.float32)
        anchor_p = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        replay_indexes = np.array([0], dtype=np.int64)

        metrics = train_module.train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=x,
            compact_p=p,
            compact_v=v,
            replay_indexes=replay_indexes,
            batch_size=1,
            device=torch.device("cpu"),
            value_loss_weight=0.0,
            value_loss="mse",
            huber_delta=1.0,
            grad_clip=None,
            behavior_anchor_x=x,
            behavior_anchor_p=anchor_p,
            behavior_anchor_replay_indexes=replay_indexes,
            behavior_loss_weight=2.0,
        )

        expected_ce = float(np.log(6.0))
        self.assertAlmostEqual(expected_ce, metrics["policy_loss"], places=5)
        self.assertAlmostEqual(expected_ce, metrics["behavior_anchor_loss"], places=5)
        self.assertAlmostEqual(expected_ce * 3.0, metrics["total_loss"], places=5)


if __name__ == "__main__":
    unittest.main()
