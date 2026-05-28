import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import evaluate_top_k_checkpoints


class EvaluateTopKCheckpointsTest(unittest.TestCase):
    def executable_python(self) -> str:
        return str(Path(__file__).resolve().parents[2] / ".venv/bin/python")

    def test_dry_run_uses_shared_default_workers_for_arena_command(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-topk-default-workers-") as tmp:
            tmp_path = Path(tmp)
            iter_dir = tmp_path / "iter1"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "summary.json"
            iter_dir.mkdir()
            current_dir.mkdir()
            (iter_dir / "checkpoint.npz").write_bytes(b"stub")

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/evaluate_top_k_checkpoints.py",
                    "--iter-dir",
                    str(iter_dir),
                    "--current-path",
                    str(current_dir),
                    "--out",
                    str(out_path),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(24, payload["workers"])
        self.assertEqual(
            "24",
            payload["candidates"][0]["arena_command"][
                payload["candidates"][0]["arena_command"].index("--workers") + 1
            ],
        )

    def test_discover_checkpoints_orders_base_then_numeric_topk(self):
        with tempfile.TemporaryDirectory(prefix="azlite-topk-discover-") as tmp:
            iter_dir = Path(tmp)
            for name in [
                "checkpoint.top10.npz",
                "checkpoint.top2.npz",
                "checkpoint.npz",
                "checkpoint.top1.npz",
                "checkpoint.topx.npz",
            ]:
                (iter_dir / name).write_bytes(b"stub")

            discovered = evaluate_top_k_checkpoints.discover_checkpoints(iter_dir)

        self.assertEqual(
            [
                "checkpoint.npz",
                "checkpoint.top1.npz",
                "checkpoint.top2.npz",
                "checkpoint.top10.npz",
            ],
            [path.name for path in discovered],
        )

    def test_dry_run_writes_summary_with_export_and_arena_commands(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="azlite-topk-dry-run-") as tmp:
            tmp_path = Path(tmp)
            iter_dir = tmp_path / "iter1"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "summary.json"
            iter_dir.mkdir()
            current_dir.mkdir()
            (iter_dir / "checkpoint.npz").write_bytes(b"stub")
            (iter_dir / "checkpoint.top2.npz").write_bytes(b"stub")
            (iter_dir / "checkpoint.top1.npz").write_bytes(b"stub")

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/evaluate_top_k_checkpoints.py",
                    "--iter-dir",
                    str(iter_dir),
                    "--current-path",
                    str(current_dir),
                    "--games",
                    "24",
                    "--workers",
                    "3",
                    "--out",
                    str(out_path),
                    "--dry-run",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["dry_run"])
        self.assertEqual(3, len(payload["candidates"]))
        self.assertEqual(
            ["checkpoint.npz", "checkpoint.top1.npz", "checkpoint.top2.npz"],
            [candidate["checkpoint_name"] for candidate in payload["candidates"]],
        )
        first = payload["candidates"][0]
        self.assertEqual(
            [
                "ml/alphazero_lite/export_artifact.py",
                "--checkpoint",
                str(iter_dir / "checkpoint.npz"),
            ],
            first["export_command"][1:4],
        )
        self.assertIn("ml/alphazero_lite/arena.py", first["arena_command"])
        self.assertIn("--challenger-simulations", first["arena_command"])
        self.assertIn("640", first["arena_command"])
        self.assertFalse(
            (iter_dir / "top_k_exports" / "checkpoint" / "metadata.json").exists()
        )
