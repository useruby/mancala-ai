from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import benchmark_exact_root_probe_reuse as benchmark
from ml.alphazero_lite import self_play


class ProbeLifecycleTests(unittest.TestCase):
    def _run(self, *, fail: bool) -> bool:
        closed = False

        class FakeProbe:
            def __init__(self, *_args, **_kwargs):
                pass

            def warm(self):
                return None

            def root_action_margins(self, game, _player):
                if fail:
                    raise RuntimeError("injected query failure")
                return {move: 0 for move in game.possible_moves()}

            def close(self):
                nonlocal closed
                closed = True

        state = self_play.standard_start_state()
        request = {
            "game_id": "g0",
            "ply": 1,
            "state": state,
            "legal_moves": list(range(6)),
            "tie_break_priors": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        with patch.object(benchmark, "NativeExactRootTablebase", FakeProbe):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "injected query failure"):
                    benchmark._run_lifetime(
                        [request],
                        [{"pair_id": "p0", "game_ids": ["g0"]}],
                        benchmark.Path("probe"),
                        benchmark.Path("tablebase"),
                        "per_game",
                    )
            else:
                benchmark._run_lifetime(
                    [request],
                    [{"pair_id": "p0", "game_ids": ["g0"]}],
                    benchmark.Path("probe"),
                    benchmark.Path("tablebase"),
                    "per_game",
                )
        return closed

    def test_probe_closes_after_success(self):
        self.assertTrue(self._run(fail=False))

    def test_probe_closes_after_query_failure(self):
        self.assertTrue(self._run(fail=True))


class LegalActionCoverageTests(unittest.TestCase):
    def test_decide_rejects_missing_legal_action_margin(self):
        class IncompleteProbe:
            def root_action_margins(self, game, _player):
                return {move: 0 for move in game.possible_moves()[:-1]}

        request = {
            "state": self_play.standard_start_state(),
            "legal_moves": list(range(6)),
            "tie_break_priors": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        with self.assertRaisesRegex(ValueError, "legal_action_coverage_mismatch"):
            benchmark._decide(IncompleteProbe(), request)

    def test_benchmark_rejects_same_incomplete_reply_from_both_variants(self):
        incomplete = {
            "action_margins": {"0": 1},
            "selected_move": 0,
        }
        run = {
            "outputs": {"g0:1": incomplete},
            "timings": {
                "startup_wall_ns": 0,
                "warmup_wall_ns": 0,
                "query_wall_ns": 0,
            },
            "native_child_cpu_ns": 1,
        }
        manifest = {
            "manifest_sha256": "fixture",
            "game_boundaries": [{"pair_id": "p0", "game_id": "g0"}],
            "requests": [
                {
                    "pair_id": "p0",
                    "game_id": "g0",
                    "ply": 1,
                    "legal_moves": [0, 1],
                    "archived_exact_row": False,
                    "archived_action_margins": None,
                    "archived_move": 0,
                }
            ],
        }
        with patch.object(benchmark, "_run_lifetime", return_value=run):
            with self.assertRaisesRegex(
                ValueError, "benchmark_legal_action_coverage_mismatch"
            ):
                benchmark.benchmark(
                    manifest, benchmark.Path("probe"), benchmark.Path("tb")
                )


class CliOutputPathTests(unittest.TestCase):
    def test_manifest_mode_rejects_existing_output_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            output.write_text("published")
            argv = [
                "benchmark_exact_root_probe_reuse",
                "manifest",
                "--manifest-output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark, "build_manifest") as build_manifest,
            ):
                with self.assertRaises(SystemExit):
                    benchmark.main()
            build_manifest.assert_not_called()
            self.assertEqual(output.read_text(), "published")

    def test_benchmark_mode_rejects_existing_output_before_loading_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("published")
            argv = [
                "benchmark_exact_root_probe_reuse",
                "benchmark",
                "--manifest",
                str(Path(directory) / "missing-manifest.json"),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    benchmark.main()
            self.assertEqual(output.read_text(), "published")


if __name__ == "__main__":
    unittest.main()
