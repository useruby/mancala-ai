from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
