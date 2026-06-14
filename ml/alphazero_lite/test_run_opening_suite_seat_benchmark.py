import tempfile
import unittest
from argparse import Namespace
from unittest import mock

from ml.alphazero_lite import run_opening_suite_seat_benchmark as benchmark


class OpeningSuiteSeatBenchmarkTest(unittest.TestCase):
    def test_deterministic_mode_uses_cli_seed(self):
        captured_seeds = []

        def fake_run_arena(**kwargs):
            captured_seeds.append(kwargs["seed"])
            return {"score": 0.5}

        def fake_parse_game_jsonl(path: str):
            seat = 0 if "starts_0" in path else 1
            return [
                {
                    "challenger_player": seat,
                    "winner": "draw",
                    "margin": 0,
                    "game_length": 1,
                    "trajectory": f"seat-{seat}",
                    "opening_prefix_moves": [0],
                }
            ]

        with tempfile.TemporaryDirectory(prefix="azlite-benchmark-") as tmp:
            args = Namespace(
                workdir=tmp,
                suite="ignored-suite.jsonl",
                current="ignored-current",
                candidates="candidate_a",
                budget_pairs="384:256",
                games_per_opening=1,
                seed=123,
                root_policy_mode="deterministic",
                root_temperatures="0.0",
                seeds="42,43",
                workers=1,
                timeout=10,
            )
            with (
                mock.patch.object(benchmark, "parse_args", return_value=args),
                mock.patch.object(
                    benchmark,
                    "load_suite",
                    return_value=[{"prefix_moves": [0]}],
                ),
                mock.patch.object(
                    benchmark,
                    "sha256_file",
                    side_effect=["current-sha", "candidate-sha", "suite-sha"],
                ),
                mock.patch.object(benchmark, "run_arena", side_effect=fake_run_arena),
                mock.patch.object(
                    benchmark,
                    "parse_game_jsonl",
                    side_effect=fake_parse_game_jsonl,
                ),
            ):
                rc = benchmark.main()

        self.assertEqual(0, rc)
        self.assertEqual([123, 123], captured_seeds)


if __name__ == "__main__":
    unittest.main()
