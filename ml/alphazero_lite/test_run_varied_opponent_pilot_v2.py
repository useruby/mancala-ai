from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import run_varied_opponent_pilot_v2 as runner


class PairProbeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = [
            {
                "game_id": f"g{seat}",
                "pair_id": "pair",
                "arm": "control",
                "opening_id": "o",
                "opening_moves": [],
                "seed": 7,
                "designated_p_seat": seat,
                "opponent": "P",
                "search_workers": 1,
            }
            for seat in (0, 1)
        ]
        self.evaluators = type(
            "Evaluators",
            (),
            {
                "runtime_policy": {
                    "runtime_assets": {
                        "native_probe": {"path": "probe"},
                        "tablebase": {"path": "tablebase"},
                    },
                    "effective_search_policy": {"exact_root_threshold": 16},
                },
                "p": object(),
                "initializer": object(),
                "t512": object(),
            },
        )()

    def _execute(self, *, fail_warm=False, fail_game=None):
        probes = []

        class Probe:
            def __init__(self, *_args, **_kwargs):
                self.closed = False
                probes.append(self)

            def warm(self):
                if fail_warm:
                    raise RuntimeError("warm failed")

            def close(self):
                self.closed = True

        games = []

        def play(spec, **kwargs):
            games.append(spec["game_id"])
            if fail_game == spec["game_id"]:
                raise RuntimeError("game failed")
            return {"spec": spec}

        job = {"game_specs": self.specs}
        with (
            patch.object(runner.pilot, "play_game", side_effect=play),
            patch.object(runner.pilot, "verify_pair_result"),
        ):
            try:
                result = runner.play_pair_with_probe(
                    job, self.evaluators, probe_factory=Probe
                )
            except RuntimeError:
                result = None
        return probes, games, result

    def test_both_seat_games_share_one_probe(self):
        probes, games, result = self._execute()
        self.assertEqual(games, ["g0", "g1"])
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0].closed)
        self.assertEqual(len(result["games"]), 2)

    def test_new_pair_owns_a_fresh_probe(self):
        first, _, _ = self._execute()
        second, _, _ = self._execute()
        self.assertIsNot(first[0], second[0])

    def test_warmup_and_either_game_failure_close_probe(self):
        for kwargs in ({"fail_warm": True}, {"fail_game": "g0"}, {"fail_game": "g1"}):
            with self.subTest(kwargs=kwargs):
                probes, _, result = self._execute(**kwargs)
                self.assertIsNone(result)
                self.assertEqual(len(probes), 1)
                self.assertTrue(probes[0].closed)

    def test_interrupted_attempt_remains_uncommitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt.json"
            job = {"game_specs": self.specs}
            with (
                patch.object(runner.pilot, "_initialize_pair_process"),
                patch.object(runner.pilot, "_PROCESS_EVALUATORS", self.evaluators),
                patch.object(
                    runner,
                    "play_pair_with_probe",
                    side_effect=RuntimeError("interrupted"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    runner._run_attempt(job, "artifacts", attempt)
            self.assertEqual(json.loads(attempt.read_text())["status"], "started")
            self.assertEqual(list(root.glob("*.json")), [attempt])


if __name__ == "__main__":
    unittest.main()
