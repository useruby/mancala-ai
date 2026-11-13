from __future__ import annotations

import unittest
import json
import hashlib
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
                    runner._run_attempt(job, "artifacts", attempt, {"schema": "v2"})
            self.assertEqual(json.loads(attempt.read_text())["status"], "started")
            self.assertEqual(list(root.glob("*.json")), [attempt])

    def test_v1_checkpoint_is_rejected_before_v2_writes_or_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "pairs"
            attempts = root / "attempts"
            checkpoints.mkdir()
            (checkpoints / "old.json").write_text(
                json.dumps({"schema": "varied_opponent_pair_checkpoint_v1"})
            )
            with self.assertRaisesRegex(ValueError, "unbound_existing_run_evidence"):
                runner.bind_run_directory(checkpoints, attempts, {"schema": "v2"})
            self.assertFalse((checkpoints / "run-identity.v2").exists())
            self.assertFalse(attempts.exists())

    def test_v1_freeze_is_not_accepted_as_a_v2_freeze(self):
        with self.assertRaisesRegex(ValueError, "pilot_v2_freeze_schema_mismatch"):
            runner.load_v2_freeze(
                Path("docs/data/varied-opponent-coverage-pilot/freeze.json"),
                Path.cwd(),
            )

    def test_both_game_callbacks_use_the_same_probe_instance(self):
        observed = []

        class Probe:
            def __init__(self, *_args, **_kwargs):
                self.closed = False

            def warm(self):
                pass

            def close(self):
                self.closed = True

        def production(_game, _evaluator, _rng, _ply, _retain, *, exact_root):
            exact_root(object(), object())
            return {"move": 0, "row": {}}

        def exact(_game, _evaluator, *, tablebase, threshold):
            observed.append((tablebase, threshold))
            return object()

        def play(spec, **kwargs):
            kwargs["search_turn"](object(), object(), object(), 0, True)
            return {"spec": spec}

        with (
            patch.object(runner.pilot, "play_game", side_effect=play),
            patch.object(
                runner.pilot, "production_search_turn", side_effect=production
            ),
            patch.object(runner.pilot, "verify_pair_result"),
            patch.object(runner.arena, "exact_root_decision", side_effect=exact),
        ):
            result = runner.play_pair_with_probe(
                {"game_specs": self.specs}, self.evaluators, probe_factory=Probe
            )
        self.assertEqual(len(result["games"]), 2)
        self.assertEqual(len(observed), 2)
        self.assertIs(observed[0][0], observed[1][0])
        self.assertEqual([threshold for _, threshold in observed], [16, 16])

    def test_replay_harness_exercises_all_frozen_boundaries_through_v2_callback(self):
        frozen_request = json.loads(
            Path(
                "docs/data/varied-opponent-coverage-pilot/probe-reuse-manifest.json"
            ).read_text()
        )["requests"][0]
        state = frozen_request["state"]
        chosen = int(frozen_request["archived_move"])
        synthetic_priors = [0.0] * 12
        synthetic_priors[chosen] = 1.0
        boundary_rows = []
        requests = []
        for index in range(256):
            game_id = f"game-{index:03d}"
            pair_id = f"pair-{index // 2:03d}"
            count = 14 if index >= 230 else 13
            boundary_rows.append(
                {"game_id": game_id, "pair_id": pair_id, "requests": count}
            )
            for ply in range(count):
                requests.append(
                    {
                        "game_id": game_id,
                        "pair_id": pair_id,
                        "ply": ply,
                        "state": state,
                        "tie_break_priors": synthetic_priors,
                        "archived_move": chosen,
                        "archived_exact_row": False,
                        "archived_action_margins": None,
                    }
                )
        self.assertEqual(len(requests), 3354)
        probes = []

        class Probe:
            def __init__(self, *_args, **_kwargs):
                probes.append(self)

            def warm(self):
                pass

            def root_action_margins(self, game, _player):
                return {int(move): 0 for move in game.possible_moves()}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "probe"
            tablebase = Path(directory) / "tablebase"
            binary.write_bytes(b"probe")
            tablebase.write_bytes(b"tablebase")
            manifest = {
                "manifest_sha256": "fixture",
                "runtime": {
                    "native_probe": {"sha256": hashlib.sha256(b"probe").hexdigest()},
                    "tablebase": {"sha256": hashlib.sha256(b"tablebase").hexdigest()},
                },
                "game_boundaries": boundary_rows,
                "requests": requests,
            }
            result = runner.replay_frozen_requests(
                manifest, binary, tablebase, probe_factory=Probe
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["requests"], 3354)
        self.assertEqual(result["game_boundaries"], 256)
        self.assertEqual(result["per_game_probe_count"], 256)
        self.assertEqual(result["per_pair_probe_count"], 128)
        self.assertEqual(len(probes), 384)


if __name__ == "__main__":
    unittest.main()
