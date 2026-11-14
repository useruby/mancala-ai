"""Synthetic fail-closed regressions for bounded tablebase search."""

from __future__ import annotations

import io
import os
import threading
import time
import unittest
from types import SimpleNamespace

from ml.alphazero_lite.exact_kalah_solver import ExactState, SearchTimeout
from ml.alphazero_lite.exact_tactical_label_diagnostic import (
    DeadlineNativeExactRootTablebase,
    TablebaseBoundedSolver,
    OUT,
    _reject_published_output,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_exact_tactical_label_diagnostic import (
    verify_rebuilt_source_roots,
)


def _small_state() -> ExactState:
    return ExactState(
        pits=(1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        stores=(3, 2),
        current_player=0,
    )


class _PipeProbeProcess:
    def __init__(self, response: bytes, delay_seconds: float) -> None:
        read_fd, self.write_fd = os.pipe()
        self.stdin = io.StringIO()
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.returncode: int | None = None
        self.writer = threading.Thread(
            target=self._write_response,
            args=(response, delay_seconds),
            daemon=True,
        )
        self.writer.start()

    def _write_response(self, response: bytes, delay_seconds: float) -> None:
        time.sleep(delay_seconds)
        try:
            os.write(self.write_fd, response)
        except OSError:
            pass
        finally:
            os.close(self.write_fd)

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            raise AssertionError("fake probe was not terminated")
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()


class TablebaseShortcutRegressionTests(unittest.TestCase):
    def test_published_output_directory_is_read_only_for_reruns(self) -> None:
        with self.assertRaisesRegex(ValueError, "published_output_directory"):
            _reject_published_output(OUT)

    def test_incomplete_tablebase_action_map_is_rejected(self) -> None:
        class PartialTablebase:
            def root_action_margins(self, *_args, **_kwargs):
                return {0: 4}

        solver = TablebaseBoundedSolver(PartialTablebase(), cache_enabled=False)
        with self.assertRaisesRegex(ValueError, "incomplete_or_invalid"):
            solver._search(_small_state(), -10_000, 10_000)

    def test_late_complete_reply_is_not_cached_as_exact(self) -> None:
        class LateTablebase:
            def __init__(self) -> None:
                self.timeout_seconds = None

            def root_action_margins(
                self, _game, perspective_player, *, timeout_seconds
            ):
                del perspective_player
                self.timeout_seconds = timeout_seconds
                time.sleep(0.02)
                return {0: 4, 1: 3}

        tablebase = LateTablebase()
        solver = TablebaseBoundedSolver(tablebase, cache_enabled=False)
        solver._deadline = time.monotonic() + 0.005
        with self.assertRaises(SearchTimeout):
            solver._search(_small_state(), -10_000, 10_000)
        self.assertTrue(
            isinstance(tablebase.timeout_seconds, float)
            and 0 < tablebase.timeout_seconds <= 0.005
        )

    def test_native_probe_times_out_on_incomplete_json_line(self) -> None:
        read_fd, write_fd = os.pipe()
        stdout = os.fdopen(read_fd, "rb", buffering=0)
        process = SimpleNamespace(
            returncode=None,
            stdin=io.StringIO(),
            stdout=stdout,
        )
        process.poll = lambda: process.returncode

        def terminate() -> None:
            process.returncode = -15
            stdout.close()

        process.terminate = terminate
        process.wait = lambda timeout=None: process.returncode
        process.kill = terminate
        probe = DeadlineNativeExactRootTablebase.__new__(
            DeadlineNativeExactRootTablebase
        )
        probe.process = process
        probe._stdout_buffer = ""
        probe._probe_poisoned = False
        writer = threading.Thread(
            target=lambda: os.write(write_fd, b'{"actions":'), daemon=True
        )
        writer.start()
        try:
            with self.assertRaisesRegex(TimeoutError, "deadline exceeded"):
                probe._probe([1] + [0] * 11, 0, timeout_seconds=0.03)
        finally:
            os.close(write_fd)
            stdout.close()
            writer.join(timeout=1)

    def test_timed_out_reply_cannot_leak_to_next_same_legal_moves_request(self) -> None:
        class RestartingProbe(DeadlineNativeExactRootTablebase):
            def _launch_fresh_probe(self) -> None:
                self.process = self.fresh_process
                self._stdout_buffer = ""
                self._probe_poisoned = False
                self.fresh_launches += 1

        first = _PipeProbeProcess(b'{"actions":{"0":40,"1":30}}\n', delay_seconds=0.06)
        second = _PipeProbeProcess(b'{"actions":{"0":5,"1":-2}}\n', delay_seconds=0)
        probe = RestartingProbe.__new__(RestartingProbe)
        probe.process = first
        probe.fresh_process = second
        probe.fresh_launches = 0
        probe._stdout_buffer = ""
        probe._probe_poisoned = False
        probe.calls = 0
        game = KalahGame.from_state(
            {
                "player_pits": [1, 1, 0, 0, 0, 0],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        try:
            with self.assertRaisesRegex(TimeoutError, "deadline exceeded"):
                probe.root_action_margins(game, 0, timeout_seconds=0.02)
            margins = probe.root_action_margins(game, 0, timeout_seconds=0.5)
            self.assertEqual(margins, {0: 5, 1: -2})
            self.assertEqual(probe.fresh_launches, 1)
            self.assertEqual(set(margins), set(game.possible_moves()))
        finally:
            probe.close()
            first.writer.join(timeout=1)
            second.writer.join(timeout=1)

    def test_source_cohort_comparison_rejects_archived_move_mutation(self) -> None:
        frozen = {
            "roots": [{"row_id": "replayed-row", "archived_move": 2}],
            "deduplication": {"candidate_rows": 1},
        }
        rebuilt = [{"row_id": "replayed-row", "archived_move": 3}]
        with self.assertRaisesRegex(ValueError, "archive_reconstruction"):
            verify_rebuilt_source_roots(frozen, rebuilt, {"candidate_rows": 1})


if __name__ == "__main__":
    unittest.main()
