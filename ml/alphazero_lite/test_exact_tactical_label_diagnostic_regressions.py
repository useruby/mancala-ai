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
)
from ml.alphazero_lite.verify_exact_tactical_label_diagnostic import (
    verify_rebuilt_source_roots,
)


def _small_state() -> ExactState:
    return ExactState(
        pits=(1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        stores=(3, 2),
        current_player=0,
    )


class TablebaseShortcutRegressionTests(unittest.TestCase):
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
            poll=lambda: None,
            stdin=io.StringIO(),
            stdout=stdout,
        )
        probe = DeadlineNativeExactRootTablebase.__new__(
            DeadlineNativeExactRootTablebase
        )
        probe.process = process
        probe._stdout_buffer = ""
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
