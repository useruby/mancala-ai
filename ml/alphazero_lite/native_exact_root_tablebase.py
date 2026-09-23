"""Worker-local adapter for the native KVTB exact root-action probe."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame


class NativeExactRootTablebase:
    """Translate native player-0 pit margins into final root-player margins."""

    implementation_identity = "native_kvtb_root_action_probe_v1"

    def __init__(
        self, binary: str | Path, tablebase: str | Path, *, warm_on_start: bool = False
    ) -> None:
        self.binary = Path(binary)
        self.tablebase = Path(tablebase)
        if not self.binary.is_file() or not self.tablebase.is_file():
            raise FileNotFoundError(
                "native exact-root probe binary or tablebase is missing"
            )
        self.process = subprocess.Popen(
            [str(self.binary), "probe", str(self.tablebase)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.calls = 0
        self.startup_warm_latency_ms: float | None = None
        if warm_on_start:
            self.warm()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        if self.process.stdin is not None and hasattr(self.process.stdin, "close"):
            self.process.stdin.close()
        if self.process.stdout is not None and hasattr(self.process.stdout, "close"):
            self.process.stdout.close()

    def __enter__(self) -> NativeExactRootTablebase:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def root_action_margins(
        self, game: KalahGame, perspective_player: int
    ) -> dict[int, int]:
        raw_actions = self._probe(game.pits, game.current_player)
        if not isinstance(raw_actions, dict):
            raise RuntimeError(f"native exact-root probe failed: {raw_actions}")

        self.calls += 1
        store_margin_p0 = game.captured_seeds[0] - game.captured_seeds[1]
        root_sign = 1 if int(perspective_player) == 0 else -1
        return {
            int(move): root_sign * (int(raw_margin) + store_margin_p0)
            for move, raw_margin in raw_actions.items()
        }

    def warm(self) -> None:
        started = time.perf_counter()
        actions = self._probe([0] * 12, 0)
        if actions != {}:
            raise RuntimeError(
                "native exact-root warm probe returned nonterminal actions"
            )
        self.startup_warm_latency_ms = (time.perf_counter() - started) * 1000.0

    def _probe(self, pits: list[int], player: int) -> object:
        if self.process.poll() is not None:
            raise RuntimeError("native exact-root probe exited")
        assert self.process.stdin is not None and self.process.stdout is not None
        request = {"pits": pits, "player": player}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        return response.get("actions")
