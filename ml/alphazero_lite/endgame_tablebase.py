from __future__ import annotations

from typing import Protocol

from ml.alphazero_lite.kalah_rules import KalahGame


class EndgameTablebaseContract(Protocol):
    def lookup(self, game: KalahGame, perspective_player: int) -> float | None:
        """Return solved win-rate for ``game`` from ``perspective_player``.

        Values must follow classic MCTS playout conventions in the ``[0.0, 1.0]`` range:
        ``1.0`` forced win, ``0.5`` forced draw, ``0.0`` forced loss for
        ``perspective_player``. Return ``None`` when state is not solved/present.
        """


class EndgameTablebase:
    def __init__(self) -> None:
        self._values: dict[tuple[int, tuple[int, ...], tuple[int, int], int], float] = {}

    def lookup(self, game: KalahGame, perspective_player: int) -> float | None:
        return self._values.get(self._key(game, perspective_player))

    def record(self, game: KalahGame, perspective_player: int, value: float) -> None:
        self._values[self._key(game, perspective_player)] = float(value)

    def _key(self, game: KalahGame, perspective_player: int) -> tuple[int, tuple[int, ...], tuple[int, int], int]:
        return (
            int(perspective_player),
            tuple(game.pits),
            (int(game.captured_seeds[0]), int(game.captured_seeds[1])),
            int(game.current_player),
        )
