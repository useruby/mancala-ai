from __future__ import annotations

import sys
from typing import Protocol

from ml.alphazero_lite.kalah_rules import KalahGame

_MAX_RECURSION_DEPTH = 20000


class EndgameTablebaseContract(Protocol):
    def lookup(self, game: KalahGame, perspective_player: int) -> float | None:
        """Return solved win-rate for ``game`` from ``perspective_player``.

        Values must follow classic MCTS playout conventions in the ``[0.0, 1.0]`` range:
        ``1.0`` forced win, ``0.5`` forced draw, ``0.0`` forced loss for
        ``perspective_player``. Return ``None`` when state is not solved/present.
        """
        raise NotImplementedError

    def lookup_cached(self, game: KalahGame, perspective_player: int) -> float | None:
        """Return cached or terminal result without invoking on-demand solving."""
        raise NotImplementedError


class EndgameTablebase:
    MAX_SOLVED_SEEDS = 16

    def __init__(self) -> None:
        sys.setrecursionlimit(_MAX_RECURSION_DEPTH)
        self._values: dict[
            tuple[int, tuple[int, ...], tuple[int, int], int], float
        ] = {}

    def lookup(self, game: KalahGame, perspective_player: int) -> float | None:
        cached = self.lookup_cached(game, perspective_player)
        if cached is not None:
            return cached

        if not self._should_solve(game):
            return None

        key = self._key(game, perspective_player)
        value = self._solve(game.clone(), perspective_player)
        self._values[key] = value
        return value

    def lookup_cached(self, game: KalahGame, perspective_player: int) -> float | None:
        key = self._key(game, perspective_player)
        value = self._values.get(key)
        if value is not None:
            return value

        if self._is_terminal(game):
            value = self._terminal_value(game, perspective_player)
            self._values[key] = value
            return value
        return None

    def clear_cache(self) -> None:
        self._values.clear()

    def record(self, game: KalahGame, perspective_player: int, value: float) -> None:
        self._values[self._key(game, perspective_player)] = float(value)

    def _should_solve(self, game: KalahGame) -> bool:
        return sum(game.pits) <= self.MAX_SOLVED_SEEDS

    def _solve(self, game: KalahGame, perspective_player: int) -> float:
        key = self._key(game, perspective_player)
        cached = self._values.get(key)
        if cached is not None:
            return cached

        if self._is_terminal(game):
            value = self._terminal_value(game, perspective_player)
            self._values[key] = value
            return value

        child_values: list[float] = []
        offset = game.current_player * 6
        for move in game.possible_moves():
            next_game = game.clone()
            next_game.move(offset + move)
            child_values.append(self._solve(next_game, perspective_player))

        if game.current_player == perspective_player:
            value = max(child_values)
        else:
            value = min(child_values)

        self._values[key] = value
        return value

    def _is_terminal(self, game: KalahGame) -> bool:
        return game.over()

    def _terminal_value(self, game: KalahGame, perspective_player: int) -> float:
        settled_scores = game.captured_seeds.copy()
        for player in (0, 1):
            start = player * 6
            settled_scores[player] += sum(game.pits[start : start + 6])

        perspective_score = settled_scores[perspective_player]
        opponent_score = settled_scores[1 - perspective_player]
        if perspective_score > opponent_score:
            return 1.0
        if perspective_score < opponent_score:
            return 0.0
        return 0.5

    def _key(
        self, game: KalahGame, perspective_player: int
    ) -> tuple[int, tuple[int, ...], tuple[int, int], int]:
        return (
            int(perspective_player),
            tuple(game.pits),
            (int(game.captured_seeds[0]), int(game.captured_seeds[1])),
            int(game.current_player),
        )
