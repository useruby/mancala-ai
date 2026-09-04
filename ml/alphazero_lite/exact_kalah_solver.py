"""Standalone exact Kalah solver used only by the feasibility preflight.

This module deliberately owns its rules implementation.  ``KalahGame`` is used
by the preflight as an oracle for parity testing, never by the solver itself.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time


PITS_PER_PLAYER = 6
TOTAL_PITS = PITS_PER_PLAYER * 2
EXACT = 0
LOWER = 1
UPPER = 2
MIN_MARGIN = -10_000
MAX_MARGIN = 10_000
PERSIST_BATCH_SIZE = 10_000


class SearchTimeout(RuntimeError):
    """Raised when a requested exact search exceeds its wall-clock limit."""


@dataclass(frozen=True)
class ExactState:
    """Compact immutable state for the independently implemented Kalah rules."""

    pits: tuple[int, ...]
    stores: tuple[int, int]
    current_player: int

    @classmethod
    def from_game_state(cls, state: dict) -> "ExactState":
        return cls(
            pits=tuple(state["player_pits"] + state["opponent_pits"]),
            stores=(int(state["player_store"]), int(state["opponent_store"])),
            current_player=int(state["current_player"]),
        )

    def to_game_state(self) -> dict:
        return {
            "player_pits": list(self.pits[:PITS_PER_PLAYER]),
            "opponent_pits": list(self.pits[PITS_PER_PLAYER:]),
            "player_store": self.stores[0],
            "opponent_store": self.stores[1],
            "current_player": self.current_player,
        }

    def key(self) -> bytes:
        """Return a fixed-width key suitable for the memory and SQLite tables."""
        return bytes((*self.pits, *self.stores, self.current_player))

    def legal_moves(self) -> tuple[int, ...]:
        offset = self.current_player * PITS_PER_PLAYER
        return tuple(i for i in range(PITS_PER_PLAYER) if self.pits[offset + i] > 0)

    def is_terminal(self) -> bool:
        return not self.legal_moves()

    def settled_margin(self) -> int:
        """Return player-zero's final margin, including a pending terminal sweep."""
        scores = list(self.stores)
        scores[0] += sum(self.pits[:PITS_PER_PLAYER])
        scores[1] += sum(self.pits[PITS_PER_PLAYER:])
        return scores[0] - scores[1]

    def play(self, move: int) -> "ExactState":
        """Apply one relative legal move under this repository's Kalah rules."""
        offset = self.current_player * PITS_PER_PLAYER
        absolute_move = offset + move
        if move not in self.legal_moves():
            raise ValueError(f"illegal move {move}")

        pits = list(self.pits)
        stores = list(self.stores)
        seeds = pits[absolute_move]
        pits[absolute_move] = 0
        index = absolute_move
        owner = self.current_player
        extra_turn = False
        for _ in range(seeds):
            next_index = (index + 1) % TOTAL_PITS
            next_owner = next_index // PITS_PER_PLAYER
            if owner != next_owner:
                take_seed_out = owner == self.current_player
                owner = next_owner
                if take_seed_out:
                    stores[self.current_player] += 1
                    extra_turn = True
                    continue
            extra_turn = False
            index = next_index
            pits[index] += 1

        current_player = self.current_player
        if not extra_turn:
            if index // PITS_PER_PLAYER == current_player and pits[index] == 1:
                opposite = TOTAL_PITS - index - 1
                if pits[opposite] > 0:
                    stores[current_player] += pits[index] + pits[opposite]
                    pits[index] = 0
                    pits[opposite] = 0
            current_player = 1 - current_player

        next_state = ExactState(tuple(pits), (stores[0], stores[1]), current_player)
        if next_state.is_terminal():
            pits = list(next_state.pits)
            stores = list(next_state.stores)
            opposite = 1 - current_player
            start = opposite * PITS_PER_PLAYER
            stores[opposite] += sum(pits[start : start + PITS_PER_PLAYER])
            return ExactState((0,) * TOTAL_PITS, (stores[0], stores[1]), current_player)
        return next_state


class ExactKalahSolver:
    """Full-depth alpha-beta solver for exact player-zero final margins."""

    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        tt_size: int = 500_000,
        move_order: str = "ascending",
        cache_enabled: bool = True,
    ) -> None:
        if move_order not in {"ascending", "descending"}:
            raise ValueError("move_order must be ascending or descending")
        self.tt_size = tt_size
        self.move_order = move_order
        self.cache_enabled = cache_enabled
        self._tt: OrderedDict[bytes, tuple[int, int]] = OrderedDict()
        self._pending_cache: dict[bytes, int] = {}
        self._connection: sqlite3.Connection | None = None
        self.nodes = 0
        self.cache_hits = 0
        self._deadline: float | None = None
        if cache_enabled and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(cache_path)
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS exact_values (state BLOB PRIMARY KEY, margin INTEGER NOT NULL) WITHOUT ROWID"
            )

    def close(self) -> None:
        self.flush_cache()
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def flush_cache(self) -> None:
        if self._connection is not None and self._pending_cache:
            self._connection.executemany(
                "INSERT OR IGNORE INTO exact_values(state, margin) VALUES (?, ?)",
                self._pending_cache.items(),
            )
            self._connection.commit()
            self._pending_cache.clear()

    def solve(
        self, state: ExactState, *, time_limit_seconds: float | None = None
    ) -> int:
        """Return the exact final player-zero store margin or raise SearchTimeout."""
        self.nodes = 0
        self.cache_hits = 0
        self._deadline = (
            None
            if time_limit_seconds is None
            else time.monotonic() + time_limit_seconds
        )
        try:
            return self._search(state, MIN_MARGIN, MAX_MARGIN)
        finally:
            self._deadline = None
            self.flush_cache()

    def action_margins(
        self, state: ExactState, *, time_limit_seconds: float | None = None
    ) -> dict[int, int]:
        """Return an independently exact final margin for every legal action."""
        self.nodes = 0
        self.cache_hits = 0
        self._deadline = (
            None
            if time_limit_seconds is None
            else time.monotonic() + time_limit_seconds
        )
        try:
            return {
                move: self._search(state.play(move), MIN_MARGIN, MAX_MARGIN)
                for move in self._ordered_moves(state)
            }
        finally:
            self._deadline = None
            self.flush_cache()

    def _search(self, state: ExactState, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise SearchTimeout()
        if state.is_terminal():
            return state.settled_margin()

        key = state.key()
        original_alpha, original_beta = alpha, beta
        entry = self._lookup(key)
        if entry is not None:
            value, bound = entry
            if bound == EXACT:
                return value
            if bound == LOWER:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value

        maximizing = state.current_player == 0
        value = MIN_MARGIN if maximizing else MAX_MARGIN
        for move in self._ordered_moves(state):
            child_value = self._search(state.play(move), alpha, beta)
            if maximizing:
                value = max(value, child_value)
                alpha = max(alpha, value)
            else:
                value = min(value, child_value)
                beta = min(beta, value)
            if alpha >= beta:
                break

        bound = EXACT
        if value <= original_alpha:
            bound = UPPER
        elif value >= original_beta:
            bound = LOWER
        self._store(key, value, bound)
        return value

    def _ordered_moves(self, state: ExactState) -> tuple[int, ...]:
        moves = state.legal_moves()
        return moves if self.move_order == "ascending" else tuple(reversed(moves))

    def _lookup(self, key: bytes) -> tuple[int, int] | None:
        if not self.cache_enabled:
            return None
        entry = self._tt.get(key)
        if entry is not None:
            self._tt.move_to_end(key)
            self.cache_hits += 1
            return entry
        if self._connection is not None:
            row = self._connection.execute(
                "SELECT margin FROM exact_values WHERE state = ?", (key,)
            ).fetchone()
            if row is not None:
                self.cache_hits += 1
                entry = (int(row[0]), EXACT)
                self._store_memory(key, entry)
                return entry
        return None

    def _store(self, key: bytes, value: int, bound: int) -> None:
        if not self.cache_enabled:
            return
        self._store_memory(key, (value, bound))
        if bound == EXACT and self._connection is not None:
            self._pending_cache[key] = value
            if len(self._pending_cache) >= PERSIST_BATCH_SIZE:
                self.flush_cache()

    def _store_memory(self, key: bytes, entry: tuple[int, int]) -> None:
        self._tt[key] = entry
        self._tt.move_to_end(key)
        if len(self._tt) > self.tt_size:
            self._tt.popitem(last=False)
