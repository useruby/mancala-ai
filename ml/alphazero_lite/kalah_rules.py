"""Python Kalah rules engine aligned with Rails Games::Kalah behavior."""

from __future__ import annotations

from dataclasses import dataclass


PITS_PER_PLAYER = 6
NUMBER_OF_PLAYERS = 2
TOTAL_PITS = PITS_PER_PLAYER * NUMBER_OF_PLAYERS


@dataclass
class KalahGame:
    pits: list[int]
    captured_seeds: list[int]
    current_player: int
    winner: int | None = None
    _over: bool = False

    @classmethod
    def from_state(cls, state: dict) -> "KalahGame":
        return cls(
            pits=list(state["player_pits"]) + list(state["opponent_pits"]),
            captured_seeds=[int(state["player_store"]), int(state["opponent_store"])],
            current_player=int(state["current_player"]),
        )

    def clone(self) -> "KalahGame":
        return KalahGame(
            pits=self.pits.copy(),
            captured_seeds=self.captured_seeds.copy(),
            current_player=self.current_player,
            winner=self.winner,
            _over=self._over,
        )

    def to_state(self) -> dict:
        return {
            "player_pits": self.pits[:PITS_PER_PLAYER],
            "opponent_pits": self.pits[PITS_PER_PLAYER:],
            "player_store": self.captured_seeds[0],
            "opponent_store": self.captured_seeds[1],
            "current_player": self.current_player,
        }

    def pit_owner(self, pit_index: int) -> int:
        return pit_index // PITS_PER_PLAYER

    def pit_index(self, relative_index: int) -> int:
        return relative_index + (self.current_player * PITS_PER_PLAYER)

    def opposite_player(self) -> int:
        return 1 - self.current_player

    def opposite_pit_index(self, index: int) -> int:
        return TOTAL_PITS - index - 1

    def possible_moves(self) -> list[int]:
        offset = self.current_player * PITS_PER_PLAYER
        return [i for i in range(PITS_PER_PLAYER) if self.pits[offset + i] > 0]

    def over(self) -> bool:
        if self._over:
            return True

        self._over = len(self.possible_moves()) == 0
        return self._over

    def move(self, absolute_index: int) -> bool:
        if not (0 <= absolute_index < TOTAL_PITS):
            return False

        if self.pit_owner(absolute_index) != self.current_player:
            return False

        if self.pits[absolute_index] == 0:
            return False

        pit_seeds = self.pits[absolute_index]
        self.pits[absolute_index] = 0

        last_pit_index, extra_turn = self._seeding(absolute_index, pit_seeds, self.current_player)
        if not extra_turn:
            self._capture(last_pit_index)
            self.current_player = self.opposite_player()

        if self.over():
            self._after_game_over()

        return True

    def _seeding(self, pit_index: int, pit_seeds: int, pit_owner: int) -> tuple[int, bool]:
        extra_turn = False

        for _ in range(pit_seeds):
            next_index = (pit_index + 1) % TOTAL_PITS
            next_owner = self.pit_owner(next_index)

            if pit_owner != next_owner:
                need_take_seed_out = pit_owner == self.current_player
                pit_owner = next_owner

                if need_take_seed_out:
                    self.captured_seeds[self.current_player] += 1
                    extra_turn = True
                    continue

            extra_turn = False
            pit_index = next_index
            self.pits[pit_index] += 1

        return pit_index, extra_turn

    def _capture(self, pit_index: int) -> None:
        if self.pit_owner(pit_index) != self.current_player:
            return

        if self.pits[pit_index] != 1:
            return

        opposite_index = self.opposite_pit_index(pit_index)
        opposite_seeds = self.pits[opposite_index]
        if opposite_seeds == 0:
            return

        self.captured_seeds[self.current_player] += self.pits[pit_index] + opposite_seeds
        self.pits[pit_index] = 0
        self.pits[opposite_index] = 0

    def _after_game_over(self) -> None:
        opposite = self.opposite_player()
        opposite_indexes = range(opposite * PITS_PER_PLAYER, (opposite + 1) * PITS_PER_PLAYER)
        self.captured_seeds[opposite] += sum(self.pits[i] for i in opposite_indexes)

        for i in range(TOTAL_PITS):
            self.pits[i] = 0

        if self.captured_seeds[0] == self.captured_seeds[1]:
            self.winner = None
        else:
            self.winner = 0 if self.captured_seeds[0] > self.captured_seeds[1] else 1
