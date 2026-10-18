"""Tests for replaying arena-written absolute pit-index trajectories."""

from __future__ import annotations

import pytest

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed488_opening_replay import replay_game


def test_replay_uses_arena_absolute_pit_indices_and_both_seats():
    opening = {
        "prefix_moves": [4, 5, 2, 4, 5],
    }
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in opening["prefix_moves"]:
        assert game.move(game.pit_index(move))
    opening["state"] = game.to_state()

    for seat in (0, 1):
        replay = KalahGame.from_state(opening["state"])
        moves = []
        first_moves = {}
        while not replay.over():
            relative = replay.possible_moves()[0]
            absolute = replay.pit_index(relative)
            first_moves.setdefault(replay.current_player, relative)
            moves.append(absolute)
            assert replay.move(absolute)
        margin = replay.captured_seeds[seat] - replay.captured_seeds[1 - seat]
        winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
        row = {
            "trajectory": ",".join(map(str, moves)),
            "game_length": len(moves),
            "winner": winner,
            "margin": margin,
            "first_move_challenger": first_moves[seat],
            "first_move_current": first_moves[1 - seat],
        }
        assert replay_game(opening, row, seat) == {
            "winner": winner,
            "margin": margin,
            "game_length": len(moves),
        }


def test_replay_rejects_relative_index_in_absolute_trajectory():
    prefix = [4, 5, 2, 4, 5]
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in prefix:
        assert game.move(game.pit_index(move))
    opening = {"prefix_moves": prefix, "state": game.to_state()}
    row = {
        "trajectory": "0",
        "game_length": 1,
        "winner": "draw",
        "margin": 0,
        "first_move_challenger": 0,
        "first_move_current": 0,
    }
    with pytest.raises(ValueError, match="trajectory_illegal:0"):
        replay_game(opening, row, 0)


def test_old_relative_trajectory_interpretation_rejects_arena_absolute_writer_value():
    prefix = [0, 1, 2, 3, 4]
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for relative in prefix:
        assert game.move(game.pit_index(relative))
    absolute = game.pit_index(game.possible_moves()[0])
    assert absolute >= 6

    def old_relative_validator(move: int) -> None:
        if game.over() or move not in game.possible_moves():
            raise ValueError("trajectory_illegal:0")

    with pytest.raises(ValueError, match="trajectory_illegal:0"):
        old_relative_validator(absolute)
