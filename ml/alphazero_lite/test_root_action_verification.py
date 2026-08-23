from __future__ import annotations

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.root_action_verification import root_perspective_value


def _after(state: dict, action: int) -> tuple[KalahGame, int]:
    game = KalahGame.from_state(state)
    root_player = game.current_player
    assert game.move(game.pit_index(action))
    return game, root_player


def test_perspective_for_ordinary_player_switch() -> None:
    child, root_player = _after(
        {
            "player_pits": [1, 1, 0, 0, 0, 0],
            "opponent_pits": [1, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
        0,
    )
    assert child.current_player != root_player
    assert (
        root_perspective_value(
            0.25, root_player=root_player, child_player=child.current_player
        )
        == -0.25
    )


def test_perspective_for_extra_turn() -> None:
    child, root_player = _after(
        {
            "player_pits": [6, 1, 0, 0, 0, 0],
            "opponent_pits": [1, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
        0,
    )
    assert child.current_player == root_player
    assert (
        root_perspective_value(
            0.25, root_player=root_player, child_player=child.current_player
        )
        == 0.25
    )


def test_perspective_for_terminal_move_uses_exact_outcome() -> None:
    child, root_player = _after(
        {
            "player_pits": [1, 0, 0, 0, 0, 0],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 5,
            "opponent_store": 0,
            "current_player": 0,
        },
        0,
    )
    exact_child_value = 1.0 if child.winner == child.current_player else -1.0
    assert exact_child_value == -1.0  # Player 0 wins from player 1's child view.
    assert (
        root_perspective_value(
            exact_child_value,
            root_player=root_player,
            child_player=child.current_player,
        )
        == 1.0
    )
