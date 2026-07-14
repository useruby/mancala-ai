"""Focused invariants for terminal-outcome replay-balance diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_terminal_outcome_replay_balance_audit import (
    HIDDEN_SIZES,
    INPUT_ENCODING,
    MODEL_TYPE,
    apply_relative_move,
    deterministic_replay_indexes,
    freeze_value_head,
    swap_move,
    swap_state,
    swapped_winner,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


class TerminalOutcomeReplayBalanceTest(unittest.TestCase):
    state = {
        "player_pits": [1, 2, 0, 4, 1, 3],
        "opponent_pits": [4, 0, 2, 1, 5, 1],
        "player_store": 9,
        "opponent_store": 15,
        "current_player": 0,
    }

    def test_player_swap_is_involution_and_maps_legal_moves(self) -> None:
        transformed = swap_state(self.state)
        self.assertEqual(self.state, swap_state(transformed))
        legal = KalahGame.from_state(self.state).possible_moves()
        transformed_legal = KalahGame.from_state(transformed).possible_moves()
        self.assertEqual(sorted(swap_move(move) for move in legal), transformed_legal)

    def test_move_transition_commutes_with_player_swap(self) -> None:
        for move in KalahGame.from_state(self.state).possible_moves():
            self.assertEqual(
                swap_state(apply_relative_move(self.state, move)),
                apply_relative_move(swap_state(self.state), swap_move(move)),
            )

    def test_terminal_winner_margin_and_target_are_antisymmetric(self) -> None:
        terminal = {
            "player_pits": [0, 0, 0, 0, 0, 0],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 30,
            "opponent_store": 18,
            "current_player": 0,
        }
        swapped = swap_state(terminal)
        self.assertEqual(1, KalahGame.from_state(swapped).current_player)
        self.assertEqual(-12, swapped["player_store"] - swapped["opponent_store"])
        self.assertEqual(1, swapped_winner(0))
        self.assertEqual(0, swapped_winner(1))
        self.assertIsNone(swapped_winner(None))
        target = 1.0
        self.assertEqual(-target, -1.0)  # z(T(s)) = -z(s) for player-0 win.

    def test_encoded_transform_has_expected_base_feature_mapping(self) -> None:
        original = np.asarray(encode_state(self.state, input_encoding=INPUT_ENCODING))
        transformed = np.asarray(
            encode_state(swap_state(self.state), input_encoding=INPUT_ENCODING)
        )
        expected = original[[*range(6, 12), *range(6), 13, 12, 14]]
        expected[-1] = 1.0 - expected[-1]
        self.assertTrue(np.allclose(transformed[:15], expected))

    def test_game_balanced_sampler_is_deterministic_and_uniform_per_game(self) -> None:
        rows = []
        for game, count in enumerate((1, 3, 5)):
            for row in range(count):
                rows.append({"game_index": game, "player": row % 2, "value": 1.0})
        first = deterministic_replay_indexes(rows, "game_balanced", seed=42, epochs=30)
        second = deterministic_replay_indexes(rows, "game_balanced", seed=42, epochs=30)
        self.assertTrue(np.array_equal(first, second))
        games = [rows[index]["game_index"] for index in first]
        counts = [games.count(game) for game in range(3)]
        self.assertLessEqual(max(counts) - min(counts), 25)

    def test_seat_outcome_sampler_covers_all_available_strata(self) -> None:
        rows = [
            {"game_index": 0, "player": 0, "value": 1.0},
            {"game_index": 1, "player": 0, "value": -1.0},
            {"game_index": 2, "player": 1, "value": 1.0},
            {"game_index": 3, "player": 1, "value": -1.0},
        ]
        indexes = deterministic_replay_indexes(
            rows, "seat_outcome_balanced", seed=7, epochs=20
        )
        strata = {(rows[index]["player"], rows[index]["value"]) for index in indexes}
        self.assertEqual({(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)}, strata)

    def test_only_value_head_is_trainable(self) -> None:
        model = PolicyValueNet(
            HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
        )
        freeze_value_head(model)
        trainable = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            {
                "value_hidden_layer.weight",
                "value_hidden_layer.bias",
                "value_head.weight",
                "value_head.bias",
            },
            trainable,
        )


if __name__ == "__main__":
    unittest.main()
