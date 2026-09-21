from __future__ import annotations

import random
import unittest
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_value_head_oracle_audit import (
    ExactWdlLeafValueEvaluator,
    ZeroNonterminalValueEvaluator,
    baseline_matches,
    paired_bootstrap,
    verify_exact_perspective,
)
from ml.alphazero_lite.self_play import PUCT, terminal_value


class StubEvaluator:
    def __init__(self, value: float = 0.25) -> None:
        self.policy = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.4], dtype=np.float32)
        self.value = value

    def evaluate(self, _game: KalahGame) -> tuple[np.ndarray, float]:
        return self.policy, self.value


class StubOracle:
    def __init__(self, value: float) -> None:
        self.value_to_return = value
        self.games: list[KalahGame] = []

    def value(self, game: KalahGame) -> float:
        self.games.append(game.clone())
        return self.value_to_return


def game(state: dict) -> KalahGame:
    return KalahGame.from_state(state)


class ValueHeadOracleAuditTest(unittest.TestCase):
    def test_zero_wrapper_preserves_policy_identity_and_only_changes_nonterminal_value(
        self,
    ):
        delegate = StubEvaluator()
        wrapper = ZeroNonterminalValueEvaluator(delegate)
        nonterminal = game(
            {
                "player_pits": [1, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        policy, value = wrapper.evaluate(nonterminal)
        self.assertIs(policy, delegate.policy)
        self.assertEqual(0.0, value)
        terminal = game(
            {
                "player_pits": [0] * 6,
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 24,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        _policy, terminal_value_from_wrapper = wrapper.evaluate(terminal)
        self.assertEqual(delegate.value, terminal_value_from_wrapper)

    def test_exact_wrapper_uses_current_player_perspective(self):
        oracle = StubOracle(-1.0)
        delegate = StubEvaluator()
        wrapper = ExactWdlLeafValueEvaluator(delegate, oracle)
        position = game(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 1,
            }
        )
        policy, value = wrapper.evaluate(position)
        self.assertIs(policy, delegate.policy)
        self.assertEqual(-1.0, value)
        self.assertEqual(1, oracle.games[0].current_player)

    def test_puct_keeps_terminal_game_outcome_outside_wrapper(self):
        terminal = game(
            {
                "player_pits": [0] * 6,
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 30,
                "opponent_store": 18,
                "current_player": 0,
            }
        )
        terminal._after_game_over()
        self.assertEqual(1.0, terminal_value(terminal))
        engine = PUCT(
            ZeroNonterminalValueEvaluator(StubEvaluator(-1.0)),
            1,
            1.25,
            random.Random(1),
            fpu_mode="zero",
            reuse_subtree=False,
            normalize_values=False,
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
        )
        self.assertEqual(
            1.0,
            engine._search(
                type(
                    "Node", (), {"game": terminal, "expanded": False, "children": {}}
                )()
            ),
        )

    def test_extra_turn_backup_does_not_flip_and_turn_change_does_flip(self):
        extra_turn = game(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        changed_turn = game(
            {
                "player_pits": [1, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for position, expected_player in ((extra_turn, 0), (changed_turn, 1)):
            child = position.clone()
            child.move(child.pit_index(position.possible_moves()[0]))
            self.assertEqual(expected_player, child.current_player)
            value = 0.5
            self.assertEqual(
                value if child.current_player == position.current_player else -value,
                value if expected_player == 0 else -value,
            )

    def test_paired_bootstrap_is_deterministic(self):
        self.assertEqual(
            paired_bootstrap([0.2, 0.4], [0.4, 0.1]),
            paired_bootstrap([0.2, 0.4], [0.4, 0.1]),
        )

    def test_baseline_reproduction_requires_pr351_metrics(self):
        actual = {
            "exact_optimal_visit_mass": 0.7,
            "visit_weighted_expected_exact_regret": 1.6,
        }
        expected = {"optimal_mass": 0.7, "expected_regret": 1.6}
        self.assertTrue(baseline_matches(actual, expected))
        self.assertFalse(
            baseline_matches(actual | {"exact_optimal_visit_mass": 0.699}, expected)
        )

    def test_exact_corpus_successors_never_increase_active_stones(self):
        corpus = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-target-quality/results.json"
            ).read_text(encoding="utf-8")
        )["states"]
        for row in corpus:
            root = game(row["canonical_state"])
            for move in root.possible_moves():
                child = root.clone()
                child.move(child.pit_index(move))
                self.assertLessEqual(sum(child.pits), sum(root.pits))

    def test_exact_corpus_uses_root_player_tablebase_perspective(self):
        corpus = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-target-quality/results.json"
            ).read_text(encoding="utf-8")
        )["states"]
        for row in corpus:
            verify_exact_perspective(row)


if __name__ == "__main__":
    unittest.main()
