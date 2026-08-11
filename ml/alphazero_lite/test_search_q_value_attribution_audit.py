"""Perspective and provenance contracts for the search-Q attribution audit."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_search_q_value_attribution_audit import (
    source_provenance,
)
from ml.alphazero_lite.run_denoised_puct_convergence_audit import (
    assert_distinct_source_domains,
    forced_continuation,
)
from ml.alphazero_lite.self_play import Evaluator, Node, PUCT


class KnownValueEvaluator(Evaluator):
    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.values = values or {}

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy = np.zeros(6, dtype=np.float32)
        legal = game.possible_moves()
        for move in legal:
            policy[move] = 1.0 / len(legal)
        return policy, self.values.get(json.dumps(game.to_state(), sort_keys=True), 0.0)


def search(evaluator: Evaluator) -> PUCT:
    return PUCT(evaluator=evaluator, simulations=1, c_puct=1.25, rng=random.Random(1))


class SearchQValueAttributionAuditTest(unittest.TestCase):
    def test_duplicate_source_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(
                '{"state":{"player_pits":[1,0,0,0,0,0],"opponent_pits":[1,0,0,0,0,0],"player_store":0,"opponent_store":0,"current_player":0}}\n'
            )
            with self.assertRaisesRegex(RuntimeError, "same file hash"):
                source_provenance(
                    {
                        "pr176": (path, "PR #176", "standard"),
                        "pr177": (path, "PR #177", "opening"),
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "same file hash"):
                assert_distinct_source_domains({"pr176": path, "pr177": path})

    def test_turn_switch_is_negated_into_parent_action_q(self) -> None:
        parent = Node(KalahGame([1] + [0] * 11, [0, 0], 0), expanded=True)
        child = Node(KalahGame([0] * 6 + [1] + [0] * 5, [0, 0], 1))
        # A direct non-zero leaf makes the perspective assertion unambiguous.
        evaluator = KnownValueEvaluator(
            {json.dumps(child.game.to_state(), sort_keys=True): 0.6}
        )
        parent = Node(parent.game, expanded=True)
        parent.children[0] = Node(child.game)
        self.assertAlmostEqual(-0.6, search(evaluator)._search(parent))
        self.assertAlmostEqual(-0.6, parent.children[0].q_value)

    def test_extra_turn_is_not_negated_into_parent_action_q(self) -> None:
        parent = Node(KalahGame([1] + [0] * 11, [0, 0], 0), expanded=True)
        child = Node(KalahGame([0, 1] + [0] * 10, [0, 0], 0))
        parent.children[0] = child
        evaluator = KnownValueEvaluator(
            {json.dumps(child.game.to_state(), sort_keys=True): 0.6}
        )
        self.assertAlmostEqual(0.6, search(evaluator)._search(parent))
        self.assertAlmostEqual(0.6, parent.children[0].q_value)

    def test_terminal_child_q_has_root_player_sign(self) -> None:
        parent = Node(KalahGame([1] + [0] * 11, [0, 0], 0), expanded=True)
        # Player 1 is to move but player 0 has already won; terminal_value is -1
        # in child perspective and must become +1 for the root action.
        terminal_child = Node(KalahGame([0] * 12, [25, 23], 1, winner=0, _over=True))
        parent.children[0] = terminal_child
        self.assertEqual(1.0, search(KnownValueEvaluator())._search(parent))
        self.assertEqual(1.0, terminal_child.q_value)

    def test_forced_continuation_outcome_uses_root_player_perspective(self) -> None:
        state = {
            "player_pits": [1, 0, 0, 0, 0, 0],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        result = forced_continuation(
            evaluator=KnownValueEvaluator(),
            task={
                "state": state,
                "state_hash": "terminal",
                "continuation_budget": 768,
                "experiment_seed": 42,
            },
            forced_move=0,
        )
        self.assertEqual(1.0, result["outcome_root"])
        self.assertGreater(result["store_margin_root"], 0)

    def test_synthetic_evaluator_produces_expected_child_q_ranking(self) -> None:
        root = KalahGame([1, 1] + [0] * 4 + [1] + [0] * 5, [0, 0], 0)
        child_values = {}
        for move, root_value in ((0, 0.2), (1, 0.8)):
            child = root.clone()
            child.move(child.pit_index(move))
            # Both moves switch turns, so evaluator values are child-player values.
            child_values[json.dumps(child.to_state(), sort_keys=True)] = -root_value
        engine = PUCT(
            KnownValueEvaluator(child_values),
            simulations=2,
            c_puct=1.25,
            rng=random.Random(1),
        )
        _visits, searched_root = engine.run(
            root, dirichlet_alpha=None, dirichlet_epsilon=0.0
        )
        self.assertGreater(
            searched_root.children[1].q_value, searched_root.children[0].q_value
        )
        self.assertAlmostEqual(0.8, searched_root.children[1].q_value)
        self.assertAlmostEqual(0.2, searched_root.children[0].q_value)


if __name__ == "__main__":
    unittest.main()
