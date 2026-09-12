import random
from unittest import mock

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, Node, PUCT


class FakeGame:
    current_player = 0
    pits = [4] * 12


def search(*, fpu_mode="zero", root_fpu_mode=None):
    return PUCT(
        evaluator=mock.Mock(),
        simulations=1,
        c_puct=1.25,
        rng=random.Random(7),
        fpu_mode=fpu_mode,
        root_fpu_mode=root_fpu_mode,
    )


def node(*, visits=0, value_sum=0.0):
    return Node(game=FakeGame(), visit_count=visits, value_sum=value_sum)


def test_root_fpu_none_preserves_historical_fpu_behavior():
    root, child = node(visits=4, value_sum=2.0), node()
    historical = search(fpu_mode="parent_q")
    isolated_default = search(fpu_mode="parent_q", root_fpu_mode=None)
    historical._active_root = root
    isolated_default._active_root = root

    assert historical._child_q_value(root, child) == 0.5
    assert isolated_default._child_q_value(root, child) == 0.5


def test_root_zero_overrides_only_root_unvisited_children():
    root, root_child = node(visits=4, value_sum=2.0), node()
    internal, internal_child = node(visits=6, value_sum=3.0), node()
    isolated = search(fpu_mode="parent_q", root_fpu_mode="zero")
    isolated._active_root = root

    assert isolated._child_q_value(root, root_child) == 0.0
    assert isolated._child_q_value(internal, internal_child) == 0.5


def test_root_parent_q_is_dynamic_and_visited_children_keep_empirical_q():
    root = node(visits=4, value_sum=2.0)
    unvisited, visited = node(), node(visits=2, value_sum=1.5)
    isolated = search(fpu_mode="zero", root_fpu_mode="parent_q")
    isolated._active_root = root

    assert isolated._child_q_value(root, unvisited) == 0.5
    root.value_sum = 3.0
    assert isolated._child_q_value(root, unvisited) == 0.75
    assert isolated._child_q_value(root, visited) == 0.75


def test_parent_q_and_parent_value_remain_aliases_at_non_root_nodes():
    parent, child = node(visits=4, value_sum=2.0), node()
    assert search(fpu_mode="parent_q")._child_q_value(parent, child) == 0.5
    assert search(fpu_mode="parent_value")._child_q_value(parent, child) == 0.5


def test_root_fpu_trajectory_is_deterministic():
    class ConstantEvaluator(Evaluator):
        def evaluate(self, game):
            del game
            return np.full(6, 1 / 6, dtype=np.float32), 0.25

    game = KalahGame.from_state(
        {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )

    def run_once():
        history = []
        probe = PUCT(
            ConstantEvaluator(),
            8,
            1.25,
            random.Random(42),
            fpu_mode="zero",
            root_fpu_mode="parent_q",
            root_snapshot_checkpoints={1, 2, 4, 8},
            root_backup_history=history,
        )
        probe.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
        return history, probe.root_summary()["root_snapshots"]

    assert run_once() == run_once()
