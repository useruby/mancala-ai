from __future__ import annotations

import random

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, Node, PUCT


class FixedEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy = np.zeros(6, dtype=np.float32)
        legal = game.possible_moves()
        for move in legal:
            policy[move] = 1.0 / len(legal)
        return policy, 0.4


def _game() -> KalahGame:
    return KalahGame([1, 1] + [0] * 4 + [1] + [0] * 5, [0, 0], 0)


def _search(
    *, simulations: int, override=None, trace=None
) -> tuple[np.ndarray, Node, PUCT]:
    search = PUCT(
        FixedEvaluator(),
        simulations,
        1.25,
        random.Random(7),
        backup_override=override,
        selection_trace=trace,
    )
    visits, root = search.run(_game(), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return visits, root, search


def test_backup_override_noop_is_exactly_normal() -> None:
    visits, root, search = _search(simulations=16)
    overridden_visits, overridden_root, overridden_search = _search(
        simulations=16, override=lambda _index, raw, _trace: raw
    )

    assert np.array_equal(overridden_visits, visits)
    assert overridden_root.value_sum == root.value_sum
    assert overridden_search.root_summary() == search.root_summary()


def test_zero_backup_override_preserves_visits_and_corrects_values() -> None:
    baseline_visits, baseline_root, _ = _search(simulations=1)
    trace: list[dict] = []
    visits, root, _ = _search(simulations=1, override=lambda *_: 0.0, trace=trace)

    assert np.array_equal(visits, baseline_visits)
    assert root.visit_count == baseline_root.visit_count == 1
    assert root.value_sum == 0.0
    for move, child in root.children.items():
        assert child.visit_count == baseline_root.children[move].visit_count
        assert child.value_sum == 0.0
    assert trace[0]["backed_up_value"] == 0.0
    assert trace[0]["raw_backed_up_value"] == baseline_root.value_sum
    assert trace[0]["backup_override_active"] is True


def test_backup_delta_uses_parent_player_identity_for_extra_turns() -> None:
    root_player = 0
    root = Node(KalahGame([1] + [0] * 11, [0, 0], root_player))
    extra_turn_parent = Node(KalahGame([1] + [0] * 11, [0, 0], root_player))
    switched_parent = Node(KalahGame([1] + [0] * 11, [0, 0], 1))
    extra_turn_child = Node(extra_turn_parent.game.clone(), value_sum=2.0)
    switched_child = Node(switched_parent.game.clone(), value_sum=2.0)

    PUCT._apply_backup_delta(
        [(extra_turn_parent, extra_turn_child), (switched_parent, switched_child)],
        root_player=root.game.current_player,
        delta_root=0.25,
    )

    assert extra_turn_child.value_sum == 2.25
    assert switched_child.value_sum == 1.75


def test_override_trace_is_absent_without_hook_and_explicit_with_hook() -> None:
    normal_trace: list[dict] = []
    override_trace: list[dict] = []
    _search(simulations=1, trace=normal_trace)
    _search(
        simulations=1, override=lambda _index, raw, _trace: raw, trace=override_trace
    )

    assert "raw_backed_up_value" not in normal_trace[0]
    assert normal_trace[0]["backed_up_value"] == override_trace[0]["backed_up_value"]
    assert (
        override_trace[0]["raw_backed_up_value"] == override_trace[0]["backed_up_value"]
    )
    assert override_trace[0]["backup_override_delta"] == 0.0


def test_override_cannot_change_current_simulation_selection_path() -> None:
    baseline_trace: list[dict] = []
    override_trace: list[dict] = []
    _search(simulations=1, trace=baseline_trace)
    _search(
        simulations=1,
        override=lambda _index, raw, _trace: raw + 10.0,
        trace=override_trace,
    )

    assert override_trace[0]["selection_path"] == baseline_trace[0]["selection_path"]
