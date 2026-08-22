from __future__ import annotations

import random
import math

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, Node, PUCT


class FixedEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        priors = np.zeros(6, dtype=np.float32)
        for move in game.possible_moves():
            priors[move] = 1.0 / len(game.possible_moves())
        return priors, 0.25


def _game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [2, 2, 0, 0, 0, 0],
            "opponent_pits": [2, 2, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def _run(override=None, observer=None):
    trace: list[dict] = []
    search = PUCT(
        FixedEvaluator(),
        12,
        1.25,
        random.Random(8),
        selection_q_override=override,
        pre_simulation_hook=observer,
        selection_trace=trace,
    )
    visits, root = search.run(_game(), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return visits, root, trace


def test_selection_q_override_absent_and_identity_are_exact_noops() -> None:
    visits, root, trace = _run()
    identity_visits, identity_root, identity_trace = _run(
        lambda _t, _state, _move, q, _visits: q
    )

    assert np.array_equal(identity_visits, visits)
    assert identity_root.value_sum == root.value_sum
    assert [row["selection_path"] for row in identity_trace] == [
        row["selection_path"] for row in trace
    ] or all(
        [entry["selection_score"] for entry in identity_node["children"]]
        == [entry["selection_score"] for entry in baseline_node["children"]]
        for identity_row, baseline_row in zip(identity_trace, trace, strict=True)
        for identity_node, baseline_node in zip(
            identity_row["selection_path"], baseline_row["selection_path"], strict=True
        )
    )


def test_selection_q_override_changes_only_selection_q_component() -> None:
    parent = Node(_game(), visit_count=4)
    child = Node(_game(), prior=0.4, visit_count=2, value_sum=0.5)
    unvisited = Node(_game(), prior=0.6)
    parent.children = {0: child, 1: unvisited}
    search = PUCT(
        FixedEvaluator(), 1, 1.25, random.Random(1), selection_q_override=lambda *_: 0.9
    )
    search._active_simulation_index = 1
    entries, _move, _child, _trust = search._selection_entries(parent, sort_moves=True)

    visited, fpu = entries
    assert visited["stored_q_value"] == child.q_value == 0.25
    assert visited["selection_q_value"] == 0.9
    assert visited["prior"] == child.prior
    assert visited["u_component"] == 1.25 * child.prior * math.sqrt(2) / 3
    assert fpu["selection_q_overridden"] is False
    assert fpu["selection_q_value"] == 0.0
    assert child.value_sum == 0.5 and child.visit_count == 2


def test_pre_simulation_observer_never_sees_current_simulation_evidence() -> None:
    observed: list[tuple[int, int]] = []
    _run(
        observer=lambda simulation, root: observed.append(
            (simulation, root.visit_count)
        )
    )

    assert observed == [(simulation, simulation - 1) for simulation in range(1, 13)]


def test_override_does_not_mutate_statistics_outside_normal_backup() -> None:
    calls: list[tuple[int, float, int]] = []
    _visits, root, trace = _run(
        lambda simulation, _state, _move, q, visits: (
            calls.append((simulation, q, visits)) or q
        )
    )

    assert calls
    assert root.visit_count == 12
    assert all(
        entry["stored_q_value"] == entry["q_value"]
        for row in trace
        for node in row["selection_path"]
        for entry in node["children"]
    )
