from __future__ import annotations

import json
import random

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_fresh_p1_context_action_q_trajectory_probe import (
    EVENT_SIZE,
    HISTORY_LENGTH,
    _history_vector,
    _trajectory_arrays,
)
from ml.alphazero_lite.self_play import Evaluator, PUCT


class FixedEvaluator(Evaluator):
    def evaluate(self, _game: KalahGame) -> tuple[np.ndarray, float]:
        return np.asarray([0.31, 0.19, 0.17, 0.13, 0.11, 0.09], np.float32), 0.2


def _game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def test_history_vector_is_oldest_first_and_left_padded() -> None:
    vector = _history_vector(
        [
            {"simulation": 1, "action": 2, "root_value": 0.25},
            {"simulation": 2, "action": 5, "root_value": -0.5},
        ],
        include_values=True,
    ).reshape(HISTORY_LENGTH, EVENT_SIZE)

    assert not vector[:-2].any()
    assert np.array_equal(vector[-2, :6], [0, 0, 1, 0, 0, 0])
    assert vector[-2, 6] == 0.25
    assert np.array_equal(vector[-1, :6], [0, 0, 0, 0, 0, 1])
    assert vector[-1, 6] == -0.5


def test_trajectory_excludes_the_current_simulation_backup(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    history = [
        {"simulation": index, "action": index % 6, "root_value": float(index)}
        for index in range(1, 1201)
    ]
    path.write_text(
        json.dumps({"state_index": 0, "state_hash": "x", "history": history}) + "\n"
    )

    actions, values = _trajectory_arrays(path, 1)

    assert actions[0, 0, -1] == 255
    assert actions[0, 1, -1] == 1
    assert values[0, 1, -1] == 1.0
    assert actions[0, 32, 0] == 1
    assert values[0, 32, 0] == 1.0


def test_live_preselection_history_matches_offline_history_features() -> None:
    completed: list[dict[str, int | float]] = []
    observed: list[np.ndarray] = []

    def capture(_simulation: int, _root: object) -> None:
        observed.append(_history_vector(completed, include_values=True))

    search = PUCT(
        FixedEvaluator(),
        48,
        1.25,
        random.Random(17),
        pre_simulation_hook=capture,
        root_backup_history=completed,
    )
    search.run(_game())

    assert len(completed) == len(observed) == 48
    for simulation, vector in enumerate(observed):
        assert np.array_equal(
            vector,
            _history_vector(completed[:simulation], include_values=True),
        )
