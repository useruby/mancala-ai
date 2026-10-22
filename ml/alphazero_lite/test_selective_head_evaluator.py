"""Tests for experiment-only selective-head evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.selective_head_evaluator import SelectiveHeadArtifactEvaluator

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"


def _game(state: dict | None = None) -> KalahGame:
    return KalahGame.from_state(
        state
        or {
            "player_pits": [4, 0, 2, 0, 3, 0],
            "opponent_pits": [1, 2, 3, 4, 5, 6],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def test_selective_heads_match_composed_outputs_for_both_players_and_extra_turn() -> (
    None
):
    initializer = arena.ArtifactEvaluator(ARTIFACTS / "initializer")
    e4 = arena.ArtifactEvaluator(ARTIFACTS / "E4")
    reference = arena.ComposedArtifactEvaluator(
        initializer, e4, policy_source="candidate", value_source="current"
    )
    optimized = SelectiveHeadArtifactEvaluator(e4, initializer)
    states = [
        _game().to_state(),
        {**_game().to_state(), "current_player": 1},
        {
            "player_pits": [0, 1, 0, 0, 0, 0],
            "opponent_pits": [0, 2, 0, 1, 0, 0],
            "player_store": 10,
            "opponent_store": 10,
            "current_player": 0,
        },
    ]
    for state in states:
        game = KalahGame.from_state(state)
        expected_policy, expected_value = reference.evaluate(game)
        actual_policy, actual_value = optimized.evaluate(game)
        assert np.array_equal(actual_policy, expected_policy)
        assert (
            np.asarray(actual_value, dtype=np.float32).tobytes()
            == np.asarray(expected_value, dtype=np.float32).tobytes()
        )


def test_only_owned_head_branches_are_computed(monkeypatch: pytest.MonkeyPatch) -> None:
    initializer = arena.ArtifactEvaluator(ARTIFACTS / "initializer")
    e4 = arena.ArtifactEvaluator(ARTIFACTS / "E4")
    optimized = SelectiveHeadArtifactEvaluator(e4, initializer)
    game = _game()
    counts = {"e4_policy": 0, "e4_value": 0, "init_policy": 0, "init_value": 0}

    def wrap(obj: object, attr: str, key: str) -> None:
        original = getattr(obj, attr)

        def counted(*args: object) -> np.ndarray:
            counts[key] += 1
            return original(*args)

        monkeypatch.setattr(obj, attr, counted)

    wrap(e4, "extract_trunk", "e4_policy")
    wrap(initializer, "extract_trunk", "init_value")
    # Poison unused branch parameters: they must never be read.
    e4.w_value_hidden = None
    e4.b_value_hidden = None
    initializer.w_policy_hidden = None
    initializer.b_policy_hidden = None
    optimized.evaluate(game)
    assert counts == {"e4_policy": 1, "e4_value": 0, "init_policy": 0, "init_value": 1}


def test_terminal_state_invokes_neither_trunk() -> None:
    initializer = arena.ArtifactEvaluator(ARTIFACTS / "initializer")
    e4 = arena.ArtifactEvaluator(ARTIFACTS / "E4")
    optimized = SelectiveHeadArtifactEvaluator(e4, initializer)
    initializer.extract_trunk = lambda _game: pytest.fail("initializer called")
    e4.extract_trunk = lambda _game: pytest.fail("E4 called")
    terminal = KalahGame.from_state(
        {
            "player_pits": [0] * 6,
            "opponent_pits": [1] * 6,
            "player_store": 24,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    policy, value = optimized.evaluate(terminal)
    assert np.array_equal(policy, np.zeros(6, dtype=np.float32))
    assert value == 0.0
