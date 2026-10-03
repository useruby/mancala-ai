"""Focused replay, selection, encoding, and evidence-accounting tests."""

from __future__ import annotations

import json

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed398_search_budget_persistence import (
    LEDGER,
    replay_state,
    selection,
)
from ml.alphazero_lite.verify_seed398_search_budget_persistence import verify


def test_replay_preserves_extra_turn_and_absolute_trajectory_encoding() -> None:
    row = next(
        item
        for item in (json.loads(line) for line in LEDGER.read_text().splitlines())
        if item["treatment"] == "SS" and item["trajectory"]
    )
    initial = replay_state({**row, "trajectory": ""}, 0)
    after_first = replay_state(row, 1)
    initial_game = KalahGame.from_state(initial)
    move = int(row["trajectory"].split(",")[0])
    assert initial_game.pit_index(move % 6) == move
    initial_game.move(move)
    assert initial_game.to_state() == after_first
    if initial_game.current_player == KalahGame.from_state(initial).current_player:
        assert not initial_game.over()

    extra_turn = KalahGame.from_state(
        {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    assert extra_turn.move(5)
    assert extra_turn.current_player == 0


def test_selection_is_deterministic_and_uses_legal_relative_actions() -> None:
    first = selection()
    second = selection()
    assert first == second
    assert len(first) == 64
    assert len({row["opening_index"] for row in first}) == 64
    for row in first:
        assert row["ss_action_384"] in row["legal_actions"]
        assert row["ff_action_384"] in row["legal_actions"]


def test_visit_ties_resolve_to_lowest_action_index() -> None:
    import numpy as np

    assert arena.choose_best_move(np.asarray([9, 9, 0, 0, 0, 0]), [0, 1]) == 0


def test_published_probe_bundle_reconciles_snapshots_and_accounting() -> None:
    result = verify()
    assert result["status"] == "verified"
    assert result["sampled_states"] == 64
    assert result["ss_ff_same_at_1536"] + result["persistent_with_margin"] <= 64
