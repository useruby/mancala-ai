from __future__ import annotations

from unittest.mock import patch

import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed398_paired_first_action import play, validate_outcome


def test_relative_action_maps_to_absolute_pit_and_preserves_extra_turn():
    game = KalahGame.from_state(
        {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    assert game.move(game.pit_index(5))
    assert game.current_player == 0
    assert game.captured_seeds[0] == 1


def test_forced_action_is_recorded_as_relative_and_absolute():
    state = {
        "player_pits": [0, 0, 0, 0, 0, 0],
        "opponent_pits": [1, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    row = {
        "state": state,
        "opening_index": 1,
        "state_hash": "hash",
        "ss_action_384": 0,
        "ff_action_384": 0,
    }
    outcome = play(
        row, "SS", 384, evaluator=None, endgame_tablebase=object()
    )  # terminal after forced move
    assert outcome["trajectory"][0]["action_relative"] == 0
    assert outcome["trajectory"][0]["action_absolute"] == 6
    assert outcome["score"] == 1.0
    assert outcome["store_margin_root_perspective"] == 1
    validate_outcome(outcome, row)
    corrupted = dict(outcome, score=0.0)
    with pytest.raises(ValueError, match="terminal_accounting"):
        validate_outcome(corrupted, row)


def test_continuation_passes_explicit_native_backend_to_root_solver():
    state = {
        "player_pits": [1, 0, 0, 0, 0, 0],
        "opponent_pits": [15, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    row = {
        "state": state,
        "opening_index": 2,
        "state_hash": "root",
        "ss_action_384": 0,
        "ff_action_384": 0,
    }
    native = object()
    seen = {}

    def stop_at_first_continuation(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("captured native handoff")

    with patch.object(arena, "evaluate_artifact_position", stop_at_first_continuation):
        with pytest.raises(RuntimeError, match="captured native handoff"):
            play(row, "SS", 1536, evaluator=object(), endgame_tablebase=native)
    assert seen["endgame_tablebase"] is native
    assert seen["exact_root_solve_threshold"] == 16


def test_seed_context_has_no_branch_identity():
    common = {
        "contract_version": "azlite_eval_seed_v2",
        "base_seed": 406,
        "suite_sha256": "suite",
        "opening_index": 3,
        "opening_state_hash": "opening",
        "challenger_player": 0,
        "game_within_opening": 0,
        "ply": 4,
        "canonical_current_state_hash": "state",
        "acting_role": "current",
    }
    left = dict(common, action="SS")
    right = dict(common, action="FF")
    left.pop("action")
    right.pop("action")
    assert derive_search_seed(**left) == derive_search_seed(**right)


def test_branch_seed_identity_is_determined_only_by_search_context():
    # In the runner, action labels are absent from the context dictionary;
    # these equal context dictionaries therefore yield equal coupled seeds.
    context = {
        "contract_version": "azlite_eval_seed_v2",
        "base_seed": 406,
        "suite_sha256": "suite",
        "opening_index": 7,
        "opening_state_hash": "opening",
        "challenger_player": 1,
        "game_within_opening": 0,
        "ply": 2,
        "canonical_current_state_hash": "same-state",
        "acting_role": "challenger",
    }
    assert derive_search_seed(**context) == derive_search_seed(**context)
