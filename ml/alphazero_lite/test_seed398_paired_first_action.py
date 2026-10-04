from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import seed398_paired_first_action as diagnostic
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


def test_default_analysis_uses_complete_native_ledger_with_fallback_present():
    report = diagnostic.analyze()
    assert len(report["paired_matrix"]) == 64
    assert report["primary"]["mean_delta"] == -0.109375
    assert report["primary"]["paired_bootstrap_95_interval"] == [
        -0.2109375,
        -0.015625,
    ]
    assert report["secondary"]["mean_delta"] == -0.046875
    assert report["classification"] == "SS-action advantage"


@pytest.fixture
def analysis_workspace(tmp_path, monkeypatch):
    out = tmp_path / "publication"
    out.mkdir()
    source_out = diagnostic.OUT
    (out / "registration.json").write_bytes(
        (source_out / "registration.json").read_bytes()
    )
    (out / "outcomes.jsonl").write_bytes((source_out / "outcomes.jsonl").read_bytes())
    ledger = out / "native-outcomes.jsonl"
    rows = [
        json.loads(line)
        for line in (source_out / "native-outcomes.jsonl").read_text().splitlines()
    ]
    monkeypatch.setattr(diagnostic, "OUT", out)
    monkeypatch.setattr(diagnostic, "NATIVE_LEDGER", ledger)
    return ledger, rows


def test_analysis_rejects_incomplete_native_ledger(analysis_workspace):
    ledger, rows = analysis_workspace
    ledger.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="outcome_ledger_count_mismatch"):
        diagnostic.analyze()


def test_analysis_rejects_duplicate_native_evidence(analysis_workspace):
    ledger, rows = analysis_workspace
    rows.append(dict(rows[0]))
    ledger.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="duplicate_outcome"):
        diagnostic.analyze()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trajectory", "forced", "forced_action"),
        ("trajectory", "backend", "native_backend"),
        ("score", 0.25, "terminal_accounting"),
    ],
)
def test_analysis_rejects_invalid_native_evidence(
    analysis_workspace, field, value, message
):
    ledger, rows = analysis_workspace
    row = rows[0]
    if field == "trajectory" and value == "forced":
        row["trajectory"][0]["action_relative"] = (
            row["trajectory"][0]["action_relative"] + 1
        ) % 6
    elif field == "trajectory":
        continuation = next(
            move for move in row["trajectory"][1:] if move["exact_root_backend"]
        )
        continuation["exact_root_backend"] = "python_fallback"
    else:
        row[field] = value
    ledger.write_text("\n".join(json.dumps(item) for item in rows))
    with pytest.raises(ValueError, match=message):
        diagnostic.analyze()
