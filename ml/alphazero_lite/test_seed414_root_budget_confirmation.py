from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite import seed414_root_budget_confirmation as runner
from ml.alphazero_lite.verify_seed414_root_budget_confirmation import verify
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame


def _state_row() -> dict:
    game = KalahGame.from_state(runner.suites.INITIAL_STATE)
    state = game.to_state()
    return {
        "opening_index": 0,
        "state": state,
        "state_hash": runner.suites.canonical_key(state),
        "legal_actions": game.possible_moves(),
        "prefix_moves": [],
        "source_prefix_moves_absolute": [],
        "opening_contract": "arena_player_relative_v2",
    }


def test_seed_identity_is_coupled_across_budget_reference_and_branch_labels() -> None:
    base = {
        "contract_version": "azlite_eval_seed_v2",
        "base_seed": 414,
        "suite_sha256": "a" * 64,
        "opening_index": 4,
        "opening_state_hash": "b" * 64,
        "challenger_player": 0,
        "game_within_opening": 0,
        "ply": 1,
        "canonical_current_state_hash": "c" * 64,
        "acting_role": "challenger",
    }
    original = derive_search_seed(**base)
    # Budget, branch, and reference are deliberately metadata only and are not
    # parameters accepted by the v2 seed derivation API.
    for budget in (384, 1536):
        for reference in ("seed455", "original_O0_E4"):
            for branch in ("FF384", "FF1536"):
                _metadata = (budget, reference, branch)
                assert derive_search_seed(**base) == original
    changed = {**base, "ply": 2}
    assert derive_search_seed(**changed) != original


def test_root_snapshot_validator_checks_shared_prefix_counts() -> None:
    row = _state_row()
    legal = row["legal_actions"]
    visits384 = {str(move): 0 for move in legal}
    visits1536 = {str(move): 0 for move in legal}
    visits384[str(legal[0])] = 384
    visits1536[str(legal[0])] = 1536
    entries = [
        {"move": move, "q_value": 0.0, "prior": 1.0 / len(legal)} for move in legal
    ]
    snapshots = {
        "384": {"visits": visits384, "moves": entries, "action": legal[0]},
        "1536": {"visits": visits1536, "moves": entries, "action": legal[0]},
    }
    runner._validate_root_snapshots(row, snapshots)
    snapshots["1536"]["visits"][str(legal[0])] = 1000
    with pytest.raises(ValueError, match="root_snapshot_visit_total:1536"):
        runner._validate_root_snapshots(row, snapshots)


def test_published_verifier_is_model_free_and_does_not_mutate_evidence() -> None:
    before = {
        path.relative_to(runner.OUT): path.read_bytes()
        for path in runner.OUT.rglob("*")
        if path.is_file()
    }
    report = verify()
    after = {
        path.relative_to(runner.OUT): path.read_bytes()
        for path in runner.OUT.rglob("*")
        if path.is_file()
    }
    assert report["status"] == "verified"
    assert before == after


def test_probe_resume_skips_completed_state(tmp_path: Path, monkeypatch) -> None:
    row = _state_row()
    suite = tmp_path / "suite.jsonl"
    suite.write_text(json.dumps(row) + "\n")
    probes = tmp_path / "root-probes.jsonl"
    registration = tmp_path / "registration.json"
    registration.write_text("{}\n")
    monkeypatch.setattr(runner, "SUITE", suite)
    monkeypatch.setattr(runner, "PROBES", probes)
    monkeypatch.setattr(runner, "REGISTRATION", registration)
    calls = 0

    class Evaluator:
        def __init__(self, _path):
            pass

    def evaluate(**kwargs):
        nonlocal calls
        calls += 1
        state = kwargs["state"]
        legal = KalahGame.from_state(state).possible_moves()
        visits = [0] * 6
        visits[legal[0]] = kwargs["simulations"]
        moves = [
            {"move": move, "q_value": 0.0, "prior": 1.0 / len(legal)} for move in legal
        ]
        snapshots = []
        for budget in (384, 1536):
            prefix_visits = [0] * 6
            prefix_visits[legal[0]] = budget
            snapshots.append(
                {
                    "simulation": budget,
                    "selected_move": legal[0],
                    "visits": prefix_visits,
                    "moves": moves,
                }
            )
        return {"root_snapshots": snapshots}

    monkeypatch.setattr(runner.arena, "ArtifactEvaluator", Evaluator)
    monkeypatch.setattr(runner.arena, "evaluate_artifact_position", evaluate)
    monkeypatch.setattr(
        runner, "frozen_runtime_identities", lambda: {"frozen": "runtime"}
    )
    reg = {
        "states": [row],
        "suite_sha256": runner.sha(suite),
        "source_sha256": runner.source_hashes(),
        "runtime_identities": {"frozen": "runtime"},
    }
    # Root snapshot counts must be valid enough for resume identity; semantic
    # prefix checks are exercised by the portable verifier's test helpers.
    runner.probe_roots(reg)
    runner.probe_roots(reg)
    assert calls == 1
    assert len(runner._load_jsonl(probes)) == 1


def test_continuation_resume_and_same_action_aliases(
    tmp_path: Path, monkeypatch
) -> None:
    row = _state_row()
    trajectories = tmp_path / "continuations.jsonl"
    aliases = tmp_path / "aliases.jsonl"
    registration = tmp_path / "registration.json"
    registration.write_text("{}\n")
    monkeypatch.setattr(runner, "TRAJECTORIES", trajectories)
    monkeypatch.setattr(runner, "ALIASES", aliases)
    monkeypatch.setattr(runner, "REGISTRATION", registration)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"frozen": "yes"})
    monkeypatch.setattr(
        runner, "frozen_runtime_identities", lambda: {"frozen": "runtime"}
    )
    monkeypatch.setattr(runner, "NativeExactRootTablebase", lambda *a, **k: _Adapter())
    probes = [
        {
            "opening_index": 0,
            "snapshots": {"384": {"action": 0}, "1536": {"action": 0}},
        }
    ]
    reg = {
        "states": [row],
        "source_sha256": {"frozen": "yes"},
        "runtime_identities": {"frozen": "runtime"},
    }
    calls = 0

    def interrupted(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulate_interruption")
        return {
            "state_hash": row["state_hash"],
            "score": 0.5,
            "store_margin_root_perspective": 0,
            "elapsed_wall_seconds": 1.0,
        }

    monkeypatch.setattr(runner, "play_continuation", interrupted)
    with pytest.raises(RuntimeError, match="simulate_interruption"):
        runner.run_continuations(reg, probes)
    assert len(runner._load_jsonl(trajectories)) == 1

    def resumed(*_args):
        nonlocal calls
        calls += 1
        return {
            "state_hash": row["state_hash"],
            "score": 0.5,
            "store_margin_root_perspective": 0,
            "elapsed_wall_seconds": 1.0,
        }

    monkeypatch.setattr(runner, "play_continuation", resumed)
    final, logical = runner.run_continuations(reg, probes)
    assert len(final) == 2
    assert len(logical) == 4
    assert sum(item["alias_of"] is not None for item in logical) == 2
    assert calls == 3


class _Adapter:
    def close(self):
        pass
