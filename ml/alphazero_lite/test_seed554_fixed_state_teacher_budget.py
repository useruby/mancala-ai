"""Contract tests for seed554's bounded authentication and restart safety."""

from __future__ import annotations

import json

import pytest

from ml.alphazero_lite import seed554_fixed_state_teacher_budget as protocol
from ml.alphazero_lite import verify_seed554_fixed_state_teacher_budget as verifier
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import build_policy_target


def _valid_search(state: dict, budget: int = 128, seed: int = 554_001) -> dict:
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    visits = [0] * 6
    for index in range(budget):
        visits[legal[index % len(legal)]] += 1
    return {
        "simulations": budget,
        "seed": seed,
        "visits": visits,
        "target": build_policy_target(visits, legal_moves=legal, temperature=1.0),
        "legal_moves": legal,
        "root_visit_count": budget,
        "selected_move": min(
            a for a in legal if visits[a] == max(visits[b] for b in legal)
        ),
        "root_children": {
            str(a): {"visits": visits[a], "prior": 1.0 / len(legal), "q_value": 0.0}
            for a in legal
        },
        "evaluator_calls": 1,
    }


def _opening_state() -> dict:
    return {
        "player_pits": [4] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }


def test_bounded_authentication_accepts_committed_source_contracts() -> None:
    result = protocol.authenticate_inputs()
    assert result["occurrences"] == 13_818
    assert result["updates"] == 108
    assert result["exposures"] == 54_896
    assert result["development_occurrences"] == 4_738
    assert result["initializer_sha256"] == protocol.INITIALIZER_SHA256
    assert (
        "full historical runtime verifier not claimed"
        in result["historical_verifier_scope"]
    )


def test_saved_search_semantic_validation_rejects_rebound_target_corruption(
    tmp_path,
) -> None:
    state = _opening_state()
    evidence = _valid_search(state)
    protocol._validate_search(evidence, state, 128, 554_001)
    evidence["target"][0] += 0.01
    with pytest.raises(ValueError, match="saved_search_target_reconstruction_mismatch"):
        protocol._validate_search(evidence, state, 128, 554_001)
    canonical = protocol.teacher.canonical_state_id(state)
    float32_id = protocol.teacher.encoded_state_id(state)
    row = {
        "state_index": 1,
        "canonical_state_sha256": canonical,
        "production_float32_sha256": float32_id,
        "simulations": 128,
        "search": evidence,
    }
    ledger = tmp_path / "rebound-search-receipt.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    # Rebinding a surrounding receipt to the corrupted row cannot make it
    # resumable: the resume path reconstructs the target from visits.
    state_row = {
        "canonical": canonical,
        "float32": float32_id,
        "state_index": 1,
        "seed": 554_001,
        "state": state,
    }
    with pytest.raises(ValueError, match="saved_search_target_reconstruction_mismatch"):
        protocol._load_search_ledger([state_row], ledger)


def test_saved_search_semantic_validation_rejects_visit_mutation() -> None:
    state = _opening_state()
    evidence = _valid_search(state)
    evidence["visits"][0] += 1
    with pytest.raises(ValueError, match="saved_search_visit_total_mismatch"):
        protocol._validate_search(evidence, state, 128, 554_001)


def test_completed_search_is_loaded_and_skipped_without_search(tmp_path) -> None:
    state = _opening_state()
    game = KalahGame.from_state(state)
    encoded = protocol.teacher.encoded_state(state)
    canonical = protocol.teacher.canonical_state_id(state)
    encoded_id = protocol.teacher.encoded_state_id(state)
    state_row = {
        "state": state,
        "canonical": canonical,
        "float32": encoded_id,
        "state_index": 1,
        "seed": 554_001,
    }
    evidence = _valid_search(state)
    row = {
        "state_index": 1,
        "canonical_state_sha256": canonical,
        "production_float32_sha256": encoded_id,
        "simulations": 128,
        "search": evidence,
    }
    ledger = tmp_path / "searches.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    completed = protocol._load_search_ledger([state_row], ledger)
    assert completed[(canonical, 128)] == row
    assert encoded.shape == (27,)
    assert game.possible_moves() == evidence["legal_moves"]


def test_freeze_refuses_existing_evidence_without_overwriting(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "seed554"
    output.mkdir()
    sentinel = output / "preserve.json"
    sentinel.write_text('{"prior":true}\n')
    monkeypatch.setattr(protocol, "OUT", output)
    before = sentinel.read_bytes()
    with pytest.raises(FileExistsError, match="seed554_output_directory_not_empty"):
        protocol.freeze()
    assert sentinel.read_bytes() == before
    assert sorted(path.name for path in output.iterdir()) == ["preserve.json"]


def test_corrupt_registration_rejected_before_any_computation_or_write(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "seed554"
    output.mkdir()
    (output / "registration.json").write_text(
        json.dumps(
            {
                "label_generation_freeze_sha256": "bad",
                "input_authentication_sha256": "bad",
            }
        )
    )
    (output / "label-generation-freeze.json").write_text("{}\n")
    (output / "input-authentication.json").write_text("{}\n")
    monkeypatch.setattr(protocol, "OUT", output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="label_generation_freeze_binding_mismatch"):
        protocol.preflight()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_conflicting_training_evidence_is_preserved(tmp_path) -> None:
    conflict = tmp_path / "control-updates.jsonl"
    conflict.write_text('{"prior":"evidence"}\n')
    before = conflict.read_bytes()
    before_mtime = conflict.stat().st_mtime_ns
    with pytest.raises(
        FileExistsError, match="seed554_training_evidence_already_exists"
    ):
        protocol.reject_existing_training_evidence(tmp_path)
    assert conflict.read_bytes() == before
    assert conflict.stat().st_mtime_ns == before_mtime


@pytest.mark.parametrize(
    ("field", "value"),
    (("value", -1.0), ("teacher_source", "exact_root_tablebase")),
)
def test_value_and_policy_weight_source_mutations_are_rejected(field, value) -> None:
    source = {
        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "value": 1.0,
        "teacher_source": "policy_model",
        "active_pit_stones": 20,
        "puct_simulations_executed": 128,
    }
    derivative = dict(source)
    derivative[field] = value
    with pytest.raises(ValueError, match="verifier_nonpolicy_source_row_mutation"):
        verifier.require_nonpolicy_source_row(source, derivative, "control", 0)


def test_occurrence_multiplicity_mutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="verifier_occurrence_multiplicity_mismatch"):
        verifier.require_unique_occurrences(
            [{"row_id": "same"}, {"row_id": "same"}], expected=2
        )
