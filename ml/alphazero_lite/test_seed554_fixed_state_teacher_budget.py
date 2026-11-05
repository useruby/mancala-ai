"""Contract tests for seed554's bounded authentication and restart safety."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from unittest.mock import Mock

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


def test_shared_seed549_training_source_remains_prerequisite_bound() -> None:
    source = protocol.ROOT / "ml/alphazero_lite/seed549_training.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "c33438f66a0569d4c61f08d8373b0c02d667b448c7162e814153dcc93291f6f3"
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


def _prepare_generation_rejection(
    tmp_path, monkeypatch
) -> tuple[Path, dict[str, bytes]]:
    output = tmp_path / "seed554"
    output.mkdir()
    shutil.copy2(protocol.OUT / "execution-environment-binding.json", output)
    shutil.copy2(protocol.OUT / "training-input-binding.json", output)
    sentinel = output / "preserve.json"
    sentinel.write_text('{"sentinel":true}\n')
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(protocol, "OUT", output)
    monkeypatch.setattr(protocol.os, "sched_getaffinity", lambda _pid: {31})
    monkeypatch.setattr(protocol.os, "nice", lambda _increment: 19)
    monkeypatch.setattr(protocol.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(protocol.torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(protocol.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(protocol.torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(
        protocol,
        "_live_cgroup_limits",
        lambda: {
            "cpu.max": "25000 100000",
            "cpu.weight": "1",
            "memory.max": "4294967296",
        },
    )
    monkeypatch.setattr(
        protocol, "preflight", Mock(side_effect=AssertionError("preflight reached"))
    )
    monkeypatch.setattr(
        protocol,
        "_states_by_occurrence",
        Mock(side_effect=AssertionError("population loaded")),
    )
    monkeypatch.setattr(
        protocol.teacher,
        "run_search",
        Mock(side_effect=AssertionError("search launched")),
    )
    monkeypatch.setattr(
        protocol.teacher.self_play,
        "CheckpointEvaluator",
        Mock(side_effect=AssertionError("evaluator constructed")),
    )
    return output, before


def test_generate_rejects_python_312_before_evaluator_or_writes(
    tmp_path, monkeypatch
) -> None:
    output, before = _prepare_generation_rejection(tmp_path, monkeypatch)
    monkeypatch.setattr(protocol.sys, "version", "3.12.0")
    with pytest.raises(ValueError, match="generation_numerical_runtime_mismatch"):
        protocol.generate()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    protocol.teacher.self_play.CheckpointEvaluator.assert_not_called()
    protocol.teacher.run_search.assert_not_called()
    protocol.preflight.assert_not_called()


def test_generate_rejects_unbounded_cgroup_before_evaluator_or_writes(
    tmp_path, monkeypatch
) -> None:
    output, before = _prepare_generation_rejection(tmp_path, monkeypatch)
    monkeypatch.setattr(protocol.os, "sched_getaffinity", lambda _pid: {31})
    monkeypatch.setattr(protocol.os, "nice", lambda _increment: 19)
    monkeypatch.setattr(
        protocol,
        "_live_cgroup_limits",
        lambda: {
            "cpu.max": "max 100000",
            "cpu.weight": "1",
            "memory.max": "4294967296",
        },
    )
    with pytest.raises(
        ValueError, match="generation_cgroup_resource_contract_mismatch"
    ):
        protocol.generate()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    protocol.teacher.self_play.CheckpointEvaluator.assert_not_called()
    protocol.teacher.run_search.assert_not_called()
    protocol.preflight.assert_not_called()


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


def test_freeze_creates_and_binds_environment_before_registration(
    tmp_path, monkeypatch
):
    output = tmp_path / "seed554"
    output.mkdir()
    shutil.copy2(protocol.OUT / "README.md", output)
    monkeypatch.setattr(protocol, "OUT", output)
    protocol.freeze()
    environment_path = output / "execution-environment-binding.json"
    environment_hash = protocol.file_sha(environment_path)
    freeze = json.loads((output / "label-generation-freeze.json").read_text())
    registration = json.loads((output / "registration.json").read_text())
    assert freeze["execution_environment_binding_sha256"] == environment_hash
    assert registration["execution_environment_binding_sha256"] == environment_hash
    assert (
        protocol.preflight()["execution_environment_binding_sha256"] == environment_hash
    )


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


def test_rebound_search_receipt_cannot_change_reconstructed_state_index_and_seed() -> (
    None
):
    canonical = "canonical-state"
    state_map = {
        canonical: {"state_index": 1, "float32": "production-input", "state": {}}
    }
    row = {
        "canonical_state_sha256": canonical,
        "production_float32_sha256": "production-input",
        "state_index": 2,
        "simulations": 128,
        "search": {"seed": 554_002},
    }
    # Rebind a receipt around the co-mutated archived index and seed.
    row["receipt_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="verifier_search_state_index_mismatch"):
        verifier.resolve_search_state(row, state_map)


def _prepare_live_gate_test(tmp_path, monkeypatch):
    output = tmp_path / "seed554"
    output.mkdir()
    shutil.copy2(protocol.OUT / "execution-environment-binding.json", output)
    shutil.copy2(protocol.OUT / "training-input-binding.json", output)
    sentinel = output / "preserve.json"
    sentinel.write_text('{"preserve":true}\n')
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    binding = json.loads((output / "execution-environment-binding.json").read_text())
    monkeypatch.setattr(protocol, "OUT", output)
    monkeypatch.setattr(protocol.os, "sched_getaffinity", lambda _pid: {31})
    monkeypatch.setattr(protocol.os, "nice", lambda _increment: 19)
    monkeypatch.setattr(protocol.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(protocol.torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(
        protocol,
        "_live_cgroup_limits",
        lambda: {
            "cpu.max": "25000 100000",
            "cpu.weight": "1",
            "memory.max": "4294967296",
        },
    )
    preflight = Mock(side_effect=AssertionError("frozen input preflight reached"))
    states = Mock(side_effect=AssertionError("source state loading reached"))
    evaluator = Mock(side_effect=AssertionError("evaluator constructed"))
    search = Mock(side_effect=AssertionError("teacher search launched"))
    monkeypatch.setattr(protocol, "preflight", preflight)
    monkeypatch.setattr(protocol, "_states_by_occurrence", states)
    monkeypatch.setattr(protocol.teacher, "run_search", search)
    monkeypatch.setattr(protocol.teacher.self_play, "CheckpointEvaluator", evaluator)
    return output, before, binding, (preflight, states, evaluator, search)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("python", "generation_numerical_runtime_mismatch"),
        ("cgroup", "generation_cgroup_resource_contract_mismatch"),
    ),
)
def test_generate_rejects_unregistered_runtime_or_scope_before_launch_or_write(
    tmp_path, monkeypatch, mismatch, message
) -> None:
    output, before, _binding, spies = _prepare_live_gate_test(tmp_path, monkeypatch)
    if mismatch == "python":
        monkeypatch.setattr(protocol.sys, "version", "3.12.0")
    else:
        monkeypatch.setattr(
            protocol,
            "_live_cgroup_limits",
            lambda: {
                "cpu.max": "max 100000",
                "cpu.weight": "1",
                "memory.max": "4294967296",
            },
        )
    with pytest.raises(ValueError, match=message):
        protocol.generate()
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    assert after == before
    for spy in spies:
        spy.assert_not_called()


def test_edited_environment_binding_rejected_by_published_hash_before_launch(
    tmp_path, monkeypatch
) -> None:
    output, _before, binding, spies = _prepare_live_gate_test(tmp_path, monkeypatch)
    binding["resource_limits"]["cpu_affinity"] = "30"
    binding["resource_limits"]["memory_max_bytes"] = 8_589_934_592
    (output / "execution-environment-binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    with pytest.raises(ValueError, match="execution_environment_binding_hash_mismatch"):
        protocol.generate()
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    assert after == before
    for spy in spies:
        spy.assert_not_called()
