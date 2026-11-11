"""Synthetic-only tests for the #512 freeze, gate, and read-only verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import seed512_policy_strength as runner
from ml.alphazero_lite import verify_seed512_policy_strength as verifier
from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame


def synthetic_evidence() -> dict:
    control = {}
    treatment = {}
    for index in range(16):
        key = f"root-{index:02d}"
        opening = index
        control[key] = {
            "opening_index": opening,
            "regret": 1,
            "wdl": 0,
            "oracle_wdl": 1,
            "selected_action": 0,
            "seed": 1000 + index,
        }
        treatment[key] = {
            "opening_index": opening,
            "regret": 0,
            "wdl": 1,
            "oracle_wdl": 1,
            "selected_action": 1,
            "seed": 1000 + index,
        }
    paired = {"control": control, "treatment": treatment}
    return {
        "schema": "seed512-policy-strength-evidence-v1",
        "pairs": paired,
        "gate": runner.reconstruct_gate(paired),
    }


def test_synthetic_gate_reconstruction_passes_all_three_conditions() -> None:
    evidence = synthetic_evidence()
    result = runner.reconstruct_gate(evidence["pairs"])
    assert (
        result["decision"] == "advance_to_separate_public_exploratory_strength_screen"
    )
    assert result["mean_treatment_minus_control_regret"] == -1.0
    assert result["opening_bootstrap_95"] == [-1.0, -1.0]
    assert (
        result["treatment_wdl_downgrade_rate"] <= result["control_wdl_downgrade_rate"]
    )


def test_synthetic_gate_failure_is_do_not_advance() -> None:
    evidence = synthetic_evidence()
    for row in evidence["pairs"]["treatment"].values():
        row["regret"] = 1
        row["wdl"] = -1
    result = runner.reconstruct_gate(evidence["pairs"])
    assert result["decision"] == "do_not_advance"


def test_gate_rejects_incomplete_pairs() -> None:
    evidence = synthetic_evidence()
    evidence["pairs"]["treatment"].pop("root-00")
    with pytest.raises(ValueError, match="incomplete_or_unpaired"):
        runner.reconstruct_gate(evidence["pairs"])


def test_verifier_reconstructs_gate_without_rewriting_evidence(tmp_path: Path) -> None:
    evidence = synthetic_evidence()
    target = tmp_path / "synthetic-evidence.json"
    original = json.dumps(evidence, sort_keys=True).encode()
    target.write_bytes(original)
    assert verifier.verify_gate_only(copy.deepcopy(evidence)) == evidence["gate"]
    assert target.read_bytes() == original


def test_mutated_arm_evidence_cannot_reuse_gate() -> None:
    evidence = synthetic_evidence()
    evidence["pairs"]["treatment"]["root-00"]["regret"] = 5
    with pytest.raises(ValueError, match="gate_reconstruction_mismatch"):
        verifier.verify_gate_only(evidence)


def test_altered_artifact_pins_and_membership_are_rejected() -> None:
    with pytest.raises(ValueError, match="artifact_hash_manifest_mismatch"):
        runner.assert_artifact_hashes({"T512/model.npz": "altered"})
    with pytest.raises(ValueError, match="qualification_membership_binding_mismatch"):
        runner.assert_membership_binding(
            {"population": 422, "membership_sha256": "not-the-locked-digest"}
        )


def test_pre_review_freeze_cannot_start_qualification(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(PermissionError, match="qualification_not_authorized"):
        runner.evaluate(runner.ROOT, output, tmp_path / "no-authorization.json")
    assert not output.exists()


def test_update512_export_parity_on_synthetic_states(tmp_path: Path) -> None:
    artifact_dir = runner.DATA / "runtime-artifacts" / "T512"
    checkpoint_path = (
        runner.ROOT
        / "docs/data/seed571-policy-prior-distillation/run-0001/seed571-policy-distillation-update-512.npz"
    )
    with np.load(checkpoint_path) as archive:
        checkpoint_weights = {name: archive[name].tolist() for name in archive.files}
    exported_weights = json.loads((artifact_dir / "weights.json").read_text())
    assert set(checkpoint_weights) == set(exported_weights)
    assert all(
        np.array_equal(
            np.asarray(checkpoint_weights[name]), np.asarray(exported_weights[name])
        )
        for name in checkpoint_weights
    )

    source = tmp_path / "checkpoint-derived"
    source.mkdir()
    (source / "metadata.json").write_bytes(
        (artifact_dir / "metadata.json").read_bytes()
    )
    (source / "weights.json").write_text(json.dumps(checkpoint_weights))
    exported = ArtifactEvaluator(artifact_dir)
    reconstructed = ArtifactEvaluator(source)
    synthetic_states = [
        {
            "current_player": 0,
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
        },
        {
            "current_player": 1,
            "player_pits": [0, 3, 1, 0, 2, 1],
            "opponent_pits": [2, 0, 1, 2, 0, 1],
            "player_store": 15,
            "opponent_store": 14,
        },
    ]
    for state in synthetic_states:
        game = KalahGame.from_state(state)
        exported_policy, exported_value = exported.evaluate(game)
        checkpoint_policy, checkpoint_value = reconstructed.evaluate(game)
        assert np.array_equal(exported_policy, checkpoint_policy)
        assert exported_value == checkpoint_value


def test_control_and_treatment_compose_the_same_frozen_value_model() -> None:
    artifacts = runner.DATA / "runtime-artifacts"
    control_policy = ArtifactEvaluator(artifacts / "P-E4")
    treatment_policy = ArtifactEvaluator(artifacts / "T512")
    shared_initializer = ArtifactEvaluator(artifacts / "initializer")
    control = runner.arena.ComposedArtifactEvaluator(
        control_policy,
        shared_initializer,
        policy_source="current",
        value_source="candidate",
    )
    treatment = runner.arena.ComposedArtifactEvaluator(
        treatment_policy,
        shared_initializer,
        policy_source="current",
        value_source="candidate",
    )
    state = {
        "current_player": 0,
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
    }
    game = KalahGame.from_state(state)
    control_policy_output, control_value = control.evaluate(game)
    treatment_policy_output, treatment_value = treatment.evaluate(game)
    assert control_value == treatment_value
    assert np.isfinite(control_policy_output).all()
    assert np.isfinite(treatment_policy_output).all()
