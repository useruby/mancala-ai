"""Mutation-focused tests for seed549 controlled-training evidence."""

from __future__ import annotations

import json
import sys

import torch
import pytest

from ml.alphazero_lite import seed549_training as runner
from ml.alphazero_lite import verify_seed549_training as verifier

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _schedule(arm: str) -> dict:
    sizes = ([512] * 26 + [412]) * 4
    batches = []
    cursor = 0
    for update, size in enumerate(sizes, 1):
        batches.append(
            {
                "update": update,
                "start_exposure": cursor,
                "end_exposure_exclusive": cursor + size,
                "row_occurrence_indices": [0] * size,
            }
        )
        cursor += size
    return {"updates": 108, "exposures": 54_896, "batches": batches}


def test_schedule_mapping_rejects_modified_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "CORPUS", tmp_path)
    for arm in runner.ARMS:
        schedule = _schedule(arm)
        schedule["batches"][0]["row_occurrence_indices"][0] = (
            13_818 if arm == "control" else 13_849
        )
        (tmp_path / f"{arm}-training-schedule.json").write_text(json.dumps(schedule))
    with pytest.raises(ValueError, match="schedule_row_index_out_of_range"):
        runner.load_schedule("control")


def test_schedule_shape_mutation_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "CORPUS", tmp_path)
    value = _schedule("control")
    value["batches"][0]["row_occurrence_indices"].pop()
    (tmp_path / "control-training-schedule.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="schedule_batch_mapping_mismatch"):
        runner.load_schedule("control")


def test_real_verifier_entrypoint_rejects_tampered_freeze_read_only(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "registration.json").write_text(
        json.dumps({"source_sha256": {"AGENTS.md": "wrong"}, "input_sha256": {}})
    )
    freeze_path = tmp_path / "execution-freeze.json"
    freeze_path.write_text(json.dumps({"registration_sha256": "wrong"}))
    before = (freeze_path.read_bytes(), freeze_path.stat().st_mtime_ns)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_seed549_training", "--root", str(runner.ROOT), "--out", str(tmp_path)],
    )
    with pytest.raises(ValueError, match="registration_freeze_mismatch"):
        verifier.main()
    assert (freeze_path.read_bytes(), freeze_path.stat().st_mtime_ns) == before


def test_objective_is_policy_ce_plus_half_huber() -> None:
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    policy = -(target * torch.log_softmax(logits, dim=1)).sum()
    value = torch.nn.functional.smooth_l1_loss(
        torch.tensor([[0.0]]), torch.tensor([[0.5]]), beta=1.0
    )
    assert float(policy + 0.5 * value) == pytest.approx(
        float(policy) + 0.5 * float(value)
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("batch_identity_sha256", "altered-batch"),
        ("policy_loss", 123.0),
        ("optimizer_state_before_sha256", "altered-adam"),
        ("parameter_state_after_sha256", "altered-parameters"),
        ("production_equivalent", False),
    ],
)
def test_real_training_replay_rejects_mutated_update_evidence(
    tmp_path, monkeypatch, field, replacement
) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(verifier, "OUT", tmp_path)
    source = runner.OUT / "control-updates.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    records[0][field] = replacement
    (tmp_path / "control-updates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    expected_error = (
        "production_equivalence_gate_missing"
        if field in {"production_equivalent", "parameter_state_after_sha256"}
        else f"training_replay_mismatch:control:1:{field}"
    )
    with pytest.raises(ValueError, match=expected_error):
        verifier.replay("control")


def test_real_training_replay_rejects_target_tensor_mutation(
    tmp_path, monkeypatch
) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(verifier, "OUT", tmp_path)
    source = runner.OUT / "control-updates.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    records[0]["tensors"]["policy_target"]["sha256"] = "altered-target"
    (tmp_path / "control-updates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    with pytest.raises(ValueError, match="training_replay_mismatch:control:1:tensors"):
        verifier.replay("control")


def test_real_training_replay_rejects_mutated_checkpoint(tmp_path, monkeypatch) -> None:
    import numpy as np

    torch.set_num_threads(1)
    monkeypatch.setattr(verifier, "OUT", tmp_path)
    source = runner.OUT / "control-updates.jsonl"
    (tmp_path / "control-updates.jsonl").write_bytes(source.read_bytes())
    checkpoint = runner.OUT / "control-update-027.npz"
    with np.load(checkpoint) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    first_name = next(iter(arrays))
    arrays[first_name].flat[0] += np.float32(1e-3)
    np.savez(tmp_path / "control-update-027.npz", **arrays)
    with pytest.raises(ValueError, match="checkpoint_mismatch:control:27"):
        verifier.replay("control")


@pytest.mark.parametrize(
    "field", ["target_policy", "source_evaluator", "cohort", "float32_input_sha256"]
)
def test_prediction_membership_mutations_are_rejected(field: str) -> None:
    expected = {
        "source_evaluator": "control",
        "cohort": ">32",
        "float32_input_sha256": "input-hash",
        "target_policy": [1.0, 0.0],
    }
    prediction = {**expected, "occurrence": 0}
    prediction[field] = "mutated" if field != "target_policy" else [0.0, 1.0]
    with pytest.raises(ValueError, match="development_prediction_membership_mismatch"):
        verifier.validate_prediction_membership(prediction, expected, 0, "C-final")


@pytest.mark.parametrize("section", ["metrics", "effects"])
def test_metrics_and_effect_mutations_are_rejected(section: str) -> None:
    published = {"metrics": {"loss": 0.5}, "effects": {"T_minus_C": 0.1}}
    altered = json.loads(json.dumps(published))
    altered[section][next(iter(altered[section]))] = "mutated"
    with pytest.raises(ValueError, match="development_mutation_detected"):
        verifier.require_json_equal(published, altered, "development_mutation_detected")
