"""Occurrence semantics and semantic tamper regressions for seed457 correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite import verify_seed457_fresh_subgroup_projection as old_verifier
from ml.alphazero_lite import verify_seed457_occurrence_correction as correction


class PositionSensitiveModel(torch.nn.Module):
    """Small deterministic model whose duplicate rows have occurrence outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.phase = 0.0

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        occurrence = torch.arange(len(states), dtype=torch.float32)
        logits = torch.zeros((len(states), train.POLICY_SIZE), dtype=torch.float32)
        logits[:, 0] = occurrence * 0.03 + self.phase
        logits[:, 1] = -occurrence * 0.01
        values = occurrence.mul(0.02).add(self.phase).reshape(-1, 1)
        return logits, values


def test_occurrence_trial_preserves_duplicate_outputs_and_compact_pre_reference() -> (
    None
):
    x = np.zeros((512, 27), dtype=np.float32)
    x[:, :6] = x[:, 6:12] = 1 / 48
    x[:, 12] = 1 / 48
    rows = np.asarray([0, 1, 0, 2], dtype=np.int64)
    model = PositionSensitiveModel()
    pre_logp, pre_values_compact = correction._portable_outputs(model, x, rows)
    pre_occurrence_values = correction._PRE_OCCURRENCE_VALUES[tuple(rows)]

    # Compact pre-policy lookup must remain identical to the frozen scatter/gather.
    frozen_pre_logp, frozen_pre_values = old_verifier._outputs(model, x, rows)
    torch.testing.assert_close(pre_logp[rows], frozen_pre_logp[rows], rtol=0, atol=0)
    np.testing.assert_array_equal(pre_values_compact[rows], frozen_pre_values[rows])
    assert pre_occurrence_values[0] != pre_occurrence_values[2]

    model.phase = 0.4
    corrected_kl, corrected_movement = correction.occurrence_trial_measure(
        model, x, rows, pre_logp, pre_values_compact
    )
    original_kl, original_movement = old_verifier._measure_trial(
        model, x, rows, pre_logp, pre_values_compact
    )
    assert corrected_kl != original_kl
    assert corrected_movement != original_movement

    # Explicit runner arithmetic over occurrence outputs is the correction oracle.
    with torch.inference_mode():
        trial_logits, trial_values = model(torch.from_numpy(x[rows]))
        legal = torch.from_numpy(
            train.legal_mask_matrix_for_encoded_states(x[rows])
        ).bool()
        trial_logp = torch.log_softmax(
            trial_logits.double().masked_fill(~legal, -torch.inf), dim=1
        )
        pre_log = pre_logp[rows]
        probability = pre_log.exp().masked_fill(~legal, 0.0)
        terms = (probability * (pre_log - trial_logp).masked_fill(~legal, 0.0)).sum(
            dim=1
        )
        expected_kl = float(
            (terms * torch.full((len(rows),), 1 / len(rows), dtype=torch.float64)).sum()
        )
    expected_delta = trial_values.reshape(-1).numpy().astype(
        np.float64
    ) - pre_occurrence_values.astype(np.float64)
    expected_movement = float(np.mean(expected_delta**2, dtype=np.float64))
    assert corrected_kl == expected_kl
    assert corrected_movement == expected_movement


def _valid_reconciliation() -> dict[str, object]:
    base = {"policy_ce": 1.0, "value_mse": 0.5}
    treatment = {"policy_ce": 0.5, "value_mse": 0.501}
    fresh_init = {"policy_ce": 0.2, "value_mse": 0.2}
    fresh_t = {"policy_ce": 0.200003, "value_mse": 0.2}
    metrics = {
        "initializer": {
            **{w: base.copy() for w in ("exposure_weighted", "equal_input")},
            "fresh": {
                w: fresh_init.copy() for w in ("exposure_weighted", "equal_input")
            },
            "full_training_objective": 1.0,
        },
        "seed442_A": {
            w: {"policy_ce": 1.0} for w in ("exposure_weighted", "equal_input")
        },
        "T": {
            **{w: treatment.copy() for w in ("exposure_weighted", "equal_input")},
            "fresh": {w: fresh_t.copy() for w in ("exposure_weighted", "equal_input")},
            "full_training_objective": 0.9,
        },
        "C": {
            **{w: base.copy() for w in ("exposure_weighted", "equal_input")},
            "fresh": {
                w: fresh_init.copy() for w in ("exposure_weighted", "equal_input")
            },
            "full_training_objective": 0.95,
        },
    }
    effects = {}
    for weighting in ("exposure_weighted", "equal_input"):
        effects[weighting] = {
            metric: metrics["T"][weighting][metric] - metrics["C"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
        effects[f"fresh_{weighting}"] = {
            metric: metrics["T"]["fresh"][weighting][metric]
            - metrics["C"]["fresh"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
    effects["full_training_objective"] = {
        "T_minus_C": metrics["T"]["full_training_objective"]
        - metrics["C"]["full_training_objective"]
    }
    gates = {
        f"pooled_ce_vs_seed442_A_{w}": True
        for w in ("exposure_weighted", "equal_input")
    }
    gates.update(
        {
            f"pooled_ce_vs_initializer_{w}": True
            for w in ("exposure_weighted", "equal_input")
        }
    )
    gates.update(
        {f"pooled_mse_{w}": True for w in ("exposure_weighted", "equal_input")}
    )
    gates.update({f"fresh_ce_{w}": False for w in ("exposure_weighted", "equal_input")})
    gates.update(
        {
            "training_objective_decreases": True,
            "no_nonzero_rejection": True,
            "additional_subgroup_protection": True,
        }
    )
    return {
        "schema": "seed457-occurrence-reconciliation-v1",
        "verification_timing": "post_execution_correction",
        "coverage": {
            "steps_per_arm": {"C": 16, "T": 16},
            "trials_per_arm": {"C": 128, "T": 128},
            "trials_total": 256,
        },
        "control_reproduction": {"matches": True, "mismatches": []},
        "metrics": metrics,
        "effects_T_minus_C": effects,
        "activation": {"activated": True},
        "decision": {
            "classification": "close_fresh_subgroup_projection_branch",
            "gates": gates,
        },
        "arm_trajectories": {
            a: {
                "verified_trials": 128,
                "steps": [
                    {
                        "verified_trials": 8,
                        "raw_proposal_norm": 1.0,
                        "rejected": False,
                    }
                    for _ in range(16)
                ],
            }
            for a in ("C", "T")
        },
    }


def _write_hashed_report(directory: Path, report: dict[str, object]) -> None:
    path = directory / "reconciliation.json"
    path.write_text(json.dumps(report, sort_keys=True))
    receipt = {
        "classification": report["decision"]["classification"],  # type: ignore[index]
        "correction_files_sha256": {
            "reconciliation.json": hashlib.sha256(path.read_bytes()).hexdigest()
        },
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("effects", "correction_effect_mismatch"),
        ("classification", "correction_classification_semantic_mismatch"),
    ],
)
def test_rehashed_summary_tampering_is_rejected_semantically(
    tmp_path: Path, change: str, error: str
) -> None:
    report = _valid_reconciliation()
    if change == "effects":
        report["effects_T_minus_C"]["equal_input"]["policy_ce"] += 0.001  # type: ignore[index]
    else:
        report["decision"]["classification"] = (
            "advance_to_separately_preregistered_strength_experiment"  # type: ignore[index]
        )
    _write_hashed_report(tmp_path, report)
    with pytest.raises(ValueError, match=error):
        correction.verify_correction_receipt(tmp_path)


def test_rehashed_decision_gate_tampering_is_rejected() -> None:
    report = _valid_reconciliation()
    report["decision"]["gates"]["fresh_ce_equal_input"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="correction_gate_semantic_mismatch"):
        correction.validate_reconciliation_semantics(report)


def test_archive_parts_are_reassembled_and_logical_hash_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = tmp_path / "parts"
    parts.mkdir()
    first, second = b"logical-", b"archive"
    (parts / "a.part").write_bytes(first)
    (parts / "b.part").write_bytes(second)
    monkeypatch.setattr(correction, "PARTS", Path("parts"))
    monkeypatch.setattr(
        correction,
        "T_PART_SHA256",
        {
            "a.part": hashlib.sha256(first).hexdigest(),
            "b.part": hashlib.sha256(second).hexdigest(),
        },
    )
    # The committed parts are individually rebound, but their joined logical
    # archive still fails the registered whole-file identity check.
    assembled = tmp_path / "assembled"
    assembled.mkdir()
    with pytest.raises(ValueError, match="logical_T_archive_identity_mismatch"):
        correction._assemble_t(tmp_path, assembled)


def test_archive_tensor_and_occurrence_measurement_tampering_rejects(
    tmp_path: Path,
) -> None:
    # Simulate a rebased ordinary artifact receipt: its byte hash now matches,
    # but exact tensor reconstruction still rejects the changed occurrence.
    archive_path = tmp_path / "step-tensors.npz"
    receipt_path = tmp_path / "receipt.json"
    expected = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)
    np.savez(archive_path, C_05_00_trial_1=expected)
    receipt = {
        "files_sha256": {
            archive_path.name: hashlib.sha256(archive_path.read_bytes()).hexdigest()
        }
    }
    receipt_path.write_text(json.dumps(receipt))
    changed = expected.copy()
    changed[1] += 0.125
    np.savez(archive_path, C_05_00_trial_1=changed)
    receipt["files_sha256"][archive_path.name] = hashlib.sha256(
        archive_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    assert (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        == receipt["files_sha256"][archive_path.name]
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        with pytest.raises(
            ValueError, match="semantic_tensor_mismatch:trial_parameters"
        ):
            correction.require_tensor_semantics(
                archive["C_05_00_trial_1"], expected, "trial_parameters"
            )

    actual_measurement = 0.032772165412068455
    changed_recorded_measurement = actual_measurement + 0.25
    measurement_path = tmp_path / "trial-measurement.json"
    measurement_path.write_text(json.dumps({"batch_kl": changed_recorded_measurement}))
    receipt["files_sha256"][measurement_path.name] = hashlib.sha256(
        measurement_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    assert (
        hashlib.sha256(measurement_path.read_bytes()).hexdigest()
        == receipt["files_sha256"][measurement_path.name]
    )
    with pytest.raises(ValueError, match="trial_measurement_mismatch:batch_kl:1.0"):
        correction.require_trial_measurement(
            actual_measurement, changed_recorded_measurement, "batch_kl", 1.0
        )


def test_ordered_occurrences_are_not_sorted_or_deduplicated(tmp_path: Path) -> None:
    expected = np.asarray([9, 3, 9, 1], dtype=np.int64)
    digest = hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest()
    changed = expected[[1, 0, 2, 3]]
    assert digest != hashlib.sha256(changed.astype("<i8").tobytes()).hexdigest()
    assert len(set(map(int, expected))) == 3
    assert np.flatnonzero(expected == 9).tolist() == [0, 2]
    with pytest.raises(ValueError, match="occurrence_order_mismatch"):
        correction.require_occurrence_order(changed, expected, "tamper")
    path = tmp_path / "ordered-occurrences.json"
    path.write_text(json.dumps(changed.tolist()))
    ordinary_receipt = {
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    }
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == ordinary_receipt["files_sha256"][path.name]
    )
    with pytest.raises(ValueError, match="occurrence_order_mismatch"):
        correction.require_occurrence_order(
            np.asarray(json.loads(path.read_text()), dtype=np.int64),
            expected,
            "rehashed",
        )


def test_archived_c_step6_occurrence_failure_fixture_if_available() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = root / ".tmp/seed457-kl-diagnosis/step6-scale1-diagnostic-arrays.npz"
    if not fixture.is_file():
        pytest.skip("local archived diagnosis arrays are not staged in this checkout")
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "ba0469cf8c99cedc533513b9ff58460acb0aca2e9f6277291d57ee6f7a30fef8"
    )
    with np.load(fixture, allow_pickle=False) as arrays:
        rows = arrays["compact_rows"]
        assert np.flatnonzero(rows == 72594).tolist() == [343, 360]
        inputs = arrays["ordered_inputs"]
        np.testing.assert_array_equal(inputs[343], inputs[360])
        logits = arrays["trial_logits_occurrence"]
        assert np.max(np.abs(logits[343] - logits[360])) == 4.76837158203125e-07
        assert (
            np.max(
                np.abs(logits[343] - arrays["trial_logits_after_scatter_gather"][343])
            )
            == 4.76837158203125e-07
        )
        runner_terms = arrays["kl_terms_runner"]
        verifier_terms = arrays["kl_terms_verifier"]
        assert np.flatnonzero(runner_terms != verifier_terms).tolist() == [343]
        assert abs(runner_terms[343] - verifier_terms[343]) == pytest.approx(
            3.96008305453277e-08, rel=0, abs=1e-22
        )
        weights = arrays["uniform_weights"]
        runner_kl = float(np.sum(runner_terms * weights, dtype=np.float64))
        verifier_kl = float(np.mean(verifier_terms, dtype=np.float64))
        assert runner_kl == pytest.approx(0.032772165412068455, rel=0, abs=1e-17)
        assert verifier_kl == pytest.approx(0.03277216548941382, rel=0, abs=1e-17)
