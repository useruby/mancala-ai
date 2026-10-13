"""Synthetic numerical and receipt-rebound coverage for seed470."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite.seed470_finite_step_transport import (
    SCALES,
    classify,
    set_parameters,
)
from ml.alphazero_lite import verify_seed470_finite_step_transport as verifier


def _quadratic(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Actual policy-like cross entropy from a two-logit linear predictor."""
    logits = np.broadcast_to(np.stack((theta[0], -theta[0])), (len(target), 2))
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -(target * logp[:, 0] + (1 - target) * logp[:, 1])


def test_real_computation_curvature_overshoot_and_improvement() -> None:
    target = np.full(2, 0.75, dtype=np.float64)
    initial = np.asarray([0.1], dtype=np.float64)
    direction = np.asarray([1.0], dtype=np.float64)
    grad = float(
        np.mean(
            (_quadratic(initial, target + 0.0) - _quadratic(initial - 1e-6, target))
            / 1e-6
        )
    )
    # A positive first-order descent direction can cross the CE minimum at scale one.
    full = float(
        np.mean(_quadratic(initial + direction, target) - _quadratic(initial, target))
    )
    small = float(
        np.mean(
            _quadratic(initial + direction / 8, target) - _quadratic(initial, target)
        )
    )
    assert grad < 0
    assert full > 1e-6
    assert small < -1e-6
    assert (
        float(
            np.mean(
                _quadratic(initial - direction / 8, target)
                - _quadratic(initial, target)
            )
        )
        > 0
    )


def test_zero_displacement_float32_writeback_and_scale_order() -> None:
    initial = np.asarray([1.0, 2.0**24], dtype=np.float32)
    proposal = np.asarray([0.0, 1.0], dtype=np.float32)
    realized = []
    for scale in SCALES:
        product = np.multiply(proposal, np.float32(scale), dtype=np.float32)
        state = np.add(initial, product, dtype=np.float32)
        realized.append(state.astype(np.float64) - initial.astype(np.float64))
    assert np.array_equal(realized[-1], np.zeros(2))
    assert realized[0][1] == 0.0  # float32 writeback at 2**24
    assert SCALES == tuple(sorted(SCALES, reverse=True))


def test_real_model_trial_restoration_preserves_initializer_and_buffers() -> None:
    class Buffered(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("sentinel", torch.tensor([3.0]))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.linear(values) + self.sentinel

    model = Buffered()
    initial = np.concatenate(
        [parameter.detach().numpy().reshape(-1) for parameter in model.parameters()]
    ).astype(np.float32)
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    set_parameters(model, initial + np.float32(0.25))
    set_parameters(model, initial)
    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in frozen.items()
    )


def test_repeated_inputs_keep_occurrence_specific_targets_and_weights() -> None:
    identities = ["same", "same", "other"]
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert identities[0] == identities[1] and not np.array_equal(targets[0], targets[1])
    exposure = np.full(3, 1 / 3)
    equal = np.asarray([1 / 4, 1 / 4, 1 / 2])
    assert exposure.sum() == pytest.approx(1)
    assert equal.sum() == pytest.approx(1)
    # Distinct targets at a repeated identity contribute distinct CE values.
    assert (
        _quadratic(np.asarray([0.7]), np.asarray([1.0]))[0]
        != _quadratic(np.asarray([0.7]), np.asarray([0.0]))[0]
    )


def test_classification_requires_12_batches_and_registered_small_scale() -> None:
    rows = []
    for batch in range(16):
        for scale_index in range(8):
            rows.append(
                {
                    "objective": "P",
                    "direction": "adam",
                    "weighting": "exposure_weighted",
                    "batch": batch,
                    "scale_index": scale_index,
                    "ce_change": ([2e-6] * 3 + [-2e-6] * 5)[scale_index],
                    "predicted_descent_at_1": batch < 12,
                }
            )
    labels = classify(rows)
    assert labels["P/adam/exposure_weighted"] == "finite_step_overshoot_consistent"
    rows = [
        dict(row, ce_change=-2e-6 if row["scale_index"] == 0 else 0.0) for row in rows
    ]
    assert (
        classify(rows)["P/adam/exposure_weighted"]
        == "finite_step_improvement_consistent"
    )


def test_registration_and_receipt_rebound_semantics(tmp_path: Path) -> None:
    """Changing a bound semantic value changes the receipt digest."""
    registration = tmp_path / "registration.json"
    diagnostic = tmp_path / "diagnostic.json"
    registration.write_text(json.dumps({"proposal": [1.0, 2.0]}))
    diagnostic.write_text(json.dumps({"classifications": {"P": "mixed"}}))
    before = registration.read_bytes()
    registration.write_text(json.dumps({"proposal": [1.0, 3.0]}))
    assert registration.read_bytes() != before
    assert diagnostic.exists()


def _fake_bound_publication(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = root / verifier.OUT
    out.mkdir(parents=True)
    for relative in verifier.INPUTS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture input")
    ids = [f"identity-{index % 851}" for index in range(859)]
    indices = list(range(859))
    vectors = {
        "seed469_batch00_P_adam_proposal": np.asarray([0.1, -0.2], np.float32),
        "baseline_exposure_weighted_predictions": np.full(
            (859, 27), 1 / 27, np.float32
        ),
        "cohort_targets": np.full((859, 27), 1 / 27, np.float32),
        "cohort_legal_masks": np.ones((859, 27), np.uint8),
        "cohort_identities": np.asarray(ids, dtype="U256"),
        "cohort_weights_exposure_weighted": np.full(859, 1 / 859, np.float64),
    }
    curves = []
    for batch in range(16):
        for objective in ("P", "J"):
            for direction in ("adam", "negative_gradient"):
                for weighting in ("exposure_weighted", "equal_input"):
                    for scale_index, scale in enumerate(SCALES):
                        ce_change = -2e-6 if scale_index == 0 else 0.0
                        curves.append(
                            {
                                "batch": batch,
                                "objective": objective,
                                "direction": direction,
                                "weighting": weighting,
                                "scale_index": scale_index,
                                "scale": scale,
                                "displacement_norm": 1.0,
                                "ce_change": ce_change,
                                "first_order_prediction": -1e-6,
                                "remainder": ce_change + 1e-6,
                                "predicted_descent": True,
                                "predicted_descent_at_1": True,
                                "other_weighting_ce_change": ce_change,
                            }
                        )
    labels = verifier.expected_classifications(curves)
    counts = {}
    for key in labels:
        selected = [
            row
            for row in curves
            if f"{row['objective']}/{row['direction']}/{row['weighting']}" == key
        ]
        counts[key] = {
            "predicted_descent_observed_ascent_at_scale_1": sum(
                row["predicted_descent_at_1"]
                and row["scale_index"] == 0
                and row["ce_change"] > 1e-6
                for row in selected
            ),
            "predicted_descent_observed_ascent_by_scale": {
                str(scale): sum(
                    row["predicted_descent"] and row["ce_change"] > 1e-6
                    for row in selected
                    if row["scale_index"] == index
                )
                for index, scale in enumerate(SCALES)
            },
            "scale_1_descent": sum(
                row["scale_index"] == 0 and row["ce_change"] < -1e-6 for row in selected
            ),
        }
    id_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    report = {
        "curve_records": curves,
        "classifications": labels,
        "counts": counts,
        "cohort": {
            "selection_indices": indices,
            "occurrences": 859,
            "identities": 851,
            "identity_sha256": id_hash,
        },
    }
    (out / "evidence.npz").write_bytes(b"")
    np.savez_compressed(out / "evidence.npz", **vectors)
    np.savez_compressed(
        out / "trial-states-00-07.npz",
        batch00_P_adam_scale0_trial_parameters=np.asarray([1.1, 1.8], np.float32),
    )
    np.savez_compressed(out / "trial-states-08-15.npz")
    (out / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    registration = {
        "input_sha256": {
            key: verifier.digest(root / relative)
            for key, relative in verifier.INPUTS.items()
        },
        "source_dependency_closure_sha256": {},
        "scales": list(SCALES),
        "seed469_fresh_cohort": {
            "ordered_identity_sha256": id_hash,
            "selection_indices": indices,
        },
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text("synthetic fixture\n")
    receipt = {
        name: verifier.digest(out / name)
        for name in (
            "registration.json",
            "diagnostic.json",
            "evidence.npz",
            "trial-states-00-07.npz",
            "trial-states-08-15.npz",
            "results.md",
        )
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setattr(
        verifier, "INIT_SHA", verifier.digest(root / verifier.INPUTS["initializer"])
    )
    monkeypatch.setattr(verifier, "verify_bundle", lambda *_args: None)
    monkeypatch.setattr(verifier, "read_authenticated_bundle", lambda *_args: None)
    monkeypatch.setattr(
        verifier,
        "independent_outputs",
        lambda _root: (vectors, curves, {"ids": ids, "indices": indices}),
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "proposal",
        "trial_state",
        "prediction",
        "target",
        "weight",
        "effect",
        "remainder",
        "count",
        "classification",
        "missing_trial",
        "duplicate_trial",
    ],
)
def test_receipt_rebound_semantic_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    _fake_bound_publication(tmp_path, monkeypatch)
    out = tmp_path / verifier.OUT
    evidence_path = out / "evidence.npz"
    report_path = out / "diagnostic.json"
    report = json.loads(report_path.read_text())
    if tamper in {"proposal", "trial_state", "prediction", "target", "weight"}:
        vector_path = (
            out / "trial-states-00-07.npz" if tamper == "trial_state" else evidence_path
        )
        with np.load(vector_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        key = {
            "proposal": "seed469_batch00_P_adam_proposal",
            "trial_state": "batch00_P_adam_scale0_trial_parameters",
            "prediction": "baseline_exposure_weighted_predictions",
            "target": "cohort_targets",
            "weight": "cohort_weights_exposure_weighted",
        }[tamper]
        arrays[key] = arrays[key].copy()
        arrays[key].flat[0] += 1
        np.savez_compressed(vector_path, **arrays)
    elif tamper in {"effect", "remainder"}:
        field = "ce_change" if tamper == "effect" else "remainder"
        report["curve_records"][0][field] += 1
    elif tamper == "count":
        report["counts"]["P/adam/exposure_weighted"]["scale_1_descent"] += 1
    elif tamper == "classification":
        report["classifications"]["P/adam/exposure_weighted"] = "forged"
    elif tamper == "missing_trial":
        report["curve_records"].pop()
    else:
        report["curve_records"].append(report["curve_records"][-1])
    if tamper not in {"proposal", "trial_state", "prediction", "target", "weight"}:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for name in (
        "registration.json",
        "diagnostic.json",
        "evidence.npz",
        "trial-states-00-07.npz",
        "trial-states-08-15.npz",
        "results.md",
    ):
        receipt[name] = verifier.digest(out / name)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)
