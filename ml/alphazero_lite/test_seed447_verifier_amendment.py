"""Semantic mutation checks for the read-only seed447 verifier."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite.seed447_joint_output_cap import (
    VALUE_CAP,
    VALUE_TOLERANCE,
    choose_largest_feasible,
    decide,
)
from ml.alphazero_lite.verify_seed447_joint_output_cap import (
    _same_metrics,
    _verify_predictions,
)


class _TinyPredictor(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = inputs[:, :1]
        return torch.cat((value, -value), dim=1), value


def test_scale_mutation_changes_largest_feasible_result() -> None:
    trials = [
        {
            "scale": 1.0,
            "batch_kl": 0.001,
            "guard_kl": 0.001,
            "batch_value_movement": 0.0,
            "guard_value_movement": 0.0,
        },
        {
            "scale": 0.5,
            "batch_kl": 0.001,
            "guard_kl": 0.001,
            "batch_value_movement": 0.0,
            "guard_value_movement": 0.0,
        },
    ]
    assert (
        choose_largest_feasible(trials, cap=VALUE_CAP, tolerance=VALUE_TOLERANCE) == 1.0
    )
    trials[0]["batch_value_movement"] = VALUE_CAP + 2 * VALUE_TOLERANCE
    assert (
        choose_largest_feasible(trials, cap=VALUE_CAP, tolerance=VALUE_TOLERANCE) == 0.5
    )


def test_moment_mutation_is_not_equal_to_replayed_adam_state() -> None:
    gradient = torch.tensor([0.25, -0.5])
    replayed = torch.zeros_like(gradient).lerp(gradient, 0.1).numpy().copy()
    archived = replayed.copy()
    np.testing.assert_array_equal(replayed, archived)
    archived[1] += 1e-6
    assert not np.array_equal(replayed, archived)


def test_prediction_population_identity_and_value_mutations_are_rejected(
    tmp_path,
) -> None:
    model = _TinyPredictor()
    x = np.array([[0.25], [0.5]], dtype=np.float32)
    rows = [
        {"compact_row": 0, "input_identity": "state-a"},
        {"compact_row": 1, "input_identity": "state-b"},
    ]
    logits, values = model(torch.from_numpy(x))
    path = tmp_path / "candidate-predictions.npz"
    np.savez_compressed(
        path,
        compact_rows=np.array([0, 1]),
        input_identity=np.array(["state-a", "state-b"]),
        policy_logits=logits.detach().numpy(),
        value_predictions=values.reshape(-1).detach().numpy(),
    )
    # The verifier checks identities before prediction tensors.
    np.savez_compressed(
        path,
        compact_rows=np.array([0, 2]),
        input_identity=np.array(["state-a", "state-b"]),
        policy_logits=logits.detach().numpy(),
        value_predictions=values.reshape(-1).detach().numpy(),
    )
    with pytest.raises(ValueError, match="prediction_population_identity_invalid"):
        _verify_predictions(tmp_path, "candidate", model, x, rows)
    np.savez_compressed(
        path,
        compact_rows=np.array([0, 1]),
        input_identity=np.array(["state-a", "state-b"]),
        policy_logits=logits.detach().numpy(),
        value_predictions=values.reshape(-1).detach().numpy() + 1e-3,
    )
    with pytest.raises(ValueError, match="prediction_tampered"):
        _verify_predictions(tmp_path, "candidate", model, x, rows)


def test_metrics_and_decision_mutations_are_detected() -> None:
    good = {
        "full_training_objective": 1.0,
        "exposure_weighted": {"policy_ce": 1.0, "value_mse": 0.5},
        "equal_input": {"policy_ce": 1.0, "value_mse": 0.5},
    }
    altered = {**good, "exposure_weighted": {"policy_ce": 1.01, "value_mse": 0.5}}
    with pytest.raises(ValueError, match="metric_mismatch"):
        _same_metrics(good, altered, "test")

    metric_table = {
        label: {
            **good,
            "exposure_weighted": {"policy_ce": policy, "value_mse": 0.5},
            "equal_input": {"policy_ce": policy, "value_mse": 0.5},
        }
        for label, policy in (
            ("initializer", 1.0),
            ("seed442_A", 1.1),
            ("C", 1.0),
            ("T", 0.98),
        )
    }
    metric_table["T"]["full_training_objective"] = 0.9
    steps = [
        {
            "value_cap_activated": True,
            "proposal_norm": 1.0,
            "rejected": False,
        }
    ]
    evidence = {"metrics": metric_table, "arms": {"T": {"steps": steps}}}
    decision = decide(evidence)
    assert (
        decision["classification"]
        == "advance_to_separately_preregistered_strength_experiment"
    )
    metric_table["T"]["equal_input"]["policy_ce"] = 1.0
    changed = decide(evidence)
    assert changed != decision
