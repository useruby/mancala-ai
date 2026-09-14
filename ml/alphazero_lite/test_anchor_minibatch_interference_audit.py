import copy

import numpy as np
import torch

from ml.alphazero_lite.run_anchor_minibatch_interference_audit import (
    CELLS,
    STEP_THRESHOLD,
    assert_primary_cells,
    classify_step,
    concentration,
    gradient_cosines,
    summarize_steps,
)
from ml.alphazero_lite.train import PolicyValueNet, set_seed, train


def test_audit_selects_exact_pr307_two_by_two() -> None:
    assert_primary_cells(list(CELLS))
    assert {f"{replay}-{seed}" for replay, seed, _ in CELLS} == {
        "R61-T61",
        "R61-T63",
        "R62-T61",
        "R62-T63",
    }


def test_step_classification_uses_preregistered_threshold() -> None:
    assert classify_step(-STEP_THRESHOLD) == "harmful_step"
    assert classify_step(STEP_THRESHOLD) == "protective_step"
    assert classify_step(0.009) == "neutral_step"


def test_top_negative_movement_accounting() -> None:
    trace = [{"anchor_mass_step_delta": value} for value in (-0.5, -0.25, -0.25, 0.1)]
    assert concentration(trace) == {"1": 0.5, "5": 1.0, "10": 1.0}
    assert summarize_steps(trace)["mechanism"] == "spike_driven"


def test_gradient_probe_does_not_mutate_parameters_or_optimizer_state() -> None:
    set_seed(7)
    model = PolicyValueNet((8, 1), "residual_v3", 21)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.rand(2, 21)
    logits, _value = model(x)
    logits.sum().backward()
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    result = gradient_cosines(model, x[:1], [0, 1, 2, 3, 4, 5])
    assert set(result) == {"policy_head", "penultimate_shared"}
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_parameters, model.parameters())
    )
    assert optimizer.state_dict() == before_optimizer


def test_step_callback_does_not_change_training_checkpoint() -> None:
    x = np.zeros((4, 21), dtype=np.float32)
    x[:, :6] = 1 / 48
    p = np.zeros((4, 6), dtype=np.float32)
    p[:, 0] = 1.0
    v = np.zeros((4, 1), dtype=np.float32)
    indexes = np.arange(4, dtype=np.int64)

    def run(callback):
        set_seed(11)
        model = PolicyValueNet((8, 1), "residual_v3", 21)
        train(
            model,
            x,
            p,
            v,
            indexes,
            epochs=1,
            batch_size=2,
            lr=0.001,
            device=torch.device("cpu"),
            value_loss_weight=0.3,
            value_loss="huber",
            huber_delta=1.0,
            val_split=0.0,
            grad_clip=1.0,
            save_top_k=0,
            lr_scheduler="none",
            step_callback=callback,
        )
        return [value.detach().clone() for value in model.state_dict().values()]

    observed = []
    plain = run(None)
    traced = run(lambda phase, row: observed.append((phase, row["batch_indexes"])))
    assert observed
    assert all(torch.equal(left, right) for left, right in zip(plain, traced))
