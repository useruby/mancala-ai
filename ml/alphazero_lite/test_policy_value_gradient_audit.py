import pytest
import torch
import numpy as np

from ml.alphazero_lite.policy_value_gradient_audit import (
    gradient_metrics,
    parameter_groups,
)
from ml.alphazero_lite.run_policy_value_shared_trunk_gradient_audit import classify
from ml.alphazero_lite.train import PolicyValueNet
from ml.alphazero_lite.train import checkpoint_from_model, set_seed, train


def metrics(policy: list[float], value: list[float]):
    p = (torch.tensor(policy),)
    v = (torch.tensor(value),)
    return gradient_metrics(p, v, v)


def test_gradient_geometry_and_zero_are_safe() -> None:
    assert metrics([1, 0], [0, 1])["cosine"] == pytest.approx(0.0)
    assert metrics([1, 0], [1, 0])["cosine"] == pytest.approx(1.0)
    assert metrics([1, 0], [-1, 0])["cosine"] == pytest.approx(-1.0)
    zero = metrics([0, 0], [1, 0])
    assert zero["cosine"] is None
    assert zero["cancellation_ratio"] == 0.0


def test_weighted_value_scale_is_reflected() -> None:
    value = torch.tensor([3.0])
    result = gradient_metrics((torch.tensor([1.0]),), (0.3 * value,), (value,))
    assert result["value_weighted_norm"] == pytest.approx(0.9)
    assert result["value_raw_norm"] == pytest.approx(3.0)


def test_classifier_prefers_value_dominance_when_conflict_rates_are_equal() -> None:
    def epoch(conflict_fraction: float, ratio: float) -> dict[str, float]:
        return {
            "conflict_fraction": conflict_fraction,
            "mean_value_policy_norm_ratio": ratio,
        }

    result = {
        "cohorts": {
            "S455": {"epoch_high": {f"E{i}": epoch(0.25, 1.0) for i in range(1, 5)}},
            "F461": {"epoch_high": {f"E{i}": epoch(0.25, 2.1) for i in range(1, 5)}},
            "U467": {"epoch_high": {f"E{i}": epoch(0.25, 2.1) for i in range(1, 5)}},
        }
    }
    assert classify(result) == "shared_trunk_value_gradient_dominance_failure_signature"


def test_classifier_preserves_published_common_fixture() -> None:
    # Published S455/F461/U467 high-stone summaries, retaining their recorded
    # common classification despite residual epoch-level conflict differences.
    summaries = {
        "S455": [
            (0.292593, 0.066547),
            (0.303704, 0.071165),
            (0.329630, 0.068451),
            (0.325926, 0.066619),
        ],
        "F461": [
            (0.304183, 0.069626),
            (0.281369, 0.071452),
            (0.376426, 0.069248),
            (0.326996, 0.067239),
        ],
        "U467": [
            (0.338403, 0.069218),
            (0.346008, 0.067754),
            (0.300380, 0.066277),
            (0.342205, 0.067647),
        ],
    }
    result = {
        "cohorts": {
            name: {
                "epoch_high": {
                    f"E{epoch}": {
                        "conflict_fraction": conflict,
                        "mean_value_policy_norm_ratio": ratio,
                    }
                    for epoch, (conflict, ratio) in enumerate(values, start=1)
                }
            }
            for name, values in summaries.items()
        }
    }
    assert classify(result) == "shared_trunk_conflict_common_to_all_generations"


def test_parameter_groups_are_complete_and_heads_are_not_trunk() -> None:
    model = PolicyValueNet((8, 3), "residual_v3", 21)
    groups = parameter_groups(model)
    assert sum(map(len, groups.values())) == len(list(model.parameters()))
    input_ids = {id(parameter) for parameter in groups["input_projection"]}
    assert all(
        not name.startswith(("policy_", "value_"))
        for name, parameter in model.named_parameters()
        if id(parameter) in input_ids
    )


def test_observer_is_bytewise_observational(tmp_path) -> None:
    x = np.zeros((20, 21), dtype=np.float32)
    x[:, 0] = 1 / 48
    x[:, 6] = 1 / 48
    policy = np.zeros((20, 6), dtype=np.float32)
    policy[:, 0] = 1.0
    value = np.zeros((20, 1), dtype=np.float32)

    active = []

    def run(observer):
        set_seed(381)
        network = PolicyValueNet((8, 3), "residual_v3", 21)
        active[:] = [network]
        history, permutations = [], []
        train(
            network,
            x,
            policy,
            value,
            np.arange(20),
            epochs=1,
            batch_size=8,
            lr=0.001,
            device=torch.device("cpu"),
            value_loss_weight=0.3,
            value_loss="huber",
            huber_delta=1.0,
            val_split=0.1,
            grad_clip=1.0,
            save_top_k=1,
            lr_scheduler="none",
            loss_observer=observer,
            epoch_history=history,
            permutation_callback=lambda _epoch, values: permutations.append(values),
        )
        return network, history, permutations

    reference, reference_history, reference_permutations = run(None)

    def observer(context) -> None:
        before_rng = torch.get_rng_state().clone()
        network = active[0]
        before_grad = [
            None if parameter.grad is None else parameter.grad.clone()
            for parameter in network.parameters()
        ]
        torch.autograd.grad(
            context["policy_loss_tensor"],
            tuple(network.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        torch.autograd.grad(
            context["value_loss_tensor"],
            tuple(network.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        assert torch.equal(before_rng, torch.get_rng_state())
        assert all(
            (expected is None and parameter.grad is None)
            or (expected is not None and torch.equal(expected, parameter.grad))
            for expected, parameter in zip(before_grad, network.parameters())
        )

    instrumented, history, permutations = run(observer)
    assert reference_history == history
    assert reference_permutations == permutations
    reference_path, observed_path = (
        tmp_path / "reference.npz",
        tmp_path / "observed.npz",
    )
    np.savez(reference_path, **checkpoint_from_model(reference))
    np.savez(observed_path, **checkpoint_from_model(instrumented))
    assert reference_path.read_bytes() == observed_path.read_bytes()
