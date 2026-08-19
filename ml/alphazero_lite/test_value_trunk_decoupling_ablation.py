from __future__ import annotations

import pytest
import torch

from ml.alphazero_lite import run_value_trunk_decoupling_ablation as ablation
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    HIDDEN_SIZES,
    MODEL_TYPE,
)
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


def _model() -> PolicyValueNet:
    return PolicyValueNet(HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding("kalah_v3"))


def test_detach_value_trunk_blocks_value_gradient_into_trunk() -> None:
    model = _model()
    x = torch.randn(3, input_size_for_encoding("kalah_v3"))
    logits, value = model(x, detach_value_trunk=True)
    value.sum().backward()
    trunk_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(("input_layer.", "residual_layers."))
    ]
    assert all(grad is None or torch.count_nonzero(grad) == 0 for grad in trunk_grads)
    # The value head still receives gradient.
    assert model.value_head.weight.grad is not None
    assert torch.count_nonzero(model.value_head.weight.grad) > 0


def test_detach_policy_trunk_blocks_policy_gradient_into_trunk() -> None:
    model = _model()
    x = torch.randn(3, input_size_for_encoding("kalah_v3"))
    logits, _value = model(x, detach_policy_trunk=True)
    logits.sum().backward()
    trunk_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(("input_layer.", "residual_layers."))
    ]
    assert all(grad is None or torch.count_nonzero(grad) == 0 for grad in trunk_grads)
    assert model.policy_head.weight.grad is not None


def test_trunk_delta_measures_relative_change() -> None:
    model = _model()
    base = {name: p.detach().clone() for name, p in model.state_dict().items()}
    changed = {
        name: (p.detach().clone() + (0.1 if name.startswith("input_layer.") else 0.0))
        for name, p in model.state_dict().items()
    }
    delta = ablation.trunk_delta(changed, base)
    assert delta > 0
    assert ablation.trunk_delta(base, base) == pytest.approx(0.0)


def test_domain_metrics_counts_changes() -> None:
    records = {
        ("current", "a"): {"selected_move": 1, "q_ranking": [1, 2]},
        ("current", "b"): {"selected_move": 2, "q_ranking": [2, 1]},
        ("variant", "a"): {"selected_move": 2, "q_ranking": [1, 2]},
        ("variant", "b"): {"selected_move": 2, "q_ranking": [1, 2]},
    }
    states = [{"state_hash": "a"}, {"state_hash": "b"}]
    metrics = ablation.domain_metrics(records, states, ["variant"])
    assert metrics["variant"]["top1_change_rate"] == 0.5
    assert metrics["variant"]["q_ranking_change_rate"] == 0.5


def _summary(
    decoupled_top1: float,
    joint_top1: float,
    heads_top1: float,
    decoupled_qrank: float,
    joint_qrank: float,
) -> dict:
    return {
        "probe_metrics": {
            "value_detached_trunk": {
                "top1_change_rate": decoupled_top1,
                "q_ranking_change_rate": decoupled_qrank,
            },
            "baseline_joint": {
                "top1_change_rate": joint_top1,
                "q_ranking_change_rate": joint_qrank,
            },
            "heads_only": {
                "top1_change_rate": heads_top1,
                "q_ranking_change_rate": heads_top1,
            },
        },
        "trunk_deltas": {
            "value_detached_trunk": 0.01,
            "baseline_joint": 0.02,
        },
    }


def test_classify_reduces_harm() -> None:
    result = ablation.classify(_summary(0.005, 0.02, 0.004, 0.05, 0.13))
    assert result["label"] == "value_trunk_decoupling_reduces_harm"


def test_classify_partial() -> None:
    result = ablation.classify(_summary(0.012, 0.02, 0.004, 0.08, 0.13))
    assert result["label"] == "value_trunk_decoupling_partial"


def test_classify_no_reduction() -> None:
    result = ablation.classify(_summary(0.02, 0.02, 0.004, 0.13, 0.13))
    assert result["label"] == "value_trunk_decoupling_no_reduction"


def test_markdown_contains_sections() -> None:
    summary = {
        "classification": {
            "label": "value_trunk_decoupling_reduces_harm",
            "next_action": "run the full arena",
            "evidence": {"decoupled_trunk_delta": 0.01},
        },
        "deterministic_reproduction": True,
        "trunk_deltas": {
            "baseline_joint": 0.02,
            "heads_only": 0.0,
            "value_detached_trunk": 0.01,
        },
        "probe_metrics": {
            name: {
                "top1_change_rate": 0.01,
                "q_ranking_change_rate": 0.05,
                "states": 1024,
            }
            for name in ("baseline_joint", "heads_only", "value_detached_trunk")
        },
    }
    report = ablation.markdown(summary)
    for section in (
        "Frozen-probe search effect",
        "Classification evidence",
        "Next action",
    ):
        assert section in report
