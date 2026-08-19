from __future__ import annotations

import pytest
import torch

from ml.alphazero_lite import run_frozen_trunk_distillation_ablation as runner


def _effect(effect: float, upper_95: float) -> dict:
    return {
        "paired_candidate_effect": effect,
        "opening_bootstrap_ci": {
            "lower_95": effect - 0.02,
            "upper_95": upper_95,
            "samples": 10_000,
        },
    }


def _drift_step(total_loss: float, ce: float, huber: float) -> dict:
    return {
        "total_loss": total_loss,
        "policy": {"replay_teacher_cross_entropy": ce},
        "value": {"huber_loss": huber},
    }


def _summary(
    *,
    full_effects: dict[str, dict],
    heads_effects: dict[str, dict],
    heads_ce_gap: float = 0.0,
    heads_fit_improves: bool = True,
) -> dict:
    incumbent_ce = 0.5
    incumbent_total = 0.5 + 0.6 * 0.3
    full_ce = 0.4
    heads_ce = full_ce + heads_ce_gap
    output = {
        "all": {"0": _drift_step(incumbent_total, incumbent_ce, 0.3)},
        "heads_only": {"0": _drift_step(incumbent_total, incumbent_ce, 0.3)},
    }
    output["all"]["16"] = _drift_step(full_ce + 0.6 * 0.25, full_ce, 0.25)
    heads_total = heads_ce + 0.6 * 0.25
    if not heads_fit_improves:
        heads_total = incumbent_total
    output["heads_only"]["16"] = _drift_step(heads_total, heads_ce, 0.25)
    return {
        "checkpoint_steps": [1, 16],
        "arena": {
            "all": {"1": {"384:256": full_effects["1"]}},
            "heads_only": {"1": {"384:256": heads_effects["1"]}},
        },
        "output_drift": output,
    }


def test_checkpoint_steps() -> None:
    assert runner.checkpoint_steps(46) == [1, 4, 16, 46]
    assert runner.checkpoint_steps(16) == [1, 4, 16]
    assert runner.checkpoint_steps(3) == [1, 3]
    with pytest.raises(ValueError):
        runner.checkpoint_steps(0)


def test_tensors_identical() -> None:
    left = {"a": torch.tensor([1.0, 2.0])}
    same = {"a": torch.tensor([1.0, 2.0])}
    different = {"a": torch.tensor([1.0, 3.0])}
    assert runner.tensors_identical(left, same)
    assert not runner.tensors_identical(left, different)
    assert not runner.tensors_identical(left, {"b": torch.tensor([1.0, 2.0])})


def test_trunk_parameters_identical() -> None:
    incumbent = {
        "input_layer.weight": torch.tensor([1.0]),
        "residual_layers.0.0.weight": torch.tensor([2.0]),
        "policy_head.weight": torch.tensor([3.0]),
    }
    unchanged = {
        "input_layer.weight": torch.tensor([1.0]),
        "residual_layers.0.0.weight": torch.tensor([2.0]),
        "policy_head.weight": torch.tensor([9.0]),
    }
    changed = {
        "input_layer.weight": torch.tensor([1.0]),
        "residual_layers.0.0.weight": torch.tensor([2.5]),
        "policy_head.weight": torch.tensor([9.0]),
    }
    assert runner.trunk_parameters_identical(unchanged, incumbent)
    assert not runner.trunk_parameters_identical(changed, incumbent)


def test_group_delta_separates_groups() -> None:
    incumbent = {
        "input_layer.weight": torch.tensor([1.0, 1.0]),
        "policy_head.weight": torch.tensor([1.0, 1.0]),
        "value_head.weight": torch.tensor([1.0, 1.0]),
    }
    changed = {
        "input_layer.weight": torch.tensor([1.0, 1.0]),
        "policy_head.weight": torch.tensor([1.0, 3.0]),
        "value_head.weight": torch.tensor([1.0, 1.0]),
    }
    delta = runner.group_delta(changed, incumbent)
    assert delta["trunk"] == pytest.approx(0.0)
    assert delta["policy_head"] > 0
    assert delta["value_head"] == pytest.approx(0.0)


def test_with_total_loss_combines_objective() -> None:
    drift = {
        "0": {
            "policy": {"replay_teacher_cross_entropy": 0.5},
            "value": {"huber_loss": 0.3},
        }
    }
    result = runner.with_total_loss(drift)
    assert result["0"]["total_loss"] == pytest.approx(0.5 + 0.6 * 0.3)


def test_classify_heads_only_success() -> None:
    summary = _summary(
        full_effects={"1": _effect(-0.05, -0.01)},
        heads_effects={"1": _effect(0.0, 0.03)},
    )
    result = runner.classify(summary)
    assert result["label"] == "heads_only_success"


def test_classify_heads_only_capacity_limited() -> None:
    summary = _summary(
        full_effects={"1": _effect(-0.05, -0.01)},
        heads_effects={"1": _effect(0.0, 0.03)},
        heads_ce_gap=0.05,
    )
    result = runner.classify(summary)
    assert result["label"] == "heads_only_capacity_limited"


def test_classify_heads_only_not_useful_remains_negative() -> None:
    summary = _summary(
        full_effects={"1": _effect(-0.05, -0.01)},
        heads_effects={"1": _effect(-0.04, -0.01)},
    )
    result = runner.classify(summary)
    assert result["label"] == "heads_only_not_useful"


def test_classify_heads_only_not_useful_no_fit() -> None:
    summary = _summary(
        full_effects={"1": _effect(-0.05, -0.01)},
        heads_effects={"1": _effect(0.0, 0.03)},
        heads_fit_improves=False,
    )
    result = runner.classify(summary)
    assert result["label"] == "heads_only_not_useful"


def test_classify_inconclusive_without_full_harm() -> None:
    summary = _summary(
        full_effects={"1": _effect(0.0, 0.03)},
        heads_effects={"1": _effect(0.0, 0.03)},
    )
    result = runner.classify(summary)
    assert result["label"] == "inconclusive"


def test_classify_inconclusive_without_arena() -> None:
    result = runner.classify({"checkpoint_steps": [1, 16]})
    assert result["label"] == "inconclusive"


def test_markdown_contains_sections() -> None:
    summary = {
        "classification": {
            "label": "heads_only_success",
            "next_action": "promote a heads-only continuation",
            "evidence": {
                "full_harmful_steps": ["1"],
                "heads_safe_where_full_harmful": True,
                "heads_nonnegative": True,
                "heads_fit_improves": True,
                "final_heads_minus_full_ce": 0.001,
                "material_ce_gap": 0.01,
            },
        },
        "deterministic_reproduction": True,
        "sanity": {
            "heads_only_trunk_zero_change": True,
            "lanes_start_identical": True,
        },
        "checkpoint_steps": [1, 4, 16, 46],
        "inputs": {
            "current_weights_sha256": "abc",
            "replay_sha256": "def",
        },
        "output_drift": {
            "incumbent": {"0": _drift_step(0.5, 0.5, 0.3)},
            "all": {"0": _drift_step(0.5, 0.5, 0.3)},
            "heads_only": {"0": _drift_step(0.5, 0.5, 0.3)},
        },
        "drift": {
            "incumbent": {"0": {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}},
            "all": {"0": {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}},
            "heads_only": {"0": {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}},
        },
        "puct": {},
        "arena": {},
    }
    report = runner.markdown(summary)
    for section in (
        "Supervised objective and drift",
        "Search diagnostics",
        "Canonical arena",
        "Classification evidence",
        "Next action",
    ):
        assert section in report
