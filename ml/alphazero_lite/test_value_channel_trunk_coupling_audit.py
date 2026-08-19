from __future__ import annotations

import pytest
import torch

from ml.alphazero_lite import run_value_channel_trunk_coupling_audit as audit
from ml.alphazero_lite.run_game_shard_gradient_stability_audit import parameter_group
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    HIDDEN_SIZES,
    MODEL_TYPE,
)
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


def _named_grads() -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]:
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding("kalah_v3")
    )
    grads = {
        name: torch.ones_like(parameter, dtype=torch.float32) * 0.5
        for name, parameter in model.named_parameters()
    }
    return grads, list(model.named_parameters())


def test_mask_gradients_keeps_only_requested_group() -> None:
    grads, named = _named_grads()
    masked = audit.mask_gradients(grads, frozenset({"shared_trunk"}))
    for name, parameter in named:
        if parameter_group(name) == "shared_trunk":
            assert torch.equal(masked[name], grads[name])
        else:
            assert torch.count_nonzero(masked[name]) == 0


def test_domain_metrics_counts_top1_and_ranking_changes() -> None:
    records = {
        ("current", "a"): {
            "selected_move": 1,
            "q_ranking": [1, 2, 3],
            "root_value": 0.0,
        },
        ("current", "b"): {
            "selected_move": 2,
            "q_ranking": [2, 3, 1],
            "root_value": 0.1,
        },
        ("variant", "a"): {
            "selected_move": 2,
            "q_ranking": [1, 2, 3],
            "root_value": 0.05,
        },
        ("variant", "b"): {
            "selected_move": 2,
            "q_ranking": [3, 2, 1],
            "root_value": 0.3,
        },
    }
    states = [{"state_hash": "a"}, {"state_hash": "b"}]
    metrics = audit.domain_metrics(records, states, ["variant"])
    entry = metrics["variant"]
    assert entry["top1_change_rate"] == 0.5
    assert entry["q_ranking_change_rate"] == 0.5
    assert entry["root_value_mean_delta"] == pytest.approx(0.125)


def _summary(
    head_top1: float, trunk_top1: float, full_top1: float, head_rank: float = 0.0
) -> dict:
    return {
        "probe_metrics": {
            "value_full": {
                "top1_change_rate": full_top1,
                "q_ranking_change_rate": full_top1,
            },
            "value_head_only": {
                "top1_change_rate": head_top1,
                "q_ranking_change_rate": head_rank,
            },
            "value_trunk_only": {
                "top1_change_rate": trunk_top1,
                "q_ranking_change_rate": trunk_top1,
            },
            "policy_trunk_only": {"top1_change_rate": 0.01},
            "policy_head_only": {"top1_change_rate": 0.01},
        }
    }


def test_classify_trunk_coupling() -> None:
    result = audit.classify(_summary(0.002, 0.018, 0.019))
    assert result["label"] == "value_channel_harm_from_trunk_coupling"
    assert "trunk" in result["next_action"]


def test_classify_q_value_change() -> None:
    result = audit.classify(_summary(0.015, 0.0, 0.015))
    assert result["label"] == "value_channel_harm_from_q_value_change"


def test_classify_inconclusive() -> None:
    result = audit.classify(_summary(0.002, 0.004, 0.006))
    assert result["label"] == "value_channel_trunk_coupling_inconclusive"


def test_markdown_contains_sections() -> None:
    summary = {
        "classification": {
            "label": "value_channel_harm_from_trunk_coupling",
            "next_action": "decouple the value head",
            "evidence": {"value_head_top1_change": 0.002},
        },
        "probe_metrics": {
            name: {
                "top1_change_rate": 0.01,
                "q_ranking_change_rate": 0.01,
                "root_value_mean_delta": 0.0,
            }
            for name in (
                "value_full",
                "value_head_only",
                "value_trunk_only",
                "policy_full",
                "policy_head_only",
                "policy_trunk_only",
                "joint_full",
            )
        },
    }
    report = audit.markdown(summary)
    for section in (
        "Probe (training) decomposition",
        "Classification evidence",
        "Next action",
    ):
        assert section in report
