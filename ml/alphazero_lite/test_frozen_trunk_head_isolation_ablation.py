from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ml.alphazero_lite import run_frozen_trunk_head_isolation_ablation as runner
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import PolicyValueNet


def _effect(effect: float, upper_95: float) -> dict:
    return {
        "paired_candidate_effect": effect,
        "opening_bootstrap_ci": {
            "lower_95": effect - 0.02,
            "upper_95": upper_95,
            "samples": 10_000,
        },
    }


def _summary(
    *,
    heads: dict[str, dict],
    policy: dict[str, dict],
    value: dict[str, dict],
) -> dict:
    return {
        "checkpoint_steps": [1, 4, 16, 46],
        "arena": {
            "heads_only": {"46": {"384:256": heads}},
            "policy_head": {"46": {"384:256": policy}},
            "value_head": {"46": {"384:256": value}},
        },
    }


_HARMS = _effect(-0.26, -0.21)
_SAFE = _effect(-0.01, 0.03)


def test_group_parameters_identical_selects_families() -> None:
    incumbent = {
        "input_layer.weight": torch.tensor([1.0]),
        "policy_head.weight": torch.tensor([2.0]),
        "value_head.weight": torch.tensor([3.0]),
        "value_hidden_layer.weight": torch.tensor([4.0]),
    }
    changed_value = dict(incumbent)
    changed_value["value_head.weight"] = torch.tensor([3.5])
    changed_policy = dict(incumbent)
    changed_policy["policy_head.weight"] = torch.tensor([2.5])

    assert runner.group_parameters_identical(
        incumbent, incumbent, runner.VALUE_STACK_PREFIXES
    )
    assert not runner.group_parameters_identical(
        changed_value, incumbent, runner.VALUE_STACK_PREFIXES
    )
    assert runner.group_parameters_identical(
        changed_policy, incumbent, runner.VALUE_STACK_PREFIXES
    )
    assert not runner.group_parameters_identical(
        changed_policy, incumbent, runner.POLICY_STACK_PREFIXES
    )
    assert runner.group_parameters_identical(
        changed_value, incumbent, runner.POLICY_STACK_PREFIXES
    )
    with pytest.raises(ValueError):
        runner.group_parameters_identical(incumbent, incumbent, ("missing_family.",))


def test_lane_trainable_scopes_are_symmetric() -> None:
    from ml.alphazero_lite.run_frozen_trunk_distillation_ablation import (
        lane_trainable_scope,
    )

    assert lane_trainable_scope("all") is None
    assert lane_trainable_scope("heads_only") == "heads_only"
    assert lane_trainable_scope("policy_head") == "policy_head"
    assert lane_trainable_scope("value_head") == "value_head"
    with pytest.raises(ValueError):
        lane_trainable_scope("unknown_lane")


def test_probe_output_drift_decomposes_policy_and_value_changes() -> None:
    row = {
        "state": np.asarray(
            encode_state(
                KalahGame([4] * 12, [0, 0], 0).to_state(),
                input_encoding="kalah_v3",
            ),
            dtype=np.float32,
        )
    }
    torch.manual_seed(7)
    incumbent = PolicyValueNet((96, 3), "residual_v3", 27)

    value_changed = copy.deepcopy(incumbent.state_dict())
    value_changed["value_head.bias"] = value_changed["value_head.bias"] + 0.25
    policy_changed = copy.deepcopy(incumbent.state_dict())
    policy_changed["policy_head.weight"] = policy_changed["policy_head.weight"] + 0.25

    drift = runner.probe_output_drift(
        [row] * 4,
        {
            0: (incumbent.state_dict(), {}),
            1: (value_changed, {}),
            2: (policy_changed, {}),
        },
    )

    for step in ("0", "1", "2"):
        for family in ("policy", "value"):
            for metric, value in drift[step][family].items():
                assert np.isfinite(value), (step, family, metric)

    assert drift["0"]["policy"]["legal_l1_from_current"] == pytest.approx(0.0)
    assert drift["0"]["value"]["mean_absolute_output_delta"] == pytest.approx(0.0)

    value_only = drift["1"]
    assert value_only["value"]["mean_absolute_output_delta"] > 0.0
    assert value_only["policy"]["legal_l1_from_current"] == pytest.approx(0.0)
    assert value_only["policy"]["top1_change_from_current"] == pytest.approx(0.0)
    assert value_only["policy"]["legal_js_from_current"] == pytest.approx(0.0)

    policy_only = drift["2"]
    assert policy_only["policy"]["legal_l1_from_current"] > 0.0
    assert policy_only["policy"]["legal_js_from_current"] > 0.0
    assert policy_only["value"]["mean_absolute_output_delta"] == pytest.approx(0.0)
    assert policy_only["value"]["signed_mean_output_delta"] == pytest.approx(0.0)


def test_classify_value_head_accumulation() -> None:
    summary = _summary(heads=_HARMS, policy=_SAFE, value=_effect(-0.24, -0.19))
    result = runner.classify(summary)
    assert result["label"] == "value_head_accumulation"
    assert "value-target" in result["next_action"]
    assert result["evidence"]["value_head_reproduces_failure"]
    assert not result["evidence"]["policy_head_reproduces_failure"]


def test_classify_policy_head_accumulation() -> None:
    summary = _summary(heads=_HARMS, policy=_effect(-0.22, -0.17), value=_SAFE)
    result = runner.classify(summary)
    assert result["label"] == "policy_head_accumulation"
    assert "policy-prior" in result["next_action"]


def test_classify_joint_head_search_interaction() -> None:
    summary = _summary(heads=_HARMS, policy=_SAFE, value=_SAFE)
    result = runner.classify(summary)
    assert result["label"] == "joint_head_search_interaction"
    assert "mixed incumbent/candidate" in result["next_action"]


def test_classify_both_heads_independently_harmful() -> None:
    summary = _summary(
        heads=_HARMS,
        policy=_effect(-0.12, -0.07),
        value=_effect(-0.18, -0.13),
    )
    result = runner.classify(summary)
    assert result["label"] == "both_heads_independently_harmful"
    assert "separately" in result["next_action"]


def test_classify_borderline_negative_is_not_material() -> None:
    summary = _summary(
        heads=_HARMS,
        policy=_effect(-0.04, -0.01),
        value=_effect(-0.03, 0.01),
    )
    result = runner.classify(summary)
    assert result["label"] == "joint_head_search_interaction"


def test_classify_heads_failure_not_reproduced_is_inconclusive() -> None:
    summary = _summary(heads=_SAFE, policy=_HARMS, value=_HARMS)
    result = runner.classify(summary)
    assert result["label"] == "inconclusive"


def test_classify_inconclusive_without_arena() -> None:
    result = runner.classify({"checkpoint_steps": [1, 16]})
    assert result["label"] == "inconclusive"
    assert result["evidence"] == {"arena_complete": False}


def test_pr199_state_hashes_are_recorded() -> None:
    assert set(runner.PR199_STATE_HASHES["heads_only"]) == {"0", "1", "4", "16", "46"}
    assert set(runner.PR199_STATE_HASHES["all"]) == {"0", "1", "4", "16", "46"}
    assert (
        runner.PR199_STATE_HASHES["heads_only"]["0"]
        == runner.PR199_STATE_HASHES["all"]["0"]
    )


def test_markdown_contains_sections() -> None:
    summary = {
        "classification": {
            "label": "value_head_accumulation",
            "next_action": "run a value-target experiment",
            "evidence": {
                "material_effect_threshold": -0.05,
                "final_step": 46,
                "final_paired_effects": {
                    "heads_only": -0.26,
                    "policy_head": -0.01,
                    "value_head": -0.24,
                },
                "final_p0_effects": {
                    "heads_only": -0.51,
                    "policy_head": -0.02,
                    "value_head": -0.49,
                },
                "final_p1_effects": {
                    "heads_only": -0.01,
                    "policy_head": 0.0,
                    "value_head": 0.01,
                },
                "final_ci_upper_95": {
                    "heads_only": -0.21,
                    "policy_head": 0.03,
                    "value_head": -0.19,
                },
                "heads_only_reproduces_failure": True,
                "value_head_reproduces_failure": True,
                "policy_head_reproduces_failure": False,
                "materially_harmful_steps": {
                    "heads_only": ["16", "46"],
                    "policy_head": [],
                    "value_head": ["46"],
                },
                "p0_failure_concentrated_in_value_head": True,
            },
        },
        "deterministic_reproduction": {"pr199_heads_only_state_hashes": True},
        "sanity": {
            "lanes_start_identical": True,
            "heads_only_trunk_zero_change": True,
            "policy_head_trunk_zero_change": True,
            "policy_head_value_stack_zero_change": True,
            "value_head_trunk_zero_change": True,
            "value_head_policy_stack_zero_change": True,
        },
        "checkpoint_steps": [1, 4, 16, 46],
        "inputs": {
            "current_weights_sha256": "abc",
            "replay_sha256": "def",
        },
        "output_drift": {
            "incumbent": {
                "0": {
                    "total_loss": 1.28,
                    "policy": {"replay_teacher_cross_entropy": 1.08},
                    "value": {"huber_loss": 0.34},
                }
            },
            "heads_only": {
                "46": {
                    "total_loss": 1.28,
                    "policy": {"replay_teacher_cross_entropy": 1.08},
                    "value": {"huber_loss": 0.34},
                }
            },
        },
        "drift": {
            "incumbent": {"0": {"trunk": 0.0, "policy_head": 0.0, "value_head": 0.0}},
            "heads_only": {
                "46": {"trunk": 0.0, "policy_head": 0.002, "value_head": 0.003}
            },
        },
        "probe_output_drift": {
            "incumbent": {
                "0": {
                    "policy": {
                        "top1_change_from_current": 0.0,
                        "legal_l1_from_current": 0.0,
                        "legal_js_from_current": 0.0,
                    },
                    "value": {
                        "mean_absolute_output_delta": 0.0,
                        "signed_mean_output_delta": 0.0,
                    },
                }
            },
            "heads_only": {
                "46": {
                    "policy": {
                        "top1_change_from_current": 0.02,
                        "legal_l1_from_current": 0.05,
                        "legal_js_from_current": 0.001,
                    },
                    "value": {
                        "mean_absolute_output_delta": 0.01,
                        "signed_mean_output_delta": 0.005,
                    },
                }
            },
        },
        "puct": {
            "heads_only": {
                "metrics": {
                    "46": {
                        "384:256": {
                            "selected_move_change_rate": 0.03,
                            "visit_js": 0.002,
                            "child_q_rank_change": -0.01,
                            "root_value_delta": 0.001,
                        }
                    }
                }
            }
        },
        "arena": {
            "heads_only": {
                "46": {
                    "384:256": {
                        "paired_candidate_effect": -0.26,
                        "opening_bootstrap_ci": {
                            "lower_95": -0.30,
                            "upper_95": -0.21,
                        },
                        "p0_effect": -0.51,
                        "p1_effect": -0.01,
                    }
                }
            }
        },
    }
    report = runner.markdown(summary)
    for section in (
        "Findings",
        "Supervised objective and parameter drift",
        "Network-output drift on the frozen probe",
        "Search diagnostics",
        "Canonical arena",
        "Classification evidence",
        "Recommended next experiment",
        "Exact commands",
    ):
        assert section in report
    assert "value_head_accumulation" in report
    assert (
        "| heads_only | 46 | 384:256 | -0.2600 | [-0.3000, -0.2100] | -0.5100 | -0.0100 |"
        in report
    )
