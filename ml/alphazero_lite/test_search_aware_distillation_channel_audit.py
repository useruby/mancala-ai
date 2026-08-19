from __future__ import annotations

import pytest

from ml.alphazero_lite import run_search_aware_distillation_channel_audit as audit


def test_reference_alignment_metrics() -> None:
    records = {
        ("current", "a"): 1,
        ("current", "b"): 2,
        ("current", "c"): 3,
        ("current", "d"): 2,
        ("candidate", "a"): 1,
        ("candidate", "b"): 3,
        ("candidate", "c"): 2,
        ("candidate", "d"): 2,
    }
    reference = {"a": 1, "b": 3, "c": 3, "d": 1}
    states = [
        {"state_hash": "a"},
        {"state_hash": "b"},
        {"state_hash": "c"},
        {"state_hash": "d"},
    ]
    result = audit.reference_alignment(records, "candidate", reference, states)
    assert result["states"] == 4
    # b flips toward reference (2 -> 3 == ref); c flips away (3 -> 2 != ref == 3).
    assert result["move_change_rate"] == pytest.approx(2 / 4)
    assert result["flips_toward_reference"] == pytest.approx(1 / 4)
    assert result["flips_away_from_reference"] == pytest.approx(1 / 4)
    assert result["net_reference_delta"] == pytest.approx(0.0)
    # candidate matches reference on a and b (2 of 4).
    assert result["reference_alignment"] == pytest.approx(2 / 4)


def _channel(move_change: float, net_delta: float) -> dict:
    return {
        "move_change_rate": move_change,
        "reference_alignment": 0.85,
        "flips_toward_reference": 0.005,
        "flips_away_from_reference": 0.005,
        "net_reference_delta": net_delta,
    }


def test_classify_policy_misaligned() -> None:
    summary = {
        "probe_metrics": {
            "joint": _channel(0.03, -0.01),
            "policy": _channel(0.02, -0.01),
            "value": _channel(0.001, 0.0),
        },
        "validation_metrics": {
            "joint": _channel(0.03, -0.01),
            "policy": _channel(0.02, -0.01),
            "value": _channel(0.001, 0.0),
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "policy_channel_search_misaligned"
    assert "policy" in result["next_action"]


def test_classify_value_misaligned() -> None:
    summary = {
        "probe_metrics": {
            "joint": _channel(0.03, -0.01),
            "policy": _channel(0.001, 0.0),
            "value": _channel(0.02, -0.01),
        },
        "validation_metrics": {
            "joint": _channel(0.03, -0.01),
            "policy": _channel(0.001, 0.0),
            "value": _channel(0.02, -0.01),
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "value_channel_search_misaligned"


def test_classify_both_misaligned() -> None:
    summary = {
        "probe_metrics": {
            "joint": _channel(0.03, -0.02),
            "policy": _channel(0.02, -0.01),
            "value": _channel(0.02, -0.01),
        },
        "validation_metrics": {
            "joint": _channel(0.03, -0.02),
            "policy": _channel(0.02, -0.01),
            "value": _channel(0.02, -0.01),
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "both_channels_search_misaligned"
    assert "do not change both" in result["next_action"]


def test_classify_not_confirmed_when_no_channel_misaligned() -> None:
    summary = {
        "probe_metrics": {
            "joint": _channel(0.03, -0.002),
            "policy": _channel(0.02, -0.004),
            "value": _channel(0.02, -0.003),
        },
        "validation_metrics": {
            "joint": _channel(0.03, -0.002),
            "policy": _channel(0.02, -0.004),
            "value": _channel(0.02, -0.003),
        },
    }
    result = audit.classify(summary)
    assert result["label"] == "search_aware_top1_misalignment_not_confirmed"


def test_markdown_contains_sections() -> None:
    summary = {
        "classification": {
            "label": "policy_channel_search_misaligned",
            "next_action": "test a search-aware policy target",
            "evidence": {"policy_probe_move_change": 0.02},
        },
        "gradient_norms": {"policy_grad_norm": 3.0, "value_grad_norm": 0.4},
        "probe_metrics": {
            key: {
                "move_change_rate": 0.01,
                "reference_alignment": 0.6,
                "flips_toward_reference": 0.005,
                "flips_away_from_reference": 0.01,
                "net_reference_delta": -0.005,
            }
            for key in ("joint", "policy", "value")
        },
        "validation_metrics": {
            key: {
                "move_change_rate": 0.01,
                "reference_alignment": 0.6,
                "flips_toward_reference": 0.005,
                "flips_away_from_reference": 0.01,
                "net_reference_delta": -0.005,
            }
            for key in ("joint", "policy", "value")
        },
    }
    report = audit.markdown(summary)
    for section in (
        "Probe (training) alignment",
        "Validation (held-out) alignment",
        "Classification evidence",
        "Next action",
    ):
        assert section in report
