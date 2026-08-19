from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    HIDDEN_SIZES,
    MODEL_TYPE,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
)
from ml.alphazero_lite import run_function_space_null_direction_audit as audit


def _model() -> PolicyValueNet:
    return PolicyValueNet(HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding("kalah_v3"))


def _standard_state() -> dict:
    return {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }


def test_decompose_is_orthogonal_and_reconstructs() -> None:
    rng = np.random.default_rng(0)
    jacobian = rng.normal(size=(40, 200))
    delta = rng.normal(size=(200,))
    result = audit.decompose(jacobian, delta)
    visible = result["visible"]
    null = result["null"]
    assert np.allclose(visible + null, delta, atol=1e-8)
    assert abs(float(np.dot(visible, null))) < 1e-6
    assert result["null_fraction"] ** 2 + result[
        "visible_fraction"
    ] ** 2 == pytest.approx(1.0, abs=1e-3)


def test_decompose_null_of_jacobian_row_space_is_zero() -> None:
    rng = np.random.default_rng(1)
    jacobian = rng.normal(size=(30, 150))
    output_grad = rng.normal(size=(30,))
    # A gradient in the row space of the Jacobian has no null component.
    delta = jacobian.T @ output_grad
    result = audit.decompose(jacobian, delta)
    assert result["null_fraction"] == pytest.approx(0.0, abs=1e-3)
    assert result["visible_fraction"] == pytest.approx(1.0, abs=1e-3)


def test_singular_spectrum_condition_number_and_energy() -> None:
    rng = np.random.default_rng(2)
    jacobian = rng.normal(size=(50, 250))
    delta = rng.normal(size=(250,))
    spectrum = audit.singular_spectrum(jacobian, delta)
    assert spectrum["eigenvalue_max"] >= spectrum["eigenvalue_min"] >= 0
    assert spectrum["condition_number"] >= 1.0
    assert 0.0 <= spectrum["bottom_half_energy_fraction"] <= 1.0


def test_flat_to_delta_round_trips_through_trunk_shapes() -> None:
    model = _model()
    names, shapes, total = audit.trunk_shapes(model)
    assert total == sum(int(np.prod(shape)) for shape in shapes)
    flat = np.arange(total, dtype=np.float32) * 0.001
    delta = audit.flat_to_delta(flat, names, shapes)
    rebuilt = np.concatenate([delta[name].reshape(-1).numpy() for name in names])
    assert np.array_equal(rebuilt, flat)


def test_compute_output_jacobian_shape_and_finite() -> None:
    model = _model()
    model.eval()
    state = encode_state(_standard_state(), input_encoding="kalah_v3")
    states = np.asarray([state], dtype=np.float32)
    masks = legal_mask_matrix_for_encoded_states(states)
    names, _shapes, _total = audit.trunk_shapes(model)
    jacobian, index = audit.compute_output_jacobian(
        model, states, masks, names, torch.device("cpu")
    )
    # 6 legal logits + 1 value = 7 outputs for the full start state.
    assert jacobian.shape[0] == 7
    assert len(index) == 7
    assert jacobian.shape[1] == _total
    assert bool(np.isfinite(jacobian).all())
    assert np.linalg.norm(jacobian, axis=1).max() > 0


def test_move_change_rate_counts_disagreements() -> None:
    records = {
        ("current", "a"): 1,
        ("current", "b"): 2,
        ("candidate", "a"): 1,
        ("candidate", "b"): 3,
    }
    states = [{"state_hash": "a"}, {"state_hash": "b"}]
    result = audit.move_change_rate(records, "current", "candidate", states)
    assert result["move_change_rate"] == 0.5
    assert result["states"] == 2


def _summary_with_search(search: dict, step_null: float, gradient_null: float) -> dict:
    return {
        "step_decomposition": {"null_fraction": step_null},
        "gradient_decomposition": {"null_fraction": gradient_null},
        "search_effect": search,
    }


def test_classify_null_direction_refuted_when_visible_reproduces_full() -> None:
    search = {
        "probe_full": {"move_change_rate": 0.02},
        "probe_visible": {"move_change_rate": 0.02},
        "probe_null": {"move_change_rate": 0.0},
        "validation_full": {"move_change_rate": 0.03},
        "validation_visible": {"move_change_rate": 0.03},
        "validation_null": {"move_change_rate": 0.002},
    }
    result = audit.classify(_summary_with_search(search, 0.77, 0.046))
    assert result["label"] == "function_space_null_direction_refuted"
    assert "null" in result["next_action"]


def test_classify_null_dominated_when_null_changes_validation() -> None:
    search = {
        "probe_full": {"move_change_rate": 0.02},
        "probe_visible": {"move_change_rate": 0.01},
        "probe_null": {"move_change_rate": 0.0},
        "validation_full": {"move_change_rate": 0.05},
        "validation_visible": {"move_change_rate": 0.01},
        "validation_null": {"move_change_rate": 0.04},
    }
    result = audit.classify(_summary_with_search(search, 0.50, 0.046))
    assert result["label"] == "harmful_direction_function_space_null_dominated"


def test_classify_inconclusive_when_null_small() -> None:
    search = {
        "probe_full": {"move_change_rate": 0.05},
        "probe_visible": {"move_change_rate": 0.05},
        "probe_null": {"move_change_rate": 0.0},
        "validation_full": {"move_change_rate": 0.06},
        "validation_visible": {"move_change_rate": 0.06},
        "validation_null": {"move_change_rate": 0.001},
    }
    result = audit.classify(_summary_with_search(search, 0.20, 0.046))
    assert result["label"] == "function_space_null_direction_inconclusive"


def test_markdown_contains_key_sections() -> None:
    summary = {
        "classification": {
            "label": "function_space_null_direction_refuted",
            "next_action": "pursue search-aware constraints",
            "evidence": {
                "step_null_fraction": 0.77,
                "validation_null_move_change": 0.002,
            },
        },
        "step_decomposition": {
            "null_fraction": 0.77,
            "visible_fraction": 0.63,
            "cosine_null": 0.77,
            "cosine_visible": 0.63,
        },
        "gradient_decomposition": {
            "null_fraction": 0.046,
            "visible_fraction": 0.999,
        },
        "singular_spectrum": {
            "condition_number": 7519.0,
            "eigenvalue_min": 0.04,
            "eigenvalue_max": 2512912.0,
            "bottom_half_energy_fraction": 0.166,
            "output_dim": 2391,
        },
        "search_effect": {
            "probe_full": {"move_change_rate": 0.02},
            "probe_visible": {"move_change_rate": 0.02},
            "probe_null": {"move_change_rate": 0.0},
            "validation_full": {"move_change_rate": 0.03},
            "validation_visible": {"move_change_rate": 0.03},
            "validation_null": {"move_change_rate": 0.002},
        },
    }
    report = audit.markdown(summary)
    for section in (
        "Harmful-step decomposition",
        "Raw-gradient decomposition",
        "Singular-value spectrum",
        "Search-effect contrast",
        "Classification evidence",
        "Next action",
    ):
        assert section in report
