import copy

import pytest
import torch

from ml.alphazero_lite.run_r61_early_trunk_replay_provenance_audit import (
    FAMILY_FIELDS,
    FORMATION_END,
    a0_movement,
    batch_hash,
    clone_adam_step,
    cohort,
    evaluate_step,
    family_summaries,
    finite_probe_descent_delta,
    hard_classification,
    input_vector,
    ranked_steps,
    select_candidate_family,
    tensor_snapshot,
)
from ml.alphazero_lite.run_r61_activation_representation_audit import (
    residual_v3_activations,
)
from ml.alphazero_lite.train import PolicyValueNet


def model() -> PolicyValueNet:
    return PolicyValueNet((8, 3), "residual_v3", 21)


def entry(membership: str = "failure_cluster") -> dict:
    return {
        "id": membership,
        "membership": membership,
        "state": {
            "player_pits": [1, 2, 3, 4, 5, 6],
            "opponent_pits": [6, 5, 4, 3, 2, 1],
            "player_store": 2,
            "opponent_store": 3,
            "current_player": 0,
        },
        "legal_actions": [0, 1, 2, 3, 4, 5],
        "exact_outcome_optimal_actions": [0],
    }


def test_batch_hash_preserves_order_and_repeated_exposures() -> None:
    assert batch_hash([1, 2, 1]) == batch_hash([1, 2, 1])
    assert batch_hash([1, 2, 1]) != batch_hash([1, 1, 2])


def test_a0_movement_reports_relu_support_changes() -> None:
    movement = a0_movement(torch.tensor([[0.0, 1.0]]), torch.tensor([[2.0, 0.0]]))
    assert movement["activated"] == movement["deactivated"] == 1
    assert movement["support_flips"] == 2


def test_a0_depends_only_on_input_layer() -> None:
    network, altered = model(), model()
    altered.load_state_dict(network.state_dict())
    x = torch.rand(2, 21)
    with torch.no_grad():
        altered.policy_head.bias.add_(1.0)
    assert torch.equal(
        residual_v3_activations(network, x)["A0"],
        residual_v3_activations(altered, x)["A0"],
    )
    with torch.no_grad():
        altered.input_layer.bias.add_(1.0)
    assert not torch.equal(
        residual_v3_activations(network, x)["A0"],
        residual_v3_activations(altered, x)["A0"],
    )


def test_fixed_pre_and_post_context_a0_effects_are_recorded() -> None:
    before, after = (
        PolicyValueNet((8, 3), "residual_v3", 27),
        PolicyValueNet((8, 3), "residual_v3", 27),
    )
    with torch.no_grad():
        after.input_layer.bias.add_(0.1)
    rows, summary = evaluate_step(
        before.eval(), after.eval(), [entry(), entry("matched_control")]
    )
    assert {row["cohort"] for row in rows} == {"cluster", "controls"}
    assert "a0_only_margin_delta" in rows[0]
    assert "a0_only_margin_delta_post_context" in rows[0]
    assert summary["cluster_specific_a0_effect"] == pytest.approx(
        summary["cluster_a0_effect"] - summary["control_a0_effect"]
    )


def test_cluster_probe_gradient_sign_matches_finite_gradient_descent() -> None:
    network = PolicyValueNet((8, 3), "residual_v3", 27).eval()
    assert finite_probe_descent_delta(network, [entry()]) <= 1e-6


def test_ranked_steps_uses_fixed_formation_boundary_and_stable_ties() -> None:
    rows = [
        {"optimizer_step": 1, "cluster_specific_a0_effect": -1.0},
        {"optimizer_step": FORMATION_END, "cluster_specific_a0_effect": -1.0},
        {"optimizer_step": FORMATION_END + 1, "cluster_specific_a0_effect": -99.0},
    ]
    ranking = ranked_steps(rows[:2])
    assert ranking["worst_1"] == [1]
    assert ranking["negative_concentration"]["1"] == pytest.approx(0.5)


def test_candidate_selection_is_mechanical() -> None:
    family = {
        "family_id": "source:dynamic",
        "harmful_batches": 5,
        "enrichment": 2.0,
        "negative_per_exposure": -2.0,
        "baseline_negative_per_exposure": -1.0,
        "median_cluster_probe_alignment": -0.2,
        "median_control_probe_alignment": -0.1,
        "negative_contribution": 3.0,
    }
    assert select_candidate_family([family]) == family
    assert select_candidate_family([family | {"harmful_batches": 4}]) is None


def test_family_definitions_are_pre_registered_and_non_conjunctive() -> None:
    assert FAMILY_FIELDS == (
        "source",
        "source_phase",
        "source_player",
        "source_legal_count",
        "source_capture_available",
        "source_extra_turn_available",
        "source_value_sign",
        "source_structural_neighbor",
    )


def test_family_selection_uses_worst_batch_prevalence_and_raw_exposures() -> None:
    metadata = [
        {
            "source": "dynamic",
            "phase": "early",
            "current_player": 0,
            "legal_action_count": 2,
            "capture_available": False,
            "extra_turn_available": False,
            "value_target_sign": "positive",
            "structural_neighbor": False,
        },
        {
            "source": "fixed:a",
            "phase": "late",
            "current_player": 1,
            "legal_action_count": 3,
            "capture_available": True,
            "extra_turn_available": True,
            "value_target_sign": "negative",
            "structural_neighbor": True,
        },
    ]
    steps = [
        {
            "optimizer_step": step,
            "batch_indexes": [0] if step < 3 else [1],
            "cluster_specific_a0_effect": -1.0 if step == 1 else 0.1,
            "input_gradient": {
                "cluster_probe_alignment": -0.2,
                "control_probe_alignment": -0.1,
            },
        }
        for step in range(1, 5)
    ]
    rows = family_summaries(steps, metadata, {1})
    dynamic = next(row for row in rows if row["family_id"] == "source:dynamic")
    assert dynamic["formation_exposures"] == 2
    assert dynamic["harmful_batch_prevalence"] == pytest.approx(1.0)


def test_hard_classification_requires_counterfactual_stability() -> None:
    ranking = {"negative_concentration": {"20": 0.7}}
    counterfactual = {
        "content_stable_fraction": 0.7,
        "state_dependent_fraction": 0.0,
        "cluster_harm_stronger_than_controls": True,
    }
    classification, _ = hard_classification(
        {"family_field": "source"}, ranking, counterfactual
    )
    assert classification == "early_trunk_replay_source_identified"
    classification, _ = hard_classification(
        None,
        {"negative_concentration": {"20": 0.1}},
        counterfactual | {"content_stable_fraction": 0.0},
    )
    assert classification == "early_trunk_diffuse_replay_interference"


def test_cloned_adam_step_reproduces_historical_update() -> None:
    network = model()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    before, optimizer_before = (
        tensor_snapshot(network),
        copy.deepcopy(optimizer.state_dict()),
    )
    x = torch.rand(2, 21).numpy()
    policy = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]).numpy()
    value = torch.zeros(2, 1).numpy()
    clone = clone_adam_step(before, optimizer_before, x, policy, value)
    assert isinstance(clone, PolicyValueNet)
    assert torch.equal(input_vector(before), input_vector(tensor_snapshot(network)))
    assert optimizer.state_dict() == optimizer_before


def test_frozen_entries_are_only_cohorted_for_evaluation() -> None:
    assert cohort(entry("cluster_anchor")) == "anchor"
    assert cohort(entry("matched_control")) == "controls"
    assert cohort(entry()) == "cluster"
