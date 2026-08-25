import numpy as np
import torch

from ml.alphazero_lite.root_q_trust_region import (
    adam_proposal_and_restore,
    apply_delta,
    choose_lambda,
    q_direction_error,
    root_q_diagnostics,
    select_guard_indexes,
)


def test_q_direction_error_handles_centering_and_degeneracy() -> None:
    assert q_direction_error(np.array([2.0, 3.0]), np.array([4.0, 5.0])) == 0.0
    assert q_direction_error(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == 0.0
    assert q_direction_error(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 1.0


def test_choose_lambda_uses_preregistered_largest_feasible_scale() -> None:
    assert (
        choose_lambda({1.0: 0.3, 0.5: 0.2, 0.25: 0.1, 0.125: 0.05, 0.0: 0.0}, 0.2)
        == 0.5
    )


def test_per_action_rank_disagreement_uses_deterministic_action_ties() -> None:
    candidate = {
        "selected_move": 0,
        "child_stats": [
            {"move": 0, "visits": 3, "q_value": 0.4},
            {"move": 1, "visits": 2, "q_value": 0.4},
            {"move": 2, "visits": 1, "q_value": 0.1},
        ],
    }
    reference = {
        "selected_move": 1,
        "child_stats": [
            {"move": 0, "visits": 3, "q_value": 0.4},
            {"move": 1, "visits": 2, "q_value": 0.5},
            {"move": 2, "visits": 1, "q_value": 0.1},
        ],
    }
    assert (
        root_q_diagnostics(candidate, reference)["per_action_q_rank_disagreement"]
        == 2 / 3
    )


def test_guard_selection_is_hash_tie_broken_and_disjoint() -> None:
    population = [{"state_hash": value} for value in ("c", "a", "b", "d")]
    primary, secondary, manifest = select_guard_indexes(population, [1.0] * 4, 2, 231)
    assert primary.tolist() == [1, 2]
    assert not set(primary) & set(secondary)
    assert len(manifest["sha256_excluding_this_field"]) == 64


def test_lambda_one_proposal_is_byte_equivalent_to_adam_step() -> None:
    torch.manual_seed(7)
    baseline = torch.nn.Parameter(torch.randn(4))
    constrained = torch.nn.Parameter(baseline.detach().clone())
    baseline_optimizer = torch.optim.Adam([baseline], lr=1e-3)
    constrained_optimizer = torch.optim.Adam([constrained], lr=1e-3)
    (baseline.square().sum()).backward()
    baseline_optimizer.step()
    (constrained.square().sum()).backward()
    delta = adam_proposal_and_restore([constrained], constrained_optimizer)
    apply_delta([constrained], delta, 1.0)
    assert torch.equal(baseline, constrained)
    assert (
        baseline_optimizer.state_dict()["param_groups"]
        == constrained_optimizer.state_dict()["param_groups"]
    )
    baseline_state = next(iter(baseline_optimizer.state.values()))
    constrained_state = next(iter(constrained_optimizer.state.values()))
    for state_key, value in baseline_state.items():
        other = constrained_state[state_key]
        assert (
            torch.equal(value, other)
            if isinstance(value, torch.Tensor)
            else value == other
        )
