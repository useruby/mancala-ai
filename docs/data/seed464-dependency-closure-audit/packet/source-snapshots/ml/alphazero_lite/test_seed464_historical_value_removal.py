"""Synthetic unit tests for the seed464 training objectives."""

from __future__ import annotations

import pytest
import torch

from ml.alphazero_lite.seed464_historical_value_removal import policy_loss, value_loss


def test_value_mask_uses_fixed_original_denominator() -> None:
    huber = torch.tensor([1.0, 2.0, 3.0, 4.0])
    fresh = torch.tensor([True, False, True, False])
    assert value_loss(huber, fresh, "C", 4).item() == pytest.approx(2.5)
    assert value_loss(huber, fresh, "T", 4).item() == pytest.approx(1.0)


def test_treatment_zeroes_only_historical_value_contributions() -> None:
    huber = torch.tensor([0.5, 7.0, 1.5], requires_grad=True)
    fresh = torch.tensor([True, False, True])
    treatment = value_loss(huber, fresh, "T", 3)
    treatment.backward()
    assert treatment.item() == pytest.approx(2.0 / 3.0)
    assert huber.grad.tolist() == pytest.approx([1 / 3, 0, 1 / 3])


def test_policy_objective_is_identical_for_both_arms_and_preserves_multiplicity() -> (
    None
):
    # Repeated compact row 8 is deliberately represented by two occurrences.
    occurrence_rows = [8, 8, 11]
    ce = torch.tensor([2.0, 2.0, 100.0])
    coeff = torch.tensor([3.0, 3.0, 4.0])
    fresh = torch.tensor([True, True, False])
    objective, mass = policy_loss(ce, coeff, fresh)
    assert occurrence_rows == [8, 8, 11]
    assert mass.item() == 6.0
    assert objective.item() == pytest.approx(2.0)
    assert torch.equal(objective, policy_loss(ce, coeff, fresh)[0])


def test_empty_fresh_policy_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty_policy_denominator"):
        policy_loss(
            torch.ones(2), torch.tensor([1.0, 2.0]), torch.tensor([False, False])
        )


def test_unknown_value_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid_arm"):
        value_loss(torch.ones(2), torch.ones(2, dtype=torch.bool), "F")
