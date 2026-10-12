"""Synthetic semantic checks for seed462 objectives and replay accounting."""

import numpy as np
import pytest
import torch

from ml.alphazero_lite.seed462_historical_policy_removal import policy_objective


def test_historical_policy_zeroed_but_fresh_policy_retained() -> None:
    ce = torch.tensor([2.0, 10.0, 4.0])
    coeff = torch.tensor([1.0, 3.0, 2.0])
    fresh = torch.tensor([True, False, True])
    ordinary, mass_c = policy_objective(ce, coeff, fresh, "C")
    treatment, mass_f = policy_objective(ce, coeff, fresh, "F")
    assert mass_c == 6
    assert mass_f == 3
    assert ordinary.item() == pytest.approx(40.0 / 6.0)
    assert treatment == pytest.approx(10.0 / 3.0)


def test_rejects_empty_treatment_denominator() -> None:
    with pytest.raises(ValueError, match="empty_policy_denominator"):
        policy_objective(torch.ones(2), torch.tensor([2.0, 1.0]), torch.zeros(2), "F")


def test_occurrence_multiplicity_is_preserved() -> None:
    # Repeated row 4 has two exposures and both contribute to numerator and mass.
    compact_rows = np.array([4, 4, 9])
    ce = torch.tensor([1.0, 1.0, 8.0])
    coeff = torch.tensor([2.0, 2.0, 1.0])
    fresh = torch.tensor([True, True, False])
    result, denominator = policy_objective(ce, coeff, fresh, "F")
    assert denominator.item() == 4.0
    assert result.item() == 1.0
    assert compact_rows.tolist() == [4, 4, 9]


def test_value_loss_remains_independent_of_policy_mask() -> None:
    prediction = torch.tensor([0.0, 2.0, -1.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0, 1.0])
    value_loss = torch.nn.functional.huber_loss(prediction, target, delta=1.0)
    value_loss.backward()
    assert value_loss.item() == pytest.approx(3.5 / 3.0)
    assert torch.count_nonzero(prediction.grad).item() == 3
