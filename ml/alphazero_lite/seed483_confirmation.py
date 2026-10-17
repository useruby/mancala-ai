"""Corrected seed483 prospective confirmation rules and analysis primitives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 481
CE_IMPROVEMENT_MINIMUM = 0.001
CE_UPPER_CONFIDENCE = 0.9875
VALUE_MSE_UPPER_CONFIDENCE = 0.975
VALUE_MSE_UPPER_MAXIMUM = 0.01


def empirical_inverse_cdf(values: Sequence[float], probability: float) -> float:
    """Return the clamped order statistic ceil(p*n)-1 (zero-based)."""
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid_quantile_arguments")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = min(max(int(np.ceil(probability * len(ordered))) - 1, 0), len(ordered) - 1)
    return float(ordered[index])


def cluster_bootstrap(
    game_ids: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample game rows with replacement, retaining duplicate draws."""
    ids = np.asarray(game_ids)
    if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
        raise ValueError("game_ids_must_be_nonempty_and_unique")
    if replicates != BOOTSTRAP_REPLICATES or seed != BOOTSTRAP_SEED:
        raise ValueError("bootstrap_protocol_mismatch")
    rng = np.random.Generator(np.random.PCG64(seed))
    output = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_rows = rng.integers(0, len(ids), size=len(ids))
        output[index] = statistic(sampled_rows)
    return output


def confirmation_decision(
    ce: dict[str, tuple[float, float]],
    value_mse_upper: dict[str, float],
) -> str:
    """Apply the six registered gates using improvement and upper-bound inputs."""
    ce_keys = {
        "F_initializer_exposure",
        "F_M_exposure",
        "F_initializer_equal_input",
        "F_M_equal_input",
    }
    value_keys = {"F_initializer_exposure", "F_initializer_equal_input"}
    if set(ce) != ce_keys or set(value_mse_upper) != value_keys:
        raise ValueError("confirmation_contrast_inventory_mismatch")
    passed = all(
        improvement >= CE_IMPROVEMENT_MINIMUM and upper < 0.0
        for improvement, upper in ce.values()
    ) and all(upper <= VALUE_MSE_UPPER_MAXIMUM for upper in value_mse_upper.values())
    return (
        "advance_to_registered_strength_evaluation"
        if passed
        else "stop_fresh_confirmation"
    )


PROTOCOL: dict[str, Any] = {
    "schema": "seed483-corrected-prospective-confirmation-v1",
    "population": "selected seed482 >32-state cohort; no pooled historical cohort",
    "replicates": BOOTSTRAP_REPLICATES,
    "rng_seed": BOOTSTRAP_SEED,
    "bit_generator": "NumPy Generator(PCG64)",
    "sampling_unit": "one selected identity per independent game",
    "resampling": "draw N game IDs with replacement; preserve duplicate draws and their records",
    "quantile": "empirical inverse CDF; sorted order statistic ceil(p*n)-1, zero-based, clamped",
    "ce": {
        "contrasts": ["F-initializer", "F-M"],
        "weightings": ["exposure", "equal_input"],
        "improvement_minimum": CE_IMPROVEMENT_MINIMUM,
        "one_sided_upper_confidence": CE_UPPER_CONFIDENCE,
        "one_sided_upper_bound_strictly_below": 0.0,
    },
    "value_mse": {
        "contrast": "F-initializer",
        "weightings": ["exposure", "equal_input"],
        "one_sided_upper_confidence": VALUE_MSE_UPPER_CONFIDENCE,
        "one_sided_upper_bound_maximum": VALUE_MSE_UPPER_MAXIMUM,
    },
    "decision": "all six pass -> advance_to_registered_strength_evaluation; else stop_fresh_confirmation",
}
