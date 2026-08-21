from __future__ import annotations

import numpy as np

from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (
    CONTEXTS,
    assemble_factorial,
    paired_contrast,
)


def _effects() -> dict[str, dict]:
    values = {
        "384:256": 0.0,
        "384:384": 0.5,
        "1200:256": -1.0,
        "384:1200": -2.0,
        "1200:1200": -4.0,
    }
    return {
        context: {"per_opening_effect": {index: value for index in range(128)}}
        for context, value in values.items()
    }


def test_factorial_contrasts_use_opening_paired_effects() -> None:
    contrasts = assemble_factorial(_effects())

    assert set(contrasts) == {
        "candidate_search_increment_low_parent",
        "parent_search_increment_low_candidate",
        "equalization_384",
        "candidate_search_increment_high_parent",
        "parent_search_increment_high_candidate",
        "high_high_interaction",
    }
    assert contrasts["candidate_search_increment_low_parent"]["effect"] == -1.0
    assert contrasts["parent_search_increment_low_candidate"]["effect"] == -2.0
    assert contrasts["equalization_384"]["effect"] == 0.5
    assert contrasts["candidate_search_increment_high_parent"]["effect"] == -2.0
    assert contrasts["parent_search_increment_high_candidate"]["effect"] == -3.0
    assert contrasts["high_high_interaction"]["effect"] == -1.0


def test_paired_contrast_rejects_unmatched_openings() -> None:
    effects = _effects()
    effects[CONTEXTS[1]]["per_opening_effect"] = {0: 0.5}

    try:
        paired_contrast(
            effects, {"384:256": 1.0, "384:384": -1.0}, np.zeros((2, 2), dtype=int)
        )
    except ValueError as error:
        assert "same openings" in str(error)
    else:
        raise AssertionError("unmatched opening sets must fail")
