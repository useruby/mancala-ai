"""Adversarial checks for seed439's production semantic validator."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ml.alphazero_lite.verify_seed439_gradient_publication import (
    validate_semantics,
)

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "docs/data/seed438-seed437-gradient-correction/corrected-results.json"
ARCHIVE = (
    ROOT
    / "docs/data/seed438-seed437-gradient-correction/corrected-gradient-vectors.npz"
)


def _publication() -> dict:
    import json

    return json.loads(RESULT.read_text())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["checkpoints"]["initializer"]["metrics"]["exposure"]["T"].update(
            cosine=-0.9
        ),
        lambda r: r["checkpoints"]["initializer"]["metrics"]["exposure"]["T"].update(
            left_norm=0.0
        ),
        lambda r: r["checkpoints"]["initializer"]["metrics"]["exposure"]["T"].update(
            unit_descent_ce_change=9.0
        ),
        lambda r: r["checkpoints"].pop("initializer"),
        lambda r: r["checkpoints"]["initializer"]["vector_evidence"].pop("P"),
        lambda r: r["checkpoints"]["initializer"]["dot_contributions"]["exposure/P"][
            "source"
        ].update(fresh=1e30),
    ],
)
def test_semantic_validator_rejects_forged_publication(mutate) -> None:
    result = copy.deepcopy(_publication())
    mutate(result)
    if "checkpoints" in result and "initializer" not in result["checkpoints"]:
        result["classification"] = "persistent_training_objective_opposition"
    with pytest.raises(ValueError):
        validate_semantics(ROOT, result, ARCHIVE)


def test_semantic_validator_accepts_actual_fixed_decision() -> None:
    checked = validate_semantics(ROOT, _publication(), ARCHIVE)
    assert checked["classification"] == "no_persistent_training_objective_opposition"
    assert len(checked["recomputed_T_U_cosines"]) == 4
