import pytest

from ml.alphazero_lite.run_r61_lr_stability_ablation import (
    BATCH_SIZE,
    EPOCHS,
    LR0,
    LR_MULTIPLIERS,
    REPLAY,
    REPLAY_WEIGHTS,
    TRAINING_SEEDS,
    assert_design,
    classify,
    sign_consistency,
    success_rule,
)


def cell(seed: str, multiplier: float, *, delta: float, correct: bool = True) -> dict:
    return {
        "training_seed": seed,
        "lr_multiplier": multiplier,
        "anchor_optimal_mass_delta": delta,
        "anchor_forgotten": not correct,
        "anchor": {"top_is_outcome_optimal": correct},
        "frozen_set": {"cluster": {"optimal_mass": 0.7}},
        "arena_material_regression": False,
        "still_learning": True,
        "new_critical_forensic_regression": False,
        "step_trace": [
            {"anchor_mass_step_delta": -0.01},
            {"anchor_mass_step_delta": 0.01},
        ],
    }


def test_design_is_exactly_r61_six_cells_and_historical_exposure() -> None:
    design = [
        (REPLAY, seed, multiplier)
        for seed in TRAINING_SEEDS.values()
        for multiplier in LR_MULTIPLIERS
    ]
    assert_design(design)
    assert REPLAY == "R61"
    assert set(TRAINING_SEEDS) == {"T61", "T63"}
    assert LR_MULTIPLIERS == (1.0, 0.5, 0.25)
    assert LR0 == 0.001
    assert (EPOCHS, BATCH_SIZE, REPLAY_WEIGHTS) == (4, 512, (1, 1, 2))


def test_design_rejects_new_seed_replay_or_lr() -> None:
    with pytest.raises(ValueError):
        assert_design([("R62", 61, 1.0)] * 6)
    with pytest.raises(ValueError):
        assert_design([("R61", 64, 1.0)] * 6)
    with pytest.raises(ValueError):
        assert_design([("R61", 61, 0.75)] * 6)


def test_sign_consistency_requires_matching_step_exposure() -> None:
    cells = [cell("T61", multiplier, delta=0.0) for multiplier in LR_MULTIPLIERS]
    result = sign_consistency(cells, "T61")
    assert result["negative_at_all_lr"] == 1
    assert result["positive_at_all_lr"] == 1
    assert result["sign_agreement_rate"] == 1.0
    cells[-1]["step_trace"].pop()
    with pytest.raises(RuntimeError, match="identical optimizer exposure"):
        sign_consistency(cells, "T61")


def test_success_rule_requires_both_seed_safety_and_learning() -> None:
    cells = []
    for multiplier in LR_MULTIPLIERS:
        cells.append(
            cell("T61", multiplier, delta=-0.2289 if multiplier == 1.0 else -0.05)
        )
        cells.append(cell("T63", multiplier, delta=0.1592))
    result = success_rule(cells, {"cluster": {"optimal_mass": 0.65}})
    assert result[0.5]["passes"]
    assert result[0.25]["passes"]
    classification, _next = classify(result, cells)
    assert classification == "reduced_lr_stabilizes_anchor_without_strength_loss"
    cells[3]["anchor"]["top_is_outcome_optimal"] = False
    assert not success_rule(cells, {"cluster": {"optimal_mass": 0.65}})[0.5]["passes"]


def test_guardrails_forbid_self_play_replay_mutation_and_promotion() -> None:
    source = open(__file__.replace("test_", "run_"), encoding="utf-8").read()
    assert 'self_play": False' in source
    assert 'replay_mutation": False' in source
    assert 'promotion": False' in source
    assert "pipeline.py" not in source
