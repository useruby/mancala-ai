import pytest

from ml.alphazero_lite.run_r61_batch_stability_ablation import (
    ANCHOR_IMPROVEMENT_THRESHOLD,
    BATCH_SIZES,
    EPOCHS,
    LR0,
    REPLAY,
    TRAINING_SEEDS,
    assert_design,
    classify,
    matched_progress,
    step_summary,
    success_rule,
)


def cell(
    seed: str, batch: int, *, delta: float, correct: bool = True, p90: float = 0.1
) -> dict:
    return {
        "training_seed": seed,
        "batch_size": batch,
        "anchor_optimal_mass_delta": delta,
        "anchor": {"top_is_outcome_optimal": correct},
        "frozen_set": {"cluster": {"optimal_mass": 0.7}},
        "arena_material_regression": False,
        "still_learning": True,
        "new_critical_forensic_regression": False,
        "step_summary": {"p90_absolute_anchor_step_delta": p90},
    }


def test_design_is_exactly_r61_t61_t63_and_three_physical_batches() -> None:
    design = [
        (REPLAY, seed, batch)
        for seed in TRAINING_SEEDS.values()
        for batch in BATCH_SIZES
    ]
    assert_design(design)
    assert (REPLAY, BATCH_SIZES, LR0, EPOCHS) == ("R61", (512, 1024, 2048), 0.001, 4)


def test_design_rejects_extra_replay_seed_or_batch() -> None:
    with pytest.raises(ValueError):
        assert_design([("R62", 61, 512)] * 6)
    with pytest.raises(ValueError):
        assert_design([("R61", 64, 512)] * 6)
    with pytest.raises(ValueError):
        assert_design([("R61", 61, 4096)] * 6)


def test_step_and_example_accounting() -> None:
    traces = [
        {
            "anchor_mass_step_delta": -0.1,
            "examples_consumed": 512,
            "optimizer_step": 1,
            "anchor_after": {"optimal_mass": 0.2},
            "policy_loss": 1.0,
            "value_loss": 0.2,
        },
        {
            "anchor_mass_step_delta": 0.2,
            "examples_consumed": 1024,
            "optimizer_step": 2,
            "anchor_after": {"optimal_mass": 0.4},
            "policy_loss": 0.8,
            "value_loss": 0.1,
        },
    ]
    assert step_summary(traces)["optimizer_steps"] == 2
    assert matched_progress(traces, 1024)[-1]["examples_consumed"] == 1024


def test_success_requires_both_seeds_and_noise_reduction() -> None:
    cells = []
    for batch in BATCH_SIZES:
        cells.extend(
            [
                cell(
                    "T61",
                    batch,
                    delta=-0.2289 if batch == 512 else -0.05,
                    p90=0.1 if batch == 512 else 0.05,
                ),
                cell("T63", batch, delta=0.1592, p90=0.1 if batch == 512 else 0.05),
            ]
        )
    result = success_rule(cells, {"cluster": {"optimal_mass": 0.65}})
    assert result[1024]["passes"]
    assert result[2048]["passes"]
    assert classify(result, cells)[0] == "larger_batch_stabilizes_anchor"
    assert ANCHOR_IMPROVEMENT_THRESHOLD == 0.15


def test_guardrails_forbid_self_play_replay_mutation_and_promotion() -> None:
    source = open(__file__.replace("test_", "run_"), encoding="utf-8").read()
    assert '"self_play": False' in source
    assert '"replay_mutation": False' in source
    assert '"promotion": False' in source
    assert "pipeline.py" not in source
