import pytest
import numpy as np
import torch

from ml.alphazero_lite.run_r61_grad_clip_stability_ablation import (
    ANCHOR_IMPROVEMENT_THRESHOLD,
    BATCH_SIZE,
    CLIP_LANES,
    EPOCHS,
    LR0,
    MATERIAL_STEP_DIFFERENCE,
    REPLAY,
    TRAINING_SEEDS,
    assert_design,
    classify,
    early_divergence,
    matched_steps,
    step_summary,
    success_rule,
)
from ml.alphazero_lite.train import PolicyValueNet, set_seed, train


def trace(delta: float, preclip: float, threshold: float | None) -> dict:
    scale = min(1.0, threshold / preclip) if threshold is not None else 1.0
    return {
        "optimizer_step": 1,
        "batch_indexes": [1, 2],
        "anchor_mass_step_delta": delta,
        "preclip_gradient_norm": preclip,
        "post_clip_gradient_norm": min(preclip, threshold)
        if threshold is not None
        else preclip,
        "clip_scale": scale,
        "clip_active": threshold is not None and preclip > threshold,
        "policy_loss": 1.0,
        "value_loss": 0.1,
        "anchor_after": {"top_action": 0, "optimal_mass": 0.4},
    }


def cell(seed: str, lane: str, *, delta: float, correct: bool = True) -> dict:
    threshold = dict(CLIP_LANES)[lane]
    preclip = 3.0
    return {
        "training_seed": seed,
        "clip_lane": lane,
        "anchor_optimal_mass_delta": delta,
        "anchor": {"top_is_outcome_optimal": correct},
        "frozen_set": {"cluster": {"optimal_mass": 0.7}},
        "arena_material_regression": False,
        "new_critical_forensic_regression": False,
        "numerical_instability": False,
        "step_trace": [trace(delta, preclip, threshold)],
        "step_summary": step_summary([trace(delta, preclip, threshold)]),
    }


def test_design_is_exactly_r61_t61_t63_b512_lr0_four_epochs_and_four_clips() -> None:
    design = [
        (REPLAY, seed, BATCH_SIZE, LR0, clip)
        for seed in TRAINING_SEEDS.values()
        for _, clip in CLIP_LANES
    ]
    assert_design(design)
    assert (BATCH_SIZE, LR0, EPOCHS, CLIP_LANES) == (
        512,
        0.001,
        4,
        (("C1", 1.0), ("C2", 2.0), ("C4", 4.0), ("CNONE", None)),
    )


def test_design_rejects_replay_seed_batch_lr_and_clip_changes() -> None:
    with pytest.raises(ValueError):
        assert_design([("R62", 61, 512, 0.001, 1.0)] * 8)
    with pytest.raises(ValueError):
        assert_design([("R61", 64, 512, 0.001, 1.0)] * 8)
    with pytest.raises(ValueError):
        assert_design([("R61", 61, 1024, 0.001, 1.0)] * 8)
    with pytest.raises(ValueError):
        assert_design([("R61", 61, 512, 0.0005, 1.0)] * 8)
    with pytest.raises(ValueError):
        assert_design([("R61", 61, 512, 0.001, 0.5)] * 8)


def test_pre_and_post_clip_accounting_and_no_clip_exposure() -> None:
    summary = step_summary([trace(-0.1, 4.0, 2.0), trace(0.2, 1.0, 2.0)])
    exposure = summary["clipping_exposure"]
    assert exposure["fraction_steps_clipped"] == 0.5
    assert exposure["total_unclipped_gradient_norm_exposure"] == 5.0
    assert exposure["total_postclip_gradient_norm_exposure"] == 3.0
    assert exposure["clip_scale"]["p50"] == 0.75
    no_clip = step_summary([trace(0.1, 4.0, None)])
    assert no_clip["clipping_exposure"]["fraction_steps_clipped"] == 0.0
    assert no_clip["clipping_exposure"]["total_postclip_gradient_norm_exposure"] == 4.0


def test_train_none_grad_clip_truly_disables_clipping() -> None:
    x = np.zeros((2, 21), dtype=np.float32)
    p = np.zeros((2, 6), dtype=np.float32)
    p[:, 0] = 1.0
    v = np.zeros((2, 1), dtype=np.float32)
    seen = []
    set_seed(17)
    model = PolicyValueNet((8, 1), "residual_v3", 21)
    train(
        model,
        x,
        p,
        v,
        np.arange(2, dtype=np.int64),
        epochs=1,
        batch_size=2,
        lr=0.001,
        device=torch.device("cpu"),
        value_loss_weight=0.3,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.0,
        grad_clip=None,
        save_top_k=0,
        lr_scheduler="none",
        step_callback=lambda phase, context: seen.append((phase, context.copy())),
    )
    before = next(context for phase, context in seen if phase == "before")
    assert before["grad_clip"] is None
    assert not before["clip_active"]
    assert before["clip_scale"] == 1.0
    assert before["post_clip_gradient_norm"] == before["gradient_norm"]


def test_matched_steps_require_identical_minibatch_order_and_register_material_difference() -> (
    None
):
    cells = []
    for lane, _ in CLIP_LANES:
        item = cell("T61", lane, delta=0.0)
        item["step_trace"] = [
            trace(0.0 if lane == "C1" else 0.06, 3.0, dict(CLIP_LANES)[lane])
        ]
        cells.append(item)
    result = matched_steps(cells, "T61")
    assert result["matched_minibatch_order"]
    assert result["clip_sensitive_step_count"] == 1
    assert result["clip_sensitive_step_threshold"] == MATERIAL_STEP_DIFFERENCE
    cells[-1]["step_trace"][0]["batch_indexes"] = [9]
    with pytest.raises(RuntimeError, match="minibatch order"):
        matched_steps(cells, "T61")


def test_success_rule_and_conservative_winner_selection() -> None:
    cells = []
    for lane, _ in CLIP_LANES:
        cells.extend(
            [
                cell("T61", lane, delta=-0.2289 if lane == "C1" else -0.05),
                cell("T63", lane, delta=0.1592),
            ]
        )
    result = success_rule(cells)
    assert result["C2"]["passes"]
    assert classify(result, cells)[0] == "relaxed_grad_clip_stabilizes_anchor"
    assert ANCHOR_IMPROVEMENT_THRESHOLD == 0.15


def test_early_divergence_reports_first_events_and_longest_interval() -> None:
    rows = [
        {
            **trace(0.0, 1.0, 1.0),
            "optimizer_step": 1,
            "anchor_after": {"top_action": 0, "optimal_mass": 0.6},
        },
        {
            **trace(0.0, 1.0, 1.0),
            "optimizer_step": 2,
            "anchor_after": {"top_action": 3, "optimal_mass": 0.1},
        },
        {
            **trace(0.0, 1.0, 1.0),
            "optimizer_step": 3,
            "anchor_after": {"top_action": 3, "optimal_mass": 0.3},
        },
    ]
    assert early_divergence(rows) == {
        "first_top_action_flip": 2,
        "first_mass_below_0_20": 2,
        "first_mass_above_0_50": 1,
        "longest_continuous_incorrect_interval": 2,
        "final_correctness": False,
    }


def test_guardrails_forbid_self_play_replay_mutation_and_promotion() -> None:
    source = open(__file__.replace("test_", "run_"), encoding="utf-8").read()
    assert '"self_play": False' in source
    assert '"replay_mutation": False' in source
    assert '"promotion": False' in source
    assert "pipeline.py" not in source
