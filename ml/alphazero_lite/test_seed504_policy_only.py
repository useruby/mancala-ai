from __future__ import annotations

import numpy as np
import torch
import json
from pathlib import Path

from ml.alphazero_lite import train
from ml.alphazero_lite import verify_seed504_policy_only as verifier


def test_policy_only_training_updates_policy_but_preserves_value_head() -> None:
    torch.manual_seed(504)
    state = np.asarray(
        [1 / 12] * 12 + [0.0] * 3 + [1.0] + [0.0] * 2 + [1 / 6, 0.5, 1.0] + [0.0] * 6,
        dtype=np.float32,
    )
    x = np.stack((state, state))
    policy = np.zeros((2, 6), dtype=np.float32)
    policy[:, 0] = 1.0
    values = np.asarray([[0.0], [1.0]], dtype=np.float32)
    model = train.PolicyValueNet((8, 3), "residual_v3", 27)
    initial = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    for name, parameter in model.named_parameters():
        if name.startswith(("value_hidden_layer.", "value_head.")):
            parameter.requires_grad_(False)
    history: list[dict] = []
    train.train(
        model,
        x,
        policy,
        values,
        np.arange(2, dtype=np.int64),
        epochs=1,
        batch_size=2,
        lr=0.0001,
        device=torch.device("cpu"),
        value_loss_weight=0.0,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.0,
        grad_clip=1.0,
        save_top_k=0,
        lr_scheduler="none",
        weight_decay=0.0,
        epoch_history=history,
        primary_order_seed=504,
        final_checkpoint="final",
    )
    current = dict(model.named_parameters())
    assert any(
        not torch.equal(current[name].detach(), before)
        for name, before in initial.items()
        if name.startswith(("input_layer.", "policy_hidden_layer.", "policy_head."))
    )
    for name, before in initial.items():
        if name.startswith(("value_hidden_layer.", "value_head.")):
            assert torch.equal(current[name].detach(), before)
    assert history[0]["total_loss"] == history[0]["policy_loss"]
    assert history[0]["optimizer_updates"] == 1


def test_development_weighting_counts_duplicate_inputs_once_per_identity() -> None:
    from ml.alphazero_lite import run_seed504

    state = (
        [1 / 12] * 12 + [0.0] * 3 + [1.0] + [0.0] * 2 + [1 / 6, 0.5, 1.0] + [0.0] * 6
    )
    rows = [
        {
            "state": state,
            "policy": [1, 0, 0, 0, 0, 0],
            "value": 0.0,
            "_source": "s",
            "_line": 1,
        },
        {
            "state": state,
            "policy": [0, 1, 0, 0, 0, 0],
            "value": 1.0,
            "_source": "s",
            "_line": 2,
        },
    ]
    model = train.PolicyValueNet((8, 3), "residual_v3", 27).eval()
    metrics, _ = run_seed504.evaluate(model, rows)
    assert metrics["pooled"]["rows"] == 2
    assert metrics["pooled"]["exact_inputs"] == 1
    assert (
        metrics["pooled"]["equal_exact_input"]["policy_ce"]
        == metrics["pooled"]["exposure"]["policy_ce"]
    )


def test_development_effects_are_reconstructed_from_verified_metrics() -> None:
    results = json.loads(
        (
            Path("docs/data/seed504-policy-only-ablation") / "development-results.json"
        ).read_text()
    )
    verifier._assert_effects(results)


def test_development_effect_mutations_are_rejected() -> None:
    path = Path("docs/data/seed504-policy-only-ablation/development-results.json")
    original = json.loads(path.read_text())
    for cohort, weighting in (("pooled", "exposure"), (">32", "equal_exact_input")):
        changed = json.loads(json.dumps(original))
        changed["effects_T_minus_C"][cohort][weighting] += 0.001
        try:
            verifier._assert_effects(changed)
        except ValueError as error:
            assert "development_effect" in str(error)
        else:
            raise AssertionError(f"accepted altered effect {cohort}:{weighting}")
