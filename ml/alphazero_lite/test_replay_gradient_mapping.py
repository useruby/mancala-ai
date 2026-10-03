"""Regression tests for replay compact-row identities and objective decomposition."""

import numpy as np
import pytest
import torch
import json

from ml.alphazero_lite.replay_gradient_mapping import (
    make_replay_mapping,
    partition_id,
    split_compact_rows,
    source_contribution_loss,
)
from ml.alphazero_lite import train
from ml.alphazero_lite.verify_fresh_historical_replay_gradient_alignment import (
    recompute,
)


def test_mapping_uses_compact_rows_and_production_split_positions() -> None:
    # Three filtered/retained fresh rows and two historical rows. The first fresh
    # and historical records may have identical states; row identity is still distinct.
    names = ["fresh", "fresh", "fresh", "historic", "historic"]
    replay = np.tile(np.arange(len(names), dtype=np.int64), [2, 4, 1]).reshape(-1)
    np.random.seed(77)
    train_positions, val_positions = train.split_replay_positions_by_source_row(
        replay, val_split=0.2
    )
    mapping = make_replay_mapping(names, replay, train_positions, val_positions)
    assert (
        mapping.train_multiplicity.sum() + mapping.validation_multiplicity.sum()
        == len(replay)
    )
    assert np.all(np.isin(mapping.train_multiplicity, [0, 8]))
    assert np.all(np.isin(mapping.validation_multiplicity, [0, 8]))
    assert np.all(mapping.train_multiplicity + mapping.validation_multiplicity == 8)
    assert not set(mapping.train_compact_ids) & set(mapping.validation_compact_ids)
    assert mapping.compact_local_row_ids.tolist() == [0, 1, 2, 0, 1]
    assert partition_id("fresh", 0) == partition_id("fresh", 0)
    assert partition_id("fresh", 0) != partition_id("historic", 0)


def test_production_loader_filtered_unequal_sources_and_duplicate_states(
    tmp_path,
) -> None:
    def row(state: float, bucket: str | None = None) -> dict:
        result = {"state": [state], "policy": [1.0, 0, 0, 0, 0, 0], "value": 0.25}
        if bucket:
            result["bucket"] = bucket
        return result

    paths = [tmp_path / "fresh.jsonl", tmp_path / "history.jsonl"]
    records = [[row(0.2), row(0.3, "discard"), row(0.4)], [row(0.2), row(0.5)]]
    for path, rows in zip(paths, records):
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )
    x, p, v, replay, q = train.load_jsonl_replay(
        paths,
        [3, 2],
        exclude_buckets={"discard"},
        policy_target_mode="default",
        replay_value_target_modes=["default", "default"],
        include_policy_loss_weights=True,
    )
    names = ["fresh", "fresh", "history", "history"]
    np.testing.assert_allclose(x[:, 0], [0.2, 0.4, 0.2, 0.5])
    assert p.shape == (4, 6) and v.shape == (4, 1) and q.shape == (4,)
    np.testing.assert_array_equal(replay, [0, 1, 0, 1, 0, 1, 2, 3, 2, 3])
    train_pos, val_pos = split_compact_rows(replay, seed=19, validation_split=0.25)
    mapping = make_replay_mapping(names, replay, train_pos, val_pos)
    np.testing.assert_array_equal(
        mapping.train_multiplicity + mapping.validation_multiplicity, [3, 3, 2, 2]
    )
    assert mapping.compact_source_ids.tolist() == [0, 0, 1, 1]
    assert mapping.compact_local_row_ids.tolist() == [0, 1, 0, 1]
    assert x[0, 0] == x[2, 0]  # Same state, distinct compact identities and sources.
    assert not (set(mapping.train_compact_ids) & set(mapping.validation_compact_ids))


def test_source_terms_sum_to_expanded_weighted_cohort_objective() -> None:
    # Fixed model outputs are sufficient here: production CE and Huber primitives
    # define the row losses, and gradients below prove exact additive decomposition.
    logits = torch.tensor(
        [[1.0, -0.2, 0.4], [0.1, 0.8, -0.5], [-0.3, 0.4, 0.7]], requires_grad=True
    )
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # The third row's only target action is legal; the first two rows are masked
    # differently to exercise the production legal-mask path.
    legal = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    masked_logits = logits.masked_fill(~legal, -1e9)
    ce = train.compute_policy_cross_entropy(masked_logits, target)
    pred = (logits[:, :1] * 0.25).requires_grad_()
    target_v = torch.tensor([[0.2], [-0.3], [0.8]])
    huber = train.compute_value_loss_vector(
        pred, target_v, value_loss="huber", huber_delta=1.0
    )
    q = torch.tensor([1.0, 0.0, 1.0])
    m = torch.tensor([2.0, 3.0, 1.0])
    source = torch.tensor([True, True, False])
    cohort = torch.ones(3, dtype=torch.bool)
    all_loss = (ce * q * m).sum() / (q * m).sum() + 0.3 * (huber * m).sum() / m.sum()
    source_p, source_v, source_total = source_contribution_loss(
        ce, 0.3 * huber, q, m, cohort, source
    )
    other_p, other_v, other_total = source_contribution_loss(
        ce, 0.3 * huber, q, m, cohort, ~source
    )
    summed = source_total + other_total
    assert float(summed.detach()) == pytest.approx(float(all_loss.detach()))
    expected_grad = torch.autograd.grad(all_loss, (logits, pred), retain_graph=True)
    actual_grad = torch.autograd.grad(summed, (logits, pred), retain_graph=True)
    for got, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, expected)
    expanded_ids = torch.repeat_interleave(torch.arange(3), m.long())
    expanded_logits = masked_logits[expanded_ids]
    expanded_targets = target[expanded_ids]
    expanded_q = q[expanded_ids]
    expanded_values = pred[expanded_ids]
    expanded_vtargets = target_v[expanded_ids]
    expanded_ce = train.compute_policy_cross_entropy(expanded_logits, expanded_targets)
    expanded_huber = train.compute_value_loss_vector(
        expanded_values, expanded_vtargets, value_loss="huber", huber_delta=1.0
    )
    expanded_loss = (
        expanded_ce * expanded_q
    ).sum() / expanded_q.sum() + 0.3 * expanded_huber.mean()
    torch.testing.assert_close(all_loss, expanded_loss)
    # Simulate chunked source numerator accumulation while retaining shared denominators.
    chunks = ((0, 1), (1, 3))

    def chunked(mask: torch.Tensor) -> torch.Tensor:
        p_num = sum((ce[a:b] * q[a:b] * m[a:b] * mask[a:b]).sum() for a, b in chunks)
        v_num = sum((0.3 * huber[a:b] * m[a:b] * mask[a:b]).sum() for a, b in chunks)
        return p_num / (q * m).sum() + v_num / m.sum()

    chunked_sum = chunked(source) + chunked(~source)
    torch.testing.assert_close(chunked_sum, all_loss)
    torch.testing.assert_close(
        torch.autograd.grad(chunked_sum, (logits, pred), retain_graph=True)[0],
        expected_grad[0],
    )
    assert float((source_p + other_p).detach()) == pytest.approx(
        float(((ce * q * m).sum() / (q * m).sum()).detach())
    )
    assert float((source_v + other_v).detach()) == pytest.approx(
        float((0.3 * (huber * m).sum() / m.sum()).detach())
    )


def test_empty_and_zero_policy_mass_are_explicit_zero() -> None:
    rows = torch.tensor([2.0, 3.0], requires_grad=True)
    values = torch.tensor([0.0, 0.0], requires_grad=True)
    q = torch.zeros(2)
    m = torch.tensor([2.0, 1.0])
    cohort = torch.ones(2, dtype=torch.bool)
    policy, value, total = source_contribution_loss(
        rows, values, q, m, cohort, torch.ones(2, dtype=torch.bool)
    )
    assert policy.item() == 0.0
    assert value.item() == 0.0
    assert torch.autograd.grad(total, rows)[0].tolist() == [0.0, 0.0]
    empty_policy, empty_value, empty_total = source_contribution_loss(
        rows,
        values,
        torch.ones(2),
        m,
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    )
    assert empty_policy.item() == empty_value.item() == empty_total.item() == 0.0
    assert torch.autograd.grad(empty_total, rows)[0].tolist() == [0.0, 0.0]


def test_mapping_rejects_repeated_row_split_leakage() -> None:
    with pytest.raises(ValueError, match="leaks"):
        make_replay_mapping(
            ["a", "b"], np.array([0, 1, 0]), np.array([0, 1]), np.array([2])
        )


def test_portable_verifier_reconstructs_metrics_from_norms_and_dots() -> None:
    report = {
        "source_norms": {"fresh": 3.0, "h1": 2.0, "h2": 2.0},
        "source_pairwise_dots": {
            "fresh": {"fresh": 9.0, "h1": -3.0, "h2": -3.0},
            "h1": {"fresh": -3.0, "h1": 4.0, "h2": 0.0},
            "h2": {"fresh": -3.0, "h1": 0.0, "h2": 4.0},
        },
    }
    result = recompute(report)
    assert result["dot"] == -6.0
    assert result["historical_norm"] == pytest.approx(2**0.5 * 2)
    assert result["retained_fresh_projection"] == pytest.approx(1 / 3)
