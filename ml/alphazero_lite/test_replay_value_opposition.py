"""Tests for complete replay/fresh-value gradient accounting."""

import pytest
import torch
import numpy as np
import hashlib
from types import SimpleNamespace

from ml.alphazero_lite.replay_value_opposition import (
    flat_gradient,
    geometry,
    parameter_layout,
    verify_sums,
)
from ml.alphazero_lite.run_replay_value_opposition import (
    _gradient,
    _metrics_for_scope,
    classify,
    run_checkpoint,
)
from ml.alphazero_lite import (
    run_fresh_historical_replay_gradient_alignment as reference,
)
from ml.alphazero_lite import train


def test_policy_trunk_and_value_head_offset():
    trunk = torch.nn.Parameter(torch.tensor([1.0]))
    value_head = torch.nn.Parameter(torch.tensor([1.0]))
    # Mixture and fresh-value gradients oppose on the trunk, but agree on the
    # value head. The full-parameter dot therefore differs from trunk-only.
    mixture = torch.cat((torch.tensor([-2.0]), torch.tensor([3.0])))
    fresh_value = torch.cat((torch.tensor([1.0]), torch.tensor([1.0])))
    contributions = {
        "policy": torch.tensor([-3.0, 0.0]),
        "weighted_value": torch.tensor([1.0, 3.0]),
    }
    verify_sums(contributions, mixture, fresh_value)
    assert geometry(mixture, fresh_value)["dot"] == pytest.approx(1.0)
    assert geometry(mixture[:1], fresh_value[:1])["dot"] == pytest.approx(-2.0)
    assert trunk.requires_grad and value_head.requires_grad


def test_unused_parameter_is_explicit_zero():
    used = torch.nn.Parameter(torch.tensor([2.0]))
    unused = torch.nn.Parameter(torch.tensor([7.0]))
    vector, missing = flat_gradient((used**2).sum(), (used, unused))
    assert missing == 1
    assert vector.tolist() == [4.0, 0.0]


def test_zero_directions_are_undefined():
    result = geometry(torch.zeros(2), torch.ones(2))
    assert result["gradient_norm"] == 0.0
    assert result["cosine"] is None


def test_parameter_groups_cover_every_trainable_parameter():
    class ResidualStyleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = torch.nn.Linear(2, 2)
            self.residual_layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(3)]
            )
            self.policy_hidden_layer = torch.nn.Linear(2, 2)
            self.policy_head = torch.nn.Linear(2, 2)
            self.value_hidden_layer = torch.nn.Linear(2, 2)
            self.value_head = torch.nn.Linear(2, 1)

    model = ResidualStyleModel()
    groups = parameter_layout(model)
    grouped = [
        parameter
        for key, values in groups.items()
        if key not in ("shared_trunk", "all_parameters")
        for parameter in values
    ]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in groups["all_parameters"]
    }
    assert {id(parameter) for parameter in groups["shared_trunk"]} == {
        id(parameter)
        for parameter in (
            *groups["input_projection"],
            *groups["residual_block_0"],
            *groups["residual_block_1"],
            *groups["residual_block_2"],
        )
    }


def test_runner_group_and_source_objective_accounting():
    # The complete vector is assembled from policy and value contributions;
    # group projections independently add back to the all-parameter result.
    components = {
        "fresh:policy": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "history:weighted_value": torch.tensor([-2.0, 0.0, 0.0, 2.0]),
    }
    fresh = torch.ones(4)
    indexes = {
        "input_projection": [0],
        "residual_block_0": [1],
        "residual_block_1": [1],
        "residual_block_2": [1],
        "shared_trunk": [0, 1],
        "policy_head": [2],
        "value_head": [3],
        "all_parameters": [0, 1, 2, 3],
    }
    result = _metrics_for_scope(components, fresh, indexes, [1, 1, 1, 1])
    all_result = result["groups"]["all_parameters"]
    assert all_result["dot"] == pytest.approx(1.0)
    assert all_result["contribution_dot_sum"] == pytest.approx(1.0)
    assert all_result["source_objective_dot_contributions"] == {
        "fresh:policy": 1.0,
        "history:weighted_value": 0.0,
    }
    assert result["groups"]["shared_trunk"]["dot"] == pytest.approx(-1.0)
    assert result["groups"]["value_head"]["dot"] == pytest.approx(2.0)
    assert all_result["cross_objective_dot_products"]["fresh:policy"][
        "history:weighted_value"
    ] == pytest.approx(-2.0)


def test_decision_requires_primary_and_three_partitions():
    def row(cosine):
        return {"groups": {"all_parameters": {"cosine": cosine}}}

    checkpoint = {
        "results": {
            ">32": {
                "all": row(-0.06),
                "partition_0": row(-0.05),
                "partition_1": row(-0.07),
                "partition_2": row(-0.08),
                "partition_3": row(None),
            }
        }
    }
    assert classify(checkpoint)["recommend_value_target_provenance_calibration_audit"]
    checkpoint["results"][">32"]["all"] = row(-0.049)
    assert not classify(checkpoint)[
        "recommend_value_target_provenance_calibration_audit"
    ]


def test_gradient_chunking_multiplicity_and_policy_mask(monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Parameter(torch.tensor(0.4))
            self.value = torch.nn.Parameter(torch.tensor(0.2))

        def forward(self, x):
            logits = torch.stack(
                (torch.zeros_like(x[:, 0]), x[:, 0] * self.policy), dim=1
            )
            return logits, x * self.value

    model = TinyModel()
    x = np.asarray([[1.0], [2.0]], dtype=np.float32)
    policy_targets = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    value_targets = np.zeros((2, 1), dtype=np.float32)
    weights = np.asarray([0.0, 1.0], dtype=np.float32)
    multiplicity = np.asarray([2.0, 1.0], dtype=np.float32)
    arrays = (
        x,
        policy_targets,
        value_targets,
        weights,
        None,
        SimpleNamespace(train_multiplicity=multiplicity),
    )

    def masks(batch):
        return np.asarray(
            [[1, 0] if state[0] == 1 else [1, 1] for state in batch],
            dtype=np.float32,
        )

    monkeypatch.setattr(train, "legal_mask_matrix_for_encoded_states", masks)
    observed = {}
    for chunk in (1, 8):
        monkeypatch.setattr(reference, "CHUNK", chunk)
        observed[chunk] = {
            objective: _gradient(
                model,
                (model.policy, model.value),
                arrays,
                np.asarray([0, 1]),
                policy_denominator=1.0,
                value_denominator=3.0,
                objective=objective,
            )[0]
            for objective in ("policy", "weighted_value")
        }
    torch.testing.assert_close(observed[1]["policy"], observed[8]["policy"])
    torch.testing.assert_close(
        observed[1]["weighted_value"], observed[8]["weighted_value"]
    )
    # q=0 masks row zero from policy, while multiplicity two still weights its
    # value gradient. Both chunk partitions preserve the global denominators.
    assert observed[1]["policy"][1].item() == 0.0
    assert observed[1]["policy"][0].item() != 0.0
    assert observed[1]["weighted_value"][0].item() == 0.0
    assert observed[1]["weighted_value"][1].item() > 0.0


def test_complete_checkpoint_runner_path_is_immutable(monkeypatch, tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self, *_args):
            super().__init__()
            self.input_layer = torch.nn.Linear(1, 1, bias=False)
            self.residual_layers = torch.nn.ModuleList(
                [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
            )
            self.policy_hidden_layer = torch.nn.Linear(1, 1, bias=False)
            self.policy_head = torch.nn.Linear(1, 2, bias=False)
            self.value_hidden_layer = torch.nn.Linear(1, 1, bias=False)
            self.value_head = torch.nn.Linear(1, 1, bias=False)

        def forward(self, x):
            trunk = self.input_layer(x)
            for layer in self.residual_layers:
                trunk = trunk + layer(trunk)
            policy = self.policy_head(self.policy_hidden_layer(trunk))
            value = self.value_head(self.value_hidden_layer(trunk))
            return policy, value

    checkpoint = tmp_path / "frozen.npz"
    checkpoint.write_bytes(b"checkpoint sentinel")

    def tiny_groups(tiny_model):
        return {
            "input_projection": (tiny_model.input_layer.weight,),
            "residual_block_0": (tiny_model.residual_layers[0].weight,),
            "residual_block_1": (tiny_model.residual_layers[1].weight,),
            "residual_block_2": (tiny_model.residual_layers[2].weight,),
            "policy_head": (
                *tiny_model.policy_hidden_layer.parameters(),
                *tiny_model.policy_head.parameters(),
            ),
            "value_head": (
                *tiny_model.value_hidden_layer.parameters(),
                *tiny_model.value_head.parameters(),
            ),
        }

    monkeypatch.setattr(reference, "PolicyValueNet", TinyModel)
    monkeypatch.setattr(
        reference.train, "load_checkpoint_into_model", lambda *_args: None
    )
    monkeypatch.setattr(
        "ml.alphazero_lite.run_replay_value_opposition.parameter_groups",
        tiny_groups,
    )
    monkeypatch.setattr(
        reference.train,
        "legal_mask_matrix_for_encoded_states",
        lambda batch: np.ones((len(batch), 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        reference,
        "_state_cohorts",
        lambda _x, _multiplicity: {
            "all": np.ones(2, dtype=bool),
            ">32": np.ones(2, dtype=bool),
            "17-32": np.zeros(2, dtype=bool),
            "<=16": np.zeros(2, dtype=bool),
        },
    )
    monkeypatch.setattr(
        reference, "_partition_masks", lambda _mapping: np.asarray([0, 1])
    )
    mapping = SimpleNamespace(
        train_multiplicity=np.asarray([2, 1]),
        compact_source_ids=np.asarray([0, 1]),
    )
    arrays = (
        np.asarray([[0.2], [0.5]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.1], [-0.2]], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        None,
        mapping,
        None,
    )
    result = run_checkpoint(
        "tiny",
        checkpoint,
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        arrays,
        [{"name": "fresh"}, {"name": "history"}],
    )
    assert result["immutability"]["checkpoint_file_unchanged"]
    assert result["immutability"]["parameters_and_buffers_unchanged"]
    primary = result["results"][">32"]["all"]
    assert primary["groups"]["all_parameters"]["mixture_sum_verified"]
    assert primary["groups"]["all_parameters"]["dot"] == pytest.approx(
        primary["groups"]["all_parameters"]["contribution_dot_sum"]
    )
