from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ml.alphazero_lite import seed571_policy_distillation as protocol
from ml.alphazero_lite.kalah_rules import KalahGame


def _synthetic_states() -> list[dict]:
    rows = []
    for current_player in (0, 1):
        game = KalahGame([2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [0, 0], current_player)
        rows.append(
            {
                "state": game.to_state(),
                "legal_absolute_actions": [
                    game.pit_index(action) for action in game.possible_moves()
                ],
            }
        )
    return rows


def _write_cohort(root: Path, states: list[dict]) -> None:
    cohort_path = root / protocol.CONFIG["training_membership"]["path"]
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(json.dumps({"states": states}))


def _authorize_synthetic_root(root: Path) -> None:
    freeze_dir = root / "docs/data/seed570-policy-prior-distillation"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    (freeze_dir / "pretraining-freeze.json").write_text("{}")
    (freeze_dir / "training-configuration.json").write_text("{}")
    runner = root / "ml/alphazero_lite/seed571_policy_distillation.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("synthetic runner bytes")
    bindings = protocol.expected_execution_bindings(root)
    (freeze_dir / "execution-freeze.json").write_text(
        json.dumps(
            {
                "schema": "seed571-policy-distillation-execution-freeze-v1",
                "status": "authorized-for-pretraining",
                "bindings": bindings,
            }
        )
    )


def test_input_mismatch_rejected_before_search_or_training(tmp_path, monkeypatch):
    path = tmp_path / "changed-input"
    path.write_bytes(b"changed")
    called = False

    def search(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(protocol.self_play.PUCT, "run", search)
    with pytest.raises(ValueError, match="input_binding_mismatch"):
        protocol.verify_bindings(
            tmp_path, {"changed-input": hashlib.sha256(b"frozen").hexdigest()}
        )
    assert not called


def test_visit_targets_respect_legality_and_zero_unvisited_legal_actions():
    target = protocol.normalized_visit_target([0, 7, 0, 0, 0, 0], [0, 1, 3])
    np.testing.assert_array_equal(target, [0, 1, 0, 0, 0, 0])
    assert target[[2, 4, 5]].sum() == 0
    with pytest.raises(ValueError, match="illegal_action_has_visits"):
        protocol.normalized_visit_target([0, 0, 1, 0, 0, 0], [0, 1])


def test_training_step_changes_only_policy_and_trunk_parameters():
    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Linear(3, 4)
            self.policy_head = nn.Linear(4, 6)
            self.value_head = nn.Linear(4, 1)

        def forward(self, x):
            hidden = torch.tanh(self.trunk(x))
            return self.policy_head(hidden), self.value_head(hidden)

    model = TinyPolicy()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("trunk.", "policy_head.")))
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    x = torch.tensor([[0.2, 0.5, 0.8], [0.7, 0.1, 0.3]], dtype=torch.float32)
    legal = torch.tensor(
        [
            [True, True, False, True, False, False],
            [False, True, True, False, True, False],
        ]
    )
    targets = torch.tensor(
        [[0.25, 0.75, 0, 0, 0, 0], [0, 0, 0.5, 0, 0.5, 0]], dtype=torch.float32
    )
    protocol.train_synthetic_step(model, optimizer, x, targets, legal)
    params = dict(model.named_parameters())
    assert all(torch.equal(params[name], value) for name, value in frozen.items())
    assert all(
        not torch.equal(params[name], value) for name, value in trainable_before.items()
    )


def test_training_order_is_frozen_and_complete():
    order = protocol.fixed_order()
    assert sorted(order) == list(range(256))
    assert order == protocol.fixed_order()


def test_changed_configuration_is_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"updates": 513}')
    frozen_digest = hashlib.sha256(b'{"updates": 512}').hexdigest()
    with pytest.raises(ValueError, match="configuration_binding_mismatch"):
        protocol.verify_configuration(config, frozen_digest)


def test_target_generation_handles_both_players_and_keeps_relative_slots(
    tmp_path, monkeypatch
):
    states = _synthetic_states() * 128
    _write_cohort(tmp_path, states)
    _authorize_synthetic_root(tmp_path)
    monkeypatch.setattr(protocol, "verify_inputs", lambda root: {})
    monkeypatch.setattr(protocol, "verify_pretraining_freeze", lambda root: None)
    searched_players = []

    class FakePUCT:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, game):
            searched_players.append(game.current_player)
            children = {
                action: SimpleNamespace(visit_count=64)
                for action in game.possible_moves()
            }
            return np.zeros(6), SimpleNamespace(children=children)

    monkeypatch.setattr(protocol.self_play, "PUCT", FakePUCT)
    targets = protocol.generate_targets(
        root=tmp_path, output_dir=tmp_path / "not-created", evaluator=object()
    )

    assert len(targets) == 256
    assert searched_players.count(0) == searched_players.count(1) == 128
    assert all(row["legal_actions"] == [0, 1] for row in targets)
    assert all(row["visits"] == [64, 64, 0, 0, 0, 0] for row in targets)
    assert all(row["target"] == [0.5, 0.5, 0, 0, 0, 0] for row in targets)


def test_missing_execution_freeze_blocks_target_and_training_entrypoints(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(protocol, "verify_inputs", lambda root: {})
    monkeypatch.setattr(protocol, "verify_pretraining_freeze", lambda root: None)
    searched = False
    trained = False

    class FakePUCT:
        def __init__(self, *_args, **_kwargs):
            nonlocal searched
            searched = True

    def fake_model(*_args, **_kwargs):
        nonlocal trained
        trained = True
        raise AssertionError("training model was constructed")

    monkeypatch.setattr(protocol.self_play, "PUCT", FakePUCT)
    monkeypatch.setattr(protocol.train, "PolicyValueNet", fake_model)
    output_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="execution_freeze_missing"):
        protocol.generate_targets(root=tmp_path, output_dir=output_dir)
    with pytest.raises(ValueError, match="execution_freeze_missing"):
        protocol.train_targets([], output_dir, root=tmp_path)

    assert not searched
    assert not trained
    assert not output_dir.exists()


def test_changed_execution_binding_blocks_target_generation(tmp_path, monkeypatch):
    _write_cohort(tmp_path, _synthetic_states())
    _authorize_synthetic_root(tmp_path)
    monkeypatch.setattr(protocol, "verify_inputs", lambda root: {})
    monkeypatch.setattr(protocol, "verify_pretraining_freeze", lambda root: None)
    config_path = (
        tmp_path
        / "docs/data/seed570-policy-prior-distillation/training-configuration.json"
    )
    config_path.write_text('{"changed": true}')
    searched = False

    class FakePUCT:
        def __init__(self, *_args, **_kwargs):
            nonlocal searched
            searched = True

    monkeypatch.setattr(protocol.self_play, "PUCT", FakePUCT)
    output_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="execution_freeze_binding_mismatch"):
        protocol.generate_targets(
            root=tmp_path, output_dir=output_dir, evaluator=object()
        )

    assert not searched
    assert not output_dir.exists()


def test_existing_output_directory_is_rejected_before_search(tmp_path, monkeypatch):
    states = _synthetic_states() * 128
    _write_cohort(tmp_path, states)
    _authorize_synthetic_root(tmp_path)
    monkeypatch.setattr(protocol, "verify_inputs", lambda root: {})
    monkeypatch.setattr(protocol, "verify_pretraining_freeze", lambda root: None)
    searched = False

    class FakePUCT:
        def __init__(self, *_args, **_kwargs):
            nonlocal searched
            searched = True

    monkeypatch.setattr(protocol.self_play, "PUCT", FakePUCT)
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    marker = output_dir / "preserve.txt"
    marker.write_text("untouched")
    with pytest.raises(ValueError, match="checkpoint_output_already_exists"):
        protocol.generate_targets(
            root=tmp_path, output_dir=output_dir, evaluator=object()
        )

    assert not searched
    assert marker.read_text() == "untouched"


def test_artifact_load_paths_use_the_authenticated_evidence_root(tmp_path, monkeypatch):
    relative_root = protocol.ARTIFACT_ROOT.relative_to(protocol.ROOT)
    bindings = {}
    for relative in (
        "P-E4/metadata.json",
        "P-E4/model.npz",
        "initializer/metadata.json",
    ):
        path = tmp_path / relative_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        bindings[str(relative_root / relative)] = hashlib.sha256(
            relative.encode()
        ).hexdigest()
    protocol.verify_bindings(tmp_path, bindings)
    loaded = []
    monkeypatch.setattr(
        protocol.arena, "ArtifactEvaluator", lambda path: loaded.append(path) or path
    )
    monkeypatch.setattr(
        protocol.arena,
        "ComposedArtifactEvaluator",
        lambda initializer, policy, **_kwargs: (initializer, policy),
    )

    protocol.make_teacher(tmp_path, bindings)
    monkeypatch.setattr(
        protocol.train,
        "load_checkpoint_into_model",
        lambda _model, path: loaded.append(path) or [],
    )
    assert protocol.load_teacher_checkpoint(object(), tmp_path, bindings) == []

    assert loaded == [
        tmp_path / relative_root / "initializer",
        tmp_path / relative_root / "P-E4",
        tmp_path / relative_root / "P-E4/model.npz",
    ]
