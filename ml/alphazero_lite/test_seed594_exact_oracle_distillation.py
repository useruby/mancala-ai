from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from ml.alphazero_lite import seed594_exact_oracle_distillation as seed594
from ml.alphazero_lite.kalah_rules import KalahGame


def _pair(seat: int = 0, margins: list[int] | None = None, nonce: int = 0):
    game = KalahGame([2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [nonce, nonce + 1], seat)
    state = game.to_state()
    legal = sorted(game.possible_moves())
    values = margins or [5] * len(legal)
    cohort_row = {
        "state": state,
        "state_hash": seed594.canonical_hash(state),
        "legal_absolute_actions": sorted(game.pit_index(move) for move in legal),
    }
    oracle_row = {
        "state_hash": cohort_row["state_hash"],
        "coverage_complete": True,
        "legal_moves": legal,
        "action_margins": {
            str(action): value for action, value in zip(legal, values, strict=True)
        },
    }
    return cohort_row, oracle_row


def _dataset(seat: int = 0, margins: list[int] | None = None):
    pairs = [_pair(seat, margins, nonce=index) for index in range(256)]
    return {"states": [pair[0] for pair in pairs]}, [pair[1] for pair in pairs]


def test_exact_targets_use_relative_action_slots_for_both_seats():
    for seat in (0, 1):
        cohort, oracles = _dataset(seat, [1, 9])
        row = seed594.exact_targets(cohort, oracles)[0]
        assert row["legal_actions"] == [0, 1]
        assert row["optimal_actions"] == [1]
        assert row["target"] == [0, 1, 0, 0, 0, 0]


def test_uniform_targets_cover_unique_and_tied_optima():
    unique_cohort, unique_oracles = _dataset(0, [3, 7])
    tied_cohort, tied_oracles = _dataset(1, [7, 7])
    roots = unique_cohort["states"][:128] + tied_cohort["states"][128:]
    oracles = unique_oracles[:128] + tied_oracles[128:]
    rows = seed594.exact_targets({"states": roots}, oracles)
    assert rows[0]["target"] == [0, 1, 0, 0, 0, 0]
    assert rows[128]["target"] == [0.5, 0.5, 0, 0, 0, 0]
    assert all(sum(row["target"]) == pytest.approx(1.0) for row in rows)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda root, row: row.update(coverage_complete=False),
            "oracle_coverage_incomplete",
        ),
        (
            lambda root, row: row["legal_moves"].append(5),
            "oracle_legal_action_mismatch",
        ),
        (
            lambda root, row: row["action_margins"].update({"5": 99}),
            "oracle_action_coverage_mismatch",
        ),
        (
            lambda root, row: row["action_margins"].pop("0"),
            "oracle_action_coverage_mismatch",
        ),
    ],
)
def test_incomplete_extra_and_illegal_oracle_actions_fail(mutation, error):
    cohort, oracles = _dataset()
    root, oracle = cohort["states"][0], oracles[0]
    mutation(root, oracle)
    with pytest.raises(ValueError, match=error):
        seed594.exact_targets(cohort, oracles)


def test_duplicate_or_missing_oracle_state_fails():
    cohort, oracles = _dataset()
    duplicate = dict(oracles[0])
    with pytest.raises(ValueError, match="duplicate_or_invalid_oracle_state"):
        seed594.exact_targets(cohort, [oracles[0], duplicate] + oracles[2:])
    with pytest.raises(ValueError, match="oracle_record_count_mismatch"):
        seed594.exact_targets(cohort, [])
    with pytest.raises(ValueError, match="oracle_record_count_mismatch"):
        seed594.exact_targets(cohort, oracles + [dict(oracles[0], state_hash="extra")])


def test_changed_root_hash_is_rejected():
    cohort, oracles = _dataset()
    root = cohort["states"][0]
    root["state_hash"] = "0" * 64
    with pytest.raises(ValueError, match="training_state_hash_mismatch"):
        seed594.exact_targets(cohort, oracles)


def test_input_source_digest_mismatch_fails_closed(tmp_path):
    path = tmp_path / "tampered"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="source_or_artifact_hash_mismatch"):
        seed594.verify_hashes(tmp_path)


def test_separation_checks_both_identity_schemes(monkeypatch, tmp_path):
    game = KalahGame([2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [0, 0], 0)
    state = game.to_state()
    train = [{"opening_index": 1, "state": state} for _ in range(256)]
    qual = [{"opening_index": 2, "state": state} for _ in range(423)]
    (tmp_path / seed594.TRAINING_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.TRAINING_PATH).write_text(json.dumps({"states": train}))
    (tmp_path / seed594.QUALIFICATION_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.QUALIFICATION_PATH).write_text(json.dumps({"roots": qual}))
    with pytest.raises(ValueError, match="qualification_training_canonical_overlap"):
        seed594.verify_separation(tmp_path)


def test_opening_overlap_is_rejected_independently(tmp_path):
    train_state = KalahGame([2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [0, 0], 0).to_state()
    qual_state = KalahGame([3, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [0, 0], 0).to_state()
    training = [{"opening_index": 7, "state": train_state} for _ in range(256)]
    qualification = [{"opening_index": 7, "state": qual_state} for _ in range(423)]
    (tmp_path / seed594.TRAINING_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.TRAINING_PATH).write_text(json.dumps({"states": training}))
    (tmp_path / seed594.QUALIFICATION_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.QUALIFICATION_PATH).write_text(
        json.dumps({"roots": qualification})
    )
    with pytest.raises(ValueError, match="qualification_training_opening_overlap"):
        seed594.verify_separation(tmp_path)


def test_production_float32_overlap_is_rejected_independently(monkeypatch, tmp_path):
    training_state = KalahGame(
        [2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [0, 0], 0
    ).to_state()
    qualification_state = KalahGame(
        [3, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0], [0, 0], 0
    ).to_state()
    training = [{"opening_index": 1, "state": training_state} for _ in range(256)]
    qualification = [
        {"opening_index": 2, "state": qualification_state} for _ in range(423)
    ]
    (tmp_path / seed594.TRAINING_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.TRAINING_PATH).write_text(json.dumps({"states": training}))
    (tmp_path / seed594.QUALIFICATION_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.QUALIFICATION_PATH).write_text(
        json.dumps({"roots": qualification})
    )
    monkeypatch.setattr(
        seed594, "production_float32_identity", lambda _state: "same-f32"
    )
    with pytest.raises(ValueError, match="qualification_training_float32_overlap"):
        seed594.verify_separation(tmp_path)


def test_batch_order_is_exact_seed571_permutation():
    import random

    expected = list(range(256))
    random.Random(571).shuffle(expected)
    config = json.loads((seed594.ROOT / seed594.CONFIG_PATH).read_bytes())
    assert seed594.fixed_order() == expected == config["training"]["batch_order"]
    assert config["training"]["updates"] == 512
    assert config["training"]["batch_size"] == 32
    assert config["training"]["learning_rate"] == 1e-5
    assert config["training"]["weight_decay"] == 0
    matched = json.loads(
        (
            seed594.ROOT
            / "docs/data/seed570-policy-prior-distillation/training-configuration.json"
        ).read_bytes()
    )
    assert config["training"]["batch_order"] == expected
    assert "one Python random.Random(571) shuffle" in matched["training"]["batch_order"]
    assert config["training"]["updates"] == matched["training"]["updates"]
    assert config["training"]["batch_size"] == matched["training"]["batch_size"]
    assert config["training"]["learning_rate"] == matched["training"]["learning_rate"]
    assert config["training"]["weight_decay"] == matched["training"]["weight_decay"]
    for key in ("seed", "betas", "epsilon", "optimizer", "gradient_clip_norm"):
        assert config["training"][key] == matched["training"][key]
    assert (
        f"{config['model']['architecture']}/{config['model']['input_encoding']}"
        == matched["model"]["type"]
    )
    assert (
        config["model"]["initialization"]
        == "exact authenticated serving P-E4 checkpoint"
    )


def test_runtime_mismatch_blocks_before_any_training(monkeypatch):
    monkeypatch.setattr(seed594.platform, "python_version", lambda: "0.0")
    with pytest.raises(ValueError, match="runtime_mismatch"):
        seed594.verify_runtime()


def test_existing_output_directory_rejected_before_model_construction(
    monkeypatch, tmp_path
):
    marker_dir = tmp_path / "present"
    marker_dir.mkdir()
    called = False

    def fail_if_constructed(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("model construction preceded output check")

    monkeypatch.setattr(seed594, "verify_pretraining_freeze", lambda root: None)
    monkeypatch.setattr(seed594.train, "PolicyValueNet", fail_if_constructed)
    with pytest.raises(ValueError, match="output_directory_exists"):
        seed594.train_frozen(marker_dir, root=tmp_path)
    assert not called


def test_preflight_freeze_is_not_authorized_for_training():
    freeze = json.loads((seed594.ROOT / seed594.FREEZE_PATH).read_bytes())
    assert freeze["status"] == "frozen-pretraining-not-authorized"
    assert freeze["scope"]["training"] is False


def test_changed_configuration_and_target_identity_are_rejected(monkeypatch, tmp_path):
    freeze = {
        "status": "frozen-pretraining-not-authorized",
        "configuration_sha256": "bad",
    }
    (tmp_path / seed594.FREEZE_PATH).parent.mkdir(parents=True)
    (tmp_path / seed594.FREEZE_PATH).write_text(json.dumps(freeze))
    (tmp_path / seed594.CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / seed594.CONFIG_PATH).write_text("{}")
    monkeypatch.setattr(seed594, "verify_hashes", lambda root: {})
    monkeypatch.setattr(seed594, "verify_runtime", lambda: None)
    monkeypatch.setattr(seed594, "verify_separation", lambda root: {})
    with pytest.raises(ValueError, match="configuration_hash_mismatch"):
        seed594.verify_pretraining_freeze(tmp_path)


def test_frozen_parameter_set_matches_seed571_contract():
    assert seed594.TRAINABLE_PREFIXES == (
        "input_layer.",
        "residual_layers.",
        "policy_hidden_layer.",
        "policy_head.",
    )


def test_frozen_parameter_mutation_is_detected():
    model = nn.Linear(2, 1)
    snapshot = {"weight": model.weight.detach().clone()}
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(ValueError, match="frozen_parameter_changed"):
        seed594.verify_frozen_parameters(model, snapshot)


def test_registered_bootstrap_is_deterministic_and_cluster_paired():
    openings = list(range(423))
    differences = [float((index % 7) - 3) for index in openings]
    first = seed594.paired_opening_bootstrap(openings, differences)
    second = seed594.paired_opening_bootstrap(openings, differences)
    assert first == second
    assert first["resamples"] == 20_000
    assert first["seed"] == 594
    assert first["cluster_count"] == 423
    assert first["quantile_method"] == "linear"


def test_bootstrap_rejects_wrong_cluster_count_or_nonfinite_values():
    with pytest.raises(ValueError, match="qualification_opening_count_mismatch"):
        seed594.paired_opening_bootstrap([1], [0.0])
    with pytest.raises(ValueError, match="bootstrap_nonfinite_difference"):
        seed594.paired_opening_bootstrap(list(range(423)), [float("nan")] + [0.0] * 422)


def test_missing_execution_authorization_stops_before_model_construction(
    monkeypatch, tmp_path
):
    called = False

    def fail_if_constructed(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("training model constructed without authorization")

    monkeypatch.setattr(seed594, "verify_pretraining_freeze", lambda root: None)
    monkeypatch.setattr(seed594.train, "PolicyValueNet", fail_if_constructed)
    with pytest.raises(FileNotFoundError):
        seed594.train_frozen(tmp_path / "new-output", root=tmp_path)
    assert not called
    assert not (tmp_path / "new-output").exists()
