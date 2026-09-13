from pathlib import Path

import pytest

from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    TRAINING_SEEDS,
    anchor_forgotten,
    assert_crossover_isolation,
    classify,
    factorial,
    original_train_config,
)


def test_crossover_generates_all_nine_replay_seed_cells(tmp_path: Path) -> None:
    commands = [
        original_train_config(
            tmp_path / f"{replay}.jsonl",
            tmp_path / "g0.npz",
            tmp_path / f"{replay}-{seed}.npz",
            seed=value,
            workdir=tmp_path / f"{replay}-{seed}",
        )
        for replay in ("R61", "R62", "R63")
        for seed, value in TRAINING_SEEDS.items()
    ]
    assert len(commands) == 9
    assert_crossover_isolation(commands)
    assert all("self_play.py" not in command for command in commands)
    assert all("pipeline.py" not in command for command in commands)
    assert all("--behavior-anchor-files" not in command for command in commands)


def test_crossover_isolation_rejects_hyperparameter_change() -> None:
    with pytest.raises(ValueError, match="beyond frozen replay"):
        assert_crossover_isolation(
            [["train", "--epochs", "4"], ["train", "--epochs", "5"]]
        )


def test_anchor_forgotten_requires_correct_g0_and_wrong_g1() -> None:
    assert anchor_forgotten(
        {"top_is_outcome_optimal": True}, {"top_is_outcome_optimal": False}
    )
    assert not anchor_forgotten(
        {"top_is_outcome_optimal": False}, {"top_is_outcome_optimal": False}
    )


def test_factorial_and_classification_are_descriptive() -> None:
    cells = [
        {
            "replay": replay,
            "training_seed": seed,
            "anchor_optimal_mass_delta": 0.2 if replay == "R62" else -0.2,
            "anchor_forgotten": replay != "R62",
        }
        for replay in ("R61", "R62", "R63")
        for seed in TRAINING_SEEDS
    ]
    decomposition = factorial(cells)
    assert (
        decomposition["sum_squares"]["replay"]
        > decomposition["sum_squares"]["training_seed"]
    )
    assert classify(cells) == "anchor_forgetting_replay_driven"


def test_epoch_metrics_audit_does_not_change_train_defaults() -> None:
    command = original_train_config(
        Path("dynamic.jsonl"),
        Path("g0.npz"),
        Path("out.npz"),
        seed=61,
        workdir=Path("work"),
    )
    assert command[command.index("--epochs") + 1] == "4"
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--replay-weights") + 1] == "1,1,2"
    assert "--lr" not in command
    assert "--behavior-loss-weight" not in command
