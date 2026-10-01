import subprocess
import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite.checkpoint_average import average_checkpoints, load_npz


def test_average_uses_float64_and_includes_all_parameter_arrays():
    left = {"trunk": np.array([1.0], dtype=np.float32), "head": np.array([4.0])}
    middle = {"trunk": np.array([2.0], dtype=np.float32), "head": np.array([5.0])}
    right = {"trunk": np.array([3.0], dtype=np.float32), "head": np.array([6.0])}
    averaged = average_checkpoints([left, middle, right])
    assert averaged["trunk"].dtype == np.float32
    np.testing.assert_array_equal(averaged["trunk"], [2.0])
    np.testing.assert_array_equal(averaged["head"], [5.0])


@pytest.mark.parametrize(
    ("checkpoints", "error"),
    [
        ([{"a": np.array([1.0])}, {"b": np.array([1.0])}], "keys_mismatch"),
        ([{"a": np.ones(1)}, {"a": np.ones(2)}], "shape_mismatch"),
        (
            [{"a": np.ones(1, dtype=np.float32)}, {"a": np.ones(1, dtype=np.float64)}],
            "dtype_mismatch",
        ),
        (
            [{"a": np.array([np.nan])}, {"a": np.array([1.0])}],
            "nonfinite",
        ),
        (
            [
                {"a": np.array([1], dtype=np.int64)},
                {"a": np.array([2], dtype=np.int64)},
            ],
            "nonfloating_mismatch",
        ),
    ],
)
def test_malformed_inputs_are_rejected(checkpoints, error):
    with pytest.raises(ValueError, match=error):
        average_checkpoints(checkpoints)


def test_average_npz_exports_and_round_trips(tmp_path):
    root = Path(__file__).resolve().parents[2]
    shapes = {
        "w_input": (27, 4),
        "b_input": (4,),
        "w_residual_1_1": (4, 4),
        "b_residual_1_1": (4,),
        "w_residual_1_2": (4, 4),
        "b_residual_1_2": (4,),
        "w_policy_hidden": (4, 3),
        "b_policy_hidden": (3,),
        "w_value_hidden": (4, 2),
        "b_value_hidden": (2,),
        "w_policy": (3, 6),
        "b_policy": (6,),
        "w_value": (2, 1),
        "b_value": (1,),
    }
    source, middle, final = [
        {key: np.full(shape, value, dtype=np.float32) for key, shape in shapes.items()}
        for value in (1.0, 2.0, 3.0)
    ]
    averaged = average_checkpoints([source, middle, final])
    checkpoint = tmp_path / "mean.npz"
    np.savez(checkpoint, **averaged)
    np.testing.assert_equal(load_npz(str(checkpoint)), averaged)
    out = tmp_path / "artifact"
    subprocess.run(
        [
            sys.executable,
            str(root / "ml/alphazero_lite/export_artifact.py"),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out),
            "--version",
            "test-mean",
            "--model-type",
            "residual_v3",
            "--rules-version",
            "kalah_v1",
            "--input-encoding",
            "kalah_v3",
        ],
        cwd=root,
        check=True,
    )
    with np.load(out / "model.npz", allow_pickle=False) as exported:
        for key, value in averaged.items():
            np.testing.assert_equal(exported[key], value)


def test_completed_arena_resume_preserves_bound_bytes_and_launches_nothing(
    tmp_path, monkeypatch
):
    import ml.alphazero_lite.run_seed461_cosine_lr_arena as runner

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    suite = data / "suite.jsonl"
    suite.write_text('{"prefix_moves": []}\n')
    opponent = tmp_path / "opponent"
    opponent.mkdir()
    for name in ("weights.json", "metadata.json", "search_policy.json"):
        (opponent / name).write_text(name)
    contract = {
        "runtime": "frozen",
        "exact_root_native_probe_sha256": "probe",
        "exact_root_tablebase_sha256": "tablebase",
    }
    registration = data / "registration.json"
    registration.write_text(
        json.dumps(
            {
                "holdout": {"sha256": digest(suite)},
                "evaluation": {"runtime_contract": contract},
            }
        )
    )
    training_dir = work / "training"
    training_dir.mkdir(parents=True)
    trajectories = {}
    candidates = {}
    reports = {}
    arena = work / "arena-e4"
    for seed in range(38611, 38616):
        for arm in ("A", "B"):
            run = f"order_{seed}_{arm}"
            checkpoint = training_dir / run / "E4.npz"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(run)
            checkpoint_hash = digest(checkpoint)
            trajectories[run] = {"epochs": {"E4": checkpoint_hash}}
            artifact = work / "artifacts-e4" / run
            artifact.mkdir(parents=True)
            hashes = {}
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            ):
                (artifact / name).write_text(
                    checkpoint.read_text() if name == "model.npz" else name
                )
                hashes[name] = digest(artifact / name)
            candidates[run] = {
                "artifact": str(artifact),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "artifact_sha256": hashes,
                "runtime_contract": contract,
                "epoch": "E4",
            }
            report, games = arena / f"{run}.json", arena / f"{run}-games.jsonl"
            arena.mkdir(exist_ok=True)
            report.write_bytes(b"completed report\n")
            games.write_bytes(b"completed games\n")
            reports[run] = {
                "report": str(report),
                "report_sha256": digest(report),
                "games": str(games),
                "games_sha256": digest(games),
            }
    training = work / "training.json"
    training.write_text(
        json.dumps(
            {"registration_sha256": digest(registration), "trajectories": trajectories}
        )
    )
    binding = data / "binding.json"
    immutable = {
        "schema": "seed461-cosine-lr-evaluation-binding-v1",
        "registration_sha256": digest(registration),
        "training_sha256": digest(training),
        "suite_sha256": digest(suite),
        "opponent": {
            "artifact": str(opponent),
            "weights_sha256": digest(opponent / "weights.json"),
            "metadata_sha256": digest(opponent / "metadata.json"),
            "sidecar_sha256": digest(opponent / "search_policy.json"),
            "native_probe_sha256": "probe",
            "tablebase_sha256": "tablebase",
        },
        "candidates": candidates,
        "reports": reports,
        "status": "completed_fixed_5120_games",
    }
    binding.write_text(json.dumps(immutable, indent=2, sort_keys=True) + "\n")
    before = {
        path: path.read_bytes()
        for path in [
            binding,
            *[
                Path(x["artifact"]) / n
                for x in candidates.values()
                for n in x["artifact_sha256"]
            ],
            *[Path(x[k]) for x in reports.values() for k in ("report", "games")],
        ]
    }
    monkeypatch.setattr(runner, "REG", registration)
    monkeypatch.setattr(runner, "TRAINING", training)
    monkeypatch.setattr(runner, "SUITE", suite)
    monkeypatch.setattr(runner, "WORK", work)
    monkeypatch.setattr(runner, "OPPONENT", opponent)
    monkeypatch.setattr(runner, "BINDING", binding)
    monkeypatch.setattr(
        runner, "resolve_strength_comparison_runtime_contract", lambda **_: contract
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: pytest.fail("unexpected process launch"),
    )
    runner.main()
    assert all(path.read_bytes() == payload for path, payload in before.items())
