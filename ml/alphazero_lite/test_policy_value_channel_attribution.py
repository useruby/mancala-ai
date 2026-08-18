from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_policy_value_channel_attribution as attribution


class Evaluator:
    def __init__(self, policy: list[float], value: float):
        self.policy = np.asarray(policy, dtype=np.float32)
        self.value = value

    def evaluate(self, _game):
        return self.policy.copy(), self.value


class NonterminalGame:
    def over(self) -> bool:
        return False


def test_composed_evaluator_selects_complete_output_channels() -> None:
    current = Evaluator([0.8, 0.2, 0, 0, 0, 0], 0.25)
    detached = Evaluator([0.1, 0.9, 0, 0, 0, 0], -0.5)
    policy_detached = arena.ComposedArtifactEvaluator(
        current, detached, policy_source="candidate", value_source="current"
    )
    value_detached = arena.ComposedArtifactEvaluator(
        current, detached, policy_source="current", value_source="candidate"
    )

    dp_cv_policy, dp_cv_value = policy_detached.evaluate(NonterminalGame())
    cp_dv_policy, cp_dv_value = value_detached.evaluate(NonterminalGame())

    assert np.array_equal(dp_cv_policy, detached.policy)
    assert dp_cv_value == current.value
    assert np.array_equal(cp_dv_policy, current.policy)
    assert cp_dv_value == detached.value


def test_composed_evaluator_returns_copies_without_channel_state_leaks() -> None:
    current = Evaluator([0.8, 0.2, 0, 0, 0, 0], 0.25)
    detached = Evaluator([0.1, 0.9, 0, 0, 0, 0], -0.5)
    evaluator = arena.ComposedArtifactEvaluator(
        current, detached, policy_source="candidate", value_source="current"
    )

    policy, _ = evaluator.evaluate(NonterminalGame())
    policy[0] = 0.0
    repeated, _ = evaluator.evaluate(NonterminalGame())

    assert repeated[0] == detached.policy[0]


def test_puct_uses_process_local_treatment_evaluators_and_configured_workers(
    tmp_path: Path, monkeypatch
) -> None:
    executor_calls = []
    submitted_tasks = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, _function, tasks):
            submitted_tasks.extend(tasks)
            return [
                (
                    task["state_hash"],
                    task["context"],
                    task["name"],
                    {
                        "move": 0,
                        "visit": [1.0],
                        "value": 0.0,
                        "margin": 1,
                        "q": {0: 1},
                    },
                )
                for task in tasks
            ]

    monkeypatch.setattr(attribution, "ProcessPoolExecutor", FakeExecutor)
    rows = [
        {
            "state": [0] * 14,
            "state_hash": "state-1",
            "manifest_index": 0,
            "player": 0,
        }
    ]

    attribution.puct(
        rows,
        workdir=tmp_path,
        step=3,
        current=tmp_path / "current",
        manifest_hash="manifest",
        workers=7,
    )

    assert executor_calls == [
        {
            "max_workers": 7,
            "initializer": attribution._init_puct_worker,
            "initargs": (str(tmp_path), 3, str(tmp_path / "current")),
        }
    ]
    assert {task["name"] for task in submitted_tasks} == {
        "C",
        "DP_DV",
        "DP_CV",
        "CP_DV",
        "H",
        "J",
    }


def test_forced_audit_uses_configured_process_workers(
    tmp_path: Path, monkeypatch
) -> None:
    executor_calls = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, _function, tasks):
            return [(0.0, 0.0) for _ in tasks]

    monkeypatch.setattr(attribution, "ProcessPoolExecutor", FakeExecutor)
    rows = [{"state": [0] * 14, "state_hash": f"state-{index}"} for index in range(32)]
    records = {
        (row["state_hash"], context, name): {"move": 0 if name == "C" else 1}
        for row in rows
        for context in attribution.CONTEXTS
        for name in ("C", "DP_CV", "CP_DV")
    }

    attribution.forced_audit(rows, records, tmp_path / "current.npz", workers=7)

    assert len(executor_calls) == len(attribution.CONTEXTS) * 2 * 2
    assert all(call["max_workers"] == 7 for call in executor_calls)
    assert all(
        call["initializer"] is attribution._init_forced_worker
        for call in executor_calls
    )


def test_canonical_arena_matrix_uses_all_channel_artifacts_and_matched_controls(
    tmp_path: Path, monkeypatch
) -> None:
    suite_path = tmp_path / "suite.jsonl"
    suite_path.write_text(
        "\n".join(
            json.dumps({"prefix_moves": [index]})
            for index in range(attribution.CANONICAL_ARENA_OPENINGS)
        )
        + "\n",
        encoding="utf-8",
    )
    detached = (
        tmp_path
        / "policy_detached_trunk"
        / "snapshot_artifacts"
        / "step_0003"
        / "artifact"
    )
    detached.mkdir(parents=True)
    (detached / "weights.json").write_text("{}", encoding="utf-8")
    current = tmp_path / "current"
    current.mkdir()
    calls = []

    def fake_run_arena(**kwargs):
        calls.append(kwargs)
        Path(kwargs["out_jsonl"]).write_text(
            "\n".join(
                json.dumps(
                    {
                        "opening_index": index,
                        "challenger_player": kwargs["challenger_starts"],
                        "winner": "draw",
                    }
                )
                for index in range(attribution.CANONICAL_ARENA_OPENINGS)
            )
            + "\n",
            encoding="utf-8",
        )
        return {}

    monkeypatch.setattr(attribution, "run_arena", fake_run_arena)

    result = attribution.canonical_arena_matrix(
        workdir=tmp_path,
        current=current,
        step=3,
        workers=24,
        suite_path=suite_path,
    )

    assert len(calls) == 32
    assert set(result["metrics"]) == {"384:256", "1200:1200"}
    assert all(call["workers"] == 24 for call in calls)
    assert all(
        call["games"] == 128 and call["games_per_opening"] == 1 for call in calls
    )
    assert all(call["current_policy_artifact"] == str(current) for call in calls)
    assert all(call["current_value_artifact"] == str(current) for call in calls)
    assert {
        (call["challenger_policy_artifact"], call["challenger_value_artifact"])
        for call in calls
    } == {
        (str(current), str(current)),
        (str(detached), str(detached)),
        (str(detached), str(current)),
        (str(current), str(detached)),
    }
    for metrics in result["metrics"].values():
        for effect in metrics.values():
            assert effect["opening_bootstrap_ci"]["unique_openings"] == 128
            assert effect["opening_bootstrap_ci"]["samples"] == 128
