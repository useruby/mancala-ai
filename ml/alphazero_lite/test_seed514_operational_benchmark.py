"""Focused tests for the seed514 local benchmark protocol."""

from __future__ import annotations


import numpy as np
import pytest

from ml.alphazero_lite import arena, seed514_operational_benchmark as benchmark
from ml.alphazero_lite.kalah_rules import KalahGame


def test_cohort_selection_is_hash_ordered_unique_and_stratified() -> None:
    first = benchmark.reconstruct_cohort()
    second = benchmark.reconstruct_cohort()
    assert first == second
    assert len(first) == 64
    assert len({row["state_sha256"] for row in first}) == 64
    assert {
        bucket: sum(row["bucket"] == bucket for row in first)
        for bucket in benchmark.COHORT_COUNTS
    } == benchmark.COHORT_COUNTS
    for row in first:
        assert row["state_sha256"] == benchmark.state_id(row["state"])
        assert (
            row["selection_order_sha256"]
            == __import__("hashlib")
            .sha256(
                f"seed514-cohort-v1:{row['state_sha256']}:{row['source_game_index']}:{row['ply']}".encode()
            )
            .hexdigest()
        )
        assert not KalahGame.from_state(row["state"]).over()


def test_composed_component_routing_uses_policy_and_value_owners() -> None:
    class Evaluator:
        def __init__(self, policy: list[float], value: float) -> None:
            self.policy = np.asarray(policy, dtype=np.float32)
            self.value = value
            self.calls = 0

        def evaluate(self, _game: KalahGame) -> tuple[np.ndarray, float]:
            self.calls += 1
            return self.policy, self.value

    initializer = Evaluator([0.1] * 6, 0.25)
    e4 = Evaluator([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], -0.5)
    composed = arena.ComposedArtifactEvaluator(
        initializer, e4, policy_source="candidate", value_source="current"
    )
    state = {
        "player_pits": [1, 0, 1, 0, 1, 0],
        "opponent_pits": [1] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    policy, value = composed.evaluate(KalahGame.from_state(state))
    assert policy[4] == pytest.approx(1.0)
    assert value == pytest.approx(0.25)
    assert (initializer.calls, e4.calls) == (1, 1)


def test_terminal_composition_does_not_invoke_evaluators() -> None:
    class NeverEvaluate:
        def evaluate(self, _game: KalahGame) -> tuple[np.ndarray, float]:
            raise AssertionError("terminal game must not evaluate models")

    composed = arena.ComposedArtifactEvaluator(
        NeverEvaluate(),
        NeverEvaluate(),
        policy_source="candidate",
        value_source="current",
    )
    terminal = KalahGame.from_state(
        {
            "player_pits": [0] * 6,
            "opponent_pits": [1] * 6,
            "player_store": 24,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    policy, value = composed.evaluate(terminal)
    assert np.array_equal(policy, np.zeros(6, dtype=np.float32))
    assert value == 0.0


def test_run_search_dispatches_complete_registered_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "player_pits": [4] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    captured: dict[str, object] = {}
    monkeypatch.setattr(benchmark, "REGISTRATION", benchmark.REGISTRATION)
    monkeypatch.setattr(
        arena,
        "evaluate_artifact_position",
        lambda **kwargs: (
            captured.update(kwargs)
            or {
                "legal_moves": [0, 1, 2, 3, 4, 5],
                "selected_move": 0,
                "policy": [1 / 6] * 6,
                "visits": [200] * 6,
            }
        ),
    )
    config = {"evaluator": object(), "artifact_path": benchmark.ROOT / "sentinel"}
    result, duration = benchmark.run_search(state, config, seed=77)
    assert duration >= 0.0
    assert result["selected_move"] == 0
    assert captured["simulations"] == 1200
    assert captured["seed"] == 77
    assert captured["c_puct"] == 1.25
    assert captured["evaluator"] is config["evaluator"]
    assert captured["exact_root_solve_threshold"] == 16
    assert captured["exact_solve_stone_threshold"] is None


def test_aggregation_and_paired_latency_ratio() -> None:
    measurements = []
    for state_id, base, p in (("a", 2.0, 3.0), ("b", 4.0, 2.0)):
        for repetition in range(5):
            measurements.extend(
                [
                    {
                        "state_sha256": state_id,
                        "config": "baseline",
                        "repetition": repetition,
                        "seed": 1,
                        "seconds": base,
                        "exact_root": False,
                    },
                    {
                        "state_sha256": state_id,
                        "config": "P",
                        "repetition": repetition,
                        "seed": 1,
                        "seconds": p,
                        "exact_root": repetition == 0,
                    },
                ]
            )
    pairs = benchmark.paired_ratio_rows(measurements)
    assert len(pairs) == 10
    assert pairs[0]["P_over_baseline"] == pytest.approx(1.5)
    summary = benchmark.summarize(measurements)
    assert summary["configs"]["baseline"]["searches"] == 10
    assert summary["configs"]["baseline"][
        "throughput_searches_per_second"
    ] == pytest.approx(10 / 30)
    assert summary["search_path_counts"]["P"] == {"exact_root": 2, "puct": 8}


def test_repeatability_comparison_discards_nested_wall_clock_telemetry() -> None:
    first = {
        "selected_move": 2,
        "exact_root_decision": {"optimal_moves": [2], "root_latency_ms": 14.0},
    }
    repeated = {
        "selected_move": 2,
        "exact_root_decision": {"optimal_moves": [2], "root_latency_ms": 18.0},
    }
    assert benchmark.repeatability_signature(
        first
    ) == benchmark.repeatability_signature(repeated)
