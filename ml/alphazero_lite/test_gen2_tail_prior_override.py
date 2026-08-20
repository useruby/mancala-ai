from __future__ import annotations

import random

import numpy as np
import pytest

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.policy_prior_localization import (
    TailPriorSubstitutionOverride,
)
from ml.alphazero_lite.run_gen2_tail_prior_override import calibrate
from ml.alphazero_lite.self_play import PUCT, build_eval_search_options


class FixedEvaluator:
    def __init__(self, policy: list[float], value: float = 0.25) -> None:
        self.policy = np.asarray(policy, dtype=np.float32)
        self.value = value

    def evaluate(self, _game):
        return self.policy.copy(), self.value


def _game() -> KalahGame:
    return KalahGame([4] * 12, [0, 0], 0)


def test_tail_threshold_gates_at_boundary_and_masks_illegal_moves() -> None:
    parent = FixedEvaluator([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    override = TailPriorSubstitutionOverride(parent, threshold=0.399)
    game = _game()
    candidate = np.asarray([0.3, 0.7, 10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = override(game=game, legal_moves=[0, 1], priors=candidate, depth=2)
    assert np.allclose(out, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert override.telemetry_log[-1]["candidate_vs_parent_legal_l1"] == pytest.approx(
        0.4
    )
    assert override.telemetry_log[-1]["substituted"] is True


def test_tail_threshold_leaves_lower_drift_candidate_prior() -> None:
    parent = FixedEvaluator([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    override = TailPriorSubstitutionOverride(parent, threshold=0.41)
    out = override(
        game=_game(),
        legal_moves=[0, 1],
        priors=np.asarray([0.3, 0.7, 10.0, 0.0, 0.0, 0.0], dtype=np.float32),
        depth=8,
    )
    assert np.allclose(out, [0.3, 0.7, 0.0, 0.0, 0.0, 0.0])
    assert override.telemetry_log[-1]["substituted"] is False


def test_incumbent_all_is_search_equivalent_when_values_match() -> None:
    candidate = FixedEvaluator([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
    parent = FixedEvaluator([0.3, 0.1, 0.1, 0.2, 0.2, 0.1])
    options = build_eval_search_options(tactical_root_bias=0.0)
    override = TailPriorSubstitutionOverride(parent, threshold=None)
    tail = PUCT(
        evaluator=candidate,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(7),
        prior_override=override,
        **options,
    )
    baseline = PUCT(
        evaluator=parent,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(7),
        **options,
    )
    assert np.array_equal(tail.run(_game())[0], baseline.run(_game())[0])


def test_tail_override_applies_at_multiple_tree_depths_and_preserves_value() -> None:
    candidate = FixedEvaluator([0.1, 0.2, 0.3, 0.15, 0.15, 0.1], value=0.37)
    parent = FixedEvaluator([0.3, 0.1, 0.1, 0.2, 0.2, 0.1], value=-0.8)
    override = TailPriorSubstitutionOverride(parent, threshold=0.0)
    search = PUCT(
        evaluator=candidate,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(7),
        prior_override=override,
        **build_eval_search_options(tactical_root_bias=0.0),
    )
    search.run(_game())
    assert {entry["depth"] for entry in override.telemetry_log} >= {0, 1}
    assert all(entry["substituted"] for entry in override.telemetry_log)
    assert search._last_root_raw_evaluation_value == pytest.approx(0.37)


def test_calibration_is_deterministic_and_deduplicates_replay_states() -> None:
    candidate = FixedEvaluator([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
    parent = FixedEvaluator([0.3, 0.1, 0.1, 0.2, 0.2, 0.1])
    state = _game().to_state()
    encoded = [
        *(seed / 48.0 for seed in state["player_pits"]),
        *(seed / 48.0 for seed in state["opponent_pits"]),
        0.0,
        0.0,
        0.0,
    ]
    rows = [{"state": encoded}, {"state": encoded.copy()}]
    assert calibrate(rows, candidate, parent) == calibrate(rows, candidate, parent)
    assert calibrate(rows, candidate, parent)["unique_state_count"] == 1
