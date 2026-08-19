from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import run_frozen_trunk_policy_prior_localization as runner
from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.policy_prior_localization import (
    MODE_DEPTH_THRESHOLD,
    PriorSubstitutionOverride,
    build_prior_substitution_override,
    mode_uses_incumbent_at_depth,
    summarize_override_telemetry,
)
from ml.alphazero_lite.self_play import PUCT, build_eval_search_options

CANDIDATE = Path(
    "/tmp/azlite_frozen_trunk_distillation/policy_head/snapshot_artifacts/step_0046/artifact"
)
INCUMBENT = Path("/home/alex/Mancala/ai/model-artifact/current")
CANDIDATE_SNAPSHOT = Path(
    "/tmp/azlite_frozen_trunk_distillation/policy_head/snapshots/step_0046.pt"
)
INCUMBENT_SNAPSHOT = Path(
    "/tmp/azlite_frozen_trunk_distillation/heads_only/snapshots/step_0000.pt"
)


def _game() -> KalahGame:
    return KalahGame([4] * 12, [0, 0], 0)


def _search_options() -> dict:
    return build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )


def _evaluators():
    return ArtifactEvaluator(CANDIDATE), ArtifactEvaluator(INCUMBENT)


# --- mode / threshold logic --------------------------------------------------


def test_mode_depth_threshold_matches_spec() -> None:
    assert MODE_DEPTH_THRESHOLD == {
        "candidate_all": -1,
        "incumbent_root": 0,
        "incumbent_depth1": 1,
        "incumbent_depth2": 2,
        "incumbent_all": 2**31 - 1,
    }


def test_mode_uses_incumbent_at_depth_gates_correctly() -> None:
    assert mode_uses_incumbent_at_depth("candidate_all", 0) is False
    assert mode_uses_incumbent_at_depth("incumbent_root", 0) is True
    assert mode_uses_incumbent_at_depth("incumbent_root", 1) is False
    assert mode_uses_incumbent_at_depth("incumbent_depth1", 1) is True
    assert mode_uses_incumbent_at_depth("incumbent_depth1", 2) is False
    assert mode_uses_incumbent_at_depth("incumbent_depth2", 2) is True
    assert mode_uses_incumbent_at_depth("incumbent_depth2", 3) is False
    assert mode_uses_incumbent_at_depth("incumbent_all", 10_000) is True


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        mode_uses_incumbent_at_depth("bogus", 0)


def test_candidate_all_returns_no_override() -> None:
    assert build_prior_substitution_override("candidate_all", object()) is None


def test_non_candidate_mode_requires_incumbent() -> None:
    with pytest.raises(ValueError):
        build_prior_substitution_override("incumbent_root", None)


# --- override masking / normalization / depth gating -------------------------


def test_override_substitutes_only_below_threshold() -> None:
    cand_ev, inc_ev = _evaluators()
    override = PriorSubstitutionOverride("incumbent_root", inc_ev)
    game = _game()
    legal = game.possible_moves()
    cand_policy, _ = cand_ev.evaluate(game)
    cand_masked = np.zeros(6, dtype=np.float32)
    cand_masked[legal] = cand_policy[legal]
    cand_masked /= cand_masked[legal].sum()

    out_root = override(
        game=game.clone(), legal_moves=list(legal), priors=cand_masked.copy(), depth=0
    )
    out_depth1 = override(
        game=game.clone(), legal_moves=list(legal), priors=cand_masked.copy(), depth=1
    )

    # Root: incumbent prior is used (must differ from candidate prior in general).
    assert not np.allclose(out_root, cand_masked)
    # Depth 1: candidate prior returned unchanged.
    assert np.allclose(out_depth1, cand_masked)
    # Both are valid legal-normalized distributions.
    assert out_root[legal].sum() == pytest.approx(1.0, abs=1e-6)
    assert out_depth1[legal].sum() == pytest.approx(1.0, abs=1e-6)


def test_override_returns_normalized_incumbent_legal_mass() -> None:
    _, inc_ev = _evaluators()
    override = PriorSubstitutionOverride("incumbent_all", inc_ev)
    game = _game()
    legal = game.possible_moves()
    priors = np.zeros(6, dtype=np.float32)
    priors[legal] = 1.0 / len(legal)
    out = override(
        game=game.clone(), legal_moves=list(legal), priors=priors.copy(), depth=3
    )
    assert out.shape == priors.shape
    assert np.all(out[~np.isin(np.arange(6), legal)] == 0.0)
    assert out[legal].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(out[legal] >= 0.0)


def test_override_telemetry_records_depths_and_substitution() -> None:
    cand_ev, inc_ev = _evaluators()
    override = PriorSubstitutionOverride("incumbent_depth1", inc_ev)
    game = _game()
    legal = game.possible_moves()
    priors = np.zeros(6, dtype=np.float32)
    priors[legal] = 1.0 / len(legal)
    for depth in (0, 1, 2, 3):
        override(
            game=game.clone(),
            legal_moves=list(legal),
            priors=priors.copy(),
            depth=depth,
        )
    summary = summarize_override_telemetry(override.telemetry_log)
    assert summary["total_expanded_nodes"] == 4
    assert summary["total_affected_nodes"] == 2
    assert summary["0"]["affected_nodes"] == 1
    assert summary["1"]["affected_nodes"] == 1
    assert summary["2"]["affected_nodes"] == 0
    assert summary["3"]["affected_nodes"] == 0


# --- PUCT depth tracking + search equivalence invariants ---------------------


def test_puct_prior_override_called_with_increasing_depth() -> None:
    cand_ev, inc_ev = _evaluators()
    override = PriorSubstitutionOverride("incumbent_all", inc_ev)
    search = PUCT(
        evaluator=cand_ev,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(0),
        prior_override=override,
        **_search_options(),
    )
    search.run(_game())
    depths = {entry["depth"] for entry in override.telemetry_log}
    assert 0 in depths
    assert max(depths) >= 1


def test_incumbent_all_is_search_equivalent_to_incumbent() -> None:
    """Required invariant: incumbent_all == pure incumbent search."""
    cand_ev, inc_ev = _evaluators()
    override = build_prior_substitution_override("incumbent_all", inc_ev)
    states = [
        KalahGame([4] * 12, [0, 0], 0),
        KalahGame([1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [3, 0], 1),
        KalahGame([0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [2, 1], 0),
    ]
    for state in states:
        legal = state.possible_moves()
        if not legal:
            continue
        seed = 12345
        cand_search = PUCT(
            evaluator=cand_ev,
            simulations=128,
            c_puct=1.25,
            rng=random.Random(seed),
            prior_override=override,
            **_search_options(),
        )
        inc_search = PUCT(
            evaluator=inc_ev,
            simulations=128,
            c_puct=1.25,
            rng=random.Random(seed),
            **_search_options(),
        )
        v_cand, _ = cand_search.run(state.clone())
        v_inc, _ = inc_search.run(state.clone())
        assert np.array_equal(v_cand, v_inc), (
            f"incumbent_all visits diverge from pure incumbent: {v_cand} vs {v_inc}"
        )


def test_candidate_all_override_is_noop() -> None:
    cand_ev, _ = _evaluators()
    seed = 7
    s1 = PUCT(
        evaluator=cand_ev,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(seed),
        **_search_options(),
    )
    s2 = PUCT(
        evaluator=cand_ev,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(seed),
        **_search_options(),
    )
    v1, _ = s1.run(_game())
    v2, _ = s2.run(_game())
    assert np.array_equal(v1, v2)


def test_prior_override_does_not_change_value_outputs() -> None:
    """The override touches priors only; the candidate value path is unchanged."""
    cand_ev, inc_ev = _evaluators()
    override = build_prior_substitution_override("incumbent_all", inc_ev)
    game = _game()
    search_options = _search_options()
    base = PUCT(
        evaluator=cand_ev,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(0),
        **search_options,
    )
    over = PUCT(
        evaluator=cand_ev,
        simulations=64,
        c_puct=1.25,
        rng=random.Random(0),
        prior_override=override,
        **search_options,
    )
    base.run(game.clone())
    over.run(game.clone())
    assert base._last_root_raw_evaluation_value == pytest.approx(
        over._last_root_raw_evaluation_value, abs=1e-9
    )


# --- invariant helpers -------------------------------------------------------


def _state_hash(path: Path) -> str:
    from ml.alphazero_lite.evaluation_seed_contract import stable_hash

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    return stable_hash(
        {k: v.detach().cpu().numpy().tobytes().hex() for k, v in state.items()}
    )


@pytest.mark.skipif(
    not CANDIDATE_SNAPSHOT.is_file(),
    reason="PR #200 step-46 policy_head snapshot not staged on this machine",
)
def test_candidate_snapshot_matches_pr200_step46() -> None:
    assert _state_hash(CANDIDATE_SNAPSHOT) == runner.PR200_CANDIDATE_STATE_HASH


@pytest.mark.skipif(
    not (CANDIDATE_SNAPSHOT.is_file() and INCUMBENT_SNAPSHOT.is_file()),
    reason="snapshots not staged on this machine",
)
def test_candidate_trunk_and_value_stack_match_incumbent() -> None:
    cand = torch.load(CANDIDATE_SNAPSHOT, map_location="cpu", weights_only=False)[
        "model"
    ]
    inc = torch.load(INCUMBENT_SNAPSHOT, map_location="cpu", weights_only=False)[
        "model"
    ]
    cand = {k: v.detach().cpu() for k, v in cand.items()}
    inc = {k: v.detach().cpu() for k, v in inc.items()}
    trunk_prefixes = ("input_layer.", "residual_layers.")
    value_prefixes = ("value_hidden_layer.", "value_head.")
    policy_prefixes = ("policy_hidden_layer.", "policy_head.", "move_projections.")
    assert all(
        np.array_equal(cand[k].numpy(), inc[k].numpy())
        for k in sorted(cand)
        if k.startswith(trunk_prefixes)
    )
    assert all(
        np.array_equal(cand[k].numpy(), inc[k].numpy())
        for k in sorted(cand)
        if k.startswith(value_prefixes)
    )
    assert any(
        not np.array_equal(cand[k].numpy(), inc[k].numpy())
        for k in sorted(cand)
        if k.startswith(policy_prefixes)
    )


# --- classification rule -----------------------------------------------------


def _effect(effect: float, upper: float, p0: float, p1: float = 0.0) -> dict:
    return {
        "paired_candidate_effect": effect,
        "opening_bootstrap_ci": {
            "lower_95": effect - 0.02,
            "upper_95": upper,
            "samples": 10000,
        },
        "p0_effect": p0,
        "p1_effect": p1,
        "orientation": "candidate_minus_incumbent",
        "win_draw_loss": {"wins": 0, "draws": 0, "losses": 0},
        "trajectory_divergence_from_candidate_all": {"divergence_rate": 0.0},
    }


def _summary(arena: dict) -> dict:
    return {"arena": arena}


def test_classify_root_prior_causal() -> None:
    arena = {
        "candidate_all": {"384:256": _effect(-0.19, -0.15, -0.37)},
        "incumbent_root": {"384:256": _effect(-0.03, 0.01, -0.05)},
        "incumbent_depth1": {"384:256": _effect(-0.02, 0.02, -0.04)},
        "incumbent_depth2": {"384:256": _effect(-0.02, 0.02, -0.04)},
        "incumbent_all": {"384:256": _effect(0.0, 0.02, 0.0)},
    }
    label = runner.classify(_summary(arena))["label"]
    assert label == "root_prior_causal"


def test_classify_shallow_prior_compounding() -> None:
    arena = {
        "candidate_all": {"384:256": _effect(-0.19, -0.15, -0.37)},
        "incumbent_root": {"384:256": _effect(-0.12, -0.08, -0.20)},
        "incumbent_depth1": {"384:256": _effect(-0.03, 0.01, -0.04)},
        "incumbent_depth2": {"384:256": _effect(-0.02, 0.02, -0.04)},
        "incumbent_all": {"384:256": _effect(0.0, 0.02, 0.0)},
    }
    label = runner.classify(_summary(arena))["label"]
    assert label == "shallow_prior_compounding"


def test_classify_distributed_prior_compounding() -> None:
    arena = {
        "candidate_all": {"384:256": _effect(-0.19, -0.15, -0.37)},
        "incumbent_root": {"384:256": _effect(-0.12, -0.08, -0.20)},
        "incumbent_depth1": {"384:256": _effect(-0.10, -0.06, -0.18)},
        "incumbent_depth2": {"384:256": _effect(-0.08, -0.04, -0.14)},
        "incumbent_all": {"384:256": _effect(0.0, 0.02, 0.0)},
    }
    label = runner.classify(_summary(arena))["label"]
    assert label == "distributed_prior_compounding"


def test_classify_unexpected_when_incumbent_all_not_equivalent() -> None:
    arena = {
        "candidate_all": {"384:256": _effect(-0.19, -0.15, -0.37)},
        "incumbent_root": {"384:256": _effect(-0.03, 0.01, -0.05)},
        "incumbent_depth1": {"384:256": _effect(-0.02, 0.02, -0.04)},
        "incumbent_depth2": {"384:256": _effect(-0.02, 0.02, -0.04)},
        # incumbent_all unexpectedly still harmful despite the frozen-family invariant.
        "incumbent_all": {"384:256": _effect(-0.10, -0.06, -0.18)},
    }
    label = runner.classify(_summary(arena))["label"]
    assert label == "unexpected_nonpolicy_difference"


def test_recovery_fraction_formula() -> None:
    # (intervention - baseline) / (0 - baseline)
    assert runner._recovery(-0.057, -0.19) == pytest.approx(0.7, abs=1e-6)
    assert runner._recovery(0.0, -0.19) == pytest.approx(1.0, abs=1e-6)
    assert runner._recovery(-0.19, -0.19) == pytest.approx(0.0, abs=1e-6)
    assert runner._recovery(-0.19, 0.0) is None
