#!/usr/bin/env python3
# ruff: noqa: E402
"""Unit tests for parent-relative hard output-space trust region projection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.hard_projection import (
    POLICY_KEYS,
    TrustStateSet,
    project_policy_head_step,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    group_parameters_identical,
    trunk_parameters_identical,
)
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (
    VALUE_STACK_PREFIXES,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _new_model


INITIAL_BOARD = {
    "player_pits": [4, 4, 4, 4, 4, 4],
    "opponent_pits": [4, 4, 4, 4, 4, 4],
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def _build_synthetic_trust_set(
    p1_model: torch.nn.Module,
    device: torch.device,
    num_states: int = 128,
) -> TrustStateSet:
    """Construct a synthetic TrustStateSet with valid Mancala board states."""
    rows = []
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    game = KalahGame.from_state(INITIAL_BOARD)
    encoded_init = encode_state(game.to_state(), input_encoding="kalah_v3")
    rows.append({"state": encoded_init})

    # Generate additional valid states via random playouts
    import random

    rng = random.Random(12345)
    for _ in range(num_states - 1):
        g = KalahGame.from_state(INITIAL_BOARD)
        moves = rng.randint(1, 15)
        for _ in range(moves):
            leg = g.possible_moves()
            if not leg or g.over():
                break
            m = rng.choice(leg)
            g.move(g.pit_index(m))
        enc = encode_state(g.to_state(), input_encoding="kalah_v3")
        rows.append({"state": enc})

    return TrustStateSet.from_replay_rows(rows, p1_model, device)


def test_projection_noop_inside_radius() -> None:
    """Inside trust radius, projection must be an exact no-op with lambda=1.0."""
    device = torch.device("cpu")
    model = _new_model(device)
    trust_set = _build_synthetic_trust_set(model, device)

    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}
    theta_old = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

    # Small perturbation that stays inside radius
    theta_raw = {
        k: theta_old[k] + torch.randn_like(theta_old[k]) * 1e-5 for k in POLICY_KEYS
    }
    raw_l1 = trust_set.compute_mean_l1(theta_raw)
    radius = raw_l1 * 2.0  # strictly larger than raw drift

    accepted_theta, telemetry = project_policy_head_step(
        model=model,
        theta_old=theta_old,
        theta_raw=theta_raw,
        theta_p1=theta_p1,
        trust_set=trust_set,
        radius=radius,
    )

    assert telemetry["lambda_accepted"] == 1.0
    assert not telemetry["projection_activated"]
    assert math.isclose(telemetry["accepted_mean_l1"], raw_l1, rel_tol=1e-5)
    for k in POLICY_KEYS:
        assert torch.allclose(accepted_theta[k], theta_raw[k])
        assert torch.allclose(model.state_dict()[k], theta_raw[k])


def test_boundary_clipping() -> None:
    """When update exceeds radius, bisection clips to the boundary within tolerance."""
    device = torch.device("cpu")
    model = _new_model(device)
    trust_set = _build_synthetic_trust_set(model, device)

    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}
    theta_old = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

    # Large perturbation exceeding radius
    theta_raw = {
        k: theta_old[k] + torch.randn_like(theta_old[k]) * 0.05 for k in POLICY_KEYS
    }
    raw_l1 = trust_set.compute_mean_l1(theta_raw)
    radius = raw_l1 * 0.25  # strictly smaller than raw drift

    accepted_theta, telemetry = project_policy_head_step(
        model=model,
        theta_old=theta_old,
        theta_raw=theta_raw,
        theta_p1=theta_p1,
        trust_set=trust_set,
        radius=radius,
        max_bisection_steps=30,
        tolerance=1e-6,
    )

    assert telemetry["projection_activated"]
    assert 0.0 < telemetry["lambda_accepted"] < 1.0
    assert telemetry["accepted_mean_l1"] <= radius + 1e-6
    # Should be tightly clipped close to radius
    assert abs(telemetry["accepted_mean_l1"] - radius) < 1e-4


def test_parent_ray_retracts_to_a_new_boundary_point() -> None:
    """Parent-ray retraction moves when the old segment is pinned at the boundary."""
    device = torch.device("cpu")
    model = _new_model(device)
    trust_set = _build_synthetic_trust_set(model, device)
    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

    torch.manual_seed(123)
    direction_a = {k: torch.randn_like(theta_p1[k]) for k in POLICY_KEYS}
    far_a = {k: theta_p1[k] + 0.05 * direction_a[k] for k in POLICY_KEYS}
    radius = trust_set.compute_mean_l1(far_a) * 0.2
    theta_old, old_telemetry = project_policy_head_step(
        model, theta_p1, far_a, theta_p1, trust_set, radius, mode="parent_ray"
    )
    assert old_telemetry["accepted_mean_l1"] <= radius + 1e-6

    direction_b = {k: torch.randn_like(theta_p1[k]) for k in POLICY_KEYS}
    theta_raw = {k: theta_old[k] + 0.05 * direction_b[k] for k in POLICY_KEYS}
    accepted, telemetry = project_policy_head_step(
        model,
        theta_old,
        theta_raw,
        theta_p1,
        trust_set,
        radius,
        mode="parent_ray",
    )

    assert telemetry["projection_activated"]
    assert telemetry["accepted_mean_l1"] <= radius + 1e-6
    assert telemetry["param_delta_acc_vs_old"] > 1e-5
    assert any(not torch.equal(accepted[k], theta_old[k]) for k in POLICY_KEYS)


def test_tangent_retract_removes_boundary_radial_component() -> None:
    """A boundary proposal is made parameter-orthogonal to its P1 radial vector."""
    device = torch.device("cpu")
    model = _new_model(device)
    trust_set = _build_synthetic_trust_set(model, device)
    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}
    torch.manual_seed(456)
    direction = {k: torch.randn_like(theta_p1[k]) for k in POLICY_KEYS}
    far = {k: theta_p1[k] + 0.05 * direction[k] for k in POLICY_KEYS}
    radius = trust_set.compute_mean_l1(far) * 0.2
    theta_old, _ = project_policy_head_step(
        model, theta_p1, far, theta_p1, trust_set, radius, mode="parent_ray"
    )
    theta_raw = {k: theta_old[k] + 0.01 * direction[k] for k in POLICY_KEYS}
    accepted, telemetry = project_policy_head_step(
        model,
        theta_old,
        theta_raw,
        theta_p1,
        trust_set,
        radius,
        mode="tangent_retract",
    )
    radial = torch.cat(
        [(theta_old[k] - theta_p1[k]).double().flatten() for k in POLICY_KEYS]
    )
    accepted_step = torch.cat(
        [(accepted[k] - theta_old[k]).double().flatten() for k in POLICY_KEYS]
    )
    assert telemetry["tangent_constraint_activated"]
    assert telemetry["accepted_mean_l1"] <= radius + 1e-6
    assert abs(float(torch.dot(radial, accepted_step))) < 1e-5


def test_cumulative_parent_relative_constraint() -> None:
    """Cumulative drift vs frozen P1 remains strictly bounded across multiple sequential steps."""
    device = torch.device("cpu")
    model = _new_model(device)
    trust_set = _build_synthetic_trust_set(model, device)

    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}
    theta_current = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}

    radius = 0.00125
    rng = (
        torch.Generator().manual_state_seed(42)
        if hasattr(torch.Generator(), "manual_state_seed")
        else torch.manual_seed(42)
    )

    for step in range(1, 10):
        # Raw proposed step pushing away from current
        theta_raw = {
            k: theta_current[k]
            + torch.randn(theta_current[k].shape, generator=rng) * 0.01
            for k in POLICY_KEYS
        }

        theta_acc, telemetry = project_policy_head_step(
            model=model,
            theta_old=theta_current,
            theta_raw=theta_raw,
            theta_p1=theta_p1,
            trust_set=trust_set,
            radius=radius,
            max_bisection_steps=30,
            tolerance=1e-6,
        )

        measured_drift_vs_p1 = trust_set.compute_mean_l1(theta_acc)
        assert measured_drift_vs_p1 <= radius + 1e-6
        assert telemetry["accepted_mean_l1"] <= radius + 1e-6
        theta_current = theta_acc


def test_deterministic_bisection() -> None:
    """Exact identical inputs produce identical lambda and parameters across repeated runs."""
    device = torch.device("cpu")
    model1 = _new_model(device)
    model2 = _new_model(device)
    trust_set = _build_synthetic_trust_set(model1, device)

    theta_p1 = {k: model1.state_dict()[k].clone() for k in POLICY_KEYS}
    theta_old = {k: model1.state_dict()[k].clone() for k in POLICY_KEYS}

    torch.manual_seed(999)
    theta_raw = {
        k: theta_old[k] + torch.randn_like(theta_old[k]) * 0.02 for k in POLICY_KEYS
    }
    radius = 0.0010

    acc1, tel1 = project_policy_head_step(
        model=model1,
        theta_old=theta_old,
        theta_raw=theta_raw,
        theta_p1=theta_p1,
        trust_set=trust_set,
        radius=radius,
        max_bisection_steps=30,
        tolerance=1e-6,
    )

    acc2, tel2 = project_policy_head_step(
        model=model2,
        theta_old=theta_old,
        theta_raw=theta_raw,
        theta_p1=theta_p1,
        trust_set=trust_set,
        radius=radius,
        max_bisection_steps=30,
        tolerance=1e-6,
    )

    assert tel1["lambda_accepted"] == tel2["lambda_accepted"]
    assert tel1["accepted_mean_l1"] == tel2["accepted_mean_l1"]
    for k in POLICY_KEYS:
        assert torch.equal(acc1[k], acc2[k])
        assert torch.equal(model1.state_dict()[k], model2.state_dict()[k])


def test_frozen_parameter_families() -> None:
    """Trunk and value stack parameters remain bit-identical during projection."""
    device = torch.device("cpu")
    model = _new_model(device)
    s0 = {k: v.clone() for k, v in model.state_dict().items()}
    trust_set = _build_synthetic_trust_set(model, device)

    theta_p1 = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}
    theta_old = {k: model.state_dict()[k].clone() for k in POLICY_KEYS}
    theta_raw = {
        k: theta_old[k] + torch.randn_like(theta_old[k]) * 0.05 for k in POLICY_KEYS
    }

    _acc, _tel = project_policy_head_step(
        model=model,
        theta_old=theta_old,
        theta_raw=theta_raw,
        theta_p1=theta_p1,
        trust_set=trust_set,
        radius=0.00125,
    )

    current_state = model.state_dict()
    assert trunk_parameters_identical(current_state, s0)
    assert group_parameters_identical(current_state, s0, VALUE_STACK_PREFIXES)
