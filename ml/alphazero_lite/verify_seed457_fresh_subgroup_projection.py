"""Independent read-only semantic checks for seed457 preparation/evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.seed457_projection import (
    COMPLEMENTARITY_TOLERANCE,
    DUAL_TOLERANCE,
    PRIMAL_TOLERANCE,
    STATIONARITY_FACTOR,
)

SCALES = tuple(2.0**-index for index in range(8))
KL_CAP = 0.005
VALUE_CAP = 0.0001
DOT_LIMIT = 1e-7

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed457-fresh-subgroup-projection")


def partition(guard: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Independently rebuild the frozen deterministic subgroup assignment."""
    identities = [row["exact_input_identity"] for row in guard]
    if len(guard) != 1399 or len(set(identities)) != 1399:
        raise ValueError("guard_identity_population_invalid")
    ordered = sorted(
        identities,
        key=lambda identity: (
            hashlib.sha256(("seed457-shard:" + identity).encode()).hexdigest(),
            identity,
        ),
    )
    ranks = {identity: rank for rank, identity in enumerate(ordered)}
    result = [[] for _ in range(8)]
    for row in guard:
        result[ranks[row["exact_input_identity"]] % 8].append(dict(row))
    if sorted(map(len, result)) != [174] + [175] * 7:
        raise ValueError("shard_count_invalid")
    return result


def _dot(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    return float(
        sum(
            np.sum(
                np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64),
                dtype=np.float64,
            )
            for a, b in zip(left, right, strict=True)
        )
    )


def verify_kkt(
    displacement: list[np.ndarray],
    projected: list[np.ndarray],
    gradients: list[list[np.ndarray]],
    source_indices: list[int],
    dual: list[float],
) -> dict[str, float]:
    """Check normalized primal/dual/stationarity/complementarity from evidence."""
    nonzero: list[list[np.ndarray]] = []
    actual_indices = []
    for index, gradient in enumerate(gradients):
        norm = np.sqrt(_dot(gradient, gradient))
        if norm:
            nonzero.append([np.asarray(g, dtype=np.float64) / norm for g in gradient])
            actual_indices.append(index)
    if source_indices != actual_indices or len(dual) != len(nonzero):
        raise ValueError("solver_constraint_binding_invalid")
    if any(value < -DUAL_TOLERANCE or not np.isfinite(value) for value in dual):
        raise ValueError("solver_dual_invalid")
    slacks = np.asarray([_dot(g, projected) for g in nonzero])
    primal = float(np.max(slacks, initial=0.0))
    complementarity = float(np.max(np.abs(slacks * dual), initial=0.0))
    correction = [np.zeros_like(np.asarray(a, dtype=np.float64)) for a in displacement]
    for coefficient, gradient in zip(dual, nonzero, strict=True):
        correction = [
            a + coefficient * b for a, b in zip(correction, gradient, strict=True)
        ]
    stationarity_vector = [
        z - d + c for z, d, c in zip(projected, displacement, correction, strict=True)
    ]
    stationarity = np.sqrt(_dot(stationarity_vector, stationarity_vector))
    if primal > PRIMAL_TOLERANCE:
        raise ValueError("solver_primal_invalid")
    if complementarity > COMPLEMENTARITY_TOLERANCE:
        raise ValueError("solver_complementarity_invalid")
    if stationarity > STATIONARITY_FACTOR * max(
        1.0, np.sqrt(_dot(displacement, displacement))
    ):
        raise ValueError("solver_stationarity_invalid")
    return {
        "normalized_primal": primal,
        "normalized_complementarity": complementarity,
        "stationarity_l2": float(stationarity),
    }


def _require_array(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError(f"semantic_tensor_mismatch:{label}")


def _require_numeric_tree(actual: Any, expected: Any, label: str) -> None:
    """Compare serialized metric trees with strict absolute double tolerance."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"semantic_metric_keys_mismatch:{label}")
        for key in actual:
            _require_numeric_tree(actual[key], expected[key], f"{label}:{key}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not np.isclose(actual, expected, atol=2e-12, rtol=0):
            raise ValueError(f"semantic_metric_mismatch:{label}")
    elif actual != expected:
        raise ValueError(f"semantic_value_mismatch:{label}")


def _gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    targets: np.ndarray,
    rows: np.ndarray,
) -> list[np.ndarray]:
    """Independent guard CE gradient with seed453 accumulation/writeback semantics."""
    params = tuple(p for p in model.parameters() if p.requires_grad)
    total = [np.zeros(tuple(p.shape), dtype=np.float64) for p in params]
    for start in range(0, len(rows), 512):
        ids = rows[start : start + 512]
        logits, _ = model(torch.from_numpy(x[ids]))
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
        ce = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(targets[ids])
        )
        grads = torch.autograd.grad(ce.sum() / len(rows), params, allow_unused=True)
        for index, grad in enumerate(grads):
            if grad is not None:
                total[index] += grad.detach().cpu().numpy().astype(np.float64)
    return [value.astype(np.float32) for value in total]


def _outputs(
    model: torch.nn.Module, x: np.ndarray, rows: np.ndarray
) -> tuple[torch.Tensor, np.ndarray]:
    logp = torch.full((len(x), train.POLICY_SIZE), -torch.inf, dtype=torch.float64)
    vals = []
    with torch.inference_mode():
        for start in range(0, len(rows), 512):
            ids = rows[start : start + 512]
            logits, value = model(torch.from_numpy(x[ids]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[ids])
            ).bool()
            logp[ids] = torch.log_softmax(
                logits.double().masked_fill(~legal, -torch.inf), dim=1
            )
            vals.append((ids, value.reshape(-1).cpu().numpy().copy()))
    value_rows = np.empty(len(x), dtype=np.float32)
    for ids, value in vals:
        value_rows[ids] = value
    return logp, value_rows


def _measure_trial(
    model: torch.nn.Module,
    x: np.ndarray,
    rows: np.ndarray,
    pre_logp: torch.Tensor,
    pre_values: np.ndarray,
) -> tuple[float, float]:
    trial_logp, trial_values = _outputs(model, x, rows)
    legal = torch.isfinite(pre_logp[rows])
    probability = pre_logp[rows].exp().masked_fill(~legal, 0.0)
    per_row = (
        probability * (pre_logp[rows] - trial_logp[rows]).masked_fill(~legal, 0.0)
    ).sum(1)
    kl = float(per_row.double().mean())
    delta = trial_values[rows].astype(np.float64) - pre_values[rows].astype(np.float64)
    return kl, float(np.mean(delta**2, dtype=np.float64))


def reconstruct_engine_step(
    *,
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    arm: str,
    record: dict[str, Any],
    archive: Any,
    full_guard_rows: np.ndarray,
    fresh_guard_rows: np.ndarray,
    subgroup_rows: list[np.ndarray],
    fixture_limits: tuple[float, float, float] | None = None,
    expected_pre: list[np.ndarray] | None = None,
    expected_moments: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct one actual engine step from its pre-state evidence."""
    if arm not in {"C", "T"}:
        raise ValueError("arm_identity_invalid")
    kl_cap, value_cap, dot_limit = fixture_limits or (
        KL_CAP,
        VALUE_CAP,
        DOT_LIMIT,
    )
    x, policy, target_value, replay, coefficients, train_positions, *_ = arrays
    params = tuple(p for p in model.parameters() if p.requires_grad)
    step = int(record["step"]) - 1
    rows = np.asarray(record["compact_rows"], dtype=np.int64)
    if rows.size != 512:
        raise ValueError("minibatch_rows_invalid")
    pre = [archive[f"{arm}_{step:02d}_{i:02d}_pre"].copy() for i in range(len(params))]
    if expected_pre is not None:
        for index, (actual, expected) in enumerate(zip(pre, expected_pre, strict=True)):
            _require_array(actual, expected, f"pre_trajectory:{index}")
    with torch.no_grad():
        for parameter, value in zip(params, pre, strict=True):
            parameter.copy_(torch.from_numpy(value.copy()))
    optimizer = torch.optim.Adam(
        params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    for index, parameter in enumerate(params):
        state = optimizer.state[parameter]
        state["step"] = torch.tensor(float(step), dtype=torch.float32)
        archived_avg = archive[f"{arm}_{step:02d}_{index:02d}_exp_avg_pre"].copy()
        archived_sq = archive[f"{arm}_{step:02d}_{index:02d}_exp_avg_sq_pre"].copy()
        if expected_moments is not None:
            _require_array(
                archived_avg, expected_moments[index][0], f"pre_moment:{index}"
            )
            _require_array(
                archived_sq, expected_moments[index][1], f"pre_moment_sq:{index}"
            )
            archived_avg, archived_sq = expected_moments[index]
        state["exp_avg"] = torch.from_numpy(archived_avg.copy())
        state["exp_avg_sq"] = torch.from_numpy(archived_sq.copy())
    logits, predicted = model(torch.from_numpy(x[rows]))
    legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[rows]))
    coeff = torch.from_numpy(coefficients[rows])
    pol = (
        train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[rows])
        )
        * coeff
    ).sum() / coeff.sum()
    val = train.compute_value_loss_vector(
        predicted,
        torch.from_numpy(target_value[rows]),
        value_loss="huber",
        huber_delta=1,
    ).mean()
    objective = pol + 0.3 * val
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    unclipped = [p.grad.detach().clone() for p in params]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    clipped = [p.grad.detach().clone() for p in params]
    aggregate_gradient = _gradient(model, x, policy, fresh_guard_rows)
    subgroup_gradients = (
        [_gradient(model, x, policy, shard) for shard in subgroup_rows]
        if arm == "T"
        else []
    )
    gradients = [aggregate_gradient] if arm == "C" else subgroup_gradients
    if any(not torch.equal(a, p.grad) for a, p in zip(clipped, params, strict=True)):
        raise ValueError("objective_gradient_overwritten")
    for i in range(len(params)):
        _require_array(
            unclipped[i].cpu().numpy(),
            archive[f"{arm}_{step:02d}_{i:02d}_objective_gradient_unclipped"],
            f"objective_gradient:{i}",
        )
        _require_array(
            clipped[i].cpu().numpy(),
            archive[f"{arm}_{step:02d}_{i:02d}_objective_gradient"],
            f"clipped_gradient:{i}",
        )
    for j, gradient in enumerate(gradients):
        for i, value in enumerate(gradient):
            key = (
                f"{arm}_{step:02d}_{i:02d}_fresh_gradient"
                if arm == "C"
                else f"{arm}_{step:02d}_{i:02d}_subgroup_{j}_gradient"
            )
            _require_array(value, archive[key], f"guard_gradient:{j}:{i}")
    optimizer.step()
    raw = [p.detach().cpu().numpy().copy() for p in params]
    for i, p in enumerate(params):
        _require_array(
            raw[i],
            archive[f"{arm}_{step:02d}_{i:02d}_raw_proposal"],
            f"raw_proposal:{i}",
        )
        _require_array(
            optimizer.state[p]["exp_avg"].cpu().numpy(),
            archive[f"{arm}_{step:02d}_{i:02d}_exp_avg"],
            f"moment:{i}",
        )
        _require_array(
            optimizer.state[p]["exp_avg_sq"].cpu().numpy(),
            archive[f"{arm}_{step:02d}_{i:02d}_exp_avg_sq"],
            f"moment_sq:{i}",
        )
    pre_np = pre
    raw_np = raw
    raw_proposal_norm = float(
        np.sqrt(
            _dot(
                [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre_np, raw_np, strict=True)
                ],
                [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre_np, raw_np, strict=True)
                ],
            )
        )
    )
    if not np.isclose(
        raw_proposal_norm, record.get("proposal_norm", np.nan), atol=2e-12, rtol=0
    ):
        raise ValueError("raw_proposal_norm_metadata_mismatch")
    projection_evidence = record["projection"]
    d = [
        b.astype(np.float64) - a.astype(np.float64)
        for a, b in zip(pre_np, raw_np, strict=True)
    ]
    if arm == "C":
        gradient = gradients[0]
        dot = _dot(gradient, d)
        norm_sq = _dot(gradient, gradient)
        coefficient = max(0.0, dot) / norm_sq if norm_sq else 0.0
        z = [
            dd - coefficient * gg.astype(np.float64)
            for dd, gg in zip(d, gradient, strict=True)
        ]
        projected = [
            np.asarray(a.astype(np.float64) + zz, dtype=np.float32)
            for a, zz in zip(pre_np, z, strict=True)
        ]
        report = projection_evidence.get("report", {})
        if not np.isclose(report.get("raw_dot", np.nan), dot, atol=1e-12, rtol=1e-12):
            raise ValueError("aggregate_projection_raw_dot_invalid")
    else:
        norms = [np.sqrt(_dot(g, g)) for g in gradients]
        nonzero = [i for i, norm in enumerate(norms) if norm != 0]
        normalized = [
            [a.astype(np.float64) / norms[i] for a in gradients[i]] for i in nonzero
        ]
        dual = [float(v) for v in projection_evidence["dual"]]
        if projection_evidence["source_indices"] != nonzero or len(dual) != len(
            nonzero
        ):
            raise ValueError("solver_constraint_binding_invalid")
        z = [a.copy() for a in d]
        for multiplier, normal in zip(dual, normalized, strict=True):
            z = [a - multiplier * b for a, b in zip(z, normal, strict=True)]
        verify_kkt(d, z, gradients, nonzero, dual)
        projected = [
            np.asarray(a.astype(np.float64) + zz, dtype=np.float32)
            for a, zz in zip(pre_np, z, strict=True)
        ]
    for i, tensor in enumerate(projected):
        _require_array(
            tensor,
            archive[f"{arm}_{step:02d}_{i:02d}_projected_proposal"],
            f"projected_proposal:{i}",
        )
    # Guard and minibatch reference distributions are always measured from the
    # reconstructed pre-state, never from Adam's raw proposal.
    with torch.no_grad():
        for parameter, value_array in zip(params, pre_np, strict=True):
            parameter.copy_(torch.from_numpy(value_array.copy()))
    batch_logp, batch_values = _outputs(model, x, rows)
    full_logp, full_values = _outputs(model, x, full_guard_rows)
    trials = []
    for scale in SCALES:
        trial = [
            p.copy()
            if scale == 1
            else np.asarray(a + (p - a) * scale, dtype=np.float32)
            for a, p in zip(pre_np, projected, strict=True)
        ]
        with torch.no_grad():
            for parameter, value_array in zip(params, trial, strict=True):
                parameter.copy_(torch.from_numpy(value_array.copy()))
        batch_kl, batch_move = _measure_trial(model, x, rows, batch_logp, batch_values)
        full_kl, full_move = _measure_trial(
            model, x, full_guard_rows, full_logp, full_values
        )
        dots = [
            _dot(
                [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre_np, trial, strict=True)
                ],
                g,
            )
            for g in gradients
        ]
        feasible = (
            batch_kl <= kl_cap + 1e-10
            and full_kl <= kl_cap + 1e-10
            and batch_move <= value_cap + 1e-12
            and full_move <= value_cap + 1e-12
            and all(dot <= dot_limit for dot in dots)
        )
        stored = record["trials"][len(trials)]
        measurements = {
            "batch_kl": batch_kl,
            "guard_kl": full_kl,
            "batch_value_movement": batch_move,
            "guard_value_movement": full_move,
        }
        for key, actual in measurements.items():
            if not np.isclose(actual, stored[key], atol=2e-12, rtol=0):
                raise ValueError(f"trial_measurement_mismatch:{key}:{scale}")
        if stored["feasible"] != feasible or not np.allclose(
            dots, stored["gradient_dots"], atol=2e-10, rtol=0
        ):
            raise ValueError(f"trial_feasibility_or_dot_mismatch:{scale}")
        for i, value_array in enumerate(trial):
            _require_array(
                value_array,
                archive[f"{arm}_{step:02d}_{i:02d}_trial_{scale:g}"],
                f"trial_parameters:{scale}:{i}",
            )
        trials.append({"scale": scale, "feasible": feasible})
    selected = next((row["scale"] for row in trials if row["feasible"]), None)
    if selected != record["selected_scale"] or (selected is None) != record["rejected"]:
        raise ValueError("selected_scale_mismatch")
    post = (
        [a.copy() for a in pre_np]
        if selected is None
        else [
            p.copy()
            if selected == 1
            else np.asarray(a + (p - a) * selected, dtype=np.float32)
            for a, p in zip(pre_np, projected, strict=True)
        ]
    )
    for i, value_array in enumerate(post):
        _require_array(
            value_array, archive[f"{arm}_{step:02d}_{i:02d}_post"], f"post_state:{i}"
        )
    return {
        "step": step + 1,
        "selected_scale": selected,
        "rejected": selected is None,
        "objective": float(objective.detach()),
        "verified_trials": len(trials),
        "post_parameters": [value.copy() for value in post],
        "moments": [
            (
                optimizer.state[p]["exp_avg"].detach().cpu().numpy().copy(),
                optimizer.state[p]["exp_avg_sq"].detach().cpu().numpy().copy(),
            )
            for p in params
        ],
        "optimizer_step": step + 1,
        "pre_parameters": [value.copy() for value in pre_np],
        "raw_proposal": [value.copy() for value in raw_np],
        "raw_proposal_norm": raw_proposal_norm,
        "accepted_displacement_norm": float(
            np.sqrt(
                _dot(
                    [
                        b.astype(np.float64) - a.astype(np.float64)
                        for a, b in zip(pre_np, post, strict=True)
                    ],
                    [
                        b.astype(np.float64) - a.astype(np.float64)
                        for a, b in zip(pre_np, post, strict=True)
                    ],
                )
            )
        ),
        "projected_proposal": [value.copy() for value in projected],
        "aggregate_gradient": [value.copy() for value in aggregate_gradient],
        "subgroup_gradients": [
            [value.copy() for value in gradient] for gradient in subgroup_gradients
        ],
        "gradient_dots": [
            _dot(
                [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre_np, post, strict=True)
                ],
                gradient,
            )
            for gradient in gradients
        ],
    }


def reconstruct_trajectory(
    *,
    model: torch.nn.Module,
    initializer: list[np.ndarray],
    arrays: tuple[np.ndarray, ...],
    arm: str,
    records: list[dict[str, Any]],
    archive: Any,
    batches: np.ndarray,
    full_guard_rows: np.ndarray,
    fresh_guard_rows: np.ndarray,
    subgroup_rows: list[np.ndarray],
    fixture_limits: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Verify all sixteen ordered steps from initializer and zero Adam state."""
    if len(records) != 16:
        raise ValueError(f"trajectory_record_count_invalid:{arm}")
    x, _policy, _values, replay, _coefficients, train_positions, *_ = arrays
    expected_compact = replay[train_positions][batches]
    if batches.shape != (8192,):
        raise ValueError("frozen_batch_vector_invalid")
    parameter_state = [
        np.asarray(value, dtype=np.float32).copy() for value in initializer
    ]
    if len(parameter_state) != len(tuple(model.parameters())):
        raise ValueError("initializer_parameter_count_invalid")
    moments = [
        (np.zeros_like(value), np.zeros_like(value)) for value in parameter_state
    ]
    verified_steps = []
    for step, record in enumerate(records):
        if record.get("step") != step + 1 or record.get("optimizer_step") != step + 1:
            raise ValueError(f"trajectory_step_order_invalid:{arm}:{step + 1}")
        expected_rows = expected_compact[step * 512 : (step + 1) * 512]
        if record.get("compact_rows") != expected_rows.tolist():
            raise ValueError(f"minibatch_identity_order_invalid:{arm}:{step + 1}")
        facts = reconstruct_engine_step(
            model=model,
            arrays=arrays,
            arm=arm,
            record=record,
            archive=archive,
            full_guard_rows=full_guard_rows,
            fresh_guard_rows=fresh_guard_rows,
            subgroup_rows=subgroup_rows,
            fixture_limits=fixture_limits,
            expected_pre=parameter_state,
            expected_moments=moments,
        )
        parameter_state = facts["post_parameters"]
        moments = facts["moments"]
        verified_steps.append(facts)
    return {
        "arm": arm,
        "steps": 16,
        "selected_scales": [record["selected_scale"] for record in records],
        "rejected_count": sum(bool(record["rejected"]) for record in records),
        "accepted_nonzero_count": sum(
            not record["rejected"] and facts["accepted_displacement_norm"] > 0
            for record, facts in zip(records, verified_steps, strict=True)
        ),
        "raw_nonzero_count": sum(
            facts["raw_proposal_norm"] > 0 for facts in verified_steps
        ),
        "rejected_count_verified": sum(facts["rejected"] for facts in verified_steps),
        "final_parameters": [value.copy() for value in parameter_state],
        "final_moments": [(a.copy(), b.copy()) for a, b in moments],
        "optimizer_step": 16,
        "verified_steps": verified_steps,
    }


def compare_c_reference(
    records: list[dict[str, Any]],
    archive: Any,
    reference_records: list[dict[str, Any]],
    reference_archive: Any,
) -> dict[str, Any]:
    """Report exact per-step scale and post-tensor C-control divergences."""
    mismatches = []
    if len(records) != 16 or len(reference_records) != 16:
        return {"matches": False, "mismatches": [{"field": "record_count"}]}
    for step, (actual, expected) in enumerate(
        zip(records, reference_records, strict=True)
    ):
        if actual.get("selected_scale") != expected.get("selected_scale"):
            mismatches.append(
                {
                    "step": step + 1,
                    "field": "selected_scale",
                    "expected": expected.get("selected_scale"),
                    "actual": actual.get("selected_scale"),
                }
            )
        for index in range(22):
            actual_key = f"C_{step:02d}_{index:02d}_post"
            reference_key = f"B_{step:02d}_{index:02d}_post"
            if not np.array_equal(
                archive[actual_key], reference_archive[reference_key]
            ):
                mismatches.append(
                    {"step": step + 1, "tensor": index, "field": "post_parameters"}
                )
    return {"matches": not mismatches, "mismatches": mismatches}


def reconstruct_endpoint_evidence(
    *,
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
    initializer: list[np.ndarray],
    trajectories: dict[str, dict[str, Any]],
    prediction_archive: Any,
    recorded_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Independently predict and recalculate endpoint metrics from verified states."""
    x, policy, value, replay, coefficients, train_positions, *_ = arrays
    strict_unseen = [
        row
        for row in membership
        if row.get("subset") == "unseen" and row.get("active_stones", 0) > 32
    ]
    fresh_unseen = [row for row in strict_unseen if row.get("source") == "fresh"]
    states = {"initializer": initializer}
    states.update(
        {arm: result["final_parameters"] for arm, result in trajectories.items()}
    )
    recomputed: dict[str, Any] = {}
    for name, parameters in states.items():
        with torch.no_grad():
            for parameter, state in zip(model.parameters(), parameters, strict=True):
                parameter.copy_(torch.from_numpy(np.asarray(state).copy()))
        logits_all = np.empty((len(x), train.POLICY_SIZE), dtype=np.float32)
        values_all = np.empty(len(x), dtype=np.float32)
        model.eval()
        with torch.inference_mode():
            for begin in range(0, len(x), 512):
                logits, predicted = model(torch.from_numpy(x[begin : begin + 512]))
                logits_all[begin : begin + len(logits)] = logits.cpu().numpy()
                values_all[begin : begin + len(logits)] = (
                    predicted.reshape(-1).cpu().numpy()
                )
        legal = train.legal_mask_matrix_for_encoded_states(x)
        losses = np.empty(len(x), dtype=np.float64)
        for begin in range(0, len(x), 512):
            stop = min(begin + 512, len(x))
            losses[begin:stop] = (
                train.compute_policy_cross_entropy(
                    torch.from_numpy(logits_all[begin:stop]).masked_fill(
                        torch.from_numpy(legal[begin:stop]) <= 0, -1e9
                    ),
                    torch.from_numpy(policy[begin:stop]),
                )
                .double()
                .numpy()
            )
        # Production full-training objective: expanded replay rows, weighted policy
        # mean and unweighted mean Huber value loss.
        expanded = replay[train_positions]
        huber = (
            train.compute_value_loss_vector(
                torch.from_numpy(values_all[expanded, None]),
                torch.from_numpy(value[expanded]),
                value_loss="huber",
                huber_delta=1.0,
            )
            .double()
            .numpy()
        )
        train_objective = float(
            np.average(losses[expanded], weights=coefficients[expanded])
            + 0.3 * np.mean(huber)
        )

        def cohort(
            rows: list[dict[str, Any]], label: str
        ) -> dict[str, dict[str, float]]:
            ids = np.asarray([int(row["compact_row"]) for row in rows], dtype=np.int64)
            identities = [row["input_identity"] for row in rows]
            cohort_logits = []
            cohort_values = []
            model.eval()
            with torch.inference_mode():
                for begin in range(0, len(ids), 512):
                    part = ids[begin : begin + 512]
                    logits, predicted = model(torch.from_numpy(x[part]))
                    cohort_logits.append(logits.cpu().numpy().copy())
                    cohort_values.append(predicted.reshape(-1).cpu().numpy().copy())
            ordered_logits = np.concatenate(cohort_logits, axis=0)
            ordered_values = np.concatenate(cohort_values, axis=0)
            groups: dict[str, list[int]] = {}
            for index, identity in enumerate(identities):
                groups.setdefault(identity, []).append(index)
            archive_fields = {
                f"{label}_compact_rows": ids,
                f"{label}_identities": np.asarray(identities, dtype="U"),
                f"{label}_targets": policy[ids].copy(),
                f"{label}_value_targets": value[ids].copy(),
                f"{label}_inputs": x[ids].copy(),
                f"{label}_logits": ordered_logits,
                f"{label}_value_predictions": ordered_values,
            }
            for key, expected in archive_fields.items():
                _require_array(
                    prediction_archive[key], expected, f"endpoint_prediction:{key}"
                )
            legal_rows = train.legal_mask_matrix_for_encoded_states(x[ids])
            exposures_ce = np.empty(len(ids), dtype=np.float64)
            for begin in range(0, len(ids), 512):
                stop = min(begin + 512, len(ids))
                exposures_ce[begin:stop] = (
                    train.compute_policy_cross_entropy(
                        torch.from_numpy(ordered_logits[begin:stop]).masked_fill(
                            torch.from_numpy(legal_rows[begin:stop]) <= 0, -1e9
                        ),
                        torch.from_numpy(policy[ids[begin:stop]]),
                    )
                    .double()
                    .numpy()
                )
            exposures_mse = np.square(
                ordered_values.astype(np.float64) - value[ids, 0].astype(np.float64)
            )
            return {
                "exposure_weighted": {
                    "policy_ce": float(np.mean(exposures_ce)),
                    "value_mse": float(np.mean(exposures_mse)),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([np.mean(exposures_ce[ix]) for ix in groups.values()])
                    ),
                    "value_mse": float(
                        np.mean([np.mean(exposures_mse[ix]) for ix in groups.values()])
                    ),
                },
            }

        pooled_label, fresh_label = f"{name}_pooled", f"{name}_fresh"
        pooled = cohort(strict_unseen, pooled_label)
        fresh = cohort(fresh_unseen, fresh_label)
        recomputed[name] = {
            **pooled,
            "fresh": fresh,
            "full_training_objective": train_objective,
        }
        for weighting in ("exposure_weighted", "equal_input"):
            for metric in ("policy_ce", "value_mse"):
                for scope, actual in (
                    ("pooled", pooled[weighting][metric]),
                    ("fresh", fresh[weighting][metric]),
                ):
                    expected = (
                        recorded_metrics[name][scope][weighting][metric]
                        if scope == "fresh"
                        else recorded_metrics[name][weighting][metric]
                    )
                    if not np.isclose(actual, expected, atol=2e-12, rtol=0):
                        raise ValueError(
                            f"endpoint_metric_mismatch:{name}:{scope}:{weighting}:{metric}"
                        )
        if not np.isclose(
            train_objective,
            recorded_metrics[name]["full_training_objective"],
            atol=2e-12,
            rtol=0,
        ):
            raise ValueError(f"training_objective_mismatch:{name}")
    effects = {}
    for weighting in ("exposure_weighted", "equal_input"):
        effects[weighting] = {
            metric: recomputed["T"][weighting][metric]
            - recomputed["C"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
        effects[f"fresh_{weighting}"] = {
            metric: recomputed["T"]["fresh"][weighting][metric]
            - recomputed["C"]["fresh"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
    effects["full_training_objective"] = {
        "T_minus_C": recomputed["T"]["full_training_objective"]
        - recomputed["C"]["full_training_objective"]
    }
    return {"metrics": recomputed, "effects_T_minus_C": effects}


def reconstruct_activation(
    t_steps: list[dict[str, Any]], verified_steps: list[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute additional subgroup protection from verified gradients and states."""
    active = []
    for record, facts in zip(t_steps, verified_steps, strict=True):
        if record["rejected"] or record["selected_scale"] is None:
            continue
        pre = facts["pre_parameters"]
        raw = facts["raw_proposal"]
        aggregate = facts["aggregate_gradient"]
        subgroups = facts["subgroup_gradients"]
        d = [
            b.astype(np.float64) - a.astype(np.float64)
            for a, b in zip(pre, raw, strict=True)
        ]
        dot = _dot(aggregate, d)
        norm_sq = _dot(aggregate, aggregate)
        coefficient = max(0.0, dot) / norm_sq if norm_sq else 0.0
        aggregate_projection = [
            np.asarray(
                a.astype(np.float64) + dd - coefficient * gg.astype(np.float64),
                dtype=np.float32,
            )
            for a, dd, gg in zip(pre, d, aggregate, strict=True)
        ]
        subgroup_counter_dots = [
            _dot(
                [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre, aggregate_projection, strict=True)
                ],
                gradient,
            )
            for gradient in subgroups
        ]
        subgroup_accepted_dots = [
            _dot(
                [
                    b.astype(np.float64) - a.astype(np.float64)
                    for a, b in zip(pre, facts["post_parameters"], strict=True)
                ],
                gradient,
            )
            for gradient in subgroups
        ]
        nonzero_displacement = any(
            np.any(a.astype(np.float64) != b.astype(np.float64))
            for a, b in zip(pre, facts["post_parameters"], strict=True)
        )
        if (
            nonzero_displacement
            and any(value > DOT_LIMIT for value in subgroup_counter_dots)
            and all(value <= DOT_LIMIT for value in subgroup_accepted_dots)
        ):
            active.append(
                {
                    "step": facts["step"],
                    "counterfactual_dots": subgroup_counter_dots,
                    "accepted_dots": subgroup_accepted_dots,
                }
            )
    return {
        "schema": "seed457-activation-v1",
        "activated": bool(active),
        "steps": active,
    }


def recalculate_gates_and_classification(
    metrics: dict[str, Any],
    activation: dict[str, Any],
    t_steps: list[dict[str, Any]],
    initializer_objective: float,
    verified_steps: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, bool], str]:
    """Independently apply fixed seed457 gates and inactive/failure/success precedence."""
    gates: dict[str, bool] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        gates[f"pooled_ce_vs_seed442_A_{weighting}"] = (
            metrics["T"][weighting]["policy_ce"]
            <= metrics["seed442_A"][weighting]["policy_ce"] - 0.01
        )
        gates[f"pooled_ce_vs_initializer_{weighting}"] = (
            metrics["T"][weighting]["policy_ce"]
            <= metrics["initializer"][weighting]["policy_ce"] - 0.005
        )
        gates[f"pooled_mse_{weighting}"] = (
            metrics["T"][weighting]["value_mse"]
            <= metrics["initializer"][weighting]["value_mse"] + 0.002
        )
        gates[f"fresh_ce_{weighting}"] = (
            metrics["T"]["fresh"][weighting]["policy_ce"]
            <= metrics["initializer"]["fresh"][weighting]["policy_ce"] + 2e-6
        )
    gates["training_objective_decreases"] = (
        metrics["T"]["full_training_objective"] < initializer_objective
    )
    gates["no_nonzero_rejection"] = not any(
        facts["raw_proposal_norm"] > 0 and facts["rejected"]
        for facts in (verified_steps or [])
    )
    gates["additional_subgroup_protection"] = bool(activation["activated"])
    if not activation["activated"]:
        classification = "subgroup_projection_inactive_no_followup"
    elif all(gates.values()):
        classification = "advance_to_separately_preregistered_strength_experiment"
    else:
        classification = "close_fresh_subgroup_projection_branch"
    return gates, classification


def finalize_reconstruction(
    *,
    endpoint: dict[str, Any],
    evidence: dict[str, Any],
    activation: dict[str, Any],
    historical_metrics: dict[str, Any],
    verified_steps: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, bool], str]:
    """Validate reconstructed effects and apply fixed gates with receipt-bound A."""
    _require_numeric_tree(
        endpoint["effects_T_minus_C"],
        evidence.get("effects_T_minus_C"),
        "effects_T_minus_C",
    )
    _require_numeric_tree(
        historical_metrics,
        evidence.get("metrics", {}).get("seed442_A"),
        "seed442_A_comparator",
    )
    metrics = {**endpoint["metrics"], "seed442_A": historical_metrics}
    return recalculate_gates_and_classification(
        metrics,
        activation,
        evidence["arms"]["T"]["steps"],
        endpoint["metrics"]["initializer"]["full_training_objective"],
        verified_steps,
    )


def verify_synthetic_publication(
    *,
    root: Path,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
    initializer: list[np.ndarray],
    model: torch.nn.Module,
    batches: np.ndarray,
    full_guard_rows: np.ndarray,
    fresh_guard_rows: np.ndarray,
    subgroup_rows: list[np.ndarray],
    reference_records: list[dict[str, Any]],
    reference_archive: Any,
    output_dir: Path | None = None,
    fixture_limits: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Run the full reconstruction core for an explicitly synthetic fixture directory."""
    out = output_dir or root.resolve() / ".tmp/seed457-synthetic-test-fixture"
    evidence = json.loads((out / "evidence.json").read_text())
    if evidence.get("status") != "synthetic-test-fixture":
        raise ValueError("synthetic_fixture_label_missing")
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("status") != "synthetic-test-fixture":
        raise ValueError("synthetic_receipt_label_invalid")
    for name, digest in receipt.get("files_sha256", {}).items():
        path = out / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"receipt_hash_mismatch:{name}")
    for section in ("sources", "inputs"):
        identities = receipt.get(f"{section[:-1]}_sha256")
        if not isinstance(identities, dict) or not identities:
            raise ValueError(f"receipt_identity_missing:{section}")
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in identities.values()
        ):
            raise ValueError(f"receipt_identity_invalid:{section}")
    expected_inventory = {
        path.name
        for path in out.iterdir()
        if path.is_file() and path.name != "receipt.json"
    }
    if set(receipt.get("files_sha256", {})) != expected_inventory:
        raise ValueError("receipt_inventory_incomplete")
    if set(evidence.get("arms", {})) != {"C", "T"}:
        raise ValueError("arm_set_invalid")
    npz_archives = {
        arm: np.load(out / f"step-tensors-{arm}.npz", allow_pickle=False)
        for arm in ("C", "T")
    }
    prediction_archive = np.load(out / "endpoint-predictions.npz", allow_pickle=False)
    try:
        trajectories = {}
        for arm in ("C", "T"):
            trajectories[arm] = reconstruct_trajectory(
                model=model,
                initializer=initializer,
                arrays=arrays,
                arm=arm,
                records=evidence["arms"][arm]["steps"],
                archive=npz_archives[arm],
                batches=batches,
                full_guard_rows=full_guard_rows,
                fresh_guard_rows=fresh_guard_rows,
                subgroup_rows=subgroup_rows,
                fixture_limits=fixture_limits,
            )
        c_control = compare_c_reference(
            evidence["arms"]["C"]["steps"],
            npz_archives["C"],
            reference_records,
            reference_archive,
        )
        if evidence.get("control_reproduction") != c_control:
            raise ValueError("control_reproduction_evidence_mismatch")
        endpoint = reconstruct_endpoint_evidence(
            model=model,
            arrays=arrays,
            membership=membership,
            initializer=initializer,
            trajectories=trajectories,
            prediction_archive=prediction_archive,
            recorded_metrics=evidence["metrics"],
        )
        _require_numeric_tree(
            endpoint["metrics"],
            {key: evidence["metrics"][key] for key in endpoint["metrics"]},
            "metrics",
        )
        _require_numeric_tree(
            endpoint["effects_T_minus_C"],
            evidence["effects_T_minus_C"],
            "effects_T_minus_C",
        )
        activation = reconstruct_activation(
            evidence["arms"]["T"]["steps"], trajectories["T"]["verified_steps"]
        )
        if activation != evidence.get("activation"):
            raise ValueError("activation_semantic_mismatch")
        gates, classification = finalize_reconstruction(
            endpoint=endpoint,
            evidence=evidence,
            activation=activation,
            historical_metrics=evidence["metrics"]["seed442_A"],
            verified_steps=trajectories["T"]["verified_steps"],
        )
        if gates != evidence.get("gates"):
            raise ValueError("decision_gates_semantic_mismatch")
        if classification != evidence.get("decision", {}).get("classification"):
            raise ValueError("classification_semantic_mismatch")
        return {
            "status": "valid_synthetic_fixture",
            "steps_per_arm": 16,
            "classification": classification,
        }
    finally:
        for archive in npz_archives.values():
            archive.close()
        prediction_archive.close()


def verify_evidence_structure(evidence: dict[str, Any]) -> dict[str, Any]:
    """Check schema prerequisites only; never represents complete verification."""
    if evidence.get("schema") != "seed457-evidence-v1":
        raise ValueError("evidence_schema_invalid")
    if set(evidence.get("arms", {})) != {"C", "T"}:
        raise ValueError("arm_set_invalid")
    for arm in ("C", "T"):
        steps = evidence["arms"][arm].get("steps", [])
        if len(steps) != 16:
            raise ValueError(f"step_count_invalid:{arm}")
        for index, step in enumerate(steps, 1):
            if step.get("step") != index or len(step.get("trials", [])) != 8:
                raise ValueError(f"trial_coverage_invalid:{arm}:{index}")
            if arm == "T":
                solver = step.get("solver")
                if not isinstance(solver, dict) or "kkt_inputs" not in solver:
                    raise ValueError(f"solver_evidence_missing:{arm}:{index}")
                kkt = solver["kkt_inputs"]
                verify_kkt(
                    kkt["displacement"],
                    kkt["projected"],
                    kkt["gradients"],
                    step["solver"]["source_indices"],
                    step["solver"]["dual"],
                )
    classification = evidence.get("decision", {}).get("classification")
    if classification not in {
        "subgroup_projection_inactive_no_followup",
        "close_fresh_subgroup_projection_branch",
        "advance_to_separately_preregistered_strength_experiment",
    }:
        raise ValueError("decision_classification_invalid")
    return {
        "status": "structure_only_not_verified",
        "classification": classification,
        "steps_per_arm": 16,
    }


def verify(root: Path) -> dict[str, Any]:
    """Independently verify a real frozen publication; synthetic data is rejected."""
    root = root.resolve()
    out = root / OUT
    registration_path = out / "registration.json"
    if not registration_path.is_file():
        raise ValueError("real_registration_missing")
    registration = json.loads(registration_path.read_text())
    if registration.get("status") != "frozen_before_either_arm":
        raise ValueError("real_registration_status_invalid")
    for section in ("input_sha256", "source_sha256"):
        bindings = registration.get(section)
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError(f"registration_bindings_missing:{section}")
        for relative, digest in bindings.items():
            path = root / relative
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"frozen_binding_mismatch:{relative}")
    from ml.alphazero_lite.seed457_fresh_subgroup_projection import (
        REQUIRED_INPUTS,
        REQUIRED_SOURCES,
    )

    if not set(REQUIRED_INPUTS).issubset(registration.get("input_sha256", {})):
        raise ValueError("required_input_binding_inventory_incomplete")
    if not set(REQUIRED_SOURCES).issubset(registration.get("source_sha256", {})):
        raise ValueError("required_source_binding_inventory_incomplete")
    evidence_path = out / "evidence.json"
    if not evidence_path.is_file():
        raise ValueError("real_evidence_missing")
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("status") == "synthetic-test-fixture":
        raise ValueError("synthetic_evidence_cannot_verify_real_experiment")
    if evidence.get("status") != "real_execution":
        raise ValueError("real_evidence_status_invalid")

    from ml.alphazero_lite import seed442_kl_capped_adam as seed442
    from ml.alphazero_lite import seed435_adam_direction as seed435
    from ml.alphazero_lite import verify_seed453_fresh_policy_projection as verify453

    seed442.configure_root(root)
    arrays = verify453._portable_inputs(root)
    full_guard = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json").read_text()
    )
    fresh_guard = json.loads((out / "fresh-guard.json").read_text())
    if len(full_guard) != 2048 or len(fresh_guard) != 1399:
        raise ValueError("guard_population_invalid")
    shards = partition(fresh_guard)
    if (
        registration.get("shards") != shards
        or registration.get("fresh_guard") != fresh_guard
        or registration.get("full_guard")
        != json.loads((out / "full-guard.json").read_text())
    ):
        raise ValueError("frozen_partition_or_guard_mismatch")
    x, _policy, _value, replay, _coeff, train_positions, _validation, sources = arrays
    training_rows = set(map(int, replay[train_positions]))
    members = seed435.membership_rows()
    unseen = [
        row
        for row in members
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    exact_eval = {row["input_identity"] for row in unseen}
    canonical_eval = {row["canonical_identity"] for row in unseen}
    for guard_row in full_guard:
        compact = int(guard_row["compact_row"])
        identity = x[compact].astype("<f4", copy=False).tobytes().hex()
        if (
            compact not in training_rows
            or identity != guard_row["exact_input_identity"]
        ):
            raise ValueError(f"guard_training_or_identity_invalid:{compact}")
        if identity in exact_eval or guard_row["canonical_identity"] in canonical_eval:
            raise ValueError(f"guard_evaluation_overlap:{compact}")
    for shard in shards:
        for row in shard:
            if sources[int(row["compact_row"])].get("source") != "fresh":
                raise ValueError("fresh_guard_provenance_invalid")
    batches = np.asarray(registration["seed442_first_16_positions"], dtype=np.int64)
    if (
        batches.shape != (8192,)
        or len(set(map(int, batches))) != 8192
        or batches.reshape(16, 512).shape != (16, 512)
    ):
        raise ValueError("frozen_minibatches_invalid")
    seed442_registration = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/registration.json").read_text()
    )
    if batches.tolist() != seed442_registration["first_16_minibatch_positions"]:
        raise ValueError("seed442_minibatch_protocol_mismatch")
    seed442_receipt = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/receipt.json").read_text()
    )
    seed442_evidence_path = (
        root / "docs/data/seed442-kl-capped-adam-screen/evidence.json"
    )
    if (
        seed442_receipt.get("evidence_files_sha256", {}).get(
            "docs/data/seed442-kl-capped-adam-screen/evidence.json"
        )
        != hashlib.sha256(seed442_evidence_path.read_bytes()).hexdigest()
    ):
        raise ValueError("seed442_historical_evidence_receipt_mismatch")
    historical_metrics = json.loads(seed442_evidence_path.read_text())["metrics"]["A"]
    if evidence.get("metrics", {}).get("seed442_A") != historical_metrics:
        raise ValueError("seed442_comparator_historical_mismatch")
    if registration.get("historical_comparator") != historical_metrics:
        raise ValueError("historical_comparator_registration_mismatch")
    fixed_protocol = {
        "steps": 16,
        "batch_size": 512,
        "arms": {
            "C": "seed453-B aggregate gradient projection",
            "T": "nearest joint eight-subgroup projection",
        },
        "kl_cap": 0.005,
        "kl_tolerance": 1e-10,
        "value_cap": 0.0001,
        "value_tolerance": 1e-12,
        "dot_allowance": 1e-7,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "initial_moments": "zero",
        "gradient_clip_norm": 1.0,
        "projection_arithmetic": "float64",
        "parameter_writeback": "float32",
        "rejection": "restore_parameters_retain_advanced_adam_state",
    }
    for key, expected in fixed_protocol.items():
        if registration.get("protocol", {}).get(key) != expected:
            raise ValueError(f"production_protocol_mismatch:{key}")
    if registration.get("model") != {
        "architecture": "residual_v3",
        "shape": [96, 3],
        "input_size": 27,
    }:
        raise ValueError("production_model_mismatch")
    if registration.get("steps") != 16 or registration.get("scales") != list(SCALES):
        raise ValueError("production_trajectory_shape_mismatch")
    guard_rows = np.asarray(
        [int(row["compact_row"]) for row in full_guard], dtype=np.int64
    )
    fresh_rows = np.asarray(
        [int(row["compact_row"]) for row in fresh_guard], dtype=np.int64
    )
    subgroup_rows = [
        np.asarray([int(row["compact_row"]) for row in shard], dtype=np.int64)
        for shard in shards
    ]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    init_path = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    if (
        hashlib.sha256(init_path.read_bytes()).hexdigest()
        != "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    ):
        raise ValueError("initializer_hash_invalid")
    train.load_checkpoint_into_model(model, init_path)
    initializer = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    archives = {
        arm: np.load(out / f"step-tensors-{arm}.npz", allow_pickle=False)
        for arm in ("C", "T")
    }
    prediction_archive = np.load(out / "endpoint-predictions.npz", allow_pickle=False)
    try:
        trajectories = {}
        for arm in ("C", "T"):
            trajectories[arm] = reconstruct_trajectory(
                model=model,
                initializer=initializer,
                arrays=arrays,
                arm=arm,
                records=evidence["arms"][arm]["steps"],
                archive=archives[arm],
                batches=batches,
                full_guard_rows=guard_rows,
                fresh_guard_rows=fresh_rows,
                subgroup_rows=subgroup_rows,
            )
        historical_ev = json.loads(
            (
                root / "docs/data/seed453-fresh-policy-projection/evidence.json"
            ).read_text()
        )
        historical_archive = np.load(
            root / "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
            allow_pickle=False,
        )
        try:
            control = compare_c_reference(
                evidence["arms"]["C"]["steps"],
                archives["C"],
                historical_ev["arms"]["B"]["steps"],
                historical_archive,
            )
        finally:
            historical_archive.close()
        if not control["matches"] or control != evidence.get("control_reproduction"):
            raise ValueError(
                f"c_control_reproduction_mismatch:{control['mismatches'][:8]}"
            )
        endpoint = reconstruct_endpoint_evidence(
            model=model,
            arrays=arrays,
            membership=members,
            initializer=initializer,
            trajectories=trajectories,
            prediction_archive=prediction_archive,
            recorded_metrics=evidence["metrics"],
        )
        activation = reconstruct_activation(
            evidence["arms"]["T"]["steps"], trajectories["T"]["verified_steps"]
        )
        if activation != evidence.get("activation"):
            raise ValueError("activation_semantic_mismatch")
        gates, classification = finalize_reconstruction(
            endpoint=endpoint,
            evidence=evidence,
            activation=activation,
            historical_metrics=historical_metrics,
            verified_steps=trajectories["T"]["verified_steps"],
        )
        if gates != evidence.get("gates") or classification != evidence.get(
            "decision", {}
        ).get("classification"):
            raise ValueError("decision_semantics_mismatch")
        if evidence.get("decision", {}).get("verified") is not True:
            raise ValueError("decision_not_verified")
        receipt = json.loads((out / "receipt.json").read_text())
        if (
            receipt.get("input_sha256") != registration["input_sha256"]
            or receipt.get("source_sha256") != registration["source_sha256"]
        ):
            raise ValueError("receipt_frozen_bindings_mismatch")
        expected_artifacts = {
            str(path.relative_to(root))
            for path in out.rglob("*")
            if path.is_file() and path.name != "receipt.json"
        }
        if set(receipt.get("files_sha256", {})) != expected_artifacts:
            raise ValueError("receipt_inventory_incomplete")
        for rel, digest in receipt["files_sha256"].items():
            path = root / rel
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"receipt_mismatch:{rel}")
        return {
            "status": "valid",
            "classification": classification,
            "control_reproduction": control,
            "steps_per_arm": 16,
        }
    finally:
        for archive in archives.values():
            archive.close()
        prediction_archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
