"""Prospective joint policy-KL and value-movement screen (seed447).

The proposal generator is seed442's frozen Adam procedure. This module adds
only a training-time value-output movement feasibility check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442

OUT_NAME = "seed447-joint-output-cap"
VALUE_CAP = 0.0001
VALUE_TOLERANCE = 1e-12
FORWARD_BATCH_SIZE = 512


def value_movement(before: np.ndarray, after: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean squared float32 prediction movement in float64 arithmetic."""
    before64 = np.asarray(before, dtype=np.float32).astype(np.float64)
    after64 = np.asarray(after, dtype=np.float32).astype(np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    if before64.shape != after64.shape or before64.shape != weights64.shape:
        raise ValueError("value_movement_shape_mismatch")
    if before64.ndim != 1 or not np.isfinite(weights64).all():
        raise ValueError("value_movement_invalid_input")
    return float(np.sum(np.square(after64 - before64) * weights64, dtype=np.float64))


def archive_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Return a defensive snapshot, never a view into mutable Torch storage."""
    return tensor.detach().cpu().numpy().copy()


def choose_largest_feasible(
    trials: list[dict[str, Any]], *, cap: float, tolerance: float
) -> float | None:
    """Return the largest feasible scale, without assuming feasibility monotonicity."""
    for trial in trials:
        if (
            trial["batch_kl"] <= seed442.CAP + seed442.KL_TOLERANCE
            and trial["guard_kl"] <= seed442.CAP + seed442.KL_TOLERANCE
            and trial["batch_value_movement"] <= cap + tolerance
            and trial["guard_value_movement"] <= cap + tolerance
        ):
            return float(trial["scale"])
    return None


def cap_activation(policy_scale: float | None, joint_scale: float | None) -> bool:
    """Activation is proposal-local: value constraints lower that proposal's scale."""
    return (
        policy_scale is not None
        and joint_scale is not None
        and joint_scale < policy_scale
    )


def decide(evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed seed447 gates and classification hierarchy."""
    metrics = evidence["metrics"]
    init, ordinary, treatment = (metrics[k] for k in ("initializer", "seed442_A", "T"))
    checks: dict[str, bool] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        checks[f"T_minus_C_{weighting}_policy"] = (
            treatment[weighting]["policy_ce"] - ordinary[weighting]["policy_ce"]
            <= -0.01
        )
        checks[f"T_minus_initializer_{weighting}_policy"] = (
            treatment[weighting]["policy_ce"] - init[weighting]["policy_ce"] <= -0.005
        )
        checks[f"T_{weighting}_value_guard"] = (
            treatment[weighting]["value_mse"] <= init[weighting]["value_mse"] + 0.002
        )
    checks["T_training_objective_decreases"] = (
        treatment["full_training_objective"] < init["full_training_objective"]
    )
    steps = evidence["arms"]["T"]["steps"]
    activated = any(
        s["value_cap_activated"] and s["proposal_norm"] > 0 and not s["rejected"]
        for s in steps
    )
    checks["value_cap_activates"] = activated
    checks["no_nonzero_proposal_rejected"] = not any(
        s["proposal_norm"] > 0 and s["rejected"] for s in steps
    )
    classification = (
        "value_cap_inactive_no_followup"
        if not activated
        else "advance_to_separately_preregistered_strength_experiment"
        if all(checks.values())
        else "close_joint_output_cap_branch"
    )
    return {"classification": classification, "checks": checks}


def register(root: Path) -> dict[str, Any]:
    """Freeze the seed442 bindings and seed447-only acceptance constants."""
    root = root.resolve()
    seed442.configure_root(root)
    source = root / "docs/data/seed442-kl-capped-adam-screen/registration.json"
    base = json.loads(source.read_text(encoding="utf-8"))
    out = root / "docs/data" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "registration.json"
    if destination.exists():
        raise ValueError("registration_is_immutable")
    registration = {
        "schema": "seed447-prospective-registration-v1",
        "status": "frozen_before_either_arm",
        "seed442_registration_sha256": seed442.sha(source),
        "seed442_registration": base,
        "seed442_evidence_sha256": seed442.sha(
            root / "docs/data/seed442-kl-capped-adam-screen/evidence.json"
        ),
        "value_movement": {
            "cap": VALUE_CAP,
            "acceptance_tolerance": VALUE_TOLERANCE,
            "arithmetic": "float32_predictions_then_float64_difference_square_weighted_mean",
            "forward_batch_size": FORWARD_BATCH_SIZE,
            "weighting": "uniform minibatch exposure; equal guard input",
            "endpoint_guarantee": False,
        },
        "arms": {
            "C": "seed442-B policy-KL-capped Adam",
            "T": "same plus value movement cap",
        },
        "decision_rules": {
            "T_policy_ce_minus_seed442_A": -0.01,
            "T_policy_ce_minus_initializer": -0.005,
            "T_value_mse_margin": 0.002,
            "T_full_training_objective_decreases": True,
            "require_value_cap_activation_on_accepted_nonzero_step": True,
            "reject_nonzero_proposals": False,
            "inactive": "value_cap_inactive_no_followup",
            "active_failure": "close_joint_output_cap_branch",
            "active_success": "advance_to_separately_preregistered_strength_experiment",
        },
        "source_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in (
                "ml/alphazero_lite/seed447_joint_output_cap.py",
                "ml/alphazero_lite/seed447_analysis.py",
                "ml/alphazero_lite/verify_seed447_joint_output_cap.py",
                "ml/alphazero_lite/test_seed447_joint_output_cap.py",
            )
        },
        "bound_input_sha256": {
            **base["frozen_input_sha256"],
            "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json": seed442.sha(
                root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"
            ),
            "docs/data/seed442-kl-capped-adam-screen/B-final.npz": seed442.sha(
                root / "docs/data/seed442-kl-capped-adam-screen/B-final.npz"
            ),
        },
        "guard_identity_sha256": seed442.sha(
            root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"
        ),
        "guard_input_identities": json.loads(
            (
                root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"
            ).read_text()
        ),
        "unseen_population": [
            {
                "compact_row": int(row["compact_row"]),
                "input_identity": row["input_identity"],
                "canonical_identity": row["canonical_identity"],
                "active_stones": row["active_stones"],
            }
            for row in seed442.seed435.membership_rows()
            if row["subset"] == "unseen" and row["active_stones"] > 32
        ],
        "train_positions_sha256": base["train_positions_sha256"],
        "validation_positions_sha256": base["validation_positions_sha256"],
    }
    snapshot_dir = out / "source-snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for relative, digest in registration["source_sha256"].items():
        snapshot = snapshot_dir / Path(relative).name
        shutil.copyfile(root / relative, snapshot)
        if seed442.sha(snapshot) != digest:
            raise ValueError(f"source_snapshot_copy_mismatch:{relative}")
    destination.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    return registration


def run(root: Path) -> dict[str, Any]:
    """Execute the two frozen sixteen-proposal arms and publish complete evidence."""
    root = root.resolve()
    seed442.configure_root(root)
    out = root / "docs/data" / OUT_NAME
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for relative, expected in registration["source_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    if (
        seed442.sha(root / "docs/data/seed442-kl-capped-adam-screen/evidence.json")
        != registration["seed442_evidence_sha256"]
    ):
        raise ValueError("seed442_evidence_binding_mismatch")
    for relative, expected in registration["bound_input_sha256"].items():
        if seed442.sha(root / relative) != expected:
            raise ValueError(f"frozen_input_hash_mismatch:{relative}")
    arrays = seed442.load_inputs()
    x, policy, value, replay, coefficients, train_positions, _validation, _source = (
        arrays
    )
    batches = np.asarray(
        registration["seed442_registration"]["first_16_minibatch_positions"],
        dtype=np.int64,
    )
    train_rows = replay[train_positions]
    guard = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json").read_text()
    )
    guard_rows = np.asarray([row["compact_row"] for row in guard], dtype=np.int64)
    membership = seed442.seed435.membership_rows()
    evidence: dict[str, Any] = {
        "schema": "seed447-evidence-v1",
        "registration_sha256": seed442.sha(registration_path),
        "arms": {},
        "metrics": {},
    }
    from ml.alphazero_lite import train

    init_model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(init_model, seed442.seed435.INIT)
    evidence["metrics"]["initializer"] = seed442._endpoint_metrics(
        init_model, arrays, membership
    )
    tensor_archive: dict[str, np.ndarray] = {}
    for arm in ("C", "T"):
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model, seed442.seed435.INIT)
        parameters = seed442._flat_parameters(model)
        optimizer = torch.optim.Adam(
            parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        seed442._CURRENT_PARAMETERS[:] = parameters
        steps = []
        for step in range(16):
            positions = batches[step * 512 : (step + 1) * 512]
            ids = train_rows[positions]
            pre = [p.detach().clone() for p in parameters]
            batch_pre = seed442._pre_policy(model, x, ids)
            guard_pre = seed442._pre_policy(model, x, guard_rows)
            with torch.inference_mode():
                batch_value_pre = _predict_values(model, x, ids)
                guard_value_pre = _predict_values(model, x, guard_rows)
            moment_pre = [
                (
                    optimizer.state[p]
                    .get("exp_avg", torch.zeros_like(p))
                    .detach()
                    .clone(),
                    optimizer.state[p]
                    .get("exp_avg_sq", torch.zeros_like(p))
                    .detach()
                    .clone(),
                )
                for p in parameters
            ]
            optimizer.zero_grad(set_to_none=True)
            bx = torch.from_numpy(x[ids])
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
            logits, predicted = model(bx)
            coeff = torch.from_numpy(coefficients[ids])
            policy_loss = (
                train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[ids])
                )
                * coeff
            ).sum() / coeff.sum()
            value_loss = train.compute_value_loss_vector(
                predicted,
                torch.from_numpy(value[ids]),
                value_loss="huber",
                huber_delta=1.0,
            ).mean()
            loss = policy_loss + 0.3 * value_loss
            loss.backward()
            unclipped = [p.grad.detach().clone() for p in parameters]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            clipped = [p.grad.detach().clone() for p in parameters]
            optimizer.step()
            proposal = [p.detach().clone() for p in parameters]
            proposed_moments = [
                (
                    optimizer.state[p]["exp_avg"].detach().clone(),
                    optimizer.state[p]["exp_avg_sq"].detach().clone(),
                )
                for p in parameters
            ]
            trial_rows = []
            for scale in seed442.SCALES:
                seed442._set_trial(pre, proposal, scale)
                batch_kl = seed442._kl(
                    model, x, ids, np.full(len(ids), 1.0 / len(ids)), batch_pre
                )
                guard_kl = seed442._kl(
                    model,
                    x,
                    guard_rows,
                    np.full(len(guard_rows), 1.0 / len(guard_rows)),
                    guard_pre,
                )
                batch_value = _predict_values(model, x, ids)
                guard_value = _predict_values(model, x, guard_rows)
                batch_move = value_movement(
                    batch_value_pre, batch_value, np.full(len(ids), 1.0 / len(ids))
                )
                guard_move = value_movement(
                    guard_value_pre,
                    guard_value,
                    np.full(len(guard_rows), 1.0 / len(guard_rows)),
                )
                policy_ok = (
                    batch_kl <= seed442.CAP + seed442.KL_TOLERANCE
                    and guard_kl <= seed442.CAP + seed442.KL_TOLERANCE
                )
                joint_ok = (
                    policy_ok
                    and batch_move <= VALUE_CAP + VALUE_TOLERANCE
                    and guard_move <= VALUE_CAP + VALUE_TOLERANCE
                )
                trial_rows.append(
                    {
                        "scale": scale,
                        "batch_kl": batch_kl,
                        "guard_kl": guard_kl,
                        "batch_value_movement": batch_move,
                        "guard_value_movement": guard_move,
                        "policy_feasible": policy_ok,
                        "joint_feasible": joint_ok,
                    }
                )
            policy_scale = next(
                (t["scale"] for t in trial_rows if t["policy_feasible"]), None
            )
            selected = (
                policy_scale
                if arm == "C"
                else choose_largest_feasible(
                    trial_rows, cap=VALUE_CAP, tolerance=VALUE_TOLERANCE
                )
            )
            if arm == "C":
                for trial in trial_rows:
                    trial["accepted"] = trial["policy_feasible"]
            else:
                for trial in trial_rows:
                    trial["accepted"] = trial["joint_feasible"]
            if selected is None:
                with torch.no_grad():
                    for p, old in zip(parameters, pre, strict=True):
                        p.copy_(old)
            else:
                seed442._set_trial(pre, proposal, selected)
            post = [p.detach().clone() for p in parameters]
            step_row = {
                "step": step + 1,
                "optimizer_step": step + 1,
                "expanded_positions": positions.tolist(),
                "compact_rows": ids.tolist(),
                "batch_loss": float(loss.detach()),
                "proposal_norm": seed442.norm(
                    [b - a for a, b in zip(pre, proposal, strict=True)]
                ),
                "pre_hash": seed442.vector_hash(pre),
                "proposal_hash": seed442.vector_hash(proposal),
                "accepted_hash": seed442.vector_hash(post),
                "selected_scale": selected,
                "policy_only_scale": policy_scale,
                "rejected": selected is None,
                "value_cap_activated": arm == "T"
                and cap_activation(policy_scale, selected),
                "moment_pre_hash": seed442.vector_hash(
                    [v for pair in moment_pre for v in pair]
                ),
                "moment_post_hash": seed442.vector_hash(
                    [v for pair in proposed_moments for v in pair]
                ),
                "trials": trial_rows,
            }
            steps.append(step_row)
            for i, (p0, proposal_i, p1, g0, g1, moments0, moments1) in enumerate(
                zip(
                    pre,
                    proposal,
                    post,
                    unclipped,
                    clipped,
                    moment_pre,
                    proposed_moments,
                    strict=True,
                )
            ):
                prefix = f"{arm}_{step:02d}"
                for name, tensor in (
                    ("pre", p0),
                    ("proposal", proposal_i),
                    ("post", p1),
                    ("gradient_unclipped", g0),
                    ("gradient", g1),
                    ("exp_avg_pre", moments0[0]),
                    ("exp_avg_sq_pre", moments0[1]),
                    ("exp_avg", moments1[0]),
                    ("exp_avg_sq", moments1[1]),
                ):
                    tensor_archive[f"{prefix}_{name}_{i:02d}"] = archive_tensor(tensor)
        checkpoint = out / f"{arm}-final.npz"
        np.savez_compressed(checkpoint, **train.checkpoint_from_model(model))
        evidence["arms"][arm] = {
            "steps": steps,
            "final_parameter_sha256": seed442.vector_hash(parameters),
            "checkpoint_sha256": seed442.sha(checkpoint),
        }
        evidence["metrics"][arm] = seed442._endpoint_metrics(model, arrays, membership)
        if arm == "C":
            expected = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(
                expected,
                root / "docs/data/seed442-kl-capped-adam-screen/B-final.npz",
            )
            if any(
                not torch.equal(actual, reference)
                for actual, reference in zip(
                    model.parameters(), expected.parameters(), strict=True
                )
            ):
                raise ValueError("control_endpoint_tensor_mismatch_seed442_B")
            old = json.loads(
                (
                    root / "docs/data/seed442-kl-capped-adam-screen/evidence.json"
                ).read_text()
            )
            old_scales = [step["selected_scale"] for step in old["arms"]["B"]["steps"]]
            if [step["selected_scale"] for step in steps] != old_scales:
                raise ValueError("control_accepted_scales_mismatch_seed442_B")
    # Preserve seed442's baseline metrics as a comparator; C has independently
    # reproduced its B trajectory and is recorded separately for T-C effects.
    base_evidence = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
    )
    evidence["metrics"]["seed442_A"] = base_evidence["metrics"]["A"]
    evidence["metrics"]["seed442_B"] = base_evidence["metrics"]["B"]
    # Decision comparator C is the published seed442 B; keep the three requested
    # decision comparators explicit in the metrics table.
    evidence["metrics"]["T_minus_C"] = _metric_effect(
        evidence["metrics"]["T"], evidence["metrics"]["C"]
    )
    evidence["decision"] = decide(evidence)
    np.savez_compressed(out / "step-tensors.npz", **tensor_archive)
    for arm in ("C", "T"):
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model, out / f"{arm}-final.npz")
        seed442._publish_predictions(out, model, arrays, arm)
    seed442._publish_predictions(out, init_model, arrays, "initializer")
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    ids = np.asarray([int(r["compact_row"]) for r in unseen], dtype=np.int64)
    np.savez_compressed(
        out / "ordered-targets.npz",
        compact_rows=ids.copy(),
        input_identity=np.asarray([r["input_identity"] for r in unseen]),
        policy_targets=policy[ids].copy(),
        value_targets=value[ids].copy(),
    )
    (out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    receipt_files = {
        str(path.relative_to(root)): seed442.sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    receipt = {
        "schema": "seed447-publication-receipt-v1",
        "registration_sha256": seed442.sha(registration_path),
        "evidence_files_sha256": receipt_files,
        "interpretation": "prospective short retrospective-population screen; not evidence of playing strength",
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return evidence


def _predict_values(
    model: torch.nn.Module, x: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(ids), FORWARD_BATCH_SIZE):
            logits, prediction = model(
                torch.from_numpy(x[ids[offset : offset + FORWARD_BATCH_SIZE]])
            )
            del logits
            values.append(prediction.reshape(-1).detach().cpu().numpy().copy())
    return np.concatenate(values).astype(np.float32, copy=False)


def _metric_effect(
    treatment: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "full_training_objective": treatment["full_training_objective"]
        - control["full_training_objective"]
    }
    for weighting in ("exposure_weighted", "equal_input"):
        result[weighting] = {
            key: treatment[weighting][key] - control[weighting][key]
            for key in ("policy_ce", "value_mse")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    result = register(args.root) if args.command == "register" else run(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
