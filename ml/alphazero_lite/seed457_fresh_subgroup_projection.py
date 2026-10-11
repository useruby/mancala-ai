"""Prospective seed457 fresh-training subgroup projection workflow.

Preparation commands are read-only with respect to experiment publication. The
``run`` command is intentionally separate and requires a finalized registration.
No run is performed by this module's preparation or readiness commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import seed453_fresh_policy_projection as seed453
from ml.alphazero_lite import seed447_joint_output_cap as seed447
from ml.alphazero_lite import train
from ml.alphazero_lite.seed457_projection import projected_proposal

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed457-fresh-subgroup-projection")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
INIT_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
SHARD_COUNT = 8


class _ConstraintLimits:
    """Immutable measurements limits; nondefault instances are fixture-only."""

    def __init__(self, kl: float = 0.005, value: float = 0.0001, dot: float = 1e-7):
        self.kl = kl
        self.value = value
        self.dot = dot


def sha(path: Path) -> str:
    """Return a file's SHA256 identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed442_positions(registration: dict[str, Any]) -> list[int]:
    """Validate the exact frozen ordered sixteen-minibatch position vector."""
    positions = registration.get("first_16_minibatch_positions")
    if (
        not isinstance(positions, list)
        or len(positions) != 16 * 512
        or any(type(value) is not int or value < 0 for value in positions)
        or len(set(positions)) != 16 * 512
    ):
        raise ValueError("seed442_minibatch_vector_invalid")
    return positions


def shard_guard(guard: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Deterministically stratify guard identities, preserving within-shard order."""
    if len(guard) != 1399:
        raise ValueError("fresh_guard_count_invalid")
    raw_identities = [row.get("exact_input_identity") for row in guard]
    if any(
        not isinstance(identity, str) or not identity for identity in raw_identities
    ):
        raise ValueError("fresh_guard_identity_missing")
    identities = [str(identity) for identity in raw_identities]
    if len(set(identities)) != len(identities):
        raise ValueError("fresh_guard_identity_duplicate")
    rank = {
        identity: index
        for index, identity in enumerate(
            sorted(
                identities,
                key=lambda identity: (
                    hashlib.sha256(("seed457-shard:" + identity).encode()).hexdigest(),
                    identity,
                ),
            )
        )
    }
    shards: list[list[dict[str, Any]]] = [[] for _ in range(SHARD_COUNT)]
    for row in guard:
        shards[rank[row["exact_input_identity"]] % SHARD_COUNT].append(dict(row))
    counts = [len(shard) for shard in shards]
    if sorted(counts) != [174] + [175] * 7:
        raise ValueError(f"fresh_shard_counts_invalid:{counts}")
    if sum(counts) != 1399 or {
        row["exact_input_identity"] for shard in shards for row in shard
    } != set(identities):
        raise ValueError("fresh_shard_coverage_invalid")
    return shards


def validate_cohorts(
    root: Path,
    arrays: tuple[np.ndarray, ...],
    fresh: list[dict[str, Any]],
    full: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Validate guard identity/provenance/training/exclusion and return shards."""
    x, _policy, _value, replay, _coefficients, train_positions, _validation, sources = (
        arrays
    )
    train_rows = set(map(int, replay[train_positions]))
    if len(full) != 2048 or len({r.get("exact_input_identity") for r in full}) != 2048:
        raise ValueError("full_guard_membership_invalid")
    if len(fresh) != 1399:
        raise ValueError("fresh_guard_membership_invalid")
    membership = seed442.seed435.membership_rows()
    eval_rows = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    exact_eval = {r["input_identity"] for r in eval_rows}
    canonical_eval = {r["canonical_identity"] for r in eval_rows}
    for row in full:
        compact = int(row["compact_row"])
        if compact not in train_rows:
            raise ValueError(f"guard_not_training:{compact}")
        actual = x[compact].astype("<f4", copy=False).tobytes().hex()
        if actual != row["exact_input_identity"]:
            raise ValueError(f"guard_identity_mismatch:{compact}")
        if actual in exact_eval or row["canonical_identity"] in canonical_eval:
            raise ValueError(f"guard_evaluation_overlap:{compact}")
    by_identity = {r["exact_input_identity"]: r for r in full}
    for row in fresh:
        identity = row["exact_input_identity"]
        compact = int(row["compact_row"])
        if identity not in by_identity or by_identity[identity] != row:
            raise ValueError(f"fresh_not_full_guard_member:{compact}")
        if sources[compact].get("source") != "fresh":
            raise ValueError(f"fresh_provenance_invalid:{compact}")
        if compact not in train_rows:
            raise ValueError(f"fresh_guard_not_training:{compact}")
    return shard_guard(fresh)


def readiness(root: Path) -> dict[str, Any]:
    """Check complete frozen bindings and semantic guard membership without training."""
    root = root.resolve()
    missing = [rel for rel in REQUIRED_INPUTS if not (root / rel).is_file()]
    if missing:
        raise ValueError(f"missing_required_inputs:{missing}")
    if sha(root / INIT) != INIT_SHA256:
        raise ValueError("initializer_hash_mismatch")
    registration_path = root / OUT / "registration.json"
    if not registration_path.is_file():
        return {
            "status": "unfrozen_ready_to_prepare",
            "required_inputs": len(REQUIRED_INPUTS),
        }
    registration = json.loads(registration_path.read_text())
    if registration.get("status") != "frozen_before_either_arm":
        raise ValueError("registration_status_invalid")
    for section in ("source_sha256", "input_sha256"):
        bindings = registration.get(section)
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError(f"incomplete_bindings:{section}")
        for relative, digest in bindings.items():
            if not (root / relative).is_file() or sha(root / relative) != digest:
                raise ValueError(f"frozen_binding_mismatch:{relative}")
    if not set(REQUIRED_INPUTS).issubset(registration.get("input_sha256", {})):
        raise ValueError("required_input_binding_inventory_incomplete")
    if not set(REQUIRED_SOURCES).issubset(registration.get("source_sha256", {})):
        raise ValueError("required_binding_inventory_incomplete")
    guard = json.loads((root / OUT / "fresh-guard.json").read_text())
    if guard != registration.get("fresh_guard"):
        raise ValueError("fresh_guard_registration_mismatch")
    shards = shard_guard(guard)
    if registration.get("shards") != shards:
        raise ValueError("shard_partition_mismatch")
    return {
        "status": "ready",
        "fresh_guard_rows": len(guard),
        "shard_counts": [len(x) for x in shards],
    }


REQUIRED_INPUTS = (
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    "docs/data/seed416-policy-target-softening/registration-v3.json",
    "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
    "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
    "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
    "docs/data/seed426-canonical-overlap/sources/fresh.jsonl.gz",
    "docs/data/seed426-canonical-overlap/sources/generic_bootstrap.jsonl.gz",
    "docs/data/seed426-canonical-overlap/sources/opening_disagreement.jsonl.gz",
    "docs/data/seed426-canonical-overlap/sources/random_teacher.jsonl.gz",
    "docs/data/seed426-canonical-overlap/sources/stability.jsonl.gz",
    "docs/data/seed435-adam-direction-screen/A-final.npz",
    "docs/data/seed442-kl-capped-adam-screen/registration.json",
    "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json",
    "docs/data/seed442-kl-capped-adam-screen/evidence.json",
    "docs/data/seed447-joint-output-cap/T-final.npz",
    "docs/data/seed447-joint-output-cap/step-tensors.npz",
    "docs/data/seed453-fresh-policy-projection/registration.json",
    "docs/data/seed453-fresh-policy-projection/evidence.json",
    "docs/data/seed453-fresh-policy-projection/fresh-guard.json",
    "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
    "docs/data/seed454-full-guard-compliance/superseding-v7/audit.json",
    "docs/data/seed455-fresh-projection-attribution/results.json",
    "docs/data/seed456-seed455-verifier-integration/receipt.json",
    "docs/data/seed442-kl-capped-adam-screen/receipt.json",
    "docs/data/seed442-kl-capped-adam-screen/A-final.npz",
    "docs/data/seed442-kl-capped-adam-screen/A-predictions.npz",
    "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
    "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
)

REQUIRED_SOURCES = (
    "ml/alphazero_lite/seed457_projection.py",
    "ml/alphazero_lite/seed457_fresh_subgroup_projection.py",
    "ml/alphazero_lite/verify_seed457_fresh_subgroup_projection.py",
    "ml/alphazero_lite/seed453_fresh_policy_projection.py",
    "ml/alphazero_lite/verify_seed453_fresh_policy_projection.py",
    "ml/alphazero_lite/seed442_kl_capped_adam.py",
    "ml/alphazero_lite/seed435_adam_direction.py",
    "ml/alphazero_lite/seed447_joint_output_cap.py",
    "ml/alphazero_lite/seed429_policy_normalization.py",
    "ml/alphazero_lite/verify_seed434_census_source.py",
    "ml/alphazero_lite/verify_seed450_seed449_correction.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/exact_root_policy_targets.py",
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/seed432_policy_target_census.py",
    "ml/alphazero_lite/seed433_census_correction.py",
)


def prepare(root: Path) -> dict[str, Any]:
    """Construct a prospective draft in memory; never writes a registration."""
    root = root.resolve()
    if any(not (root / item).is_file() for item in REQUIRED_INPUTS):
        missing = [item for item in REQUIRED_INPUTS if not (root / item).is_file()]
        raise ValueError(f"missing_required_inputs:{missing}")
    seed442.configure_root(root)
    seed447.seed442.configure_root(root)
    arrays = seed453.portable_inputs(root)
    fresh = seed453.fresh_guard(root, arrays)
    full = seed442.guard_cohort(root, arrays)
    shards = validate_cohorts(root, arrays, fresh, full)
    sources = REQUIRED_SOURCES
    seed442_registration = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/registration.json").read_text()
    )
    batches = seed442_positions(seed442_registration)
    return {
        "schema": "seed457-registration-v1",
        "status": "draft_not_frozen",
        "source_sha256": {rel: sha(root / rel) for rel in sources},
        "input_sha256": {rel: sha(root / rel) for rel in REQUIRED_INPUTS},
        "initializer_sha256": INIT_SHA256,
        "full_guard_count": len(full),
        "fresh_guard": fresh,
        "shards": shards,
        "seed442_first_16_positions": batches,
        "steps": 16,
        "scales": list(seed453.SCALES),
        "protocol": {
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
        },
        "decision_labels": [
            "subgroup_projection_inactive_no_followup",
            "close_fresh_subgroup_projection_branch",
            "advance_to_separately_preregistered_strength_experiment",
        ],
    }


def shard_gradients(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    shards: list[list[dict[str, Any]]],
) -> list[list[np.ndarray]]:
    """Compute mean legal-action CE gradients using seed453 writeback arithmetic."""
    x, policy = arrays[:2]
    gradients = []
    for shard in shards:
        rows = np.asarray([int(row["compact_row"]) for row in shard], dtype=np.int64)
        gradients.append(seed453.fresh_policy_gradient(model, x, policy, rows))
    return gradients


def arm_proposal(
    arm: str,
    pre: list[np.ndarray],
    raw: list[np.ndarray],
    aggregate_gradient: list[np.ndarray],
    subgroup_gradients: list[list[np.ndarray]],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Construct the registered aggregate-control or nearest subgroup proposal."""
    if arm == "C":
        proposal, report = seed453.project_proposal(pre, raw, aggregate_gradient)
        return proposal, {"method": "aggregate_halfspace", "report": report}
    if arm != "T":
        raise ValueError(f"invalid_arm:{arm}")
    if len(subgroup_gradients) != SHARD_COUNT:
        raise ValueError("subgroup_gradient_count_invalid")
    proposal, report = projected_proposal(pre, raw, subgroup_gradients)
    report["method"] = "nearest_joint_halfspaces"
    report["aggregate_counterfactual"] = seed453.project_proposal(
        pre, raw, aggregate_gradient
    )[0]
    report["aggregate_counterfactual_dots"] = [
        seed453.realized_dot(pre, report["aggregate_counterfactual"], gradient)
        for gradient in subgroup_gradients
    ]
    return proposal, report


def trial_state(
    pre: list[np.ndarray], proposal: list[np.ndarray], scale: float
) -> list[np.ndarray]:
    """Match seed453 float32 interpolation operation order exactly."""
    if scale not in seed453.SCALES:
        raise ValueError("unregistered_trial_scale")
    return [
        b.copy() if scale == 1.0 else np.asarray(a + (b - a) * scale, dtype=np.float32)
        for a, b in zip(pre, proposal, strict=True)
    ]


def choose_scale(trials: list[dict[str, Any]]) -> float | None:
    """Select largest feasible registered scale without a monotonicity assumption."""
    if [row.get("scale") for row in trials] != list(seed453.SCALES):
        raise ValueError("trial_scale_coverage_invalid")
    return next(
        (float(row["scale"]) for row in trials if row.get("feasible") is True), None
    )


def decide(
    metrics: dict[str, Any],
    steps: list[dict[str, Any]],
    activation: dict[str, Any] | None = None,
) -> str:
    """Apply the fixed seed457 T-only gates and activation hierarchy."""
    checks = []
    for weighting in ("exposure_weighted", "equal_input"):
        t, init, ordinary = (
            metrics["T"][weighting],
            metrics["initializer"][weighting],
            metrics["seed442_A"][weighting],
        )
        checks.extend(
            (
                t["policy_ce"] <= ordinary["policy_ce"] - 0.01,
                t["policy_ce"] <= init["policy_ce"] - 0.005,
                t["value_mse"] <= init["value_mse"] + 0.002,
                metrics["T"]["fresh"][weighting]["policy_ce"]
                <= metrics["initializer"]["fresh"][weighting]["policy_ce"] + 2e-6,
            )
        )
    checks.extend(
        (
            metrics["T"]["full_training_objective"]
            < metrics["initializer"]["full_training_objective"],
            not any(s["proposal_norm"] > 0 and s["rejected"] for s in steps),
            (
                bool(activation["activated"])
                if activation is not None
                else any(
                    s.get("additional_subgroup_protection") and not s["rejected"]
                    for s in steps
                )
            ),
        )
    )
    active = (
        bool(activation["activated"])
        if activation is not None
        else any(
            s.get("additional_subgroup_protection") and not s["rejected"] for s in steps
        )
    )
    if not active:
        return "subgroup_projection_inactive_no_followup"
    return (
        "advance_to_separately_preregistered_strength_experiment"
        if all(checks)
        else "close_fresh_subgroup_projection_branch"
    )


def _run_arm_engine(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    batches: np.ndarray,
    full_guard_rows: np.ndarray,
    fresh_shards: list[list[dict[str, Any]]],
    arm: str,
    *,
    proposal_count: int = 16,
    limits: _ConstraintLimits | None = None,
    fresh_guard_rows: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run fixed Adam/projection proposals; proposal_count is internal test injection."""
    if arm not in {"C", "T"} or not 1 <= proposal_count <= 16:
        raise ValueError("invalid_internal_arm_context")
    x, policy, value, replay, coefficients, train_positions, *_ = arrays
    train_rows = replay[train_positions]
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    limits = limits or _ConstraintLimits()
    optimizer = torch.optim.Adam(
        params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    seed442._CURRENT_PARAMETERS[:] = params
    archive: dict[str, np.ndarray] = {}
    records = []
    fresh_rows = (
        np.asarray(fresh_guard_rows, dtype=np.int64)
        if fresh_guard_rows is not None
        else np.asarray(
            [int(row["compact_row"]) for shard in fresh_shards for row in shard],
            dtype=np.int64,
        )
    )
    for step in range(proposal_count):
        ids = train_rows[batches[step * 512 : (step + 1) * 512]]
        pre = [p.detach().clone() for p in params]
        pre_np = [p.detach().cpu().numpy().copy() for p in pre]
        batch_policy = seed442._pre_policy(model, x, ids)
        full_policy = seed442._pre_policy(model, x, full_guard_rows)
        batch_value_pre = seed453._values(model, x, ids)
        full_value_pre = seed453._values(model, x, full_guard_rows)
        optimizer.zero_grad(set_to_none=True)
        logits, predicted = model(torch.from_numpy(x[ids]))
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ids]))
        coeff = torch.from_numpy(coefficients[ids])
        policy_loss = (
            train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[ids])
            )
            * coeff
        ).sum() / coeff.sum()
        value_loss = train.compute_value_loss_vector(
            predicted, torch.from_numpy(value[ids]), value_loss="huber", huber_delta=1.0
        ).mean()
        objective = policy_loss + 0.3 * value_loss
        objective.backward()
        unclipped = [p.grad.detach().clone() for p in params]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        clipped = [p.grad.detach().clone() for p in params]
        aggregate = seed453.fresh_policy_gradient(model, x, policy, fresh_rows)
        subgroup = shard_gradients(model, arrays, fresh_shards) if arm == "T" else []
        if any(
            not torch.equal(old, p.grad) for old, p in zip(clipped, params, strict=True)
        ):
            raise ValueError("guard_gradient_overwrote_objective_gradient")
        moments_pre = [
            (
                optimizer.state[p].get("exp_avg", torch.zeros_like(p)).clone(),
                optimizer.state[p].get("exp_avg_sq", torch.zeros_like(p)).clone(),
            )
            for p in params
        ]
        optimizer.step()
        raw = [p.detach().cpu().numpy().copy() for p in params]
        moments_post = [
            (
                optimizer.state[p]["exp_avg"].clone(),
                optimizer.state[p]["exp_avg_sq"].clone(),
            )
            for p in params
        ]
        proposal, projection = arm_proposal(arm, pre_np, raw, aggregate, subgroup)
        gradients = [aggregate] if arm == "C" else subgroup
        trials = []
        for scale in seed453.SCALES:
            trial = trial_state(pre_np, proposal, scale)
            with torch.no_grad():
                for p, value_array in zip(params, trial, strict=True):
                    p.copy_(torch.from_numpy(value_array.copy()))
            batch_kl = seed442._kl(
                model, x, ids, np.full(len(ids), 1 / len(ids)), batch_policy
            )
            full_kl = seed442._kl(
                model,
                x,
                full_guard_rows,
                np.full(len(full_guard_rows), 1 / len(full_guard_rows)),
                full_policy,
            )
            batch_move = seed447.value_movement(
                batch_value_pre,
                seed453._values(model, x, ids),
                np.full(len(ids), 1 / len(ids)),
            )
            full_move = seed447.value_movement(
                full_value_pre,
                seed453._values(model, x, full_guard_rows),
                np.full(len(full_guard_rows), 1 / len(full_guard_rows)),
            )
            realized = [p.detach().cpu().numpy().copy() for p in params]
            dots = [seed453.realized_dot(pre_np, realized, g) for g in gradients]
            trials.append(
                {
                    "scale": scale,
                    "batch_kl": batch_kl,
                    "guard_kl": full_kl,
                    "batch_value_movement": batch_move,
                    "guard_value_movement": full_move,
                    "gradient_dots": dots,
                    "feasible": batch_kl <= limits.kl + 1e-10
                    and full_kl <= limits.kl + 1e-10
                    and batch_move <= limits.value + 1e-12
                    and full_move <= limits.value + 1e-12
                    and all(dot <= limits.dot for dot in dots),
                }
            )
            for i, tensor in enumerate(realized):
                archive[f"{arm}_{step:02d}_{i:02d}_trial_{scale:g}"] = tensor.copy()
        selected = choose_scale(trials)
        post = (
            [a.copy() for a in pre_np]
            if selected is None
            else trial_state(pre_np, proposal, selected)
        )
        with torch.no_grad():
            for p, tensor in zip(params, post, strict=True):
                p.copy_(torch.from_numpy(tensor.copy()))
        for i, (a, b, c, g, m0, m1) in enumerate(
            zip(raw, proposal, post, clipped, moments_pre, moments_post, strict=True)
        ):
            prefix = f"{arm}_{step:02d}_{i:02d}"
            for name, tensor in (
                ("pre", pre_np[i]),
                ("raw_proposal", a),
                ("projected_proposal", b),
                ("post", c),
                ("objective_gradient_unclipped", unclipped[i].cpu().numpy()),
                ("objective_gradient", g.cpu().numpy()),
                ("fresh_gradient", aggregate[i]),
                ("exp_avg_pre", m0[0].cpu().numpy()),
                ("exp_avg_sq_pre", m0[1].cpu().numpy()),
                ("exp_avg", m1[0].cpu().numpy()),
                ("exp_avg_sq", m1[1].cpu().numpy()),
            ):
                archive[f"{prefix}_{name}"] = np.asarray(tensor).copy()
            for shard_index, gradient in enumerate(subgroup):
                archive[f"{prefix}_subgroup_{shard_index}_gradient"] = gradient[
                    i
                ].copy()
        records.append(
            {
                "step": step + 1,
                "optimizer_step": step + 1,
                "objective_gradient_preserved": True,
                "compact_rows": ids.tolist(),
                "objective": float(objective.detach()),
                "selected_scale": selected,
                "rejected": selected is None,
                "accepted_update_nonzero": any(
                    np.any(before != after)
                    for before, after in zip(pre_np, post, strict=True)
                ),
                "trials": trials,
                "projection": projection,
                "proposal_norm": float(
                    np.sqrt(
                        sum(
                            np.sum((b.astype(np.float64) - a.astype(np.float64)) ** 2)
                            for a, b in zip(pre_np, raw, strict=True)
                        )
                    )
                ),
            }
        )
    return records, archive


def run_arm_engine(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    batches: np.ndarray,
    full_guard_rows: np.ndarray,
    fresh_shards: list[list[dict[str, Any]]],
    arm: str,
    *,
    fresh_guard_rows: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Production engine fixed to the registered sixteen-proposal protocol."""
    return _run_arm_engine(
        model,
        arrays,
        batches,
        full_guard_rows,
        fresh_shards,
        arm,
        proposal_count=16,
        fresh_guard_rows=fresh_guard_rows,
    )


def endpoint_predictions(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    rows: list[dict[str, Any]],
    label: str,
) -> dict[str, np.ndarray]:
    """Capture ordered endpoint outputs and row-level target/identity evidence."""
    x, policy, value = arrays[:3]
    ids = np.asarray([int(row["compact_row"]) for row in rows], dtype=np.int64)
    logits_out, values_out = [], []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(ids), 512):
            part = ids[begin : begin + 512]
            logits, prediction = model(torch.from_numpy(x[part]))
            logits_out.append(logits.cpu().numpy().copy())
            values_out.append(prediction.reshape(-1).cpu().numpy().copy())
    return {
        f"{label}_compact_rows": ids.copy(),
        f"{label}_identities": np.asarray(
            [
                row.get("input_identity", row.get("exact_input_identity"))
                for row in rows
            ],
            dtype="U",
        ),
        f"{label}_targets": policy[ids].copy(),
        f"{label}_value_targets": value[ids].copy(),
        f"{label}_logits": np.concatenate(logits_out, axis=0),
        f"{label}_value_predictions": np.concatenate(values_out, axis=0),
    }


def endpoint_metric_rows(
    outputs: dict[str, np.ndarray], label: str
) -> dict[str, dict[str, float]]:
    """Compute exposure and equal-input CE/MSE, retaining all exposure targets."""
    logits = outputs[f"{label}_logits"]
    targets = outputs[f"{label}_targets"]
    values = outputs[f"{label}_value_predictions"].astype(np.float64)
    value_targets = outputs[f"{label}_value_targets"][:, 0].astype(np.float64)
    identities = outputs[f"{label}_identities"].tolist()
    ce = np.empty(len(logits), dtype=np.float64)
    for start in range(0, len(logits), 512):
        stop = min(start + 512, len(logits))
        part = logits[start:stop]
        # Inputs are carried in the prediction bundle solely for legal masking.
        states = outputs[f"{label}_inputs"][start:stop]
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(states))
        ce[start:stop] = (
            train.compute_policy_cross_entropy(
                torch.from_numpy(part).masked_fill(legal <= 0, -1e9),
                torch.from_numpy(targets[start:stop]),
            )
            .double()
            .numpy()
        )
    mse = np.square(values - value_targets)
    groups: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        groups.setdefault(identity, []).append(index)
    return {
        "exposure_weighted": {
            "policy_ce": float(np.mean(ce)),
            "value_mse": float(np.mean(mse)),
        },
        "equal_input": {
            "policy_ce": float(np.mean([np.mean(ce[ix]) for ix in groups.values()])),
            "value_mse": float(np.mean([np.mean(mse[ix]) for ix in groups.values()])),
        },
    }


def compare_control(
    c_steps: list[dict[str, Any]],
    c_tensors: dict[str, np.ndarray],
    historical_evidence: dict[str, Any],
    historical_tensors: Any,
) -> dict[str, Any]:
    """Compare C scales and every post-step array with seed453 B evidence."""
    mismatches = []
    old_steps = historical_evidence["arms"]["B"]["steps"]
    for step, (current, old) in enumerate(zip(c_steps, old_steps, strict=True)):
        if current["selected_scale"] != old["selected_scale"]:
            mismatches.append(
                {
                    "step": step + 1,
                    "field": "selected_scale",
                    "expected": old["selected_scale"],
                    "actual": current["selected_scale"],
                }
            )
        for index in range(22):
            key = f"C_{step:02d}_{index:02d}_post"
            expected_key = f"B_{step:02d}_{index:02d}_post"
            if not np.array_equal(c_tensors[key], historical_tensors[expected_key]):
                mismatches.append(
                    {"step": step + 1, "tensor": index, "field": "post_parameters"}
                )
    return {"matches": not mismatches, "mismatches": mismatches}


def subgroup_activation(
    t_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Find accepted update where aggregate counterfactual harms a shard."""
    activated_steps = []
    for record in t_steps:
        if record["rejected"] or record.get("selected_scale") is None:
            continue
        counter = record["projection"].get("aggregate_counterfactual")
        dots = record["projection"].get("aggregate_counterfactual_dots", [])
        accepted_dots = record["trials"][
            next(
                i
                for i, row in enumerate(record["trials"])
                if row["scale"] == record["selected_scale"]
            )
        ]["gradient_dots"]
        if not record.get("accepted_update_nonzero", False):
            continue
        if (
            counter is not None
            and record.get("proposal_norm", 0) > 0
            and any(dot > 1e-7 for dot in dots)
            and all(dot <= 1e-7 for dot in accepted_dots)
        ):
            activated_steps.append(
                {
                    "step": record["step"],
                    "counterfactual_dots": dots,
                    "accepted_dots": accepted_dots,
                }
            )
    return {
        "schema": "seed457-activation-v1",
        "activated": bool(activated_steps),
        "steps": activated_steps,
    }


def _jsonable(value: Any) -> Any:
    """Convert nested ndarray/scalar evidence to deterministic JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_synthetic_publication(
    root: Path,
    models: dict[str, torch.nn.Module],
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
    arm_steps: dict[str, list[dict[str, Any]]],
    tensors: dict[str, dict[str, np.ndarray]],
    *,
    seed442_a_metrics: dict[str, Any],
    initializer_training_objective: float,
    final_training_objectives: dict[str, float],
    control: dict[str, Any],
    identities: dict[str, dict[str, str]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a clearly synthetic fixture publication using the production metric paths."""
    if set(identities) != {"sources", "inputs"} or any(
        len(digest) != 64
        for section in identities.values()
        for digest in section.values()
    ):
        raise ValueError("synthetic_source_input_identities_incomplete")
    out = output_dir or root / ".tmp/seed457-synthetic-test-fixture"
    out.mkdir(parents=True, exist_ok=True)
    pooled_rows = [
        row
        for row in membership
        if row.get("subset") == "unseen" and row.get("active_stones", 0) > 32
    ]
    fresh_rows = [row for row in pooled_rows if row.get("source") == "fresh"]
    predictions = {}
    metrics: dict[str, Any] = {"seed442_A": seed442_a_metrics}
    for name, model in (
        ("initializer", models["initializer"]),
        ("C", models["C"]),
        ("T", models["T"]),
    ):
        pooled_label, fresh_label = f"{name}_pooled", f"{name}_fresh"
        pooled = endpoint_predictions(model, arrays, pooled_rows, pooled_label)
        pooled[f"{pooled_label}_inputs"] = arrays[0][
            pooled[f"{pooled_label}_compact_rows"]
        ].copy()
        fresh = endpoint_predictions(model, arrays, fresh_rows, fresh_label)
        fresh[f"{fresh_label}_inputs"] = arrays[0][
            fresh[f"{fresh_label}_compact_rows"]
        ].copy()
        predictions.update(pooled)
        predictions.update(fresh)
        metrics[name] = endpoint_metric_rows(predictions, pooled_label)
        metrics[name]["fresh"] = endpoint_metric_rows(predictions, fresh_label)
        metrics[name]["full_training_objective"] = final_training_objectives[name]
    effects = {}
    for weight in ("exposure_weighted", "equal_input"):
        effects[weight] = {
            metric: metrics["T"][weight][metric] - metrics["C"][weight][metric]
            for metric in ("policy_ce", "value_mse")
        }
        effects[f"fresh_{weight}"] = {
            metric: metrics["T"]["fresh"][weight][metric]
            - metrics["C"]["fresh"][weight][metric]
            for metric in ("policy_ce", "value_mse")
        }
    effects["full_training_objective"] = {
        "T_minus_C": metrics["T"]["full_training_objective"]
        - metrics["C"]["full_training_objective"]
    }
    activation = subgroup_activation(arm_steps["T"])
    gates = decision_gates(
        metrics, initializer_training_objective, activation, arm_steps["T"]
    )
    evidence = {
        "schema": "seed457-evidence-v1",
        "status": "synthetic-test-fixture",
        "metrics": metrics,
        "effects_T_minus_C": effects,
        "control_reproduction": control,
        "activation": activation,
        "gates": gates,
        "decision": {
            "classification": decide(metrics, arm_steps["T"], activation)
            if control["matches"]
            else "control_reproduction_mismatch"
        },
        "arms": {arm: {"steps": _jsonable(arm_steps[arm])} for arm in ("C", "T")},
        "identities": identities,
    }
    np.savez_compressed(out / "endpoint-predictions.npz", **predictions)
    for arm in ("C", "T"):
        np.savez_compressed(out / f"step-tensors-{arm}.npz", **tensors[arm])
    (out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    inventory = {
        p.name: sha(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "receipt.json"
    }
    receipt = {
        "schema": "seed457-synthetic-receipt-v1",
        "status": "synthetic-test-fixture",
        "source_sha256": identities["sources"],
        "input_sha256": identities["inputs"],
        "files_sha256": inventory,
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return evidence


def verify_synthetic_receipt(out: Path) -> None:
    """Verify all synthetic fixture file hashes without trusting the receipt."""
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("status") != "synthetic-test-fixture":
        raise ValueError("synthetic_receipt_status_invalid")
    for name, expected in receipt["files_sha256"].items():
        path = out / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"synthetic_receipt_hash_mismatch:{name}")


def decision_gates(
    metrics: dict[str, Any],
    initializer_objective: float,
    activation: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, bool]:
    """Return individually auditable fixed advancement gates."""
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
        row["proposal_norm"] > 0 and row["rejected"] for row in steps
    )
    gates["additional_subgroup_protection"] = bool(activation["activated"])
    return gates


def run(root: Path) -> dict[str, Any]:
    """Execute registered C/T arms and publish trajectory evidence."""
    root = root.resolve()
    status = readiness(root)
    if status["status"] != "ready":
        raise ValueError("registration_not_frozen_ready")
    registration = json.loads((root / OUT / "registration.json").read_text())
    seed442.configure_root(root)
    seed447.seed442.configure_root(root)
    arrays = seed453.portable_inputs(root)
    x, _policy, _value, replay, _coeff, train_positions, *_ = arrays
    full_guard = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json").read_text()
    )
    full_rows = np.asarray(
        [int(row["compact_row"]) for row in full_guard], dtype=np.int64
    )
    shards = registration["shards"]
    fresh_rows = np.asarray(
        [int(row["compact_row"]) for row in registration["fresh_guard"]],
        dtype=np.int64,
    )
    batches = np.asarray(registration["seed442_first_16_positions"], dtype=np.int64)
    results, tensor_archives = {}, {}
    endpoint_models = {}
    for arm in ("C", "T"):
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model, root / INIT)
        records, tensors = run_arm_engine(
            model,
            arrays,
            batches,
            full_rows,
            shards,
            arm,
            fresh_guard_rows=fresh_rows,
        )
        results[arm] = records
        tensor_archives[arm] = tensors
        endpoint_models[arm] = model
        np.savez_compressed(root / OUT / f"step-tensors-{arm}.npz", **tensors)
    # Exact control comparison uses parameter arrays, never checkpoint bytes.
    historical_evidence = json.loads(
        (root / "docs/data/seed453-fresh-policy-projection/evidence.json").read_text()
    )
    with np.load(
        root / "docs/data/seed453-fresh-policy-projection/step-tensors-B.npz",
        allow_pickle=False,
    ) as historical_tensors:
        control = compare_control(
            results["C"], tensor_archives["C"], historical_evidence, historical_tensors
        )
    initializer_model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(initializer_model, root / INIT)
    membership = seed442.seed435.membership_rows()
    metrics = {
        "seed442_A": json.loads(
            (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
        )["metrics"]["A"]
    }
    predictions = {}
    for label, model in (
        ("initializer", initializer_model),
        ("C", endpoint_models["C"]),
        ("T", endpoint_models["T"]),
    ):
        pooled = [
            row
            for row in membership
            if row["subset"] == "unseen" and row["active_stones"] > 32
        ]
        fresh = [row for row in pooled if row.get("source") == "fresh"]
        for scope, cohort in (("pooled", pooled), ("fresh", fresh)):
            name = f"{label}_{scope}"
            result = endpoint_predictions(model, arrays, cohort, name)
            result[f"{name}_inputs"] = x[result[f"{name}_compact_rows"]].copy()
            predictions.update(result)
            scored = endpoint_metric_rows(predictions, name)
            if scope == "pooled":
                metrics.setdefault(label, {}).update(scored)
            else:
                metrics[label]["fresh"] = scored
        metrics[label]["full_training_objective"] = seed453._endpoint_metrics(
            model, arrays, membership
        )["full_training_objective"]
    effects = {}
    for weighting in ("exposure_weighted", "equal_input"):
        effects[weighting] = {
            metric: metrics["T"][weighting][metric] - metrics["C"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
        effects[f"fresh_{weighting}"] = {
            metric: metrics["T"]["fresh"][weighting][metric]
            - metrics["C"]["fresh"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
    effects["full_training_objective"] = {
        "T_minus_C": metrics["T"]["full_training_objective"]
        - metrics["C"]["full_training_objective"]
    }
    activation = subgroup_activation(results["T"])
    gates = decision_gates(
        metrics,
        metrics["initializer"]["full_training_objective"],
        activation,
        results["T"],
    )
    evidence = {
        "schema": "seed457-evidence-v1",
        "status": "real_execution",
        "registration_sha256": sha(root / OUT / "registration.json"),
        "metrics": metrics,
        "effects_T_minus_C": effects,
        "control_reproduction": control,
        "activation": activation,
        "gates": gates,
        "decision": {
            "verified": bool(control["matches"]),
            "classification": decide(metrics, results["T"], activation)
            if control["matches"]
            else None,
        },
        "arms": {arm: {"steps": _jsonable(results[arm])} for arm in ("C", "T")},
    }
    np.savez_compressed(root / OUT / "endpoint-predictions.npz", **predictions)
    (root / OUT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    (root / OUT / "results.md").write_text(
        "# Seed457 — fresh-training subgroup projection\n\n"
        "Prospectively registered synthetic/real execution record. Validation was not used for training constraints or scale selection.\n\n"
        f"**Classification:** `{evidence['decision']['classification']}`\n\n"
        "See evidence.json, endpoint-predictions.npz, and per-arm step tensor archives for numerical details.\n"
    )
    receipt_files = {
        str(path.relative_to(root)): sha(path)
        for path in sorted((root / OUT).rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    receipt = {
        "schema": "seed457-receipt-v1",
        "input_sha256": registration["input_sha256"],
        "source_sha256": registration["source_sha256"],
        "files_sha256": receipt_files,
    }
    (root / OUT / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return {
        "status": "arms_executed",
        "steps_per_arm": 16,
        "decision": evidence["decision"],
        "control_reproduction": control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "readiness", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "readiness":
        result = readiness(args.root)
    else:
        result = run(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
