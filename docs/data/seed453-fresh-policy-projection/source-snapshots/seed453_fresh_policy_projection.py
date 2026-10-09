"""Numerical primitives and fixed decision rule for seed453.

The experiment is prospective and uses only the fresh-source training guard.
Historical seed447/452 artifacts are inputs and are never rewritten.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import seed447_joint_output_cap as seed447
from ml.alphazero_lite import train

DOT_ALLOWANCE = 1e-7
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
OUT = "docs/data/seed453-fresh-policy-projection"
VALUE_CAP = 0.0001
VALUE_TOLERANCE = 1e-12
CONTROL_T_FINAL_SHA256 = (
    "bcf3e76b12896cd4017aac5a8eda0b938088c9bee216b59d07cfde05c095f026"
)
INIT_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"


def sha(path: Any) -> str:
    """Hash a file using SHA256."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Copy a tensor into independent host storage for immutable evidence."""
    return tensor.detach().cpu().numpy().copy()


def portable_inputs(root: Any) -> tuple[np.ndarray, ...]:
    """Rebuild seed416 Lane-A JSONL under checkout scratch; ignore absolute paths."""
    import json
    import tempfile
    from pathlib import Path

    root = Path(root).resolve()
    reg_path = root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    registration = json.loads(reg_path.read_text())
    scratch = root / ".tmp"
    scratch.mkdir(exist_ok=True)
    from ml.alphazero_lite.verify_seed450_seed449_correction import (
        _reconstruct_derivatives,
    )

    with tempfile.TemporaryDirectory(prefix="seed453-lane-a-", dir=scratch) as temp:
        derivatives = _reconstruct_derivatives(root, registration, Path(temp))
        original = train.load_jsonl_replay

        def rooted_loader(paths: Any, weights: Any = None, **kwargs: Any) -> Any:
            mapped = [derivatives[row["name"]] for row in registration["replays"]]
            return original(mapped, weights, **kwargs)

        train.load_jsonl_replay = rooted_loader
        try:
            return seed442.load_inputs()
        finally:
            train.load_jsonl_replay = original


def fresh_guard(root: Any, arrays: tuple[np.ndarray, ...]) -> list[dict[str, Any]]:
    """Filter seed442's frozen guard to unique fresh-source training inputs."""
    x, _p, _v, replay, _coeff, train_positions, _val, reconstructed = arrays
    train_rows = set(map(int, replay[train_positions]))
    original = seed442.guard_cohort(root, arrays)
    membership = seed442.seed435.membership_rows()
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    unseen_input = {r["input_identity"] for r in unseen}
    unseen_canonical = {r["canonical_identity"] for r in unseen}
    result = []
    seen = set()
    for row in original:
        compact = int(row["compact_row"])
        source = reconstructed[compact]
        identity = x[compact].astype("<f4", copy=False).tobytes().hex()
        if source.get("source") != "fresh":
            continue
        if compact not in train_rows:
            raise ValueError(f"fresh_guard_not_training:{compact}")
        if identity in seen or identity != row["exact_input_identity"]:
            raise ValueError(f"fresh_guard_identity_invalid:{compact}")
        if identity in unseen_input or row["canonical_identity"] in unseen_canonical:
            raise ValueError(f"fresh_guard_evaluation_overlap:{compact}")
        seen.add(identity)
        result.append(dict(row))
    if not result:
        raise ValueError("fresh_guard_empty")
    return result


def register(root: Any) -> dict[str, Any]:
    """Prospectively bind all inputs, code, frozen guard, and decision rules."""
    import json
    import shutil
    from pathlib import Path

    root = Path(root).resolve()
    seed442.configure_root(root)
    seed447.seed442.configure_root(root)
    arrays = portable_inputs(root)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    reg = out / "registration.json"
    if reg.exists():
        raise ValueError("registration_is_immutable")
    guard = fresh_guard(root, arrays)
    guard_path = out / "fresh-guard.json"
    guard_path.write_text(json.dumps(guard, indent=2, sort_keys=True) + "\n")
    base_path = root / "docs/data/seed442-kl-capped-adam-screen/registration.json"
    base = json.loads(base_path.read_text())
    sources = [
        "ml/alphazero_lite/seed453_fresh_policy_projection.py",
        "ml/alphazero_lite/test_seed453_fresh_policy_projection.py",
        "ml/alphazero_lite/verify_seed453_fresh_policy_projection.py",
        "ml/alphazero_lite/seed442_kl_capped_adam.py",
        "ml/alphazero_lite/seed435_adam_direction.py",
        "ml/alphazero_lite/seed447_joint_output_cap.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
    ]
    bound = dict(base["frozen_input_sha256"])
    bound["docs/data/seed442-kl-capped-adam-screen/registration.json"] = sha(base_path)
    bound["docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"] = sha(
        root / "docs/data/seed442-kl-capped-adam-screen/guard-cohort.json"
    )
    bound["docs/data/seed442-kl-capped-adam-screen/evidence.json"] = sha(
        root / "docs/data/seed442-kl-capped-adam-screen/evidence.json"
    )
    bound["docs/data/seed442-kl-capped-adam-screen/B-final.npz"] = sha(
        root / "docs/data/seed442-kl-capped-adam-screen/B-final.npz"
    )
    for relative in (
        "docs/data/seed447-joint-output-cap/registration.json",
        "docs/data/seed447-joint-output-cap/evidence.json",
        "docs/data/seed447-joint-output-cap/T-final.npz",
        "docs/data/seed447-joint-output-cap/step-tensors.npz",
        "docs/data/seed451-source-policy-alignment/results.json",
        "docs/data/seed452-fresh-policy-step-attribution/results.json",
    ):
        bound[relative] = sha(root / relative)
    unseen = [
        {
            key: row[key]
            for key in (
                "compact_row",
                "input_identity",
                "canonical_identity",
                "active_stones",
            )
        }
        for row in seed442.seed435.membership_rows()
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    protocol = {
        "schema": "seed453-registration-v1",
        "status": "frozen_before_either_arm",
        "seed442_registration_sha256": sha(base_path),
        "seed442_evidence_sha256": bound[
            "docs/data/seed442-kl-capped-adam-screen/evidence.json"
        ],
        "input_sha256": bound,
        "source_sha256": {rel: sha(root / rel) for rel in sources},
        "initializer_sha256": INIT_SHA256,
        "seed447_T_checkpoint_sha256": CONTROL_T_FINAL_SHA256,
        "seed447_registration_sha256": sha(
            root / "docs/data/seed447-joint-output-cap/registration.json"
        ),
        "guard": guard,
        "guard_count": len(guard),
        "guard_sha256": sha(guard_path),
        "unseen_population": unseen,
        "seed442_first_16_positions": base["first_16_minibatch_positions"],
        "protocol": {
            "arms": {
                "C": "seed447 T exact joint cap reproduction",
                "B": "C plus fresh-training policy direction projection",
            },
            "steps": 16,
            "batch_size": 512,
            "optimizer": "Adam(lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0)",
            "clip_norm": 1.0,
            "objective": "seed447 weighted legal-action policy CE + .3 mean Huber(delta=1)",
            "projection": "float64 proposal-pre displacement; remove max(0,dot(gF,d))/||gF||^2 times gF; float32(pre64+d_projected); no moment projection/reset",
            "guard_weighting": "uniform unique exact encoded input; frozen seed442 order filtered by reconstructed source=fresh",
            "scales": list(SCALES),
            "policy_kl_cap": 0.005,
            "policy_kl_tolerance": 1e-10,
            "value_movement_cap": VALUE_CAP,
            "value_movement_tolerance": VALUE_TOLERANCE,
            "realized_projection_dot_allowance": DOT_ALLOWANCE,
            "trials": "all 8 descending scales; largest feasible, no monotonicity assumption; float32 seed447 arithmetic",
            "rejection": "retain pre-step parameters but advance moments once; continue; any rejected nonzero proposal disqualifies advancement",
        },
        "decision": {
            "B_minus_seed442_A_policy_ce": -0.01,
            "B_minus_initializer_policy_ce": -0.005,
            "B_value_mse_margin": 0.002,
            "B_fresh_ce_margin": 2e-6,
            "full_training_objective_decreases": True,
            "require_projection_activation_on_accepted_nonzero_step": True,
            "reject_nonzero_proposals": False,
            "inactive": "fresh_projection_inactive_no_followup",
            "active_failure": "close_fresh_policy_projection_branch",
            "active_success": "advance_to_separately_preregistered_strength_experiment",
        },
    }
    for rel in sources:
        target = out / "source-snapshots" / Path(rel).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
        if sha(target) != protocol["source_sha256"][rel]:
            raise ValueError(f"snapshot_copy_mismatch:{rel}")
    reg.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def fresh_policy_gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    policy: np.ndarray,
    rows: np.ndarray,
) -> list[np.ndarray]:
    """Compute mean legal-action fresh CE gradients without modifying .grad."""
    params = tuple(p for p in model.parameters() if p.requires_grad)
    accumulated = [np.zeros(tuple(p.shape), dtype=np.float64) for p in params]
    for start in range(0, len(rows), 512):
        selected = rows[start : start + 512]
        logits, _ = model(torch.from_numpy(x[selected]))
        legal = torch.from_numpy(
            train.legal_mask_matrix_for_encoded_states(x[selected])
        )
        losses = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[selected])
        )
        grads = torch.autograd.grad(losses.sum() / len(rows), params, allow_unused=True)
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is not None:
                accumulated[index] += grad.detach().cpu().numpy().astype(np.float64)
    return [value.astype(np.float32) for value in accumulated]


def _values(model: torch.nn.Module, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return seed447._predict_values(model, x, rows)


def run(root: Any) -> dict[str, Any]:
    """Execute both frozen sixteen-proposal arms and publish evidence."""
    import json
    from pathlib import Path

    root = Path(root).resolve()
    seed442.configure_root(root)
    seed447.seed442.configure_root(root)
    out = root / OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    if registration.get("status") != "frozen_before_either_arm":
        raise ValueError("registration_not_frozen")
    for rel, digest in registration["source_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"execution_source_hash_mismatch:{rel}")
    for rel, digest in registration["input_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"frozen_input_hash_mismatch:{rel}")
    if (
        sha(
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        != INIT_SHA256
    ):
        raise ValueError("initializer_identity_mismatch")
    arrays = portable_inputs(root)
    x, policy, value, replay, coefficients, train_positions, _val_positions, _source = (
        arrays
    )
    batches = np.asarray(registration["seed442_first_16_positions"], dtype=np.int64)
    train_rows = replay[train_positions]
    guard = json.loads((out / "fresh-guard.json").read_text())
    if guard != registration["guard"]:
        raise ValueError("guard_changed")
    guard_rows = np.asarray([r["compact_row"] for r in guard], dtype=np.int64)
    membership = seed442.seed435.membership_rows()
    init_model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(init_model, seed442.seed435.INIT)
    evidence: dict[str, Any] = {
        "schema": "seed453-evidence-v1",
        "registration_sha256": sha(registration_path),
        "arms": {},
        "metrics": {"initializer": _endpoint_metrics(init_model, arrays, membership)},
    }
    tensors: dict[str, np.ndarray] = {}
    for arm in ("C", "B"):
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(model, seed442.seed435.INIT)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        seed442._CURRENT_PARAMETERS[:] = params
        steps = []
        for step in range(16):
            positions = batches[step * 512 : (step + 1) * 512]
            ids = train_rows[positions]
            pre = [p.detach().clone() for p in params]
            batch_policy = seed442._pre_policy(model, x, ids)
            guard_policy = seed442._pre_policy(model, x, guard_rows)
            batch_value_pre = _values(model, x, ids)
            guard_value_pre = _values(model, x, guard_rows)
            moments_pre = [
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
                for p in params
            ]
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
                predicted,
                torch.from_numpy(value[ids]),
                value_loss="huber",
                huber_delta=1.0,
            ).mean()
            objective = policy_loss + 0.3 * value_loss
            objective.backward()
            objective_before_guard = [p.grad.detach().clone() for p in params]
            guard_gradient = (
                fresh_policy_gradient(model, x, policy, guard_rows)
                if arm == "B"
                else [np.zeros_like(p.detach().cpu().numpy()) for p in params]
            )
            if any(
                not torch.equal(before, parameter.grad)
                for before, parameter in zip(
                    objective_before_guard, params, strict=True
                )
            ):
                raise ValueError("guard_gradient_overwrote_objective_gradient")
            objective_unclipped = [p.grad.detach().clone() for p in params]
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            objective_clipped = [p.grad.detach().clone() for p in params]
            # Confirm diagnostic autograd did not replace or modify Adam's objective gradient.
            if any(
                not torch.equal(a, p.grad)
                for a, p in zip(objective_clipped, params, strict=True)
            ):
                raise ValueError("guard_gradient_overwrote_objective_gradient")
            optimizer.step()
            raw = [p.detach().clone() for p in params]
            moments_post = [
                (
                    optimizer.state[p]["exp_avg"].detach().clone(),
                    optimizer.state[p]["exp_avg_sq"].detach().clone(),
                )
                for p in params
            ]
            raw_np = [p.detach().cpu().numpy().copy() for p in raw]
            pre_np = [p.detach().cpu().numpy().copy() for p in pre]
            guard_np = [g.copy() for g in guard_gradient]
            projected_np, projection_report = (
                project_proposal(pre_np, raw_np, guard_np)
                if arm == "B"
                else (
                    raw_np,
                    {
                        "raw_dot": 0.0,
                        "gradient_norm": 0.0,
                        "projected_dot": 0.0,
                        "coefficient": 0.0,
                        "projected_proposal_dot": 0.0,
                    },
                )
            )
            proposal = [torch.from_numpy(a.copy()) for a in projected_np]
            projection_changed = any(
                not np.array_equal(a, b)
                for a, b in zip(projected_np, raw_np, strict=True)
            )
            trials = []
            for scale in SCALES:
                seed442._set_trial(pre, proposal, scale)
                batch_kl = seed442._kl(
                    model, x, ids, np.full(len(ids), 1 / len(ids)), batch_policy
                )
                guard_kl = seed442._kl(
                    model,
                    x,
                    guard_rows,
                    np.full(len(guard_rows), 1 / len(guard_rows)),
                    guard_policy,
                )
                batch_values = _values(model, x, ids)
                guard_values = _values(model, x, guard_rows)
                batch_move = seed447.value_movement(
                    batch_value_pre, batch_values, np.full(len(ids), 1 / len(ids))
                )
                guard_move = seed447.value_movement(
                    guard_value_pre,
                    guard_values,
                    np.full(len(guard_rows), 1 / len(guard_rows)),
                )
                post_trial = [p.detach().cpu().numpy().copy() for p in params]
                for tensor_index, trial_tensor in enumerate(post_trial):
                    tensors[f"{arm}_{step:02d}_{tensor_index:02d}_trial_{scale:g}"] = (
                        trial_tensor.copy()
                    )
                dot = realized_dot(pre_np, post_trial, guard_np) if arm == "B" else 0.0
                feasible = (
                    batch_kl <= seed442.CAP + seed442.KL_TOLERANCE
                    and guard_kl <= seed442.CAP + seed442.KL_TOLERANCE
                    and batch_move <= VALUE_CAP + VALUE_TOLERANCE
                    and guard_move <= VALUE_CAP + VALUE_TOLERANCE
                    and (arm == "C" or dot <= DOT_ALLOWANCE)
                )
                trials.append(
                    {
                        "scale": scale,
                        "batch_kl": batch_kl,
                        "guard_kl": guard_kl,
                        "batch_value_movement": batch_move,
                        "guard_value_movement": guard_move,
                        "realized_fresh_dot": dot,
                        "feasible": feasible,
                    }
                )
            selected = seed447.choose_largest_feasible(
                trials, cap=VALUE_CAP, tolerance=VALUE_TOLERANCE
            )
            if selected is None:
                with torch.no_grad():
                    for p, old in zip(params, pre, strict=True):
                        p.copy_(old)
            else:
                seed442._set_trial(pre, proposal, selected)
            post = [p.detach().clone() for p in params]
            activation = (
                arm == "B"
                and projection_report["raw_dot"] > 0
                and projection_changed
                and selected is not None
                and any(
                    not np.array_equal(a, b.detach().cpu().numpy())
                    for a, b in zip(raw_np, post, strict=True)
                )
            )
            record = {
                "step": step + 1,
                "optimizer_step": step + 1,
                "expanded_positions": positions.tolist(),
                "compact_rows": ids.tolist(),
                "objective": float(objective.detach()),
                "selected_scale": selected,
                "rejected": selected is None,
                "proposal_norm": float(
                    np.sqrt(
                        sum(
                            np.sum((b.astype(np.float64) - a.astype(np.float64)) ** 2)
                            for a, b in zip(pre_np, raw_np, strict=True)
                        )
                    )
                ),
                "activation": bool(activation),
                "projection_changed": bool(projection_changed),
                "projection": projection_report,
                "trials": trials,
            }
            steps.append(record)
            for index, (p0, raw_i, proj_i, p1, g0, g1, gf, m0, m1) in enumerate(
                zip(
                    pre,
                    raw,
                    proposal,
                    post,
                    objective_unclipped,
                    objective_clipped,
                    guard_gradient,
                    moments_pre,
                    moments_post,
                    strict=True,
                )
            ):
                prefix = f"{arm}_{step:02d}_{index:02d}"
                for key, tensor in (
                    ("pre", p0),
                    ("raw_proposal", raw_i),
                    ("projected_proposal", proj_i),
                    ("post", p1),
                    ("objective_gradient_unclipped", g0),
                    ("objective_gradient", g1),
                    ("fresh_gradient", torch.from_numpy(gf)),
                    ("exp_avg_pre", m0[0]),
                    ("exp_avg_sq_pre", m0[1]),
                    ("exp_avg", m1[0]),
                    ("exp_avg_sq", m1[1]),
                ):
                    tensors[f"{prefix}_{key}"] = tensor.detach().cpu().numpy().copy()
            record["pre_hash"] = seed442.vector_hash(pre)
            record["raw_proposal_hash"] = seed442.vector_hash(raw)
            record["projected_proposal_hash"] = seed442.vector_hash(proposal)
            record["post_hash"] = seed442.vector_hash(post)
            record["moment_pre_hash"] = seed442.vector_hash(
                [x for pair in moments_pre for x in pair]
            )
            record["moment_post_hash"] = seed442.vector_hash(
                [x for pair in moments_post for x in pair]
            )
        checkpoint = out / f"{arm}-final.npz"
        np.savez_compressed(checkpoint, **train.checkpoint_from_model(model))
        evidence["arms"][arm] = {
            "steps": steps,
            "final_parameter_sha256": seed442.vector_hash(params),
            "checkpoint_sha256": sha(checkpoint),
        }
        evidence["metrics"][arm] = _endpoint_metrics(model, arrays, membership)
        if arm == "C":
            control = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(
                control, root / "docs/data/seed447-joint-output-cap/T-final.npz"
            )
            if sha(
                root / "docs/data/seed447-joint-output-cap/T-final.npz"
            ) != CONTROL_T_FINAL_SHA256 or any(
                not torch.equal(a, b)
                for a, b in zip(model.parameters(), control.parameters(), strict=True)
            ):
                raise ValueError("control_final_tensor_mismatch_seed447_T")
            old = json.loads(
                (root / "docs/data/seed447-joint-output-cap/evidence.json").read_text()
            )
            if [s["selected_scale"] for s in steps] != [
                s["selected_scale"] for s in old["arms"]["T"]["steps"]
            ]:
                raise ValueError("control_scales_mismatch_seed447_T")
            with np.load(
                root / "docs/data/seed447-joint-output-cap/step-tensors.npz",
                allow_pickle=False,
            ) as archive:
                for step in range(16):
                    if any(
                        not np.array_equal(
                            archive[f"T_{step:02d}_post_{i:02d}"],
                            tensors[f"C_{step:02d}_{i:02d}_post"],
                        )
                        for i in range(len(params))
                    ):
                        raise ValueError(
                            f"control_post_tensor_mismatch_seed447_T:{step + 1}"
                        )
    evidence["metrics"]["seed442_A"] = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
    )["metrics"]["A"]
    evidence["metrics"]["B_minus_C"] = seed447._metric_effect(
        evidence["metrics"]["B"], evidence["metrics"]["C"]
    )
    evidence["decision"] = {
        "classification": decide(evidence["metrics"], evidence["arms"]["B"]["steps"])
    }
    np.savez_compressed(out / "step-tensors.npz", **tensors)
    for arm in ("B", "C"):
        endpoint = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(endpoint, out / f"{arm}-final.npz")
        seed442._publish_predictions(out, endpoint, arrays, arm)
    seed442._publish_predictions(out, init_model, arrays, "initializer")
    (out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text(
        "# Seed453 — fresh-policy direction projection\n\n"
        "Prospective sixteen-update screen. No validation data entered projection or acceptance.\n\n"
        f"**Fixed classification:** `{evidence['decision']['classification']}`\n\n"
        f"**Fresh guard:** {registration['guard_count']} unique training inputs.\n\n"
        "The independent verifier must pass before interpreting this screen. Historical seed447 and seed448 decisions remain unchanged.\n"
    )
    receipt_files = {
        str(path.relative_to(root)): sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "seed453-receipt-v1",
                "registration_sha256": sha(registration_path),
                "files_sha256": receipt_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return evidence


def _endpoint_metrics(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate pooled and fresh unseen metrics with both registered weightings."""
    x, targets, target_values, *_ = arrays
    selected = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    policy_rows = np.empty(len(x), dtype=np.float64)
    value_rows = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            stop = min(start + 512, len(x))
            states = x[start:stop]
            logits, predicted = model(torch.from_numpy(states))
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(states))
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9),
                torch.from_numpy(targets[start:stop]),
            )
            policy_rows[start:stop] = losses.double().cpu().numpy()
            pred = predicted.reshape(-1).double().cpu().numpy()
            value_rows[start:stop] = np.square(
                pred - target_values[start:stop, 0].astype(np.float64)
            )

    def cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_identity: dict[str, list[int]] = {}
        ids = []
        for row in rows:
            compact = int(row["compact_row"])
            ids.append(compact)
            by_identity.setdefault(row["input_identity"], []).append(compact)
        return {
            "exposure_weighted": {
                "policy_ce": float(np.mean(policy_rows[ids])),
                "value_mse": float(np.mean(value_rows[ids])),
            },
            "equal_input": {
                "policy_ce": float(
                    np.mean(
                        [np.mean(policy_rows[group]) for group in by_identity.values()]
                    )
                ),
                "value_mse": float(
                    np.mean(
                        [np.mean(value_rows[group]) for group in by_identity.values()]
                    )
                ),
            },
        }

    pooled = seed442._endpoint_metrics(model, arrays, membership)
    fresh = cohort([r for r in selected if r.get("source") == "fresh"])
    pooled["fresh"] = fresh
    return pooled


def project_direction(
    raw: list[np.ndarray], gradient: list[np.ndarray]
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Project a proposal displacement onto the non-harmful gradient halfspace."""
    if len(raw) != len(gradient) or not raw:
        raise ValueError("direction_parameter_count_mismatch")
    raw64 = [np.asarray(item, dtype=np.float64) for item in raw]
    grad64 = [np.asarray(item, dtype=np.float64) for item in gradient]
    if any(a.shape != b.shape for a, b in zip(raw64, grad64, strict=True)):
        raise ValueError("direction_shape_mismatch")
    dot = float(
        sum(np.sum(a * b, dtype=np.float64) for a, b in zip(raw64, grad64, strict=True))
    )
    norm_sq = float(sum(np.sum(b * b, dtype=np.float64) for b in grad64))
    coefficient = max(0.0, dot) / norm_sq if norm_sq > 0 else 0.0
    projected = [a - coefficient * b for a, b in zip(raw64, grad64, strict=True)]
    return projected, {
        "raw_dot": dot,
        "gradient_norm": float(np.sqrt(norm_sq)),
        "projected_dot": float(
            sum(
                np.sum(a.astype(np.float64) * b, dtype=np.float64)
                for a, b in zip(projected, grad64, strict=True)
            )
        ),
        "coefficient": coefficient,
    }


def project_proposal(
    pre: list[np.ndarray],
    raw_proposal: list[np.ndarray],
    gradient: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Project the float64 Adam displacement, then add it to float64 pre-state."""
    if len(pre) != len(raw_proposal):
        raise ValueError("proposal_parameter_count_mismatch")
    displacement = [
        b.astype(np.float64) - a.astype(np.float64)
        for a, b in zip(pre, raw_proposal, strict=True)
    ]
    projected, report = project_direction(displacement, gradient)
    result = [
        np.asarray(a.astype(np.float64) + d, dtype=np.float32)
        for a, d in zip(pre, projected, strict=True)
    ]
    report["projected_proposal_dot"] = float(
        sum(
            np.sum(
                (b.astype(np.float64) - a.astype(np.float64)) * g.astype(np.float64),
                dtype=np.float64,
            )
            for a, b, g in zip(pre, result, gradient, strict=True)
        )
    )
    return result, report


def trial_displacement(
    pre: list[np.ndarray], proposal: list[np.ndarray], scale: float
) -> list[np.ndarray]:
    """Match seed442/447 float32 trial construction exactly."""
    return [
        b.copy() if scale == 1.0 else np.asarray(a + (b - a) * scale, dtype=np.float32)
        for a, b in zip(pre, proposal, strict=True)
    ]


def realized_dot(
    pre: list[np.ndarray], trial: list[np.ndarray], gradient: list[np.ndarray]
) -> float:
    return float(
        sum(
            np.sum(
                (b.astype(np.float64) - a.astype(np.float64)) * g.astype(np.float64),
                dtype=np.float64,
            )
            for a, b, g in zip(pre, trial, gradient, strict=True)
        )
    )


def choose_largest_feasible(trials: list[dict[str, Any]]) -> float | None:
    """Select highest tested feasible scale; feasibility need not be monotone."""
    for row in trials:
        if row["feasible"]:
            return float(row["scale"])
    return None


def decide(metrics: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    """Apply the frozen seed453 advancement gates."""
    checks = []
    for weighting in ("exposure_weighted", "equal_input"):
        b = metrics["B"][weighting]
        init = metrics["initializer"][weighting]
        ordinary = metrics["seed442_A"][weighting]
        checks.extend(
            (
                b["policy_ce"] <= ordinary["policy_ce"] - 0.01,
                b["policy_ce"] <= init["policy_ce"] - 0.005,
                b["value_mse"] <= init["value_mse"] + 0.002,
                metrics["B"]["fresh"][weighting]["policy_ce"]
                <= metrics["initializer"]["fresh"][weighting]["policy_ce"] + 2e-6,
            )
        )
    checks.extend(
        (
            metrics["B"]["full_training_objective"]
            < metrics["initializer"]["full_training_objective"],
            any(s["activation"] and not s["rejected"] for s in steps),
            not any(s["proposal_norm"] > 0 and s["rejected"] for s in steps),
        )
    )
    active = any(s["activation"] and not s["rejected"] for s in steps)
    if not active:
        return "fresh_projection_inactive_no_followup"
    if not all(checks):
        return "close_fresh_policy_projection_branch"
    return "advance_to_separately_preregistered_strength_experiment"


def main() -> None:
    """Register frozen inputs or execute the prospective arms."""
    import argparse
    import json
    from pathlib import Path

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
