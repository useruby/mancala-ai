"""Read-only occurrence-level attribution of the archived seed471 T trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed472-trajectory-attribution")
SEED471 = Path("docs/data/seed471-fresh-only-adam-ablation")
INPUTS = {
    "manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "training": "docs/data/seed466-replay-input-bundle/training.npz",
    "evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    "seed471_registration": str(SEED471 / "registration.json"),
    "seed471_receipt": str(SEED471 / "receipt.json"),
    "seed471_T_steps": str(SEED471 / "T-steps.npz"),
    "seed471_T_trajectory": str(SEED471 / "T-trajectory.json"),
    "seed471_endpoint": str(SEED471 / "endpoint-evidence.json"),
}
SOURCES = (
    "ml/alphazero_lite/seed472_trajectory_attribution.py",
    "ml/alphazero_lite/publish_seed472_trajectory_attribution.py",
    "ml/alphazero_lite/verify_seed472_trajectory_attribution.py",
    "ml/alphazero_lite/test_seed472_trajectory_attribution.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/verify_seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/exact_root_policy_targets.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed472_output_exists")
    arrays = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    ev = arrays["evaluation"]
    hist = root / SEED471
    historical_receipt = json.loads((hist / "receipt.json").read_text())
    if historical_receipt.get("status") != "post_execution_bound_receipt":
        raise ValueError("seed471_receipt_status_mismatch")
    for name, expected in historical_receipt.items():
        if name != "status" and sha(hist / name) != expected:
            raise ValueError(f"seed471_historical_receipt_mismatch:{name}")
    selected = np.flatnonzero(ev["source_ids"] == 0)
    identities = ev["input_identity"][selected].tolist()
    if (len(selected), len(set(identities))) != (859, 851):
        raise ValueError("fresh_unseen_cohort_mismatch")
    if not out.exists():
        out.mkdir(parents=True)
    registration = {
        "schema": "seed472-trajectory-attribution-registration-v1",
        "status": "frozen_before_execution",
        "interpretation": "retrospective local attribution; descriptive, not proof of a causal mechanism",
        "authorization": "no new training trajectory; read-only reconstruction of archived seed471 T",
        "inputs_sha256": {name: sha(root / path) for name, path in INPUTS.items()},
        "source_closure_sha256": {name: sha(root / name) for name in SOURCES},
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "cohort": {
            "selection": "authenticated evaluation source_ids==0",
            "occurrences": 859,
            "identities": 851,
            "ordered_identities_sha256": hashlib.sha256(
                "\n".join(identities).encode()
            ).hexdigest(),
        },
        "weightings": {
            "exposure_weighted": "1/859 per occurrence",
            "equal_input": "1/(851*k_identity) per occurrence",
        },
        "steps": 16,
        "forward_schedule": "eval mode, contiguous occurrence order, chunks of 128, final chunk 91; same schedule at every pre/post state",
        "arithmetic": "float32 model forward/autograd; per-occurrence CE float64; realized displacement float64(post)-float64(pre); weighted reductions and dot products float64 in occurrence order; counterfactual scale and float32 writeback materialized",
        "tolerances": {
            "rtol": 2e-6,
            "atol": 2e-8,
            "counterfactual_writeback_atol": 2e-7,
            "classification_threshold": 1e-6,
        },
        "classifications": {
            "remainder_dominant": "Delta>1e-6 and R>0 and R>|A|",
            "direction_dominant": "Delta>1e-6 and A>0 and A>=|R|",
            "otherwise": "mixed_or_inconclusive",
            "shared": "only if both weightings agree; otherwise weighting_dependent",
        },
        "counterfactuals": "at each archived pre-state, zero-moment first-step Adam from reconstructed clipped J gradient and negative J gradient; independently norm-scaled to recorded carried-Adam realized displacement; float32 writeback; never carried forward",
        "seed471_original_endpoint": "retained unchanged; schedule-specific difference reported separately",
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )


def _load_arrays(root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    return bundle["training"], bundle["evaluation"]


def _set(model: torch.nn.Module, state: list[np.ndarray]) -> None:
    with torch.no_grad():
        for param, value in zip(model.parameters(), state, strict=True):
            param.copy_(torch.from_numpy(value.copy()))


def _flat(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([v.reshape(-1).astype(np.float64) for v in values])


def calculate(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = root.resolve()
    torch.set_num_threads(1)
    tr, ev = _load_arrays(root)
    select = np.flatnonzero(ev["source_ids"] == 0)
    identities = ev["input_identity"][select].tolist()
    compact = ev["compact_rows"][select]
    x, targets, masks = (
        tr["inputs"][compact],
        ev["policy_targets"][select],
        tr["legal_masks"][compact],
    )
    counts = {identity: identities.count(identity) for identity in set(identities)}
    weights = {
        "exposure_weighted": np.full(859, 1 / 859, dtype=np.float64),
        "equal_input": np.asarray(
            [1 / (851 * counts[i]) for i in identities], dtype=np.float64
        ),
    }
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        model,
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    archive: dict[str, np.ndarray] = {
        "occurrence_identity": np.asarray(identities, dtype="U256"),
        "occurrence_target": targets.copy(),
        "occurrence_legal_mask": masks.copy(),
    }
    for name, w in weights.items():
        archive[f"weight_{name}"] = w
    records: list[dict[str, Any]] = []
    initial_buffers = [buffer.detach().clone() for buffer in model.buffers()]
    previous_post: list[np.ndarray] | None = None
    with np.load(root / SEED471 / "T-steps.npz", allow_pickle=False) as steps:
        for step in range(16):
            pre = [
                steps[f"{step:02d}_pre_{i:02d}"].copy()
                for i in range(len(list(model.parameters())))
            ]
            post = [steps[f"{step:02d}_post_{i:02d}"].copy() for i in range(len(pre))]
            grad = [
                steps[f"{step:02d}_gradient_{i:02d}"].copy() for i in range(len(pre))
            ]
            moment = [
                steps[f"{step:02d}_exp_avg_{i:02d}"].copy() for i in range(len(pre))
            ]
            second = [
                steps[f"{step:02d}_exp_avg_sq_{i:02d}"].copy() for i in range(len(pre))
            ]
            displacement = [
                b.astype(np.float64) - a.astype(np.float64)
                for a, b in zip(pre, post, strict=True)
            ]
            realized_norm = float(np.linalg.norm(_flat(displacement)))
            # Reconstruct the J minibatch gradient and clipped Adam update from the archived pre-state.
            _set(model, pre)
            if step == 0:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.0000625,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0,
                )
            for p, archived in zip(model.parameters(), pre, strict=True):
                if not np.array_equal(p.detach().cpu().numpy(), archived):
                    raise ValueError(f"seed471_pre_state_mismatch:{step}")
            if previous_post is not None and any(
                not np.array_equal(pre[i], previous_post[i]) for i in range(len(pre))
            ):
                raise ValueError(f"seed471_state_discontinuity:{step}")
            positions = np.asarray(
                json.loads((root / SEED471 / "registration.json").read_text())[
                    "minibatches"
                ]["positions"][step * 512 : (step + 1) * 512],
                dtype=np.int64,
            )
            rows = tr["expanded_replay_indices"][tr["train_positions"][positions]]
            xb = torch.from_numpy(tr["inputs"][rows].copy())
            target = torch.from_numpy(tr["policy_targets"][rows].copy())
            mask = torch.from_numpy(tr["legal_masks"][rows].copy())
            fresh = torch.from_numpy(tr["fresh_source_mask"][rows].astype(np.float32))
            coeff = torch.from_numpy(tr["policy_coefficients"][rows].copy())
            value_target = torch.from_numpy(
                tr["value_targets"][rows].reshape(-1).copy()
            )
            model.zero_grad(set_to_none=True)
            logits, value = model(xb)
            ce = -(
                target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
            ).sum(1)
            policy_weight = coeff * fresh
            huber = F.smooth_l1_loss(value.reshape(-1), value_target, reduction="none")
            objective = (ce * policy_weight).sum() / policy_weight.sum() + 0.3 * (
                huber * fresh
            ).sum() / 512
            objective.backward()
            raw_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            reconstructed = [
                p.grad.detach().cpu().numpy().copy() for p in model.parameters()
            ]
            if any(
                not np.allclose(a, b, rtol=2e-6, atol=2e-8)
                for a, b in zip(reconstructed, grad, strict=True)
            ):
                raise ValueError(f"seed471_gradient_mismatch:{step}")
            optimizer.step()
            previous_post = post
            optimizer_model_post = [
                p.detach().cpu().numpy().copy() for p in model.parameters()
            ]
            optimizer_state = [optimizer.state[p] for p in model.parameters()]
            if any(
                not np.allclose(a, b, rtol=2e-6, atol=2e-8)
                for a, b in zip(optimizer_model_post, post, strict=True)
            ):
                raise ValueError(f"seed471_post_state_mismatch:{step}")
            if any(
                not np.allclose(s["exp_avg"].cpu().numpy(), m, rtol=2e-6, atol=2e-8)
                or not np.allclose(
                    s["exp_avg_sq"].cpu().numpy(), v, rtol=2e-6, atol=2e-8
                )
                for s, m, v in zip(optimizer_state, moment, second, strict=True)
            ):
                raise ValueError(f"seed471_adam_moment_mismatch:{step}")
            for key, vals in (
                ("pre", pre),
                ("post", post),
                ("J_gradient", grad),
                ("carried_exp_avg", moment),
                ("carried_exp_avg_sq", second),
                ("realized_displacement", displacement),
            ):
                for i, val in enumerate(vals):
                    archive[f"step{step:02d}_{key}_{i:02d}"] = val
            first_adam = [
                (-g / (np.sqrt(g.astype(np.float64) ** 2) + 1e-8)).astype(np.float32)
                for g in grad
            ]
            counterfactual_directions: dict[str, list[np.ndarray]] = {}
            counterfactual_scales: dict[str, float] = {}
            counterfactual_mismatch: dict[str, float] = {}
            for label, raw in (
                ("reset_adam", first_adam),
                ("negative_J", [-g for g in grad]),
            ):
                norm = float(np.linalg.norm(_flat(raw)))
                scale = realized_norm / norm if norm else 0.0
                trial = [
                    (a + np.asarray(r * scale, dtype=np.float32)).astype(np.float32)
                    for a, r in zip(pre, raw, strict=True)
                ]
                direction = [
                    t.astype(np.float64) - a.astype(np.float64)
                    for t, a in zip(trial, pre, strict=True)
                ]
                for i, val in enumerate(direction):
                    archive[f"step{step:02d}_{label}_direction_{i:02d}"] = val
                counterfactual_directions[label] = direction
                counterfactual_scales[label] = scale
                counterfactual_mismatch[label] = float(
                    np.linalg.norm(_flat(direction)) - realized_norm
                )
                archive[f"step{step:02d}_{label}_scale"] = np.asarray(
                    scale, dtype=np.float64
                )
                archive[f"step{step:02d}_{label}_norm_mismatch"] = np.asarray(
                    counterfactual_mismatch[label], dtype=np.float64
                )
            # Frozen occurrence forward schedule, pre and post, with gradients of weighted CE.
            model.eval()
            for state_name, state in (("pre", pre), ("post", post)):
                _set(model, state)
                losses_parts: list[np.ndarray] = []
                pred_parts: list[np.ndarray] = []
                for start in range(0, 859, 128):
                    xx = torch.from_numpy(x[start : start + 128].copy())
                    tt = torch.from_numpy(targets[start : start + 128].copy())
                    mm = torch.from_numpy(masks[start : start + 128].copy())
                    lg, _ = model(xx)
                    logp = F.log_softmax(lg.masked_fill(mm == 0, -1e9), dim=1)
                    pred_parts.append(logp.exp().detach().cpu().numpy())
                    losses_parts.append(
                        (-(tt * logp).sum(1)).detach().cpu().numpy().astype(np.float64)
                    )
                predictions, losses = (
                    np.concatenate(pred_parts),
                    np.concatenate(losses_parts),
                )
                archive[f"step{step:02d}_{state_name}_prediction"] = predictions
                archive[f"step{step:02d}_{state_name}_loss"] = losses
            # Fresh-unseen policy gradients from each weighting, independently at archived pre-state.
            _set(model, pre)
            for weighting, w in weights.items():
                model.zero_grad(set_to_none=True)
                total = 0.0
                for start in range(0, 859, 128):
                    lg, _ = model(torch.from_numpy(x[start : start + 128].copy()))
                    mm = torch.from_numpy(masks[start : start + 128].copy())
                    tt = torch.from_numpy(targets[start : start + 128].copy())
                    row_loss = -(
                        tt * F.log_softmax(lg.masked_fill(mm == 0, -1e9), dim=1)
                    ).sum(1)
                    (
                        row_loss
                        * torch.from_numpy(w[start : start + 128].astype(np.float32))
                    ).sum().backward()
                    total += float(
                        (
                            row_loss.detach().double()
                            * torch.from_numpy(w[start : start + 128])
                        ).sum()
                    )
                g_unseen = [
                    (
                        p.grad.detach().cpu().numpy().copy()
                        if p.grad is not None
                        else np.zeros_like(p.detach().cpu().numpy())
                    )
                    for p in model.parameters()
                ]
                for i, val in enumerate(g_unseen):
                    archive[f"step{step:02d}_unseen_gradient_{weighting}_{i:02d}"] = val
                d = _flat(displacement)
                g = _flat(g_unseen)
                delta = float(
                    np.dot(
                        archive[f"step{step:02d}_post_loss"].astype(np.float64)
                        - archive[f"step{step:02d}_pre_loss"].astype(np.float64),
                        w,
                    )
                )
                first = float(np.dot(g, d))
                cf_metrics = {}
                for cf_name, cf_direction in counterfactual_directions.items():
                    cf_dot = float(np.dot(g, _flat(cf_direction)))
                    cf_norm = float(np.linalg.norm(_flat(cf_direction)))
                    cf_metrics[f"{cf_name}_first_order"] = cf_dot
                    cf_metrics[f"{cf_name}_alignment"] = (
                        cf_dot / (np.linalg.norm(g) * cf_norm)
                        if np.linalg.norm(g) * cf_norm
                        else 0.0
                    )
                    cf_metrics[f"{cf_name}_norm_mismatch"] = counterfactual_mismatch[
                        cf_name
                    ]
                records.append(
                    {
                        "step": step,
                        "weighting": weighting,
                        "observed_ce_change": delta,
                        "first_order": first,
                        "remainder": delta - first,
                        "pre_ce": float(np.dot(archive[f"step{step:02d}_pre_loss"], w)),
                        "post_ce": float(
                            np.dot(archive[f"step{step:02d}_post_loss"], w)
                        ),
                        "gradient_dot_displacement": first,
                        "carried_direction_alignment": first
                        / (np.linalg.norm(g) * realized_norm)
                        if np.linalg.norm(g) * realized_norm
                        else 0.0,
                        "j_preclip_norm": raw_norm,
                        **cf_metrics,
                    }
                )
            model.train()
    totals = {}
    classifications = {}
    for weighting in weights:
        rows = [r for r in records if r["weighting"] == weighting]
        A, R, delta = (
            float(sum(r[k] for r in rows))
            for k in ("first_order", "remainder", "observed_ce_change")
        )
        label = (
            "remainder_dominant"
            if delta > 1e-6 and R > 0 and R > abs(A)
            else "direction_dominant"
            if delta > 1e-6 and A > 0 and A >= abs(R)
            else "mixed_or_inconclusive"
        )
        totals[weighting] = {
            "A_signed": A,
            "A_absolute": abs(A),
            "R_signed": R,
            "R_absolute": abs(R),
            "Delta_signed": delta,
            "Delta_absolute": abs(delta),
            "classification": label,
        }
        classifications[weighting] = label
    classification = (
        classifications["exposure_weighted"]
        if len(set(classifications.values())) == 1
        else "weighting_dependent"
    )
    endpoint_schedule = {
        "pre_exposure_weighted": float(
            np.dot(archive["step00_pre_loss"], weights["exposure_weighted"])
        ),
        "post_exposure_weighted": float(
            np.dot(archive["step15_post_loss"], weights["exposure_weighted"])
        ),
        "pre_equal_input": float(
            np.dot(archive["step00_pre_loss"], weights["equal_input"])
        ),
        "post_equal_input": float(
            np.dot(archive["step15_post_loss"], weights["equal_input"])
        ),
    }
    historical = json.loads((root / SEED471 / "endpoint-evidence.json").read_text())
    if any(
        not torch.equal(buffer, original)
        for buffer, original in zip(model.buffers(), initial_buffers, strict=True)
    ):
        raise ValueError("model_buffer_mutation")
    historical_fresh = {
        weighting: historical["arms"]["T"]["fresh"][key]["policy_ce"]
        for weighting, key in (
            ("exposure_weighted", "exposure_weighted"),
            ("equal_input", "equal_input"),
        )
    }
    endpoint_differences = {
        weighting: endpoint_schedule[f"post_{weighting}"] - historical_fresh[weighting]
        for weighting in weights
    }
    return archive, {
        "schema": "seed472-trajectory-attribution-v1",
        "steps": records,
        "totals": totals,
        "classification": classification,
        "weighting_classifications": classifications,
        "endpoint_reconciliation": {
            w: {
                "sum_step_changes": totals[w]["Delta_signed"],
                "endpoint_change": endpoint_schedule[f"post_{w}"]
                - endpoint_schedule[f"pre_{w}"],
                "reconciliation_error": totals[w]["Delta_signed"]
                - (endpoint_schedule[f"post_{w}"] - endpoint_schedule[f"pre_{w}"]),
            }
            for w in weights
        },
        "forward_schedule_endpoint": endpoint_schedule,
        "seed471_historical_endpoint_fresh_ce": historical_fresh,
        "forward_schedule_difference_from_seed471": endpoint_differences,
        "counterfactual_alignment_note": "local first-order comparisons only; do not establish optimizer-memory causality",
        "interpretation": "retrospective local attribution, not proof of a causal mechanism",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "calculate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    else:
        archive, report = calculate(args.root)
        out = args.root.resolve() / OUT
        np.savez_compressed(out / "evidence.npz", **archive)
        (out / "diagnostic.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
