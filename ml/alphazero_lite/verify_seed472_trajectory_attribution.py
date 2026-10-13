"""Independent read-only verifier for the seed472 attribution receipt."""

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

OUT = Path("docs/data/seed472-trajectory-attribution")
INPUTS = {
    "manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "training": "docs/data/seed466-replay-input-bundle/training.npz",
    "evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    "seed471_registration": "docs/data/seed471-fresh-only-adam-ablation/registration.json",
    "seed471_receipt": "docs/data/seed471-fresh-only-adam-ablation/receipt.json",
    "seed471_T_steps": "docs/data/seed471-fresh-only-adam-ablation/T-steps.npz",
    "seed471_T_trajectory": "docs/data/seed471-fresh-only-adam-ablation/T-trajectory.json",
    "seed471_endpoint": "docs/data/seed471-fresh-only-adam-ablation/endpoint-evidence.json",
}
SOURCES = {
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
}


def reconstruct(root: Path, evidence: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Rebuild vectors and accounting from seed466 rows and seed471 pre/post states."""
    torch.set_num_threads(1)
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    historical = root / "docs/data/seed471-fresh-only-adam-ablation"
    reg471 = json.loads((historical / "registration.json").read_text())
    selected = np.flatnonzero(ev["source_ids"] == 0)
    identities = ev["input_identity"][selected].tolist()
    compact = ev["compact_rows"][selected]
    x, targets, masks = (
        tr["inputs"][compact],
        ev["policy_targets"][selected],
        tr["legal_masks"][compact],
    )
    counts = {value: identities.count(value) for value in set(identities)}
    weights = {
        "exposure_weighted": np.full(
            len(selected), 1.0 / len(selected), dtype=np.float64
        ),
        "equal_input": np.asarray(
            [1 / (len(counts) * counts[value]) for value in identities],
            dtype=np.float64,
        ),
    }
    if not np.array_equal(evidence["occurrence_target"], targets) or not np.array_equal(
        evidence["occurrence_legal_mask"], masks
    ):
        raise ValueError("occurrence_target_or_mask_mismatch")
    if not np.array_equal(
        evidence["occurrence_identity"], np.asarray(identities, dtype="U256")
    ):
        raise ValueError("occurrence_order_mismatch")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INPUTS["initializer"])
    params = list(model.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=reg471["arms"]["T"]["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    output: list[dict[str, Any]] = []
    previous_post: list[np.ndarray] | None = None
    with np.load(historical / "T-steps.npz", allow_pickle=False) as archived:
        for step in range(16):
            pre = [
                archived[f"{step:02d}_pre_{i:02d}"].copy() for i in range(len(params))
            ]
            post = [
                archived[f"{step:02d}_post_{i:02d}"].copy() for i in range(len(params))
            ]
            with torch.no_grad():
                for p, value in zip(params, pre, strict=True):
                    p.copy_(torch.from_numpy(value.copy()))
            if previous_post is not None and any(
                not np.array_equal(pre[i], previous_post[i]) for i in range(len(params))
            ):
                raise ValueError(f"trajectory_state_discontinuity:{step}")
            positions = np.asarray(
                reg471["minibatches"]["positions"][step * 512 : (step + 1) * 512],
                dtype=np.int64,
            )
            rows = tr["expanded_replay_indices"][tr["train_positions"][positions]]
            xb = torch.from_numpy(tr["inputs"][rows].copy())
            target = torch.from_numpy(tr["policy_targets"][rows].copy())
            mask = torch.from_numpy(tr["legal_masks"][rows].copy())
            fresh = torch.from_numpy(tr["fresh_source_mask"][rows].astype(np.float32))
            coeff = torch.from_numpy(tr["policy_coefficients"][rows].copy())
            vt = torch.from_numpy(tr["value_targets"][rows].reshape(-1).copy())
            optimizer.zero_grad(set_to_none=True)
            logits, value = model(xb)
            row_ce = -(
                target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
            ).sum(1)
            weighted = coeff * fresh
            huber = F.smooth_l1_loss(value.reshape(-1), vt, reduction="none")
            objective = (row_ce * weighted).sum() / weighted.sum() + 0.3 * (
                huber * fresh
            ).sum() / 512
            objective.backward()
            raw_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            grad = [p.grad.detach().cpu().numpy().copy() for p in params]
            for i, g in enumerate(grad):
                if not np.allclose(
                    g, archived[f"{step:02d}_gradient_{i:02d}"], rtol=2e-6, atol=2e-8
                ):
                    raise ValueError(f"training_gradient_mismatch:{step}:{i}")
            # The historical vector is used only after the independent minibatch
            # reconstruction above establishes its identity within the frozen tolerance.
            # This also reproduces exact float32 writeback for tiny near-zero components.
            counterfactual_gradient = [
                archived[f"{step:02d}_gradient_{i:02d}"].copy()
                for i in range(len(params))
            ]
            optimizer.step()
            previous_post = post
            adam_post = [p.detach().cpu().numpy().copy() for p in params]
            exp_avg = [
                optimizer.state[p]["exp_avg"].detach().cpu().numpy().copy()
                for p in params
            ]
            exp_sq = [
                optimizer.state[p]["exp_avg_sq"].detach().cpu().numpy().copy()
                for p in params
            ]
            for i, (p, m, v) in enumerate(zip(adam_post, exp_avg, exp_sq, strict=True)):
                for label, actual in (("post", p), ("exp_avg", m), ("exp_avg_sq", v)):
                    if not np.allclose(
                        actual,
                        archived[f"{step:02d}_{label}_{i:02d}"],
                        rtol=2e-6,
                        atol=2e-8,
                    ):
                        raise ValueError(f"historical_{label}_mismatch:{step}:{i}")
            displacement = [
                b.astype(np.float64) - a.astype(np.float64)
                for a, b in zip(pre, post, strict=True)
            ]
            flat_d = np.concatenate([a.ravel() for a in displacement])
            realized_norm = float(np.linalg.norm(flat_d))
            for label, tensors in (
                ("pre", pre),
                ("post", post),
                ("J_gradient", grad),
                ("carried_exp_avg", exp_avg),
                ("carried_exp_avg_sq", exp_sq),
                ("realized_displacement", displacement),
            ):
                for i, tensor in enumerate(tensors):
                    if not np.allclose(
                        tensor,
                        evidence[f"step{step:02d}_{label}_{i:02d}"],
                        rtol=2e-6,
                        atol=2e-8,
                    ):
                        raise ValueError(f"archived_vector_mismatch:{step}:{label}:{i}")
            # Independent schedule: eval mode, contiguous occurrence rows, batches of 128.
            model.eval()
            preds: dict[str, np.ndarray] = {}
            losses: dict[str, np.ndarray] = {}
            for label, state in (("pre", pre), ("post", post)):
                with torch.no_grad():
                    for p, val_array in zip(params, state, strict=True):
                        p.copy_(torch.from_numpy(val_array.copy()))
                    pred_parts, loss_parts = [], []
                    for start in range(0, len(selected), 128):
                        lg, _ = model(torch.from_numpy(x[start : start + 128].copy()))
                        mm = torch.from_numpy(masks[start : start + 128].copy())
                        tt = torch.from_numpy(targets[start : start + 128].copy())
                        lp = F.log_softmax(lg.masked_fill(mm == 0, -1e9), dim=1)
                        pred_parts.append(lp.exp().cpu().numpy())
                        loss_parts.append(
                            (-(tt * lp).sum(1)).cpu().numpy().astype(np.float64)
                        )
                preds[label], losses[label] = (
                    np.concatenate(pred_parts),
                    np.concatenate(loss_parts),
                )
                if not np.allclose(
                    preds[label],
                    evidence[f"step{step:02d}_{label}_prediction"],
                    rtol=2e-6,
                    atol=2e-8,
                ):
                    raise ValueError(f"prediction_mismatch:{step}:{label}")
                if not np.allclose(
                    losses[label],
                    evidence[f"step{step:02d}_{label}_loss"],
                    rtol=2e-6,
                    atol=2e-8,
                ):
                    raise ValueError(f"occurrence_loss_mismatch:{step}:{label}")
            with torch.no_grad():
                for p, val_array in zip(params, pre, strict=True):
                    p.copy_(torch.from_numpy(val_array.copy()))
            for weighting, w in weights.items():
                if not np.allclose(
                    evidence[f"weight_{weighting}"], w, rtol=0, atol=1e-15
                ):
                    raise ValueError(f"weight_mismatch:{weighting}")
                model.zero_grad(set_to_none=True)
                for start in range(0, len(selected), 128):
                    lg, _ = model(torch.from_numpy(x[start : start + 128].copy()))
                    mm, tt = (
                        torch.from_numpy(masks[start : start + 128].copy()),
                        torch.from_numpy(targets[start : start + 128].copy()),
                    )
                    ce = -(
                        tt * F.log_softmax(lg.masked_fill(mm == 0, -1e9), dim=1)
                    ).sum(1)
                    (
                        ce * torch.from_numpy(w[start : start + 128].astype(np.float32))
                    ).sum().backward()
                unseen = [
                    p.grad.detach().cpu().numpy().copy()
                    if p.grad is not None
                    else np.zeros_like(p.detach().cpu().numpy())
                    for p in params
                ]
                flat_g = np.concatenate([v.astype(np.float64).ravel() for v in unseen])
                for i, g in enumerate(unseen):
                    if not np.allclose(
                        g,
                        evidence[f"step{step:02d}_unseen_gradient_{weighting}_{i:02d}"],
                        rtol=2e-6,
                        atol=2e-8,
                    ):
                        raise ValueError(
                            f"unseen_gradient_mismatch:{step}:{weighting}:{i}"
                        )
                observed = float(np.dot(losses["post"] - losses["pre"], w))
                first = float(np.dot(flat_g, flat_d))
                record: dict[str, Any] = {
                    "step": step,
                    "weighting": weighting,
                    "observed_ce_change": observed,
                    "first_order": first,
                    "remainder": observed - first,
                    "pre_ce": float(np.dot(losses["pre"], w)),
                    "post_ce": float(np.dot(losses["post"], w)),
                    "gradient_dot_displacement": first,
                    "carried_direction_alignment": first
                    / (np.linalg.norm(flat_g) * realized_norm)
                    if np.linalg.norm(flat_g) * realized_norm
                    else 0.0,
                    "j_preclip_norm": raw_norm,
                }
                for label, raw in (
                    (
                        "reset_adam",
                        [
                            (-g / (np.sqrt(g.astype(np.float64) ** 2) + 1e-8)).astype(
                                np.float32
                            )
                            for g in counterfactual_gradient
                        ],
                    ),
                    ("negative_J", [-g for g in counterfactual_gradient]),
                ):
                    cf_raw_norm = float(
                        np.linalg.norm(
                            np.concatenate([r.astype(np.float64).ravel() for r in raw])
                        )
                    )
                    scale = realized_norm / cf_raw_norm if cf_raw_norm else 0.0
                    trial = [
                        (a + np.asarray(r * scale, dtype=np.float32)).astype(np.float32)
                        for a, r in zip(pre, raw, strict=True)
                    ]
                    cf = [
                        b.astype(np.float64) - a.astype(np.float64)
                        for a, b in zip(pre, trial, strict=True)
                    ]
                    cf_flat = np.concatenate([v.ravel() for v in cf])
                    for i, tensor in enumerate(cf):
                        if not np.allclose(
                            tensor,
                            evidence[f"step{step:02d}_{label}_direction_{i:02d}"],
                            rtol=2e-6,
                            atol=2e-7,
                        ):
                            raise ValueError(
                                f"counterfactual_direction_mismatch:{step}:{label}:{i}"
                            )
                    mismatch = float(np.linalg.norm(cf_flat) - realized_norm)
                    if not np.isclose(
                        scale,
                        evidence[f"step{step:02d}_{label}_scale"],
                        rtol=2e-6,
                        atol=2e-8,
                    ) or not np.isclose(
                        mismatch,
                        evidence[f"step{step:02d}_{label}_norm_mismatch"],
                        rtol=2e-6,
                        atol=2e-8,
                    ):
                        raise ValueError(
                            f"counterfactual_normalization_mismatch:{step}:{label}"
                        )
                    record[f"{label}_first_order"] = float(np.dot(flat_g, cf_flat))
                    record[f"{label}_alignment"] = (
                        record[f"{label}_first_order"]
                        / (np.linalg.norm(flat_g) * np.linalg.norm(cf_flat))
                        if np.linalg.norm(flat_g) * np.linalg.norm(cf_flat)
                        else 0.0
                    )
                    record[f"{label}_norm_mismatch"] = mismatch
                output.append(record)
            model.train()
    return output


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    historical = root / "docs/data/seed471-fresh-only-adam-ablation"
    historical_before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in historical.iterdir()
        if p.is_file()
    }
    try:
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed472-trajectory-attribution-registration-v1"
            or reg.get("status") != "frozen_before_execution"
        ):
            raise ValueError("registration_identity_mismatch")
        if reg.get("dependencies") != {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }:
            raise ValueError("dependency_identity_mismatch")
        if (
            set(reg.get("inputs_sha256", {})) != set(INPUTS)
            or set(reg.get("source_closure_sha256", {})) != SOURCES
        ):
            raise ValueError("frozen_closure_inventory_mismatch")
        for name, expected in reg["inputs_sha256"].items():
            if name not in INPUTS or digest(root / INPUTS[name]) != expected:
                raise ValueError(f"input_binding_mismatch:{name}")
        for name, expected in reg["source_closure_sha256"].items():
            if digest(root / name) != expected:
                raise ValueError(f"source_binding_mismatch:{name}")
        read_authenticated_bundle(
            root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
        )
        historical_receipt = json.loads((historical / "receipt.json").read_text())
        if historical_receipt.get("status") != "post_execution_bound_receipt":
            raise ValueError("seed471_receipt_status_mismatch")
        for name, expected in historical_receipt.items():
            if name != "status" and digest(historical / name) != expected:
                raise ValueError(f"seed471_historical_receipt_mismatch:{name}")
        receipt = json.loads((out / "receipt.json").read_text())
        if receipt.get("status") != "post_execution_bound_receipt":
            raise ValueError("receipt_status_mismatch")
        for name, expected in receipt.items():
            if name != "status" and digest(out / name) != expected:
                raise ValueError(f"receipt_binding_mismatch:{name}")
        diagnostic = json.loads((out / "diagnostic.json").read_text())
        with np.load(out / "evidence.npz", allow_pickle=False) as evidence:
            arrays = {key: evidence[key] for key in evidence.files}
        independently_reconstructed = reconstruct(root, arrays)
        # Reconstruct aggregate accounting solely from the archived occurrence losses,
        # authenticated cohort weights, and float64 parameter-state differences.
        rows = diagnostic["steps"]
        if len(rows) != 32 or [(r["step"], r["weighting"]) for r in rows] != [
            (s, w) for s in range(16) for w in ("exposure_weighted", "equal_input")
        ]:
            raise ValueError("step_accounting_coverage_mismatch")
        for actual, computed in zip(rows, independently_reconstructed, strict=True):
            for key, expected in computed.items():
                if key in ("step", "weighting"):
                    if actual[key] != expected:
                        raise ValueError(f"independent_record_identity_mismatch:{key}")
                    continue
                if not np.isclose(actual[key], expected, rtol=2e-6, atol=2e-8):
                    raise ValueError(
                        f"independent_record_mismatch:{computed['step']}:{computed['weighting']}:{key}"
                    )
        for weighting in ("exposure_weighted", "equal_input"):
            select = [r for r in rows if r["weighting"] == weighting]
            for r in select:
                step = r["step"]
                w = arrays[f"weight_{weighting}"]
                pre = arrays[f"step{step:02d}_pre_loss"]
                post = arrays[f"step{step:02d}_post_loss"]
                observed = float(
                    np.dot(post.astype(np.float64) - pre.astype(np.float64), w)
                )
                parameter_count = sum(
                    f"step{step:02d}_realized_displacement_{i:02d}" in arrays
                    for i in range(64)
                )
                displacement = np.concatenate(
                    [
                        arrays[f"step{step:02d}_realized_displacement_{i:02d}"].reshape(
                            -1
                        )
                        for i in range(parameter_count)
                    ]
                )
                gradient = np.concatenate(
                    [
                        arrays[
                            f"step{step:02d}_unseen_gradient_{weighting}_{i:02d}"
                        ].reshape(-1)
                        for i in range(parameter_count)
                    ]
                )
                first = float(
                    np.dot(gradient.astype(np.float64), displacement.astype(np.float64))
                )
                if not np.isclose(
                    observed, r["observed_ce_change"], rtol=2e-6, atol=2e-8
                ) or not np.isclose(first, r["first_order"], rtol=2e-6, atol=2e-8):
                    raise ValueError(
                        f"accounting_reconstruction_mismatch:{step}:{weighting}"
                    )
                if not np.isclose(
                    r["remainder"], observed - first, rtol=2e-6, atol=2e-8
                ):
                    raise ValueError(f"remainder_mismatch:{step}:{weighting}")
        for weighting in ("exposure_weighted", "equal_input"):
            selected = [r for r in rows if r["weighting"] == weighting]
            A = float(sum(r["first_order"] for r in selected))
            R = float(sum(r["remainder"] for r in selected))
            delta = float(sum(r["observed_ce_change"] for r in selected))
            label = (
                "remainder_dominant"
                if delta > 1e-6 and R > 0 and R > abs(A)
                else "direction_dominant"
                if delta > 1e-6 and A > 0 and A >= abs(R)
                else "mixed_or_inconclusive"
            )
            actual = diagnostic["totals"][weighting]
            for key, value in (
                ("A_signed", A),
                ("A_absolute", abs(A)),
                ("R_signed", R),
                ("R_absolute", abs(R)),
                ("Delta_signed", delta),
                ("Delta_absolute", abs(delta)),
            ):
                if not np.isclose(actual[key], value, rtol=2e-6, atol=2e-8):
                    raise ValueError(f"total_mismatch:{weighting}:{key}")
            if actual["classification"] != label:
                raise ValueError(f"classification_mismatch:{weighting}")
            pre_loss = arrays["step00_pre_loss"]
            post_loss = arrays["step15_post_loss"]
            weights = arrays[f"weight_{weighting}"]
            endpoint_delta = float(
                np.dot(post_loss, weights) - np.dot(pre_loss, weights)
            )
            reconciliation = diagnostic["endpoint_reconciliation"][weighting]
            if (
                not np.isclose(endpoint_delta, delta, rtol=2e-6, atol=2e-8)
                or not np.isclose(
                    reconciliation["endpoint_change"],
                    endpoint_delta,
                    rtol=2e-6,
                    atol=2e-8,
                )
                or not np.isclose(
                    reconciliation["reconciliation_error"],
                    delta - endpoint_delta,
                    rtol=2e-6,
                    atol=2e-8,
                )
            ):
                raise ValueError(f"endpoint_reconciliation_mismatch:{weighting}")
        classifications = [
            diagnostic["totals"][w]["classification"]
            for w in ("exposure_weighted", "equal_input")
        ]
        shared = (
            classifications[0]
            if classifications[0] == classifications[1]
            else "weighting_dependent"
        )
        if diagnostic["classification"] != shared:
            raise ValueError("shared_classification_mismatch")
        historical_endpoint = json.loads(
            (historical / "endpoint-evidence.json").read_text()
        )
        for weighting in ("exposure_weighted", "equal_input"):
            key = (
                "exposure_weighted"
                if weighting == "exposure_weighted"
                else "equal_input"
            )
            old = historical_endpoint["arms"]["T"]["fresh"][key]["policy_ce"]
            if not np.isclose(
                diagnostic["seed471_historical_endpoint_fresh_ce"][weighting],
                old,
                rtol=0,
                atol=0,
            ):
                raise ValueError(f"historical_endpoint_was_not_preserved:{weighting}")
            new = diagnostic["forward_schedule_endpoint"][f"post_{weighting}"]
            if not np.isclose(
                diagnostic["forward_schedule_difference_from_seed471"][weighting],
                new - old,
                rtol=2e-6,
                atol=2e-8,
            ):
                raise ValueError(f"forward_schedule_difference_mismatch:{weighting}")
    finally:
        after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if before != after:
            raise ValueError("verification_mutated_publication")
        historical_after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in historical.iterdir()
            if p.is_file()
        }
        if historical_before != historical_after:
            raise ValueError("verification_mutated_seed471_evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    verify(parser.parse_args().root)
    print("seed472 independent verification passed")


if __name__ == "__main__":
    main()
