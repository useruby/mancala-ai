"""Publish the post-execution accounting supplement for seed476."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_OUT = Path("docs/data/seed476-direction-transport")
FIRST_SUPPLEMENT = Path("docs/data/seed476-direction-transport-supplement")
OUT = Path("docs/data/seed476-direction-transport-supplement-v2")
SEED474 = Path("docs/data/seed474-fresh-value-removal")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
PUBLISHER = "ml/alphazero_lite/publish_seed476_transport_supplement.py"
VERIFIER = "ml/alphazero_lite/verify_seed476_transport_supplement.py"
TESTS = "ml/alphazero_lite/test_seed476_transport_supplement.py"
BASE_SOURCES = (
    "ml/alphazero_lite/seed476_direction_transport.py",
    "ml/alphazero_lite/publish_seed476_direction_transport.py",
    "ml/alphazero_lite/verify_seed476_direction_transport.py",
    "ml/alphazero_lite/test_seed476_direction_transport.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/verify_seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
)
SOURCES = (*BASE_SOURCES, PUBLISHER, VERIFIER, TESTS)
INPUTS = {
    "manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "training": "docs/data/seed466-replay-input-bundle/training.npz",
    "evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": str(INIT),
    "seed474_registration": str(SEED474 / "registration.json"),
    "seed474_receipt": str(SEED474 / "receipt.json"),
    "seed474_P_steps": str(SEED474 / "P-steps.npz"),
    "seed474_P_trajectory": str(SEED474 / "P-trajectory.json"),
    "seed474_endpoint": str(SEED474 / "endpoint-evidence.json"),
    "seed476_registration": str(BASE_OUT / "registration.json"),
    "seed476_receipt": str(BASE_OUT / "receipt.json"),
    "seed476_evidence": str(BASE_OUT / "evidence.json"),
    "seed476_vectors": str(BASE_OUT / "directions.npz"),
    "prior_supplement_registration": str(FIRST_SUPPLEMENT / "registration.json"),
    "prior_supplement_receipt": str(FIRST_SUPPLEMENT / "receipt.json"),
    "prior_supplement_evidence": str(FIRST_SUPPLEMENT / "evidence.json"),
    "prior_supplement_accounting": str(FIRST_SUPPLEMENT / "accounting.npz"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path) -> dict:
    """Freeze supplemental source/input identities and schedule before recalculation."""
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("supplement_output_exists")
    base = json.loads((root / INPUTS["seed476_registration"]).read_text())
    receipt = json.loads((root / INPUTS["seed476_receipt"]).read_text())
    for name, digest in receipt.items():
        if name != "status" and sha(root / BASE_OUT / name) != digest:
            raise ValueError(f"seed476_receipt_mismatch:{name}")
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    positions = np.asarray(base["population"]["minibatch_positions"], dtype=np.int64)
    rows = tr["expanded_replay_indices"][tr["train_positions"]]
    selected = rows[tr["fresh_source_mask"][rows].astype(bool)]
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    if (len(selected), len(unseen), len(positions)) != (60015, 859, 8192):
        raise ValueError("frozen_population_mismatch")
    out.mkdir(parents=True, exist_ok=True)
    registration = {
        "schema": "seed476-direction-transport-supplement-registration-v1",
        "status": "post_execution_supplement_frozen_before_reconstruction",
        "timing": "post-execution retrospective verification supplement; original seed476 freeze/publication preserved and not rerun",
        "interpretation": "exploratory descriptive attribution only; no causal or strength claim",
        "original_attempt_deliverables_absent_at_execution": [
            "independent reset-Adam reconstruction including clipping, zero moments, step-one bias correction, float32 writeback, and post-state comparison",
            "pre/post occurrence predictions and policy losses under a declared forward schedule",
            "exposure/equal-input observed changes, first-order contributions, remainders, signed totals, endpoint reconciliation",
            "optimizer-update, loss-accounting, and full receipt-rebound corruption verification",
        ],
        "original_seed476_registration_sha256": sha(
            root / INPUTS["seed476_registration"]
        ),
        "original_seed476_receipt_sha256": sha(root / INPUTS["seed476_receipt"]),
        "original_seed476_evidence_sha256": sha(root / INPUTS["seed476_evidence"]),
        "original_seed476_vectors_sha256": sha(root / INPUTS["seed476_vectors"]),
        "inputs_sha256": {key: sha(root / rel) for key, rel in INPUTS.items()},
        "source_closure_sha256": {rel: sha(root / rel) for rel in SOURCES},
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "authenticated_manifest_sha256": MANIFEST_SHA256,
        "population": {
            "training_occurrences": int(len(selected)),
            "strict_unseen_occurrences": int(len(unseen)),
            "strict_unseen_exact_inputs": len(
                set(ev["input_identity"][unseen].tolist())
            ),
            "training_rows_sha256": hashlib.sha256(
                selected.astype("<i8").tobytes()
            ).hexdigest(),
            "unseen_occurrence_indices_sha256": hashlib.sha256(
                unseen.astype("<i8").tobytes()
            ).hexdigest(),
            "minibatch_positions_sha256": hashlib.sha256(
                positions.astype("<i8").tobytes()
            ).hexdigest(),
            "training_denominator": float(
                tr["policy_coefficients"][selected].sum(dtype=np.float64)
            ),
        },
        "forward_schedule": {
            "state_order": "step 1 pre, step 1 post, ..., step 16 pre, step 16 post; each state loaded independently from archived tensors",
            "training": "contiguous preserved occurrence order, batches of 512, eval mode, float32 forward; float64 occurrence loss reductions",
            "unseen": "preserved strict-unseen occurrence order, batches of 128, eval mode, float32 forward; float64 occurrence loss reductions",
            "predictions": "legal-action probabilities after float32 masked log-softmax; saved per occurrence and endpoint",
            "threads": 1,
            "comparison_rtol": 2e-6,
            "comparison_atol": 2e-8,
            "sign_threshold": 1e-6,
        },
        "optimizer_reconstruction": "each minibatch gradient recomputed from authenticated occurrence indices, own policy targets, legal masks, coefficients, and fresh mask; clip global norm 1; explicit zero grad for unused parameters; clear Adam state; one step at lr 0.0000625, betas (0.9,0.999), eps 1e-8, no weight decay; compare clipped gradients, exp_avg, exp_avg_sq, step counter, float32 post-state to P-steps archive",
        "weights": {
            "exposure": "1/859 for each occurrence",
            "equal_input": "1/(851*k_identity) per occurrence; identity groups derive from authenticated input_identity",
        },
        "accounting": "for each step and weighting, observed=post loss-pre loss; first_order=pre-state loss gradient dot float64 recorded displacement; remainder=observed-first_order; totals sum signed values over 16 steps and reconcile against endpoint loss difference",
        "counterfactuals": "norm-match only within each local archived pre-state; float32 writeback then float64 realized displacement; never carry counterfactual state forward",
        "forward_schedule_difference": "Original seed476 gradient calculation used 512-occurrence training chunks and 128-occurrence unseen chunks. Supplemental prediction/loss schedule uses those same chunk sizes; evaluation runs on all 32 archived endpoints in explicit pre/post order. No historical evidence or metric is rewritten.",
        "prior_supplement_attempt": "The first post-execution supplement attempt was retained unchanged. Its independent verifier found an NPZ key-inventory naming mismatch before semantic comparisons; it is bound by the prior_supplement_* input hashes and superseded by this explicitly versioned v2 supplement.",
        "limitation": "retrospective schedule and arithmetic reconstruction; no causal attribution to optimizer or sampling and no independent confirmation",
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    return registration


def _flatten(parameters: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in parameters]
    )


def calculate(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Calculate supplement using frozen seed476 states; no state is advanced."""
    root = root.resolve()
    torch.set_num_threads(1)
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    reg = json.loads((root / OUT / "registration.json").read_text())
    base_reg = json.loads((root / BASE_OUT / "registration.json").read_text())
    positions = np.asarray(
        base_reg["population"]["minibatch_positions"], dtype=np.int64
    )
    all_rows = tr["expanded_replay_indices"][tr["train_positions"]]
    train_rows = all_rows[tr["fresh_source_mask"][all_rows].astype(bool)]
    coeff = tr["policy_coefficients"][train_rows].astype(np.float64)
    train_denominator = float(coeff.sum(dtype=np.float64))
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    unseen_rows = ev["compact_rows"][unseen]
    identities = ev["input_identity"][unseen].tolist()
    frequencies = {identity: identities.count(identity) for identity in set(identities)}
    held_weights = {
        "exposure": np.full(len(unseen), 1 / len(unseen), dtype=np.float64),
        "equal_input": np.asarray(
            [1 / (len(frequencies) * frequencies[key]) for key in identities],
            dtype=np.float64,
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "training_occurrence_rows": train_rows.astype(np.int64),
        "training_coefficients": tr["policy_coefficients"][train_rows].copy(),
        "training_targets": tr["policy_targets"][train_rows].copy(),
        "training_legal_masks": tr["legal_masks"][train_rows].copy(),
        "unseen_occurrences": unseen.astype(np.int64),
        "unseen_rows": unseen_rows.astype(np.int64),
        "unseen_targets": ev["policy_targets"][unseen].copy(),
        "unseen_legal_masks": tr["legal_masks"][unseen_rows].copy(),
        "unseen_exposure_weights": held_weights["exposure"],
        "unseen_equal_input_weights": held_weights["equal_input"],
    }
    steps = []
    with np.load(root / SEED474 / "P-steps.npz", allow_pickle=False) as states:
        for step in range(16):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(model, root / INIT)
            params = list(model.parameters())
            pre_tensors = [
                torch.from_numpy(states[f"{step:02d}_pre_{i:02d}"].copy())
                for i in range(len(params))
            ]
            post_tensors = [
                torch.from_numpy(states[f"{step:02d}_post_{i:02d}"].copy())
                for i in range(len(params))
            ]
            with torch.no_grad():
                for p, before in zip(params, pre_tensors, strict=True):
                    p.copy_(before)
            model.eval()
            batch_rows = all_rows[positions[step * 512 : (step + 1) * 512]]
            bx = torch.from_numpy(tr["inputs"][batch_rows].copy())
            by = torch.from_numpy(tr["policy_targets"][batch_rows].copy())
            bm = torch.from_numpy(tr["legal_masks"][batch_rows].copy())
            bc = torch.from_numpy(tr["policy_coefficients"][batch_rows].copy())
            fresh = torch.from_numpy(
                tr["fresh_source_mask"][batch_rows].astype(np.float32)
            )
            logits, _ = model(bx)
            batch_ce = -(
                by * F.log_softmax(logits.masked_fill(bm == 0, -1e9), dim=1)
            ).sum(1)
            minibatch_mass = (bc * fresh).sum()
            (batch_ce.mul(bc).mul(fresh).sum() / minibatch_mass).backward()
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            preclip_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            clipped_grads = [p.grad.detach().clone() for p in params]
            optimizer = torch.optim.Adam(
                params, lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            optimizer.state.clear()
            optimizer.step()
            update_checks = {
                "preclip_norm": float(preclip_norm),
                "clipped": bool(float(preclip_norm) > 1),
                "step_counter": 1,
                "state_reset": True,
                "parameter_checks": [],
            }
            for i, (p, grad, pre, expected_post) in enumerate(
                zip(params, clipped_grads, pre_tensors, post_tensors, strict=True)
            ):
                state = optimizer.state[p]
                expected_grad = states[f"{step:02d}_gradient_{i:02d}"]
                expected_m = states[f"{step:02d}_exp_avg_{i:02d}"]
                expected_v = states[f"{step:02d}_exp_avg_sq_{i:02d}"]
                comparisons = {
                    "gradient": np.array_equal(grad.numpy(), expected_grad),
                    "exp_avg": np.array_equal(state["exp_avg"].numpy(), expected_m),
                    "exp_avg_sq": np.array_equal(
                        state["exp_avg_sq"].numpy(), expected_v
                    ),
                    "post": np.array_equal(p.detach().numpy(), expected_post.numpy()),
                    "step": float(state["step"]) == 1.0,
                }
                update_checks["parameter_checks"].append(
                    {"parameter": i, **comparisons}
                )
                arrays[f"{step:02d}_optimizer_gradient_{i:02d}"] = grad.numpy().copy()
                arrays[f"{step:02d}_optimizer_exp_avg_{i:02d}"] = (
                    state["exp_avg"].numpy().copy()
                )
                arrays[f"{step:02d}_optimizer_exp_avg_sq_{i:02d}"] = (
                    state["exp_avg_sq"].numpy().copy()
                )
                arrays[f"{step:02d}_optimizer_pre_{i:02d}"] = pre.numpy().copy()
                arrays[f"{step:02d}_optimizer_post_{i:02d}"] = p.detach().numpy().copy()
                arrays[f"{step:02d}_optimizer_counter_{i:02d}"] = np.asarray(
                    float(state["step"]), dtype=np.float64
                )
                arrays[f"{step:02d}_optimizer_zero_m_{i:02d}"] = np.zeros_like(
                    state["exp_avg"].numpy()
                )
                arrays[f"{step:02d}_optimizer_zero_v_{i:02d}"] = np.zeros_like(
                    state["exp_avg_sq"].numpy()
                )
                if not all(comparisons.values()):
                    raise ValueError(
                        f"reset_adam_reconstruction_mismatch:{step + 1}:{i}:{comparisons}"
                    )
            update_checks["passed"] = True

            # The diagnostic objective and all direction gradients are evaluated
            # independently at this archived pre-state, not at the local proposal.
            with torch.no_grad():
                for p, before in zip(params, pre_tensors, strict=True):
                    p.copy_(before)

            full_accum = [torch.zeros_like(p) for p in params]
            for start in range(0, len(train_rows), 512):
                rr = train_rows[start : start + 512]
                x = torch.from_numpy(tr["inputs"][rr].copy())
                y = torch.from_numpy(tr["policy_targets"][rr].copy())
                legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                logits_full, _ = model(x)
                ce = -(
                    y * F.log_softmax(logits_full.masked_fill(legal == 0, -1e9), dim=1)
                ).sum(1)
                weight = torch.from_numpy(
                    (coeff[start : start + len(rr)] / train_denominator).astype(
                        np.float32
                    )
                )
                gs = torch.autograd.grad((ce * weight).sum(), params, allow_unused=True)
                for acc, g in zip(full_accum, gs, strict=True):
                    if g is not None:
                        acc.add_(g.detach())
            full_g = _flatten(full_accum)
            batch_g = _flatten(clipped_grads)
            full_pre = np.concatenate(
                [a.numpy().reshape(-1).astype(np.float64) for a in pre_tensors]
            )
            full_post = np.concatenate(
                [a.numpy().reshape(-1).astype(np.float64) for a in post_tensors]
            )
            adam_delta = full_post - full_pre
            held_grads = {}
            for name, occurrence_weights in held_weights.items():
                acc = [torch.zeros_like(p) for p in params]
                for start in range(0, len(unseen), 128):
                    sl = slice(start, start + 128)
                    rr = unseen_rows[sl]
                    logits_held, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                    y = torch.from_numpy(ev["policy_targets"][unseen[sl]].copy())
                    legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                    ce = -(
                        y
                        * F.log_softmax(
                            logits_held.masked_fill(legal == 0, -1e9), dim=1
                        )
                    ).sum(1)
                    gs = torch.autograd.grad(
                        (
                            ce
                            * torch.from_numpy(
                                occurrence_weights[sl].astype(np.float32)
                            )
                        ).sum(),
                        params,
                        allow_unused=True,
                    )
                    for dst, g in zip(acc, gs, strict=True):
                        if g is not None:
                            dst.add_(g.detach())
                held_grads[name] = _flatten(acc)

            pre_model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(pre_model, root / INIT)
            with torch.no_grad():
                for p, before in zip(pre_model.parameters(), pre_tensors, strict=True):
                    p.copy_(before)
            pre_model.eval()
            state_losses: dict[str, dict[str, float]] = {}
            for endpoint, state_model, endpoint_params in (
                ("pre", pre_model, pre_tensors),
                ("post", model, post_tensors),
            ):
                with torch.no_grad():
                    for p, tensor in zip(
                        state_model.parameters(), endpoint_params, strict=True
                    ):
                        p.copy_(tensor)
                state_model.eval()
                tr_ce, tr_pred, _ = _predict(
                    state_model,
                    tr["inputs"][train_rows],
                    tr["policy_targets"][train_rows],
                    tr["legal_masks"][train_rows],
                    512,
                )
                tr_loss = float(np.dot(tr_ce, coeff) / train_denominator)
                arrays[f"{step:02d}_{endpoint}_training_ce"] = tr_ce
                arrays[f"{step:02d}_{endpoint}_training_probability"] = tr_pred
                state_losses[endpoint] = {"training_policy_ce": tr_loss}
                ue = ev["policy_targets"][unseen]
                um = tr["legal_masks"][unseen_rows]
                ue_ce, ue_pred, _ = _predict(
                    state_model, tr["inputs"][unseen_rows], ue, um, 128
                )
                arrays[f"{step:02d}_{endpoint}_unseen_ce"] = ue_ce
                arrays[f"{step:02d}_{endpoint}_unseen_probability"] = ue_pred
                for weighting, w in held_weights.items():
                    state_losses[endpoint][f"unseen_{weighting}_policy_ce"] = float(
                        np.dot(ue_ce, w)
                    )
            realized = {"recorded_adam": adam_delta}
            target_norm = float(np.linalg.norm(adam_delta))
            for name, raw in (
                ("full_negative", -full_g),
                ("minibatch_negative", -batch_g),
            ):
                raw_norm = float(np.linalg.norm(raw))
                arrays[f"{step:02d}_counterfactual_scale_{name}"] = np.asarray(
                    0.0 if raw_norm == 0 else target_norm / raw_norm, dtype=np.float64
                )
                scaled = (
                    np.zeros_like(raw)
                    if raw_norm == 0
                    else (raw * target_norm / raw_norm).astype(np.float32)
                )
                pieces = []
                offset = 0
                for pre in pre_tensors:
                    n = pre.numel()
                    after = pre + torch.from_numpy(
                        scaled[offset : offset + n].reshape(pre.shape).copy()
                    )
                    pieces.append((after.double() - pre.double()).numpy().reshape(-1))
                    offset += n
                realized[name] = np.concatenate(pieces)
            for key, vector in {
                "full_gradient": full_g,
                "minibatch_gradient": batch_g,
                "adam_displacement": adam_delta,
                **{f"unseen_{k}_gradient": v for k, v in held_grads.items()},
                **{f"realized_{k}": v for k, v in realized.items()},
            }.items():
                arrays[f"{step:02d}_{key}"] = vector
            step_record = {
                "step": step + 1,
                "optimizer_reconstruction": update_checks,
                "losses": state_losses,
                "directions": {},
            }
            observed_train = (
                state_losses["post"]["training_policy_ce"]
                - state_losses["pre"]["training_policy_ce"]
            )
            step_record["training_accounting"] = {
                "observed_change": observed_train,
                "first_order_contribution": float(np.dot(full_g, adam_delta)),
                "remainder": observed_train - float(np.dot(full_g, adam_delta)),
            }
            for weighting, g in held_grads.items():
                obs = (
                    state_losses["post"][f"unseen_{weighting}_policy_ce"]
                    - state_losses["pre"][f"unseen_{weighting}_policy_ce"]
                )
                step_record["directions"][weighting] = {}
                for direction_name, direction in realized.items():
                    norm_g = float(np.linalg.norm(g))
                    norm_d = float(np.linalg.norm(direction))
                    dot = float(np.dot(g, direction))
                    cosine = dot / (norm_g * norm_d) if norm_g and norm_d else 0.0
                    sign = (
                        "zero_norm"
                        if not norm_g or not norm_d
                        else "descent"
                        if cosine <= -1e-6
                        else "ascent"
                        if cosine >= 1e-6
                        else "ambiguous"
                    )
                    step_record["directions"][weighting][direction_name] = {
                        "norm": norm_d,
                        "dot": dot,
                        "cosine": cosine,
                        "sign": sign,
                        "first_order_contribution": dot,
                        "observed_loss_change": obs,
                        "remainder": obs - dot,
                    }
                score = step_record["directions"][weighting]
                signs = [
                    score[k]["sign"]
                    for k in ("full_negative", "minibatch_negative", "recorded_adam")
                ]
                score["category"] = (
                    "minibatch_transport_failure"
                    if signs[0] == "descent" and signs[1] == "ascent"
                    else "adam_coordinate_transport_failure"
                    if signs[1] == "descent" and signs[2] == "ascent"
                    else "both_descent"
                    if signs == ["descent"] * 3
                    else "mixed_or_ambiguous"
                )
                score["observed_change"] = obs
                score["actual_first_order_contribution"] = float(np.dot(g, adam_delta))
                score["actual_remainder"] = (
                    obs - score["actual_first_order_contribution"]
                )
            steps.append(step_record)
    totals = {}
    for weighting in held_weights:
        records = [s["directions"][weighting] for s in steps]
        categories = [r["category"] for r in records]
        signed = {
            key: float(sum(r[key]["dot"] for r in records))
            for key in ("full_negative", "minibatch_negative", "recorded_adam")
        }
        observed = float(sum(r["observed_change"] for r in records))
        first = float(sum(r["actual_first_order_contribution"] for r in records))
        rem = float(sum(r["actual_remainder"] for r in records))
        start = steps[0]["losses"]["pre"][f"unseen_{weighting}_policy_ce"]
        end = steps[-1]["losses"]["post"][f"unseen_{weighting}_policy_ce"]
        totals[weighting] = {
            "category_counts": {
                c: categories.count(c)
                for c in (
                    "minibatch_transport_failure",
                    "adam_coordinate_transport_failure",
                    "both_descent",
                    "mixed_or_ambiguous",
                )
            },
            "signed_direction_contribution_totals": signed,
            "observed_change_total": observed,
            "first_order_total": first,
            "remainder_total": rem,
            "endpoint_change": end - start,
            "endpoint_reconciliation_residual": observed - (end - start),
        }
    training_observed = float(
        sum(s["training_accounting"]["observed_change"] for s in steps)
    )
    training_first = float(
        sum(s["training_accounting"]["first_order_contribution"] for s in steps)
    )
    tr_start = steps[0]["losses"]["pre"]["training_policy_ce"]
    tr_end = steps[-1]["losses"]["post"]["training_policy_ce"]
    report = {
        "schema": "seed476-direction-transport-supplement-evidence-v1",
        "post_execution": True,
        "forward_schedule": reg["forward_schedule"],
        "record_coverage": {
            "steps": len(steps),
            "required_steps": list(range(1, 17)),
            "unseen_occurrences": len(unseen),
            "training_occurrences": len(train_rows),
        },
        "steps": steps,
        "totals": totals,
        "training_totals": {
            "observed_change_total": training_observed,
            "first_order_total": training_first,
            "remainder_total": training_observed - training_first,
            "endpoint_change": tr_end - tr_start,
            "endpoint_reconciliation_residual": training_observed - (tr_end - tr_start),
        },
        "limitations": reg["limitation"],
    }
    return report, arrays


def _predict(model, inputs, targets, masks, batch_size):
    losses, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            sl = slice(start, start + batch_size)
            logits, _ = model(torch.from_numpy(inputs[sl].copy()))
            legal = torch.from_numpy(masks[sl].copy())
            logp = F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
            target = torch.from_numpy(targets[sl].copy())
            losses.append((-(target * logp).sum(1)).cpu().numpy().astype(np.float64))
            probabilities.append(logp.exp().cpu().numpy().astype(np.float32))
    return np.concatenate(losses), np.concatenate(probabilities), None


def publish(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    if reg.get("status") != "post_execution_supplement_frozen_before_reconstruction":
        raise ValueError("supplement_not_frozen")
    for name, digest in reg["source_closure_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"source_changed:{name}")
    for name, digest in reg["inputs_sha256"].items():
        if sha(root / INPUTS[name]) != digest:
            raise ValueError(f"input_changed:{name}")
    report, arrays = calculate(root)
    np.savez_compressed(out / "accounting.npz", **arrays)
    (out / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Seed476 post-execution transport supplement",
        "",
        "Retrospective exploratory read-only supplement. No causal or strength claim.",
        "",
        "## Category counts",
    ]
    for weighting, total in report["totals"].items():
        lines.append(f"- {weighting}: {total['category_counts']}")
    lines += [
        "",
        "## Endpoint reconciliation",
        f"- Training: observed {report['training_totals']['observed_change_total']:.12g}; endpoint {report['training_totals']['endpoint_change']:.12g}; residual {report['training_totals']['endpoint_reconciliation_residual']:.4g}",
    ]
    for weighting, total in report["totals"].items():
        lines.append(
            f"- {weighting}: observed {total['observed_change_total']:.12g}; endpoint {total['endpoint_change']:.12g}; residual {total['endpoint_reconciliation_residual']:.4g}"
        )
    lines += [
        "",
        "Original seed476 evidence remains byte-for-byte separate. The supplemental forward schedule and its limitations are recorded in the registration.",
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))
    receipt = {
        p.name: sha(p)
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "receipt.json"
    }
    receipt["status"] = "seed476_supplement_bound"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "publish"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    else:
        publish(args.root)
    print(f"seed476 supplement {args.action} complete")


if __name__ == "__main__":
    main()
