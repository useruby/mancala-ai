"""Seed478 finite-step evaluation of seed476 local policy directions."""

from __future__ import annotations

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
OUT = Path("docs/data/seed478-finite-step-evaluation")
BASE = Path("docs/data/seed476-direction-transport")
SUP = Path("docs/data/seed476-direction-transport-supplement-v2")
HIST = Path("docs/data/seed474-fresh-value-removal")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
SCALES = (1.0, 0.5, 0.25, 0.125)
SOURCE_PATHS = (
    "ml/alphazero_lite/seed478_finite_step_evaluation.py",
    "ml/alphazero_lite/publish_seed478_finite_step_evaluation.py",
    "ml/alphazero_lite/verify_seed478_finite_step_evaluation.py",
    "ml/alphazero_lite/test_seed478_finite_step_evaluation.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
)
INPUT_PATHS = {
    "seed466_manifest": Path("docs/data/seed466-replay-input-bundle/manifest.json"),
    "seed466_training": Path("docs/data/seed466-replay-input-bundle/training.npz"),
    "seed466_evaluation": Path("docs/data/seed466-replay-input-bundle/evaluation.npz"),
    "initializer": INIT,
    "seed474_registration": HIST / "registration.json",
    "seed474_receipt": HIST / "receipt.json",
    "seed474_P_steps": HIST / "P-steps.npz",
    "seed474_P_trajectory": HIST / "P-trajectory.json",
    "seed474_endpoint": HIST / "endpoint-evidence.json",
    "seed476_registration": BASE / "registration.json",
    "seed476_receipt": BASE / "receipt.json",
    "seed476_evidence": BASE / "evidence.json",
    "seed476_directions": BASE / "directions.npz",
    "seed476_supplement_registration": SUP / "registration.json",
    "seed476_supplement_receipt": SUP / "receipt.json",
    "seed476_supplement_evidence": SUP / "evidence.json",
    "seed476_supplement_accounting": SUP / "accounting.npz",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat(values):
    return np.concatenate(
        [v.detach().cpu().numpy().reshape(-1).astype(np.float64) for v in values]
    )


def _prob_loss(model, x, y, legal, batch):
    probabilities, losses = [], []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            sl = slice(start, start + batch)
            logits, _ = model(torch.from_numpy(x[sl].copy()))
            mask = torch.from_numpy(legal[sl].copy())
            logp = F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
            target = torch.from_numpy(y[sl].copy())
            probabilities.append(logp.exp().cpu().numpy().astype(np.float32))
            losses.append((-(target * logp).sum(1)).cpu().numpy().astype(np.float64))
    return np.concatenate(probabilities), np.concatenate(losses)


def _build_registration(root: Path) -> dict:
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    ids = ev["input_identity"][unseen].tolist()
    groups = {value: ids.count(value) for value in set(ids)}
    return {
        "schema": "seed478-finite-step-evaluation-registration-v1",
        "status": "frozen_before_trial_evaluation",
        "interpretation": "retrospective local diagnostic only; no trajectory, causal, strength, promotion, or training authorization",
        "canonical_seed476_verifier": "passed in .venv-azlite using frozen PyTorch 2.13.0+cu130 / NumPy 2.5.2 environment",
        "seed477_semantic_verifier": "passed; semantic correction only, not numerical replay",
        "inputs_sha256": {k: sha(root / v) for k, v in INPUT_PATHS.items()},
        "sources_sha256": {v: sha(root / v) for v in SOURCE_PATHS},
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "population": {
            "training": "expanded_replay_indices[train_positions], preserve each fresh occurrence, own target, legal mask, coefficient, and multiplicity",
            "training_occurrences": 60015,
            "training_weight_denominator": float(
                tr["policy_coefficients"][
                    tr["expanded_replay_indices"][tr["train_positions"]][
                        tr["fresh_source_mask"][
                            tr["expanded_replay_indices"][tr["train_positions"]].astype(
                                int
                            )
                        ].astype(bool)
                    ]
                ].sum(dtype=np.float64)
            ),
            "unseen_selection": "evaluation source_ids==0; occurrence order and own target preserved",
            "unseen_occurrences": len(unseen),
            "unseen_exact_inputs": len(groups),
        },
        "directions": "archived reset-Adam float32 post minus pre; negative full fresh training-policy gradient; negative archived minibatch policy gradient using own coefficient/fresh denominator; latter two norm-matched to Adam displacement",
        "scales": list(SCALES),
        "trial_independence": "all trials start from their own archived P pre-state; materialize endpoint in float32; no state carry-forward",
        "forward": "float32 eval mode, contiguous occurrence batches: training 512, unseen 128; legal masked log-softmax; per-occurrence probabilities and CE",
        "gradient_accumulation": "float32 autograd and ordered contiguous 512-occurrence accumulation; full denominator sum of fresh occurrence coefficients; minibatch own fresh coefficient mass",
        "weighting": {
            "exposure": "1/859 per occurrence",
            "equal_input": "1/(851 * identity occurrence count) per occurrence",
        },
        "geometry": "float64 realized displacement, first_order=pre-state loss gradient dot this endpoint's realized displacement; remainder=endpoint CE change-first_order",
        "dtypes": {
            "forward": "float32",
            "gradient": "float32",
            "writeback": "float32",
            "reductions_and_geometry": "float64",
        },
        "tolerances": {"rtol": 2e-6, "atol": 2e-8, "classification": 1e-6},
        "reporting": "all endpoint trials retained; counts classify change below -1e-6, above +1e-6, else ambiguous; report mean and median; no scale selection or counterfactual effect summation",
    }


def freeze(root: Path) -> dict:
    root = root.resolve()
    (root / OUT).mkdir(parents=True, exist_ok=True)
    registration = _build_registration(root)
    (root / OUT / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    return registration


def calculate(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Calculate registered local trials. Never advances a trial state."""
    root = root.resolve()
    torch.set_num_threads(1)
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    rawreg = json.loads((root / BASE / "registration.json").read_text())
    positions = np.asarray(rawreg["population"]["minibatch_positions"], dtype=np.int64)
    occurrences = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh = occurrences[tr["fresh_source_mask"][occurrences].astype(bool)]
    coeff = tr["policy_coefficients"][fresh].astype(np.float64)
    denom = float(coeff.sum(dtype=np.float64))
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    urows = ev["compact_rows"][unseen]
    uy = ev["policy_targets"][unseen]
    umask = tr["legal_masks"][urows]
    ids = ev["input_identity"][unseen].tolist()
    groups = {key: ids.count(key) for key in set(ids)}
    uw = {
        "exposure": np.full(len(unseen), 1 / len(unseen), np.float64),
        "equal_input": np.asarray(
            [1 / (len(groups) * groups[x]) for x in ids], np.float64
        ),
    }
    tx, ty, tm = (
        tr["inputs"][fresh],
        tr["policy_targets"][fresh],
        tr["legal_masks"][fresh],
    )
    tw = coeff / denom
    records, arrays = [], {}
    arrays["unseen_targets"] = uy.copy()
    arrays["unseen_exposure_weights"] = uw["exposure"].copy()
    arrays["unseen_equal_input_weights"] = uw["equal_input"].copy()
    saved = np.load(root / HIST / "P-steps.npz", allow_pickle=False)
    for s in range(16):
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, root / INIT)
        params = list(model.parameters())
        preparts = [
            torch.from_numpy(saved[f"{s:02d}_pre_{i:02d}"].copy())
            for i in range(len(params))
        ]
        postparts = [
            torch.from_numpy(saved[f"{s:02d}_post_{i:02d}"].copy())
            for i in range(len(params))
        ]
        with torch.no_grad():
            for p, value in zip(params, preparts, strict=True):
                p.copy_(value)
        model.eval()
        preflat, postflat = _flat(preparts), _flat(postparts)
        adam = postflat - preflat
        full_acc = [torch.zeros_like(p) for p in params]
        for start in range(0, len(fresh), 512):
            sl = slice(start, start + 512)
            logits, _ = model(torch.from_numpy(tx[sl].copy()))
            logp = F.log_softmax(
                logits.masked_fill(torch.from_numpy(tm[sl].copy()) == 0, -1e9), 1
            )
            ce = -(torch.from_numpy(ty[sl].copy()) * logp).sum(1)
            grad = torch.autograd.grad(
                (ce * torch.from_numpy(tw[sl].astype(np.float32))).sum(),
                params,
                allow_unused=True,
            )
            for dst, src in zip(full_acc, grad, strict=True):
                if src is not None:
                    dst.add_(src.detach())
        full = -_flat(
            [
                torch.zeros_like(p) if g is None else g
                for p, g in zip(params, full_acc, strict=True)
            ]
        )
        batch_rows = occurrences[positions[s * 512 : (s + 1) * 512]]
        bc = tr["policy_coefficients"][batch_rows].astype(np.float64)
        bf = tr["fresh_source_mask"][batch_rows].astype(np.float64)
        bm = float(np.sum(bc * bf, dtype=np.float64))
        logits, _ = model(torch.from_numpy(tr["inputs"][batch_rows].copy()))
        logp = F.log_softmax(
            logits.masked_fill(
                torch.from_numpy(tr["legal_masks"][batch_rows].copy()) == 0, -1e9
            ),
            1,
        )
        bce = -(torch.from_numpy(tr["policy_targets"][batch_rows].copy()) * logp).sum(1)
        bg = torch.autograd.grad(
            (bce * torch.from_numpy((bc * bf / bm).astype(np.float32))).sum(),
            params,
            allow_unused=True,
        )
        minibatch = -_flat(
            [
                torch.zeros_like(p) if g is None else g
                for p, g in zip(params, bg, strict=True)
            ]
        )
        optgrad = np.concatenate(
            [
                saved[f"{s:02d}_gradient_{i:02d}"].reshape(-1).astype(np.float64)
                for i in range(len(params))
            ]
        )
        historical = np.load(root / BASE / "directions.npz", allow_pickle=False)
        comparisons = {}
        for label, value, hkey in (
            ("full_negative", -full, "full_gradient"),
            ("minibatch_negative", -minibatch, "minibatch_gradient"),
        ):
            h = historical[f"{s:02d}_{hkey}"].reshape(-1).astype(np.float64)
            comparisons[label] = {
                "max_abs": float(np.max(np.abs(value - h))),
                "l2": float(np.linalg.norm(value - h)),
            }
        comparisons["recorded_adam_gradient"] = {
            "max_abs": float(
                np.max(
                    np.abs(
                        optgrad
                        - np.concatenate(
                            [
                                saved[f"{s:02d}_gradient_{i:02d}"].reshape(-1)
                                for i in range(len(params))
                            ]
                        )
                    )
                )
            ),
            "l2": 0.0,
        }
        if not np.array_equal(
            _flat(postparts),
            postflat := np.concatenate(
                [
                    saved[f"{s:02d}_post_{i:02d}"].reshape(-1).astype(np.float64)
                    for i in range(len(params))
                ]
            ),
        ):
            raise ValueError("recorded_adam_post_state_mismatch")
        records.append(
            _trials_for_step(
                s,
                model,
                params,
                preparts,
                postparts,
                adam,
                full,
                minibatch,
                tr["inputs"][urows],
                tx,
                ty,
                tm,
                tw,
                uy,
                umask,
                uw,
                comparisons,
                arrays,
            )
        )
    report = {
        "schema": "seed478-finite-step-evaluation-evidence-v1",
        "scales": list(SCALES),
        "steps": records,
        "curves": _curves(records),
    }
    return report, arrays


def _trials_for_step(
    s,
    model,
    params,
    preparts,
    postparts,
    adam,
    full,
    minibatch,
    ux,
    tx,
    ty,
    tm,
    tw,
    uy,
    umask,
    uw,
    comparisons,
    arrays,
):
    preflat = _flat(preparts)
    norm = float(np.linalg.norm(adam))
    trials = []
    # Pre-state losses and gradients, computed at this archived local origin.
    preup, preul = _prob_loss(model, ux, uy, umask, 128)
    pretp, pret_loss = _prob_loss(model, tx, ty, tm, 512)
    held_grad = {}
    for weighting, weights in uw.items():
        acc = [torch.zeros_like(p) for p in params]
        for start in range(0, len(ux), 128):
            sl = slice(start, start + 128)
            logits, _ = model(torch.from_numpy(ux[sl].copy()))
            lp = F.log_softmax(
                logits.masked_fill(torch.from_numpy(umask[sl].copy()) == 0, -1e9), 1
            )
            ce = -(torch.from_numpy(uy[sl].copy()) * lp).sum(1)
            gs = torch.autograd.grad(
                (ce * torch.from_numpy(weights[sl].astype(np.float32))).sum(),
                params,
                allow_unused=True,
            )
            for dst, grad in zip(acc, gs, strict=True):
                if grad is not None:
                    dst.add_(grad.detach())
        held_grad[weighting] = _flat(
            [
                torch.zeros_like(p) if g is None else g
                for p, g in zip(params, acc, strict=True)
            ]
        )
    train_grad = full * -1.0
    for label, raw in (
        ("recorded_adam", adam),
        ("full_negative", full),
        ("minibatch_negative", minibatch),
    ):
        raw_norm = float(np.linalg.norm(raw))
        unit = (
            raw.copy()
            if label == "recorded_adam"
            else (np.zeros_like(raw) if raw_norm == 0 else raw * norm / raw_norm)
        )
        for scale in SCALES:
            # Per-parameter float32 proposal and writeback is explicit.
            parts = []
            cursor = 0
            for i, (p, base) in enumerate(zip(params, preparts, strict=True)):
                n = p.numel()
                if label == "recorded_adam":
                    if scale == 1:
                        part = postparts[i].clone()
                    else:
                        proposal = (
                            (unit[cursor : cursor + n] * scale)
                            .astype(np.float32)
                            .reshape(base.shape)
                        )
                        part = base + torch.from_numpy(proposal.copy())
                else:
                    delta = (
                        (unit[cursor : cursor + n] * scale)
                        .astype(np.float32)
                        .reshape(base.shape)
                    )
                    part = base + torch.from_numpy(delta.copy())
                parts.append(part)
                cursor += n
            with torch.no_grad():
                for p, value in zip(params, parts, strict=True):
                    p.copy_(value)
            realized = _flat(params) - preflat
            up, ul = _prob_loss(model, ux, uy, umask, 128)
            tp, tl = _prob_loss(model, tx, ty, tm, 512)
            index = f"{s:02d}_{label}_{scale:g}"
            arrays[f"{index}_unseen_probabilities"] = up
            arrays[f"{index}_unseen_occurrence_losses"] = ul
            arrays[f"{index}_training_occurrence_losses"] = tl
            arrays[f"{index}_state"] = _flat(params)
            weighting_records = {}
            for weighting, weights in uw.items():
                change = float(np.dot(ul - preul, weights))
                first = float(np.dot(held_grad[weighting], realized))
                weighting_records[weighting] = {
                    "pre_ce": float(np.dot(preul, weights)),
                    "post_ce": float(np.dot(ul, weights)),
                    "change": change,
                    "first_order": first,
                    "remainder": change - first,
                }
            train_change = float(np.dot(tl - pret_loss, tw))
            train_first = float(np.dot(train_grad, realized))
            trials.append(
                {
                    "direction": label,
                    "scale": scale,
                    "state_sha256": hashlib.sha256(
                        np.asarray(_flat(params), dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "realized_norm": float(np.linalg.norm(realized)),
                    "training": {
                        "pre_ce": float(np.dot(pret_loss, tw)),
                        "post_ce": float(np.dot(tl, tw)),
                        "change": train_change,
                        "first_order": train_first,
                        "remainder": train_change - train_first,
                    },
                    "weightings": weighting_records,
                    "comparison": comparisons.get(label, {}),
                    "array_prefix": index,
                }
            )
    return {"step": s + 1, "trials": trials}


def _curves(steps):
    result = {}
    for direction in ("recorded_adam", "full_negative", "minibatch_negative"):
        for scale in SCALES:
            for weighting in ("exposure", "equal_input"):
                vals = [
                    t["weightings"][weighting]["change"]
                    for step in steps
                    for t in step["trials"]
                    if t["direction"] == direction and t["scale"] == scale
                ]
                result[f"{direction}/{scale}/{weighting}"] = (
                    {
                        "improvement": sum(x < -1e-6 for x in vals),
                        "regression": sum(x > 1e-6 for x in vals),
                        "ambiguous": sum(abs(x) <= 1e-6 for x in vals),
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                    }
                    if vals
                    else {
                        "improvement": 0,
                        "regression": 0,
                        "ambiguous": 0,
                        "mean": None,
                        "median": None,
                    }
                )
    return result
