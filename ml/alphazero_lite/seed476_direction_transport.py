"""Retrospective seed476 full-population/minibatch direction transport."""

from __future__ import annotations

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
OUT = Path("docs/data/seed476-direction-transport")
SEED474 = Path("docs/data/seed474-fresh-value-removal")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
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
}
SOURCES = (
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path) -> dict[str, Any]:
    """Freeze all identities and numerical rules before calculating gradients."""
    root = root.resolve()
    out = root / OUT
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    reg474 = json.loads((root / INPUTS["seed474_registration"]).read_text())
    positions = np.asarray(reg474["minibatches"]["positions"], dtype=np.int64)
    occurrences = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh = occurrences[tr["fresh_source_mask"][occurrences].astype(bool)]
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    if (len(fresh), len(unseen), len(positions)) != (60015, 859, 8192):
        raise ValueError("authenticated_population_mismatch")
    out.mkdir(parents=True, exist_ok=True)
    registration = {
        "schema": "seed476-direction-transport-registration-v1",
        "status": "frozen_before_gradient_calculation",
        "interpretation": "retrospective and exploratory descriptive attribution; no causal or strength claim",
        "authorization": "read-only diagnostic; no new training arm, optimizer sweep, searches, games, runtime export, or promotion",
        "inputs_sha256": {key: sha(root / rel) for key, rel in INPUTS.items()},
        "source_closure_sha256": {rel: sha(root / rel) for rel in SOURCES},
        "authenticated_manifest_sha256": MANIFEST_SHA256,
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "population": {
            "training_selection": "expanded_replay_indices[train_positions], filter fresh_source_mask, preserve occurrence order, coefficients, targets, legal masks, and multiplicity",
            "training_occurrences": len(fresh),
            "training_policy_denominator": float(
                tr["policy_coefficients"][fresh].sum(dtype=np.float64)
            ),
            "minibatch_positions": positions.tolist(),
            "minibatches": 16,
            "batch_size": 512,
            "strict_unseen_selection": "evaluation source_ids==0, preserve all occurrences and occurrence-specific targets",
            "strict_unseen_occurrences": len(unseen),
            "strict_unseen_exact_inputs": len(
                set(ev["input_identity"][unseen].tolist())
            ),
        },
        "objective": "training policy CE = sum(c_i*CE_i)/sum(c_i) over all 60,015 fresh occurrences; actual minibatch uses each archived seed474 batch and its own sum(c_i) denominator; unseen CE uses own targets",
        "state_and_optimizer": "sixteen archived seed474 P pre-states; independently rebuild reset Adam per step (lr=0.0000625, betas=(0.9,0.999), eps=1e-8, weight_decay=0), clip global norm at 1, zero unused gradients, preserve float32 update/writeback",
        "directions": "negative full and minibatch raw gradients are scaled to the recorded displacement norm; add to float32 pre-state, then compute realized displacement in float64; Adam direction is archived float32 post minus pre in float64; no counterfactual state carry-forward",
        "weighting": {
            "exposure": "1/859 per occurrence",
            "equal_input": "1/(number of exact input identities * identity occurrence count) per occurrence",
        },
        "numeric_protocol": {
            "forward": "float32 eval mode",
            "gradient_accumulation": "ordered contiguous occurrence batches of 512; autograd float32, accumulate chunk gradients in float32",
            "geometry_and_losses": "float64",
            "threads": 1,
            "sign_tolerance": 1e-6,
            "comparison_rtol": 2e-6,
            "comparison_atol": 2e-8,
            "buffers": "must remain bitwise unchanged",
        },
        "categories": {
            "descent": "cosine <= -1e-6",
            "ascent": "cosine >= +1e-6",
            "minibatch_transport_failure": "full negative direction descends and minibatch negative direction ascends",
            "adam_coordinate_transport_failure": "minibatch negative direction descends and recorded Adam direction ascends",
            "both_descent": "all three descend",
            "otherwise": "mixed_or_ambiguous; zero norms reported separately",
            "interpretation": "descriptive only; no optimizer or sampling causality assigned",
        },
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    return registration


def _vector(grads: list[torch.Tensor | None], params: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [
            (torch.zeros_like(p) if g is None else g)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
            for p, g in zip(params, grads, strict=True)
        ]
    )


def _gradient(model, rows, tr, coefficients, denominator, fresh=None):
    params = list(model.parameters())
    accum = [torch.zeros_like(p) for p in params]
    for start in range(0, len(rows), 512):
        rr = rows[start : start + 512]
        x = torch.from_numpy(tr["inputs"][rr].copy())
        y = torch.from_numpy(tr["policy_targets"][rr].copy())
        mask = torch.from_numpy(tr["legal_masks"][rr].copy())
        logits, _ = model(x)
        ce = -(y * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)).sum(1)
        selected = (
            np.ones(len(rr), dtype=np.float64)
            if fresh is None
            else fresh[start : start + len(rr)]
        )
        weights = torch.from_numpy(
            (coefficients[start : start + len(rr)] * selected / denominator).astype(
                np.float32
            )
        )
        objective = (ce * weights).sum()
        grads = torch.autograd.grad(objective, params, allow_unused=True)
        for total, grad in zip(accum, grads, strict=True):
            if grad is not None:
                total.add_(grad.detach())
    return _vector(accum, params)


def _unseen_gradient(model, rows, ev, tr, weights):
    params = list(model.parameters())
    accum = [torch.zeros_like(p) for p in params]
    for start in range(0, len(rows), 128):
        sl = slice(start, start + 128)
        rr = rows[sl]
        logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
        target = torch.from_numpy(
            ev["policy_targets"][np.flatnonzero(ev["source_ids"] == 0)[sl]].copy()
        )
        mask = torch.from_numpy(tr["legal_masks"][rr].copy())
        ce = -(target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)).sum(
            1
        )
        grads = torch.autograd.grad(
            (ce * torch.from_numpy(weights[sl].astype(np.float32))).sum(),
            params,
            allow_unused=True,
        )
        for total, grad in zip(accum, grads, strict=True):
            if grad is not None:
                total.add_(grad.detach())
    return _vector(accum, params)


def calculate(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Calculate each independently reconstructed state without state carry-forward."""
    root = root.resolve()
    torch.set_num_threads(1)
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    registration = json.loads((root / OUT / "registration.json").read_text())
    positions = np.asarray(
        registration["population"]["minibatch_positions"], dtype=np.int64
    )
    all_rows = tr["expanded_replay_indices"][tr["train_positions"]]
    full_rows = all_rows[tr["fresh_source_mask"][all_rows].astype(bool)]
    coefficients = tr["policy_coefficients"][full_rows].astype(np.float64)
    denominator = coefficients.sum(dtype=np.float64)
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    ids = ev["input_identity"][unseen].tolist()
    counts = {key: ids.count(key) for key in set(ids)}
    unseen_rows = ev["compact_rows"][unseen]
    unseen_weights = {
        "exposure": np.full(len(unseen), 1 / len(unseen), dtype=np.float64),
        "equal_input": np.asarray(
            [1 / (len(counts) * counts[key]) for key in ids], dtype=np.float64
        ),
    }
    archive = {}
    steps = []
    with np.load(root / SEED474 / "P-steps.npz", allow_pickle=False) as saved:
        for step in range(16):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(model, root / INIT)
            params = list(model.parameters())
            with torch.no_grad():
                for i, p in enumerate(params):
                    p.copy_(torch.from_numpy(saved[f"{step:02d}_pre_{i:02d}"].copy()))
            model.eval()
            buffer_before = [b.detach().clone() for b in model.buffers()]
            full_gradient = _gradient(model, full_rows, tr, coefficients, denominator)
            # Actual minibatch gradient is reconstructed from its own targets, masks,
            # coefficients and denominator; compare it with the archived gradient.
            batch_positions = positions[step * 512 : (step + 1) * 512]
            batch_rows = all_rows[batch_positions]
            batch_coeff = tr["policy_coefficients"][batch_rows].astype(np.float64)
            batch_fresh = tr["fresh_source_mask"][batch_rows].astype(np.float64)
            batch_mass = float(np.sum(batch_coeff * batch_fresh, dtype=np.float64))
            batch_gradient = _gradient(
                model, batch_rows, tr, batch_coeff, batch_mass, batch_fresh
            )
            unseen_gradient = {
                name: _unseen_gradient(model, unseen_rows, ev, tr, weight)
                for name, weight in unseen_weights.items()
            }
            pre = np.concatenate(
                [
                    saved[f"{step:02d}_pre_{i:02d}"].reshape(-1).astype(np.float64)
                    for i in range(len(params))
                ]
            )
            post = np.concatenate(
                [
                    saved[f"{step:02d}_post_{i:02d}"].reshape(-1).astype(np.float64)
                    for i in range(len(params))
                ]
            )
            displacement = post - pre
            directions = {
                "full_negative": -full_gradient,
                "minibatch_negative": -batch_gradient,
                "recorded_adam": displacement,
            }
            norm = float(np.linalg.norm(displacement))
            matched = {}
            for name in ("full_negative", "minibatch_negative"):
                raw = directions[name]
                raw_norm = float(np.linalg.norm(raw))
                if raw_norm == 0:
                    candidate = pre.copy()
                else:
                    scaled = (raw * (norm / raw_norm)).astype(np.float32)
                    candidate = np.concatenate(
                        [
                            (
                                torch.from_numpy(
                                    saved[f"{step:02d}_pre_{i:02d}"].copy()
                                ).reshape(-1)
                                + torch.from_numpy(
                                    scaled[offset : offset + p.numel()]
                                    .reshape(-1)
                                    .copy()
                                )
                            )
                            .numpy()
                            .reshape(-1)
                            .astype(np.float64)
                            for i, p in enumerate(params)
                            for offset in [sum(q.numel() for q in params[:i])]
                        ]
                    )
                matched[name] = candidate - pre
            for name, value in [
                ("full_gradient", full_gradient),
                ("minibatch_gradient", batch_gradient),
                ("adam_displacement", displacement),
                *[
                    (f"unseen_{key}_gradient", value)
                    for key, value in unseen_gradient.items()
                ],
                *[(f"matched_{key}", value) for key, value in matched.items()],
            ]:
                archive[f"{step:02d}_{name}"] = value
            norms = {
                name: float(np.linalg.norm(value)) for name, value in directions.items()
            }
            geometry = {}
            for weighting, ug in unseen_gradient.items():
                ug_norm = float(np.linalg.norm(ug))
                scores = {}
                for name, direction in directions.items():
                    dnorm = float(np.linalg.norm(direction))
                    dot = float(np.dot(ug, direction))
                    cosine = dot / (ug_norm * dnorm) if ug_norm and dnorm else 0.0
                    scores[name] = {
                        "dot": dot,
                        "cosine": cosine,
                        "sign": "zero_norm"
                        if not ug_norm or not dnorm
                        else "descent"
                        if cosine <= -1e-6
                        else "ascent"
                        if cosine >= 1e-6
                        else "ambiguous",
                    }
                signs = [
                    scores[k]["sign"]
                    for k in ("full_negative", "minibatch_negative", "recorded_adam")
                ]
                category = (
                    "minibatch_transport_failure"
                    if signs[0] == "descent" and signs[1] == "ascent"
                    else "adam_coordinate_transport_failure"
                    if signs[1] == "descent" and signs[2] == "ascent"
                    else "both_descent"
                    if signs == ["descent"] * 3
                    else "mixed_or_ambiguous"
                )
                geometry[weighting] = {
                    "directions": scores,
                    "category": category,
                    "signed_contributions": {k: scores[k]["dot"] for k in scores},
                }
            steps.append(
                {
                    "step": step + 1,
                    "direction_norms": norms,
                    "matched_residual_norms": {
                        k: float(np.linalg.norm(v)) - norm for k, v in matched.items()
                    },
                    "unseen": geometry,
                }
            )
            if any(
                not torch.equal(a, b)
                for a, b in zip(buffer_before, model.buffers(), strict=True)
            ):
                raise ValueError("model_buffer_mutated")
    report = {
        "schema": "seed476-direction-transport-evidence-v1",
        "population": registration["population"],
        "steps": steps,
        "summary": {},
    }
    for weighting in unseen_weights:
        categories = [step["unseen"][weighting]["category"] for step in steps]
        report["summary"][weighting] = {
            name: categories.count(name)
            for name in (
                "minibatch_transport_failure",
                "adam_coordinate_transport_failure",
                "both_descent",
                "mixed_or_ambiguous",
            )
        }
    return report, archive
