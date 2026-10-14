"""Retrospective seed475 fresh-policy gradient attribution by active stones."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
)
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed475-policy-population-attribution")
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
    "ml/alphazero_lite/seed475_policy_population_attribution.py",
    "ml/alphazero_lite/publish_seed475_policy_population_attribution.py",
    "ml/alphazero_lite/verify_seed475_policy_population_attribution.py",
    "ml/alphazero_lite/test_seed475_policy_population_attribution.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/verify_seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/verify_seed474_fresh_value_removal.py",
    "ml/alphazero_lite/seed474_verification_supplement.py",
)
BUCKETS = ("<=16", "17-32", ">32")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat(parameters: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in parameters]
    )


def _state(root: Path, checkpoint: str) -> tuple[train.PolicyValueNet, dict[str, Any]]:
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    metadata: dict[str, Any] = {
        "checkpoint": "original_initializer",
        "initializer_sha256": sha(root / INIT),
    }
    if checkpoint == "seed474_P":
        with np.load(root / SEED474 / "P-steps.npz", allow_pickle=False) as archive:
            with torch.no_grad():
                for i, parameter in enumerate(model.parameters()):
                    parameter.copy_(
                        torch.from_numpy(archive[f"15_post_{i:02d}"].copy())
                    )
        metadata = {
            "checkpoint": "seed474_P_archived_endpoint",
            "seed474_P_steps_sha256": sha(root / SEED474 / "P-steps.npz"),
        }
    return model, metadata


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _geometry(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    names = list(vectors)
    norms = {name: _norm(vectors[name]) for name in names}
    dot = {a: {b: float(np.dot(vectors[a], vectors[b])) for b in names} for a in names}
    cosine = {
        a: {
            b: (dot[a][b] / (norms[a] * norms[b]) if norms[a] and norms[b] else 0.0)
            for b in names
        }
        for a in names
    }
    return {"norms": norms, "dot_products": dot, "cosines": cosine}


def calculate(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = root.resolve()
    torch.set_num_threads(1)
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    train_rows = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh = tr["fresh_source_mask"][train_rows]
    rows = train_rows[fresh]
    stones = np.asarray(
        [active_pit_stones_from_encoded_state(x) for x in tr["inputs"][rows]],
        dtype=np.int64,
    )
    buckets = np.select([stones <= 16, stones <= 32], [0, 1], default=2)
    coeff = tr["policy_coefficients"][rows].astype(np.float64)
    denominator = float(np.sum(coeff, dtype=np.float64))
    if denominator <= 0:
        raise ValueError("empty_fresh_training_policy_mass")
    # Independently retain strict-unseen evaluation occurrences and each occurrence's target.
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    if len(unseen) != 859:
        raise ValueError("strict_unseen_population_mismatch")
    identities = ev["input_identity"][unseen].tolist()
    group_counts = {
        identity: identities.count(identity) for identity in set(identities)
    }
    weights = {
        "exposure": np.full(len(unseen), 1 / len(unseen), dtype=np.float64),
        "equal_input": np.asarray(
            [1 / (len(group_counts) * group_counts[i]) for i in identities],
            dtype=np.float64,
        ),
    }
    metrics: dict[str, Any] = {
        "training_occurrences": len(rows),
        "unseen_occurrences": len(unseen),
        "unseen_exact_inputs": len(group_counts),
        "common_policy_denominator": denominator,
        "checkpoints": {},
    }
    vectors: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {
        "training_stones": stones,
        "training_bucket": buckets,
        "training_rows": rows,
        "unseen_rows": unseen,
    }
    for checkpoint in ("initializer", "seed474_P"):
        model, checkpoint_id = _state(root, checkpoint)
        model.eval()
        params = list(model.parameters())
        buffers_before = [b.detach().clone() for b in model.buffers()]
        names = ["full", *[f"bucket_{b}" for b in BUCKETS]]
        accum = {name: [torch.zeros_like(p) for p in params] for name in names}
        numerators = {name: 0.0 for name in names}
        # Fixed contiguous 512-row chunks; occurrence order and multiplicity are untouched.
        for start in range(0, len(rows), 512):
            ix = np.arange(start, min(start + 512, len(rows)))
            data_rows = rows[ix]
            xb = torch.from_numpy(tr["inputs"][data_rows].copy())
            target = torch.from_numpy(tr["policy_targets"][data_rows].copy())
            mask = torch.from_numpy(tr["legal_masks"][data_rows].copy())
            weight = torch.from_numpy((coeff[ix] / denominator).astype(np.float32))
            logits, _ = model(xb)
            ce = -(
                target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
            ).sum(1)
            for bucket_index, bucket in enumerate(BUCKETS):
                selected = torch.from_numpy((buckets[ix] == bucket_index))
                objective = (ce * weight * selected).sum()
                numerators[f"bucket_{bucket}"] += float(
                    (
                        ce.detach().double() * torch.from_numpy(coeff[ix]) * selected
                    ).sum()
                )
                if objective.requires_grad and bool(selected.any()):
                    grads = torch.autograd.grad(
                        objective, params, retain_graph=True, allow_unused=True
                    )
                    for acc, grad in zip(accum[f"bucket_{bucket}"], grads, strict=True):
                        if grad is not None:
                            acc.add_(grad.detach())
            objective = (ce * weight).sum()
            numerators["full"] += float(
                (ce.detach().double() * torch.from_numpy(coeff[ix])).sum()
            )
            grads = torch.autograd.grad(objective, params, allow_unused=True)
            for acc, grad in zip(accum["full"], grads, strict=True):
                if grad is not None:
                    acc.add_(grad.detach())
        grad_vectors = {name: _flat(value) for name, value in accum.items()}
        for name, vector in grad_vectors.items():
            vectors[f"{checkpoint}:{name}"] = vector
            arrays[f"{checkpoint}_{name}_gradient"] = vector
        # Held-out population gradients use their own targets and occurrence-specific weights.
        urows = ev["compact_rows"][unseen]
        unseen_vectors: dict[str, np.ndarray] = {}
        for weighting, occurrence_weights in weights.items():
            acc = [torch.zeros_like(p) for p in params]
            for start in range(0, len(unseen), 128):
                sl = slice(start, min(start + 128, len(unseen)))
                rr = urows[sl]
                logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                target = torch.from_numpy(ev["policy_targets"][unseen[sl]].copy())
                mask = torch.from_numpy(tr["legal_masks"][rr].copy())
                ce = -(
                    target * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
                ).sum(1)
                objective = (
                    ce * torch.from_numpy(occurrence_weights[sl].astype(np.float32))
                ).sum()
                grads = torch.autograd.grad(objective, params, allow_unused=True)
                for a, g in zip(acc, grads, strict=True):
                    if g is not None:
                        a.add_(g.detach())
            v = _flat(acc)
            unseen_vectors[weighting] = v
            vectors[f"{checkpoint}:unseen_{weighting}"] = v
            arrays[f"{checkpoint}_unseen_{weighting}_gradient"] = v
        checkpoint_metrics: dict[str, Any] = {
            **checkpoint_id,
            "training": {},
            "unseen": {},
        }
        for bucket_index, bucket in enumerate(BUCKETS):
            selection = buckets == bucket_index
            name = f"bucket_{bucket}"
            g = grad_vectors[name]
            checkpoint_metrics["training"][bucket] = {
                "occurrences": int(selection.sum()),
                "exact_inputs": len(
                    {tr["inputs"][row].tobytes() for row in rows[selection]}
                ),
                "coefficient_mass": float(coeff[selection].sum(dtype=np.float64)),
                "loss_numerator": numerators[name],
                "gradient_norm": _norm(g),
            }
        checkpoint_metrics["training"]["full"] = {
            "occurrences": int(len(rows)),
            "coefficient_mass": denominator,
            "loss_numerator": numerators["full"],
            "gradient_norm": _norm(grad_vectors["full"]),
            "component_sum_residual_norm": _norm(
                grad_vectors["full"]
                - sum(
                    (grad_vectors[f"bucket_{b}"] for b in BUCKETS),
                    np.zeros_like(grad_vectors["full"]),
                )
            ),
        }
        for weighting, ug in unseen_vectors.items():
            neg = -grad_vectors["full"]
            cos = (
                float(np.dot(ug, neg) / (_norm(ug) * _norm(neg)))
                if _norm(ug) and _norm(neg)
                else 0.0
            )
            classification = (
                "full_population_opposition"
                if cos >= 1e-6
                else "full_population_descent"
                if cos <= -1e-6
                else "ambiguous_or_zero"
            )
            contrib = {
                b: float(np.dot(ug, -grad_vectors[f"bucket_{b}"])) for b in BUCKETS
            }
            checkpoint_metrics["unseen"][weighting] = {
                "occurrences": len(unseen),
                "exact_inputs": len(group_counts),
                "gradient_norm": _norm(ug),
                "bucket_negative_gradient_contributions": contrib,
                "full_negative_gradient_contribution": float(np.dot(ug, neg)),
                "cosine_with_full_negative_training_gradient": cos,
                "classification": classification,
                "additive_contribution_residual": float(
                    np.dot(ug, neg) - sum(contrib.values())
                ),
            }
        all_geom = {
            "full_training": grad_vectors["full"],
            "training_<=16": grad_vectors["bucket_<=16"],
            "training_17-32": grad_vectors["bucket_17-32"],
            "training_>32": grad_vectors["bucket_>32"],
            **{f"unseen_{k}": v for k, v in unseen_vectors.items()},
        }
        checkpoint_metrics["pairwise_geometry"] = _geometry(all_geom)
        metrics["checkpoints"][checkpoint] = checkpoint_metrics
        if any(
            not torch.equal(a, b)
            for a, b in zip(buffers_before, model.buffers(), strict=True)
        ):
            raise ValueError("model_buffer_mutated")
    return metrics, arrays
