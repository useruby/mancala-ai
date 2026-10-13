"""Independent read-only reconstruction of the seed468 transport publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)
from ml.alphazero_lite.verify_seed466_replay_bundle import verify as reconstruct_bundle

INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
OUT = Path("docs/data/seed468-first-step-optimizer-direction-transport")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gradient(model: torch.nn.Module, loss: torch.Tensor) -> np.ndarray:
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return np.concatenate(
        [
            (torch.zeros_like(p) if g is None else g).detach().cpu().numpy().reshape(-1)
            for p, g in zip(params, grads, strict=True)
        ]
    ).astype(np.float32)


def _adam(g: np.ndarray) -> tuple[np.ndarray, bool]:
    value = torch.from_numpy(g.copy())
    norm = torch.linalg.vector_norm(value)
    clipped = bool(norm > 1)
    if clipped:
        value = value / norm
    # Explicit first step from the registered PyTorch Adam equation.
    update = -0.001 * value / (value.abs() + 1e-8)
    return update.numpy(), clipped


def verify(root: Path) -> None:
    output = root / OUT
    registration = json.loads((output / "registration.json").read_text())
    source_bindings = {
        registration["inputs"]["diagnostic_source"]: registration["inputs"][
            "diagnostic_source_sha256"
        ],
        registration["inputs"]["verifier_source"]: registration["inputs"][
            "verifier_source_sha256"
        ],
        registration["inputs"]["tests_source"]: registration["inputs"][
            "tests_source_sha256"
        ],
        registration["inputs"]["authenticated_reader"]: registration["inputs"][
            "reader_manifest_sha256"
        ],
        registration["inputs"]["model_source"]: registration["inputs"][
            "model_source_sha256"
        ],
        "ml/alphazero_lite/seed466_replay_bundle.py": registration["inputs"][
            "bundle_reader_sha256"
        ],
        "ml/alphazero_lite/verify_seed466_replay_bundle.py": registration["inputs"][
            "independent_bundle_reconstruction_sha256"
        ],
        **registration["project_source_closure_sha256"],
    }
    for relative, expected_hash in source_bindings.items():
        if digest(root / relative) != expected_hash:
            raise ValueError(f"frozen_source_identity_mismatch:{relative}")
    initial = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    bundle = root / "docs/data/seed466-replay-input-bundle"
    if digest(initial) != INIT_SHA:
        raise ValueError("initializer_identity_mismatch")
    arrays = read_authenticated_bundle(bundle, MANIFEST_SHA256)
    reconstruct_bundle(bundle, root)
    tr, ev = arrays["training"], arrays["evaluation"]
    em = json.loads((output / "evaluation-manifest.json").read_text())
    selected = np.flatnonzero(ev["source_ids"] == 0)
    target_ids = ev["input_identity"][selected].tolist()
    compact = ev["compact_rows"][selected]
    expected_selection = {
        "selection_indices": selected.tolist(),
        "identities": target_ids,
        "source_ids": ev["source_ids"][selected].astype(np.uint8).tolist(),
        "raw_lines": ev["raw_lines"][selected].astype(np.int32).tolist(),
        "compact_rows": compact.astype(np.int32).tolist(),
        "policy_targets": ev["policy_targets"][selected].tolist(),
        "legal_masks": tr["legal_masks"][compact].tolist(),
    }
    if any(em[k] != value for k, value in expected_selection.items()):
        raise ValueError("evaluation_manifest_semantic_mismatch")
    if len(selected) != 859 or len(set(target_ids)) != 851:
        raise ValueError("evaluation_cohort_count_mismatch")
    count_by_id = {i: target_ids.count(i) for i in set(target_ids)}
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, initial)
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    evalx = tr["inputs"][compact]
    evalp = ev["policy_targets"][selected]
    evalm = tr["legal_masks"][compact]
    eval_grads = {}
    for weighting in ("exposure_weighted", "equal_input"):
        logits, _ = model(torch.from_numpy(evalx.copy()))
        ce = -(
            torch.from_numpy(evalp.copy())
            * F.log_softmax(
                logits.masked_fill(torch.from_numpy(evalm.copy()) == 0, -1e9), 1
            )
        ).sum(1)
        w = np.asarray(
            [
                1 / 859
                if weighting == "exposure_weighted"
                else 1 / (851 * count_by_id[i])
                for i in target_ids
            ],
            dtype=np.float32,
        )
        eval_grads[weighting] = _gradient(model, (ce * torch.from_numpy(w)).sum())
    report = json.loads((output / "diagnostic.json").read_text())
    with np.load(output / "vectors.npz", allow_pickle=False) as archive:
        vectors = {k: archive[k] for k in archive.files}
    expected = []
    for b in range(16):
        pos = tr["first16_minibatch_positions"][512 * b : 512 * (b + 1)]
        rows = tr["expanded_replay_indices"][tr["train_positions"][pos]]
        fresh = tr["fresh_source_mask"][rows].astype(np.float32)
        coeff = tr["policy_coefficients"][rows] * fresh
        x = torch.from_numpy(tr["inputs"][rows].copy())
        p = torch.from_numpy(tr["policy_targets"][rows].copy())
        mask = torch.from_numpy(tr["legal_masks"][rows].copy())
        value_target = torch.from_numpy(tr["value_targets"][rows].reshape(-1).copy())
        for obj in ("P", "J"):
            model.load_state_dict(start)
            logits, value = model(x)
            ce = -(p * F.log_softmax(logits.masked_fill(mask == 0, -1e9), 1)).sum(1)
            mass = float(coeff.sum())
            if mass <= 0:
                raise ValueError(f"empty_fresh_policy_mass:batch{b}")
            loss = (ce * torch.from_numpy(coeff)).sum() / mass
            if obj == "J":
                loss = (
                    loss
                    + 0.3
                    * (
                        F.smooth_l1_loss(
                            value.reshape(-1), value_target, reduction="none"
                        )
                        * torch.from_numpy(fresh)
                    ).sum()
                    / 512
                )
            g = _gradient(model, loss)
            gn = torch.from_numpy(g.copy())
            norm = torch.linalg.vector_norm(gn)
            if norm > 1:
                gn /= norm
            adam = -0.001 * gn / (gn.abs() + 1e-8)
            plain = (
                -g
                * (
                    float(np.linalg.norm(adam.numpy().astype(np.float64)))
                    / float(np.linalg.norm(g.astype(np.float64)))
                )
                if np.linalg.norm(g)
                else np.zeros_like(g)
            )
            for direction, displacement in (
                ("adam", adam.numpy()),
                ("negative_gradient", plain),
            ):
                key = f"batch{b:02d}_{obj}_{direction}"
                if key not in vectors or not np.allclose(
                    vectors[key], displacement, rtol=2e-6, atol=2e-8
                ):
                    raise ValueError(f"displacement_reconstruction_mismatch:{key}")
                for weighting, ug in eval_grads.items():
                    gd, dd = ug.astype(np.float64), displacement.astype(np.float64)
                    dot = float(np.dot(gd, dd))
                    gnorm = float(np.linalg.norm(gd))
                    dnorm = float(np.linalg.norm(dd))
                    expected.append(
                        {
                            "batch": b,
                            "objective": obj,
                            "direction": direction,
                            "weighting": weighting,
                            "dot": dot,
                            "cosine": dot / (gnorm * dnorm)
                            if gnorm and dnorm
                            else None,
                            "gradient_norm": gnorm,
                            "displacement_norm": dnorm,
                            "fresh_count": int(fresh.sum()),
                            "policy_mass": float(coeff.sum()),
                            "batch_gradient_norm": float(
                                np.linalg.norm(g.astype(np.float64))
                            ),
                            "clipped": bool(np.linalg.norm(g.astype(np.float64)) > 1),
                        }
                    )
    if expected != report["comparisons"]:
        for a, b in zip(expected, report["comparisons"], strict=True):
            mismatch = next(
                (
                    key
                    for key, value in a.items()
                    if key not in b
                    or (
                        value != b[key]
                        if not isinstance(value, float)
                        else not np.isclose(value, b[key], rtol=2e-6, atol=1e-9)
                    )
                ),
                None,
            )
            if mismatch is not None:
                raise ValueError(
                    f"comparison_reconstruction_mismatch:{mismatch}:{a['batch']}:{a['objective']}:{a['direction']}:{a['weighting']}:{a.get(mismatch)}:{b.get(mismatch)}"
                )

    def consistent(obj: str, direction: str, wanted: str) -> bool:
        for weighting in ("exposure_weighted", "equal_input"):
            selected_rows = [
                r
                for r in expected
                if r["objective"] == obj
                and r["direction"] == direction
                and r["weighting"] == weighting
            ]
            cs = [r["cosine"] for r in selected_rows]
            desc = sum(c is not None and c <= -1e-6 for c in cs)
            asc = sum(c is not None and c >= 1e-6 for c in cs)
            state = "descent" if desc >= 12 else "ascent" if asc >= 12 else "mixed"
            if state != wanted:
                return False
        return True

    if consistent("P", "negative_gradient", "ascent"):
        classification = "minibatch_policy_transport_failure"
    elif consistent("P", "negative_gradient", "descent") and consistent(
        "J", "negative_gradient", "ascent"
    ):
        classification = "fresh_value_transport_failure"
    elif any(
        consistent(obj, "negative_gradient", "descent")
        and consistent(obj, "adam", "ascent")
        for obj in ("P", "J")
    ):
        classification = "adam_direction_transport_failure"
    elif all(
        consistent(obj, direction, "descent")
        for obj in ("P", "J")
        for direction in ("negative_gradient", "adam")
    ):
        classification = "first_order_descent_consistent"
    else:
        classification = "mixed_or_inconclusive_transport"
    if report["classification"] != classification:
        raise ValueError("classification_reconstruction_mismatch")
    receipt = json.loads((output / "receipt.json").read_text())
    if receipt["registration_sha256"] != digest(output / "registration.json"):
        raise ValueError("receipt_registration_binding_mismatch")
    for field, filename in (
        ("evaluation_manifest_sha256", "evaluation-manifest.json"),
        ("results_sha256", "results.md"),
    ):
        if receipt[field] != digest(output / filename):
            raise ValueError(f"receipt_binding_mismatch:{filename}")
    if receipt["diagnostic_sha256"] != digest(output / "diagnostic.json") or receipt[
        "vectors_sha256"
    ] != digest(output / "vectors.npz"):
        raise ValueError("receipt_binding_mismatch")
    if any(not torch.equal(model.state_dict()[k], v) for k, v in start.items()):
        raise ValueError("initializer_state_mutated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    verify(args.root.resolve())
    print("seed468 independent verification passed")


if __name__ == "__main__":
    main()
