"""Independent, read-only reconstruction of seed475 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
SEED474 = Path("docs/data/seed474-fresh-value-removal")
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual, expected, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"inventory_mismatch:{name}")
        for key, value in expected.items():
            _close(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=2e-6, atol=2e-8):
            raise ValueError(f"value_mismatch:{name}")
    elif actual != expected:
        raise ValueError(f"value_mismatch:{name}")


def _vector(grads: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [g.detach().numpy().reshape(-1).astype(np.float64) for g in grads]
    )


def _independent(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    occurrences = tr["expanded_replay_indices"][tr["train_positions"]]
    occurrences = occurrences[tr["fresh_source_mask"][occurrences]]
    stone_counts = np.asarray(
        [active_pit_stones_from_encoded_state(x) for x in tr["inputs"][occurrences]]
    )
    bucket_ids = np.where(stone_counts <= 16, 0, np.where(stone_counts <= 32, 1, 2))
    bucket_names = ("<=16", "17-32", ">32")
    coefficients = tr["policy_coefficients"][occurrences].astype(np.float64)
    mass = float(coefficients.sum(dtype=np.float64))
    unseen_indices = np.flatnonzero(ev["source_ids"] == 0)
    if len(unseen_indices) != 859:
        raise ValueError("strict_unseen_eligibility_mismatch")
    keys = ev["input_identity"][unseen_indices].tolist()
    frequencies = {key: keys.count(key) for key in set(keys)}
    weights = {
        "exposure": np.full(859, 1 / 859),
        "equal_input": np.asarray(
            [1 / (len(frequencies) * frequencies[k]) for k in keys]
        ),
    }
    evidence: dict = {
        "training_occurrences": int(len(occurrences)),
        "unseen_occurrences": 859,
        "unseen_exact_inputs": len(frequencies),
        "common_policy_denominator": mass,
        "checkpoints": {},
    }
    archive: dict[str, np.ndarray] = {
        "training_stones": stone_counts,
        "training_bucket": bucket_ids,
        "training_rows": occurrences,
        "unseen_rows": unseen_indices,
    }
    for label in ("initializer", "seed474_P"):
        net = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(net, root / INIT)
        checkpoint = {
            "checkpoint": "original_initializer",
            "initializer_sha256": _sha(root / INIT),
        }
        if label == "seed474_P":
            with (
                np.load(root / SEED474 / "P-steps.npz", allow_pickle=False) as saved,
                torch.no_grad(),
            ):
                for i, p in enumerate(net.parameters()):
                    p.copy_(torch.from_numpy(saved[f"15_post_{i:02d}"].copy()))
            checkpoint = {
                "checkpoint": "seed474_P_archived_endpoint",
                "seed474_P_steps_sha256": _sha(root / SEED474 / "P-steps.npz"),
            }
        net.eval()
        parameters = list(net.parameters())
        initial_buffers = [b.detach().clone() for b in net.buffers()]
        accum = {
            name: [torch.zeros_like(p) for p in parameters]
            for name in ["full", *[f"bucket_{b}" for b in bucket_names]]
        }
        nums = {name: 0.0 for name in accum}
        for start in range(0, len(occurrences), 512):
            idx = np.arange(start, min(start + 512, len(occurrences)))
            rows = occurrences[idx]
            logits, _ = net(torch.from_numpy(tr["inputs"][rows].copy()))
            y = torch.from_numpy(tr["policy_targets"][rows].copy())
            legal = torch.from_numpy(tr["legal_masks"][rows].copy())
            loss = -(
                y * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
            ).sum(1)
            scaled = torch.from_numpy((coefficients[idx] / mass).astype(np.float32))
            for n, bucket in enumerate(bucket_names):
                selected_np = bucket_ids[idx] == n
                selected = torch.from_numpy(selected_np)
                obj = (loss * scaled * selected).sum()
                nums[f"bucket_{bucket}"] += float(
                    (
                        loss.detach().double()
                        * torch.from_numpy(coefficients[idx])
                        * selected
                    ).sum()
                )
                if selected_np.any():
                    gs = torch.autograd.grad(
                        obj, parameters, retain_graph=True, allow_unused=True
                    )
                    for dest, grad in zip(accum[f"bucket_{bucket}"], gs, strict=True):
                        if grad is not None:
                            dest.add_(grad.detach())
            obj = (loss * scaled).sum()
            nums["full"] += float(
                (loss.detach().double() * torch.from_numpy(coefficients[idx])).sum()
            )
            gs = torch.autograd.grad(obj, parameters, allow_unused=True)
            for dest, grad in zip(accum["full"], gs, strict=True):
                if grad is not None:
                    dest.add_(grad.detach())
        flat = {name: _vector(grads) for name, grads in accum.items()}
        for name, vector in flat.items():
            archive[f"{label}_{name}_gradient"] = vector
        ue_rows = ev["compact_rows"][unseen_indices]
        unseen_gradients = {}
        for weighting, occurrence_weights in weights.items():
            sums = [torch.zeros_like(p) for p in parameters]
            for start in range(0, 859, 128):
                sl = slice(start, min(start + 128, 859))
                rr = ue_rows[sl]
                logits, _ = net(torch.from_numpy(tr["inputs"][rr].copy()))
                y = torch.from_numpy(ev["policy_targets"][unseen_indices[sl]].copy())
                legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                row_loss = -(
                    y * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
                ).sum(1)
                obj = (
                    row_loss
                    * torch.from_numpy(occurrence_weights[sl].astype(np.float32))
                ).sum()
                gs = torch.autograd.grad(obj, parameters, allow_unused=True)
                for dest, grad in zip(sums, gs, strict=True):
                    if grad is not None:
                        dest.add_(grad.detach())
            unseen_gradients[weighting] = _vector(sums)
            archive[f"{label}_unseen_{weighting}_gradient"] = unseen_gradients[
                weighting
            ]

        def norm(v):
            return float(np.linalg.norm(v))

        cm = {**checkpoint, "training": {}, "unseen": {}}
        for n, bucket in enumerate(bucket_names):
            chosen = bucket_ids == n
            g = flat[f"bucket_{bucket}"]
            cm["training"][bucket] = {
                "occurrences": int(chosen.sum()),
                "exact_inputs": len(
                    {tr["inputs"][row].tobytes() for row in occurrences[chosen]}
                ),
                "coefficient_mass": float(coefficients[chosen].sum(dtype=np.float64)),
                "loss_numerator": nums[f"bucket_{bucket}"],
                "gradient_norm": norm(g),
            }
        residual = flat["full"] - sum(
            (flat[f"bucket_{b}"] for b in bucket_names), np.zeros_like(flat["full"])
        )
        cm["training"]["full"] = {
            "occurrences": int(len(occurrences)),
            "coefficient_mass": mass,
            "loss_numerator": nums["full"],
            "gradient_norm": norm(flat["full"]),
            "component_sum_residual_norm": norm(residual),
        }
        geom = {
            "full_training": flat["full"],
            **{f"training_{b}": flat[f"bucket_{b}"] for b in bucket_names},
            **{f"unseen_{w}": v for w, v in unseen_gradients.items()},
        }
        norms = {k: norm(v) for k, v in geom.items()}
        cm["pairwise_geometry"] = {"norms": norms, "dot_products": {}, "cosines": {}}
        for a, va in geom.items():
            cm["pairwise_geometry"]["dot_products"][a] = {
                b: float(np.dot(va, vb)) for b, vb in geom.items()
            }
            cm["pairwise_geometry"]["cosines"][a] = {
                b: (
                    float(np.dot(va, vb) / (norms[a] * norms[b]))
                    if norms[a] and norms[b]
                    else 0.0
                )
                for b, vb in geom.items()
            }
        for weighting, ug in unseen_gradients.items():
            negative = -flat["full"]
            cosine = (
                float(np.dot(ug, negative) / (norm(ug) * norm(negative)))
                if norm(ug) and norm(negative)
                else 0.0
            )
            parts = {b: float(np.dot(ug, -flat[f"bucket_{b}"])) for b in bucket_names}
            category = (
                "full_population_opposition"
                if cosine >= 1e-6
                else "full_population_descent"
                if cosine <= -1e-6
                else "ambiguous_or_zero"
            )
            cm["unseen"][weighting] = {
                "occurrences": 859,
                "exact_inputs": len(frequencies),
                "gradient_norm": norm(ug),
                "bucket_negative_gradient_contributions": parts,
                "full_negative_gradient_contribution": float(np.dot(ug, negative)),
                "cosine_with_full_negative_training_gradient": cosine,
                "classification": category,
                "additive_contribution_residual": float(
                    np.dot(ug, negative) - sum(parts.values())
                ),
            }
        evidence["checkpoints"][label] = cm
        if any(
            not torch.equal(a, b)
            for a, b in zip(initial_buffers, net.buffers(), strict=True)
        ):
            raise ValueError("buffer_mutation")
    return evidence, archive


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        registration = json.loads((out / "registration.json").read_text())
        if (
            registration.get("schema")
            != "seed475-policy-population-attribution-registration-v1"
            or registration.get("status") != "frozen_before_gradient_calculation"
        ):
            raise ValueError("registration_mismatch")
        for name, expected in registration["inputs_sha256"].items():
            if _sha(root / INPUTS[name]) != expected:
                raise ValueError(f"input_mismatch:{name}")
        for name, expected in registration["source_closure_sha256"].items():
            if _sha(root / name) != expected:
                raise ValueError(f"source_mismatch:{name}")
        receipt = json.loads((out / "receipt.json").read_text())
        for name, expected in receipt.items():
            if name != "status" and _sha(out / name) != expected:
                raise ValueError(f"receipt_mismatch:{name}")
        expected_json, expected_arrays = _independent(root)
        _close(
            json.loads((out / "evidence.json").read_text()), expected_json, "evidence"
        )
        with np.load(out / "gradients.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(expected_arrays):
                raise ValueError("gradient_inventory_mismatch")
            for name, expected in expected_arrays.items():
                actual = archive[name]
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not np.allclose(actual, expected, rtol=2e-6, atol=2e-8)
                ):
                    raise ValueError(f"gradient_or_population_mismatch:{name}")
    finally:
        after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if before != after:
            raise ValueError("verification_mutated_evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
    print("seed475 independent verification passed")


if __name__ == "__main__":
    main()
