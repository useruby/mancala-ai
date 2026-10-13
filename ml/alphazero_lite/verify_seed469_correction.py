"""Independent read-only reconstruction verifier for the seed469 correction."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from ml.alphazero_lite.verify_seed466_replay_bundle import verify as verify_bundle
from ml.alphazero_lite.verify_seed468_transport import verify as verify_seed468

OUT = Path("docs/data/seed469-optimizer-direction-correction")
OLD = Path("docs/data/seed468-first-step-optimizer-direction-transport")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
INIT_SHA = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
ATOL = 2e-8
RTOL = 2e-6


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gradient(model: torch.nn.Module, loss: torch.Tensor) -> np.ndarray:
    params = list(model.parameters())
    values = torch.autograd.grad(loss, params, allow_unused=True)
    return np.concatenate(
        [
            (torch.zeros_like(p) if g is None else g).detach().cpu().numpy().reshape(-1)
            for p, g in zip(params, values, strict=True)
        ]
    ).astype(np.float32)


def adam(grad: np.ndarray) -> tuple[np.ndarray, bool]:
    g = torch.from_numpy(grad.copy())
    norm = torch.linalg.vector_norm(g)
    clipped = bool(norm > 1.0)
    if clipped:
        g = g / norm
    m, v = g * 0.1, g.square() * 0.001
    raw = -(0.001 / 0.1) * m / (v.sqrt() / (1.0 - 0.999) ** 0.5 + 1e-8)
    return raw.numpy().astype(np.float32), clipped


def realized(initial: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    theta = initial.astype(np.float32)
    proposed = (theta + proposal.astype(np.float32)).astype(np.float32)
    return proposed.astype(np.float64) - theta.astype(np.float64)


def classify(rows: list[dict[str, Any]]) -> str:
    def state(obj: str, direction: str, weighting: str) -> str:
        selected = [
            r["cosine"]
            for r in rows
            if (r["objective"], r["direction"], r["weighting"])
            == (obj, direction, weighting)
        ]
        down = sum(c is not None and c <= -1e-6 for c in selected)
        up = sum(c is not None and c >= 1e-6 for c in selected)
        return "descent" if down >= 12 else "ascent" if up >= 12 else "mixed"

    def consistent(obj: str, direction: str, wanted: str) -> bool:
        return all(
            state(obj, direction, w) == wanted
            for w in ("exposure_weighted", "equal_input")
        )

    if consistent("P", "negative_gradient", "ascent"):
        return "minibatch_policy_transport_failure"
    if consistent("P", "negative_gradient", "descent") and consistent(
        "J", "negative_gradient", "ascent"
    ):
        return "fresh_value_transport_failure"
    if any(
        consistent(o, "negative_gradient", "descent")
        and consistent(o, "adam", "ascent")
        for o in ("P", "J")
    ):
        return "adam_direction_transport_failure"
    if all(
        consistent(o, d, "descent")
        for o in ("P", "J")
        for d in ("negative_gradient", "adam")
    ):
        return "first_order_descent_consistent"
    return "mixed_or_inconclusive_transport"


def _equal_vector(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not np.isfinite(actual).all()
    ):
        raise ValueError(f"vector_schema_mismatch:{name}")
    if not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        raise ValueError(f"vector_reconstruction_mismatch:{name}")


def reconstruct(root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    bundle = root / "docs/data/seed466-replay-input-bundle"
    arrays = read_authenticated_bundle(bundle, MANIFEST_SHA256)
    verify_bundle(bundle, root)
    if digest(root / INIT) != INIT_SHA:
        raise ValueError("initializer_identity_mismatch")
    tr, ev = arrays["training"], arrays["evaluation"]
    selection = np.flatnonzero(ev["source_ids"] == 0)
    identities = ev["input_identity"][selection].tolist()
    if len(selection) != 859 or len(set(identities)) != 851:
        raise ValueError("fresh_unseen_cohort_mismatch")
    compact_eval = ev["compact_rows"][selection]
    eval_x = ev["inputs"][selection] if "inputs" in ev else tr["inputs"][compact_eval]
    eval_p, eval_m = ev["policy_targets"][selection], tr["legal_masks"][compact_eval]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial = np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in model.parameters()]
    ).astype(np.float32)
    groups = {identity: identities.count(identity) for identity in set(identities)}
    eval_gradients: dict[str, np.ndarray] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        logits, _ = model(torch.from_numpy(eval_x.copy()))
        ce = -(
            torch.from_numpy(eval_p.copy())
            * F.log_softmax(
                logits.masked_fill(torch.from_numpy(eval_m.copy()) == 0, -1e9), dim=1
            )
        ).sum(dim=1)
        w = np.asarray(
            [
                1 / 859 if weighting == "exposure_weighted" else 1 / (851 * groups[i])
                for i in identities
            ],
            dtype=np.float32,
        )
        eval_gradients[weighting] = gradient(model, (ce * torch.from_numpy(w)).sum())
    vectors = {f"unseen_gradient_{w}": g for w, g in eval_gradients.items()}
    rows: list[dict[str, Any]] = []
    for batch in range(16):
        pos = tr["first16_minibatch_positions"][batch * 512 : (batch + 1) * 512]
        compact = tr["expanded_replay_indices"][tr["train_positions"][pos]]
        fresh = tr["fresh_source_mask"][compact].astype(np.float32)
        coeff = tr["policy_coefficients"][compact] * fresh
        x, p, mask = (
            torch.from_numpy(tr[key][compact].copy())
            for key in ("inputs", "policy_targets", "legal_masks")
        )
        value_target = torch.from_numpy(tr["value_targets"][compact].reshape(-1).copy())
        for objective in ("P", "J"):
            model.load_state_dict(initial_state)
            logits, value = model(x)
            ce = -(p * F.log_softmax(logits.masked_fill(mask == 0, -1e9), dim=1)).sum(
                dim=1
            )
            mass = float(coeff.sum())
            if mass <= 0:
                raise ValueError(f"empty_policy_mass:{batch}")
            loss = (ce * torch.from_numpy(coeff)).sum() / mass
            if objective == "J":
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
            grad = gradient(model, loss)
            vectors[f"batch{batch:02d}_{objective}_gradient"] = grad
            raw_adam, clipped = adam(grad)
            realized_adam = realized(initial, raw_adam)
            norm_adam = float(np.linalg.norm(realized_adam))
            norm_grad = float(np.linalg.norm(grad.astype(np.float64)))
            raw_plain = (
                (-grad * np.float32(norm_adam / norm_grad)).astype(np.float32)
                if norm_grad
                else np.zeros_like(grad)
            )
            proposals = {"adam": raw_adam, "negative_gradient": raw_plain}
            for direction, proposal in proposals.items():
                delta = realized(initial, proposal)
                stem = f"batch{batch:02d}_{objective}_{direction}"
                vectors[f"{stem}_proposal"] = proposal
                vectors[f"{stem}_realized"] = delta
                for weighting, eg in eval_gradients.items():
                    gd = eg.astype(np.float64)
                    dot = float(np.dot(gd, delta))
                    gnorm, dnorm = (
                        float(np.linalg.norm(gd)),
                        float(np.linalg.norm(delta)),
                    )
                    rows.append(
                        {
                            "batch": batch,
                            "objective": objective,
                            "direction": direction,
                            "weighting": weighting,
                            "dot": dot,
                            "cosine": dot / (gnorm * dnorm)
                            if gnorm and dnorm
                            else None,
                            "gradient_norm": gnorm,
                            "displacement_norm": dnorm,
                            "fresh_count": int(fresh.sum()),
                            "policy_mass": mass,
                            "batch_gradient_norm": norm_grad,
                            "clipped": clipped,
                            "proposal_norm": float(
                                np.linalg.norm(proposal.astype(np.float64))
                            ),
                            "adam_realized_norm": norm_adam,
                            "writeback_norm_discrepancy": float(
                                dnorm
                                - float(np.linalg.norm(proposal.astype(np.float64)))
                            ),
                        }
                    )
    if any(not torch.equal(model.state_dict()[k], v) for k, v in initial_state.items()):
        raise ValueError("initializer_or_buffer_mutated")
    return vectors, rows


def verify(root: Path) -> None:
    verify_seed468(root)
    out = root / OUT
    receipt = json.loads((out / "receipt.json").read_text())
    for name in ("registration.json", "diagnostic.json", "vectors.npz", "results.md"):
        if receipt.get(name) != digest(out / name):
            raise ValueError(f"receipt_binding_mismatch:{name}")
    if receipt.get("status") != "post_execution_correction_receipt":
        raise ValueError("receipt_status_mismatch")
    expected_vectors, expected_rows = reconstruct(root)
    for directory in (root / OLD, out):
        with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
            found = {key: archive[key] for key in archive.files}
        wanted = set(expected_vectors)
        if directory == root / OLD:
            # seed468 stores the raw optimizer proposal under its historical direction key.
            wanted |= {
                f"batch{b:02d}_{o}_{d}"
                for b in range(16)
                for o in ("P", "J")
                for d in ("adam", "negative_gradient")
            }
            wanted -= {
                f"batch{b:02d}_{o}_{d}_proposal"
                for b in range(16)
                for o in ("P", "J")
                for d in ("adam", "negative_gradient")
            }
            wanted -= {
                f"batch{b:02d}_{o}_{d}_realized"
                for b in range(16)
                for o in ("P", "J")
                for d in ("adam", "negative_gradient")
            }
        if set(found) != wanted:
            raise ValueError(f"vector_key_set_mismatch:{directory.name}")
        for name, expected in expected_vectors.items():
            if name in found:
                _equal_vector(found[name], expected, name)
        if directory == root / OLD:
            for b in range(16):
                for o in ("P", "J"):
                    for d in ("adam", "negative_gradient"):
                        name = f"batch{b:02d}_{o}_{d}"
                        _equal_vector(
                            found[name], expected_vectors[f"{name}_proposal"], name
                        )
    report = json.loads((out / "diagnostic.json").read_text())
    records = report.get("comparisons")

    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["batch"],
            row["objective"],
            row["direction"],
            row["weighting"],
        )

    if len(records) != 128 or len({identity(row) for row in records}) != 128:
        raise ValueError("comparison_identity_mismatch")
    if [identity(row) for row in records] != [identity(row) for row in expected_rows]:
        raise ValueError("comparison_order_mismatch")
    for actual, expected in zip(records, expected_rows, strict=True):
        if actual.keys() != expected.keys():
            raise ValueError("comparison_fields_mismatch")
        for key, value in expected.items():
            if isinstance(value, float):
                if not np.isclose(actual[key], value, rtol=RTOL, atol=ATOL):
                    raise ValueError(f"comparison_value_mismatch:{key}")
            elif actual[key] != value:
                raise ValueError(f"comparison_value_mismatch:{key}")
    counts = {}
    for obj in ("P", "J"):
        for direction in ("adam", "negative_gradient"):
            for weighting in ("exposure_weighted", "equal_input"):
                values = [
                    r["cosine"]
                    for r in expected_rows
                    if (r["objective"], r["direction"], r["weighting"])
                    == (obj, direction, weighting)
                ]
                counts[f"{obj}/{direction}/{weighting}"] = {
                    "descent": sum(c is not None and c <= -1e-6 for c in values),
                    "ascent": sum(c is not None and c >= 1e-6 for c in values),
                    "ambiguous_or_zero_norm": sum(
                        c is None or -1e-6 < c < 1e-6 for c in values
                    ),
                }
    if report.get("classification_counts") != counts:
        raise ValueError("classification_counts_mismatch")
    if report.get("classification") != classify(expected_rows):
        raise ValueError("classification_mismatch")
    expected_summary = {
        "proposal_realized_vector_differences": sum(
            not np.array_equal(
                expected_vectors[f"batch{b:02d}_{o}_{d}_proposal"].astype(np.float64),
                expected_vectors[f"batch{b:02d}_{o}_{d}_realized"],
            )
            for b in range(16)
            for o in ("P", "J")
            for d in ("adam", "negative_gradient")
        ),
        "maximum_absolute_norm_discrepancy": max(
            abs(row["writeback_norm_discrepancy"]) for row in expected_rows
        ),
    }
    if report.get("writeback_summary") != expected_summary:
        raise ValueError("writeback_summary_mismatch")
    reg = json.loads((out / "registration.json").read_text())
    if (
        reg.get("comparison_tolerances", {}).get("rtol") != RTOL
        or reg.get("comparison_tolerances", {}).get("atol") != ATOL
    ):
        raise ValueError("comparison_tolerances_mismatch")
    if reg["seed468_registration_sha256"] != digest(
        root / OLD / "registration.json"
    ) or reg["seed468_receipt_sha256"] != digest(root / OLD / "receipt.json"):
        raise ValueError("historical_binding_mismatch")
    for relative, expected_hash in reg["correction_sources_sha256"].items():
        if digest(root / relative) != expected_hash:
            raise ValueError(f"correction_source_binding_mismatch:{relative}")
    for key, relative in {
        "seed466_manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
        "seed466_training": "docs/data/seed466-replay-input-bundle/training.npz",
        "seed466_evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
        "initializer": str(INIT),
    }.items():
        if digest(root / relative) != reg["input_sha256"][key]:
            raise ValueError(f"correction_input_binding_mismatch:{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    verify(args.root.resolve())
    print("seed469 independent verification passed")


if __name__ == "__main__":
    main()
