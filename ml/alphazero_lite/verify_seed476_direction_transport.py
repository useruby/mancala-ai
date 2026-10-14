"""Independent read-only verifier for seed476 direction transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed476_direction_transport import INPUTS, OUT, SOURCES

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(actual, expected, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"inventory_mismatch:{label}")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"value_mismatch:{label}")
        for i, value in enumerate(expected):
            _same(actual[i], value, f"{label}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=2e-6, atol=2e-8):
            raise ValueError(f"value_mismatch:{label}")
    elif actual != expected:
        raise ValueError(f"value_mismatch:{label}")


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed476-direction-transport-registration-v1"
            or reg.get("status") != "frozen_before_gradient_calculation"
        ):
            raise ValueError("registration_mismatch")
        if set(reg["source_closure_sha256"]) != set(SOURCES):
            raise ValueError("source_inventory_mismatch")
        for name, expected in reg["source_closure_sha256"].items():
            if _sha(root / name) != expected:
                raise ValueError(f"source_mismatch:{name}")
        for name, expected in reg["inputs_sha256"].items():
            if _sha(root / INPUTS[name]) != expected:
                raise ValueError(f"input_mismatch:{name}")
        receipt = json.loads((out / "receipt.json").read_text())
        for name, expected in receipt.items():
            if name != "status" and _sha(out / name) != expected:
                raise ValueError(f"receipt_mismatch:{name}")
        # Independent implementation is deliberately kept separate from publisher
        # arithmetic; imports here are limited to immutable model/data contracts.
        expected_report, expected_arrays = _reconstruct(root)
        actual_report = json.loads((out / "evidence.json").read_text())
        _same(actual_report, expected_report, "evidence")
        with np.load(out / "directions.npz", allow_pickle=False) as actual:
            if set(actual.files) != set(expected_arrays):
                raise ValueError("direction_inventory_mismatch")
            for name, wanted in expected_arrays.items():
                got = actual[name]
                if (
                    got.dtype != wanted.dtype
                    or got.shape != wanted.shape
                    or not np.allclose(got, wanted, rtol=2e-6, atol=2e-8)
                ):
                    raise ValueError(f"direction_mismatch:{name}")
    finally:
        after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if before != after:
            raise ValueError("verification_mutated_publication")


def _reconstruct(root: Path):
    """Independent gradient, optimizer, geometry, and category reconstruction."""
    # This verifier owns its forward/backward loops and does not call publisher
    # calculation functions or read published vectors as inputs.
    import torch
    import torch.nn.functional as F

    from ml.alphazero_lite import train
    from ml.alphazero_lite.seed466_runtime_bundle import (
        MANIFEST_SHA256,
        read_authenticated_bundle,
    )

    out = root / OUT
    reg = json.loads((out / "registration.json").read_text())
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    positions = np.asarray(reg["population"]["minibatch_positions"], dtype=np.int64)
    all_rows = tr["expanded_replay_indices"][tr["train_positions"]]
    selected = all_rows[tr["fresh_source_mask"][all_rows].astype(bool)]
    coeff = tr["policy_coefficients"][selected].astype(np.float64)
    mass = coeff.sum(dtype=np.float64)
    unseen = np.flatnonzero(ev["source_ids"] == 0)
    held_rows = ev["compact_rows"][unseen]
    identities = ev["input_identity"][unseen].tolist()
    frequency = {identity: identities.count(identity) for identity in set(identities)}
    weights = {
        "exposure": np.full(len(unseen), 1 / len(unseen)),
        "equal_input": np.asarray(
            [1 / (len(frequency) * frequency[i]) for i in identities]
        ),
    }
    arrays = {}
    records = []
    torch.set_num_threads(1)
    with np.load(root / (SEED474 / "P-steps.npz"), allow_pickle=False) as states:
        for step in range(16):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(model, root / INIT)
            params = list(model.parameters())
            with torch.no_grad():
                for i, p in enumerate(params):
                    p.copy_(torch.from_numpy(states[f"{step:02d}_pre_{i:02d}"].copy()))
            model.eval()
            buffers = [b.detach().clone() for b in model.buffers()]

            def grad(rows, row_coeff, row_fresh, denominator):
                accum = [torch.zeros_like(p) for p in params]
                for start in range(0, len(rows), 512):
                    rr = rows[start : start + 512]
                    logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                    y = torch.from_numpy(tr["policy_targets"][rr].copy())
                    legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                    loss = -(
                        y * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
                    ).sum(1)
                    w = torch.from_numpy(
                        (
                            row_coeff[start : start + len(rr)]
                            * row_fresh[start : start + len(rr)]
                            / denominator
                        ).astype(np.float32)
                    )
                    pieces = torch.autograd.grad(
                        (loss * w).sum(), params, allow_unused=True
                    )
                    for dst, part in zip(accum, pieces, strict=True):
                        if part is not None:
                            dst.add_(part.detach())
                return np.concatenate(
                    [v.cpu().numpy().reshape(-1).astype(np.float64) for v in accum]
                )

            full_g = grad(selected, coeff, np.ones(len(selected)), mass)
            batch_pos = positions[step * 512 : (step + 1) * 512]
            batch_rows = all_rows[batch_pos]
            batch_coeff = tr["policy_coefficients"][batch_rows].astype(np.float64)
            batch_fresh = tr["fresh_source_mask"][batch_rows].astype(np.float64)
            batch_mass = float(np.sum(batch_coeff * batch_fresh, dtype=np.float64))
            batch_g = grad(batch_rows, batch_coeff, batch_fresh, batch_mass)
            flat_pre = np.concatenate(
                [
                    states[f"{step:02d}_pre_{i:02d}"].reshape(-1).astype(np.float64)
                    for i in range(len(params))
                ]
            )
            flat_post = np.concatenate(
                [
                    states[f"{step:02d}_post_{i:02d}"].reshape(-1).astype(np.float64)
                    for i in range(len(params))
                ]
            )
            adam = flat_post - flat_pre
            held_grads = {}
            for name, ww in weights.items():
                accum = [torch.zeros_like(p) for p in params]
                for start in range(0, len(unseen), 128):
                    sl = slice(start, start + 128)
                    rr = held_rows[sl]
                    logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                    y = torch.from_numpy(ev["policy_targets"][unseen[sl]].copy())
                    legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                    loss = -(
                        y * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
                    ).sum(1)
                    gs = torch.autograd.grad(
                        (loss * torch.from_numpy(ww[sl].astype(np.float32))).sum(),
                        params,
                        allow_unused=True,
                    )
                    for dst, g in zip(accum, gs, strict=True):
                        if g is not None:
                            dst.add_(g.detach())
                held_grads[name] = np.concatenate(
                    [v.cpu().numpy().reshape(-1).astype(np.float64) for v in accum]
                )
            dirs = {
                "full_negative": -full_g,
                "minibatch_negative": -batch_g,
                "recorded_adam": adam,
            }
            delta_norm = float(np.linalg.norm(adam))
            matched = {}
            for label, direction in list(dirs.items())[:2]:
                dn = float(np.linalg.norm(direction))
                scaled = (
                    np.zeros_like(direction)
                    if not dn
                    else (direction * delta_norm / dn).astype(np.float32)
                )
                realized = []
                offset = 0
                for pidx, p in enumerate(params):
                    size = p.numel()
                    before = torch.from_numpy(
                        states[f"{step:02d}_pre_{pidx:02d}"].copy()
                    )
                    after = before + torch.from_numpy(
                        scaled[offset : offset + size].reshape(p.shape).copy()
                    )
                    realized.extend(
                        (
                            after.numpy().reshape(-1).astype(np.float64)
                            - before.numpy().reshape(-1).astype(np.float64)
                        ).tolist()
                    )
                    offset += size
                matched[label] = np.asarray(realized)
            for label, vector in [
                ("full_gradient", full_g),
                ("minibatch_gradient", batch_g),
                ("adam_displacement", adam),
                *[(f"unseen_{n}_gradient", v) for n, v in held_grads.items()],
                *[(f"matched_{n}", v) for n, v in matched.items()],
            ]:
                arrays[f"{step:02d}_{label}"] = vector
            by_weight = {}
            for name, held in held_grads.items():
                hnorm = float(np.linalg.norm(held))
                scored = {}
                for label, direction in dirs.items():
                    dnorm = float(np.linalg.norm(direction))
                    dot = float(np.dot(held, direction))
                    cosine = dot / (hnorm * dnorm) if hnorm and dnorm else 0.0
                    sign = (
                        "zero_norm"
                        if not hnorm or not dnorm
                        else "descent"
                        if cosine <= -1e-6
                        else "ascent"
                        if cosine >= 1e-6
                        else "ambiguous"
                    )
                    scored[label] = {"dot": dot, "cosine": cosine, "sign": sign}
                signs = [
                    scored[key]["sign"]
                    for key in ("full_negative", "minibatch_negative", "recorded_adam")
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
                by_weight[name] = {
                    "directions": scored,
                    "category": category,
                    "signed_contributions": {k: v["dot"] for k, v in scored.items()},
                }
            records.append(
                {
                    "step": step + 1,
                    "direction_norms": {
                        k: float(np.linalg.norm(v)) for k, v in dirs.items()
                    },
                    "matched_residual_norms": {
                        k: float(np.linalg.norm(v)) - delta_norm
                        for k, v in matched.items()
                    },
                    "unseen": by_weight,
                }
            )
            if any(
                not torch.equal(a, b)
                for a, b in zip(buffers, model.buffers(), strict=True)
            ):
                raise ValueError("model_buffer_mutated")
    totals = {}
    for weighting in weights:
        cats = [r["unseen"][weighting]["category"] for r in records]
        totals[weighting] = {
            c: cats.count(c)
            for c in (
                "minibatch_transport_failure",
                "adam_coordinate_transport_failure",
                "both_descent",
                "mixed_or_ambiguous",
            )
        }
    return {
        "schema": "seed476-direction-transport-evidence-v1",
        "population": reg["population"],
        "steps": records,
        "summary": totals,
    }, arrays


SEED474 = Path("docs/data/seed474-fresh-value-removal")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
    print("seed476 independent verification passed")


if __name__ == "__main__":
    main()
