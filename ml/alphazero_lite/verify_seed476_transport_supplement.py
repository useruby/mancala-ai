"""Independent read-only reconstruction of the seed476 supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("docs/data/seed476-direction-transport")
FIRST_SUPPLEMENT = Path("docs/data/seed476-direction-transport-supplement")
OUT = Path("docs/data/seed476-direction-transport-supplement-v2")
SEED474 = Path("docs/data/seed474-fresh-value-removal")
INITIAL = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
PUBLISHER = "ml/alphazero_lite/publish_seed476_transport_supplement.py"
VERIFIER = "ml/alphazero_lite/verify_seed476_transport_supplement.py"
TESTS = "ml/alphazero_lite/test_seed476_transport_supplement.py"
SOURCE_CLOSURE = (
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
    PUBLISHER,
    VERIFIER,
    TESTS,
)
INPUTS = {
    "manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "training": "docs/data/seed466-replay-input-bundle/training.npz",
    "evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": str(INITIAL),
    "seed474_registration": str(SEED474 / "registration.json"),
    "seed474_receipt": str(SEED474 / "receipt.json"),
    "seed474_P_steps": str(SEED474 / "P-steps.npz"),
    "seed474_P_trajectory": str(SEED474 / "P-trajectory.json"),
    "seed474_endpoint": str(SEED474 / "endpoint-evidence.json"),
    "seed476_registration": str(BASE / "registration.json"),
    "seed476_receipt": str(BASE / "receipt.json"),
    "seed476_evidence": str(BASE / "evidence.json"),
    "seed476_vectors": str(BASE / "directions.npz"),
    "prior_supplement_registration": str(FIRST_SUPPLEMENT / "registration.json"),
    "prior_supplement_receipt": str(FIRST_SUPPLEMENT / "receipt.json"),
    "prior_supplement_evidence": str(FIRST_SUPPLEMENT / "evidence.json"),
    "prior_supplement_accounting": str(FIRST_SUPPLEMENT / "accounting.npz"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deep_equal(actual, expected, path="evidence") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"inventory_mismatch:{path}")
        for key, value in expected.items():
            _deep_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"length_mismatch:{path}")
        for index, value in enumerate(expected):
            _deep_equal(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isclose(
            actual, expected, rtol=2e-6, atol=2e-8
        ):
            raise ValueError(f"value_mismatch:{path}")
    elif actual != expected:
        raise ValueError(f"value_mismatch:{path}")


def verify(root: Path) -> None:
    """Verify source/input bindings, receipt, predictions, optimizer and accounting."""
    root = root.resolve()
    out = root / OUT
    protected = [root / BASE, root / SEED474, root / FIRST_SUPPLEMENT, out]
    before = {
        str(directory.relative_to(root)) + "/" + file.name: (
            file.read_bytes(),
            file.stat().st_mtime_ns,
        )
        for directory in protected
        for file in directory.iterdir()
        if file.is_file()
    }
    try:
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema")
            != "seed476-direction-transport-supplement-registration-v1"
            or reg.get("status")
            != "post_execution_supplement_frozen_before_reconstruction"
        ):
            raise ValueError("supplement_registration_mismatch")
        import torch

        if reg["dependencies"] != {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }:
            raise ValueError("supplement_dependency_identity_mismatch")
        if set(reg["source_closure_sha256"]) != set(SOURCE_CLOSURE):
            raise ValueError("supplement_source_inventory_mismatch")
        if set(reg["inputs_sha256"]) != set(INPUTS):
            raise ValueError("supplement_input_inventory_mismatch")
        for name, expected in reg["source_closure_sha256"].items():
            if _sha(root / name) != expected:
                raise ValueError(f"supplement_source_mismatch:{name}")
        for name, expected in reg["inputs_sha256"].items():
            if _sha(root / INPUTS[name]) != expected:
                raise ValueError(f"supplement_input_mismatch:{name}")
        for field, file in (
            ("original_seed476_registration_sha256", "registration.json"),
            ("original_seed476_receipt_sha256", "receipt.json"),
            ("original_seed476_evidence_sha256", "evidence.json"),
            ("original_seed476_vectors_sha256", "directions.npz"),
        ):
            if _sha(root / BASE / file) != reg[field]:
                raise ValueError(f"original_publication_mismatch:{file}")
        receipt = json.loads((out / "receipt.json").read_text())
        if receipt.get("status") != "seed476_supplement_bound":
            raise ValueError("supplement_receipt_status_mismatch")
        if set(receipt) != {
            "status",
            "registration.json",
            "evidence.json",
            "accounting.npz",
            "results.md",
        }:
            raise ValueError("supplement_receipt_inventory_mismatch")
        for name, expected in receipt.items():
            if name != "status" and _sha(out / name) != expected:
                raise ValueError(f"supplement_receipt_mismatch:{name}")
        expected_report, expected_arrays = _independent_reconstruct(root)
        _deep_equal(json.loads((out / "evidence.json").read_text()), expected_report)
        with np.load(out / "accounting.npz", allow_pickle=False) as published:
            if set(published.files) != set(expected_arrays):
                raise ValueError("supplement_array_inventory_mismatch")
            for name, expected in expected_arrays.items():
                actual = published[name]
                equal = (
                    np.array_equal(actual, expected)
                    if expected.dtype.kind in "US"
                    else np.allclose(
                        actual, expected, rtol=2e-6, atol=2e-8, equal_nan=True
                    )
                )
                if (
                    actual.dtype != expected.dtype
                    or actual.shape != expected.shape
                    or not equal
                ):
                    raise ValueError(f"supplement_array_mismatch:{name}")
    finally:
        after = {
            str(directory.relative_to(root)) + "/" + file.name: (
                file.read_bytes(),
                file.stat().st_mtime_ns,
            )
            for directory in protected
            for file in directory.iterdir()
            if file.is_file()
        }
        if before != after:
            raise ValueError("verification_mutated_historical_or_published_evidence")


def _independent_reconstruct(root: Path):
    """Independent implementation; never calls publisher calculations or reads vectors."""
    import torch
    import torch.nn.functional as F

    from ml.alphazero_lite import train
    from ml.alphazero_lite.seed466_runtime_bundle import (
        MANIFEST_SHA256,
        read_authenticated_bundle,
    )

    reg = json.loads((root / OUT / "registration.json").read_text())
    rawreg = json.loads((root / BASE / "registration.json").read_text())
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    perm = np.asarray(rawreg["population"]["minibatch_positions"], dtype=np.int64)
    occurrences = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh_rows = occurrences[tr["fresh_source_mask"][occurrences].astype(bool)]
    coefficient = tr["policy_coefficients"][fresh_rows].astype(np.float64)
    denominator = float(coefficient.sum(dtype=np.float64))
    held = np.flatnonzero(ev["source_ids"] == 0)
    held_rows = ev["compact_rows"][held]
    identities = ev["input_identity"][held].tolist()
    groups = {key: identities.count(key) for key in set(identities)}
    weighting = {
        "exposure": np.full(len(held), 1 / len(held), dtype=np.float64),
        "equal_input": np.asarray(
            [1 / (len(groups) * groups[key]) for key in identities], dtype=np.float64
        ),
    }
    theta = {}
    outputs = {
        "training_occurrence_rows": fresh_rows.astype(np.int64),
        "training_coefficients": tr["policy_coefficients"][fresh_rows].copy(),
        "training_targets": tr["policy_targets"][fresh_rows].copy(),
        "training_legal_masks": tr["legal_masks"][fresh_rows].copy(),
        "unseen_occurrences": held.astype(np.int64),
        "unseen_rows": held_rows.astype(np.int64),
        "unseen_targets": ev["policy_targets"][held].copy(),
        "unseen_legal_masks": tr["legal_masks"][held_rows].copy(),
        "unseen_exposure_weights": weighting["exposure"],
        "unseen_equal_input_weights": weighting["equal_input"],
    }
    torch.set_num_threads(1)
    with np.load(root / SEED474 / "P-steps.npz", allow_pickle=False) as archive:
        trajectory = json.loads((root / SEED474 / "P-trajectory.json").read_text())
        for s in range(16):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            train.load_checkpoint_into_model(model, root / INITIAL)
            parameters = list(model.parameters())
            pre = [
                torch.from_numpy(archive[f"{s:02d}_pre_{i:02d}"].copy())
                for i in range(len(parameters))
            ]
            wanted_post = [
                torch.from_numpy(archive[f"{s:02d}_post_{i:02d}"].copy())
                for i in range(len(parameters))
            ]
            with torch.no_grad():
                for p, value in zip(parameters, pre, strict=True):
                    p.copy_(value)
            model.eval()
            batch_rows = occurrences[perm[s * 512 : (s + 1) * 512]]
            if (
                trajectory.get("arm") != "P"
                or trajectory["steps"][s].get("occurrence_indices")
                != batch_rows.tolist()
            ):
                raise ValueError(f"archived_minibatch_record_mismatch:{s + 1}")
            x = torch.from_numpy(tr["inputs"][batch_rows].copy())
            y = torch.from_numpy(tr["policy_targets"][batch_rows].copy())
            legal = torch.from_numpy(tr["legal_masks"][batch_rows].copy())
            c = torch.from_numpy(tr["policy_coefficients"][batch_rows].copy())
            f = torch.from_numpy(tr["fresh_source_mask"][batch_rows].astype(np.float32))
            logits, _ = model(x)
            minibatch_ce = -(
                y * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
            ).sum(1)
            minibatch_mass = (c * f).sum()
            (minibatch_ce.mul(c).mul(f).sum() / minibatch_mass).backward()
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            clip_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            grad_tensors = [p.grad.detach().clone() for p in parameters]
            opt = torch.optim.Adam(
                parameters, lr=0.0000625, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            opt.state.clear()
            opt.step()
            checks = []
            for i, (p, g, post) in enumerate(
                zip(parameters, grad_tensors, wanted_post, strict=True)
            ):
                st = opt.state[p]
                check = {
                    "gradient": np.array_equal(
                        g.numpy(), archive[f"{s:02d}_gradient_{i:02d}"]
                    ),
                    "exp_avg": np.array_equal(
                        st["exp_avg"].numpy(), archive[f"{s:02d}_exp_avg_{i:02d}"]
                    ),
                    "exp_avg_sq": np.array_equal(
                        st["exp_avg_sq"].numpy(), archive[f"{s:02d}_exp_avg_sq_{i:02d}"]
                    ),
                    "post": np.array_equal(p.detach().numpy(), post.numpy()),
                    "step": float(st["step"]) == 1,
                }
                if not all(check.values()):
                    raise ValueError(f"optimizer_reconstruction:{s + 1}:{i}")
                checks.append({"parameter": i, **check})
                outputs[f"{s:02d}_optimizer_gradient_{i:02d}"] = g.numpy().copy()
                outputs[f"{s:02d}_optimizer_exp_avg_{i:02d}"] = (
                    st["exp_avg"].numpy().copy()
                )
                outputs[f"{s:02d}_optimizer_exp_avg_sq_{i:02d}"] = (
                    st["exp_avg_sq"].numpy().copy()
                )
                outputs[f"{s:02d}_optimizer_pre_{i:02d}"] = pre[i].numpy().copy()
                outputs[f"{s:02d}_optimizer_post_{i:02d}"] = p.detach().numpy().copy()
                outputs[f"{s:02d}_optimizer_counter_{i:02d}"] = np.asarray(
                    float(st["step"]), dtype=np.float64
                )
                outputs[f"{s:02d}_optimizer_zero_m_{i:02d}"] = np.zeros_like(
                    st["exp_avg"].numpy()
                )
                outputs[f"{s:02d}_optimizer_zero_v_{i:02d}"] = np.zeros_like(
                    st["exp_avg_sq"].numpy()
                )
            opt_rec = {
                "preclip_norm": float(clip_norm),
                "clipped": bool(float(clip_norm) > 1),
                "step_counter": 1,
                "state_reset": True,
                "passed": True,
                "parameter_checks": checks,
            }
            with torch.no_grad():
                for p, value in zip(parameters, pre, strict=True):
                    p.copy_(value)

            # All differentiable quantities below are reconstructed at the frozen pre-state.
            full_acc = [torch.zeros_like(p) for p in parameters]
            for start in range(0, len(fresh_rows), 512):
                rr = fresh_rows[start : start + 512]
                logits_train, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                target = torch.from_numpy(tr["policy_targets"][rr].copy())
                mask = torch.from_numpy(tr["legal_masks"][rr].copy())
                ce = -(
                    target
                    * F.log_softmax(logits_train.masked_fill(mask == 0, -1e9), dim=1)
                ).sum(1)
                w = torch.from_numpy(
                    (coefficient[start : start + len(rr)] / denominator).astype(
                        np.float32
                    )
                )
                gs = torch.autograd.grad((ce * w).sum(), parameters, allow_unused=True)
                for a, g in zip(full_acc, gs, strict=True):
                    if g is not None:
                        a.add_(g.detach())
            full_gradient = np.concatenate(
                [g.cpu().numpy().reshape(-1).astype(np.float64) for g in full_acc]
            )
            batch_gradient = np.concatenate(
                [g.cpu().numpy().reshape(-1).astype(np.float64) for g in grad_tensors]
            )
            preflat = np.concatenate(
                [v.numpy().reshape(-1).astype(np.float64) for v in pre]
            )
            postflat = np.concatenate(
                [v.numpy().reshape(-1).astype(np.float64) for v in wanted_post]
            )
            adam = postflat - preflat
            held_gradients = {}
            for name, weights in weighting.items():
                acc = [torch.zeros_like(p) for p in parameters]
                for start in range(0, len(held), 128):
                    sl = slice(start, start + 128)
                    rr = held_rows[sl]
                    logits_unseen, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                    target = torch.from_numpy(ev["policy_targets"][held[sl]].copy())
                    mask = torch.from_numpy(tr["legal_masks"][rr].copy())
                    ce = -(
                        target
                        * F.log_softmax(
                            logits_unseen.masked_fill(mask == 0, -1e9), dim=1
                        )
                    ).sum(1)
                    gs = torch.autograd.grad(
                        (ce * torch.from_numpy(weights[sl].astype(np.float32))).sum(),
                        parameters,
                        allow_unused=True,
                    )
                    for a, g in zip(acc, gs, strict=True):
                        if g is not None:
                            a.add_(g.detach())
                held_gradients[name] = np.concatenate(
                    [g.cpu().numpy().reshape(-1).astype(np.float64) for g in acc]
                )

            # evaluate 32 actual endpoints in the fixed schedule; each row stays an occurrence.
            loss_record = {}
            for endpoint, state in (("pre", pre), ("post", wanted_post)):
                with torch.no_grad():
                    for p, value in zip(parameters, state, strict=True):
                        p.copy_(value)
                train_loss, train_prob = _predict(
                    model,
                    tr["inputs"][fresh_rows],
                    tr["policy_targets"][fresh_rows],
                    tr["legal_masks"][fresh_rows],
                    512,
                )
                held_loss, held_prob = _predict(
                    model,
                    tr["inputs"][held_rows],
                    ev["policy_targets"][held],
                    tr["legal_masks"][held_rows],
                    128,
                )
                outputs[f"{s:02d}_{endpoint}_training_ce"] = train_loss
                outputs[f"{s:02d}_{endpoint}_training_probability"] = train_prob
                outputs[f"{s:02d}_{endpoint}_unseen_ce"] = held_loss
                outputs[f"{s:02d}_{endpoint}_unseen_probability"] = held_prob
                loss_record[endpoint] = {
                    "training_policy_ce": float(
                        np.dot(train_loss, coefficient) / denominator
                    )
                }
                for name, w in weighting.items():
                    loss_record[endpoint][f"unseen_{name}_policy_ce"] = float(
                        np.dot(held_loss, w)
                    )

            realized = {"recorded_adam": adam}
            scale_norm = float(np.linalg.norm(adam))
            for label, raw in (
                ("full_negative", -full_gradient),
                ("minibatch_negative", -batch_gradient),
            ):
                raw_norm = float(np.linalg.norm(raw))
                outputs[f"{s:02d}_counterfactual_scale_{label}"] = np.asarray(
                    0.0 if raw_norm == 0 else scale_norm / raw_norm,
                    dtype=np.float64,
                )
                scaled = (
                    np.zeros_like(raw)
                    if raw_norm == 0
                    else (raw * scale_norm / raw_norm).astype(np.float32)
                )
                parts = []
                cursor = 0
                for value in pre:
                    size = value.numel()
                    after = value + torch.from_numpy(
                        scaled[cursor : cursor + size].reshape(value.shape).copy()
                    )
                    parts.append((after.double() - value.double()).numpy().reshape(-1))
                    cursor += size
                realized[label] = np.concatenate(parts)
            for key, value in {
                f"{s:02d}_full_gradient": full_gradient,
                f"{s:02d}_minibatch_gradient": batch_gradient,
                f"{s:02d}_adam_displacement": adam,
                **{
                    f"{s:02d}_unseen_{k}_gradient": v for k, v in held_gradients.items()
                },
                **{f"{s:02d}_realized_{k}": v for k, v in realized.items()},
            }.items():
                outputs[key] = value

            train_observed = (
                loss_record["post"]["training_policy_ce"]
                - loss_record["pre"]["training_policy_ce"]
            )
            train_first = float(np.dot(full_gradient, adam))
            step_record = {
                "step": s + 1,
                "optimizer_reconstruction": opt_rec,
                "losses": loss_record,
                "training_accounting": {
                    "observed_change": train_observed,
                    "first_order_contribution": train_first,
                    "remainder": train_observed - train_first,
                },
                "directions": {},
            }
            for name, g in held_gradients.items():
                observed = (
                    loss_record["post"][f"unseen_{name}_policy_ce"]
                    - loss_record["pre"][f"unseen_{name}_policy_ce"]
                )
                outputs[f"{s:02d}_counterfactual_scale_full_negative"] = np.asarray(
                    0.0
                    if np.linalg.norm(-full_gradient) == 0
                    else scale_norm / np.linalg.norm(-full_gradient),
                    dtype=np.float64,
                )
                outputs[f"{s:02d}_counterfactual_scale_minibatch_negative"] = (
                    np.asarray(
                        0.0
                        if np.linalg.norm(-batch_gradient) == 0
                        else scale_norm / np.linalg.norm(-batch_gradient),
                        dtype=np.float64,
                    )
                )
                ds = {}
                for label, delta in realized.items():
                    ng = float(np.linalg.norm(g))
                    nd = float(np.linalg.norm(delta))
                    dot = float(np.dot(g, delta))
                    cosine = dot / (ng * nd) if ng and nd else 0.0
                    sign = (
                        "zero_norm"
                        if not ng or not nd
                        else "descent"
                        if cosine <= -1e-6
                        else "ascent"
                        if cosine >= 1e-6
                        else "ambiguous"
                    )
                    ds[label] = {
                        "norm": nd,
                        "dot": dot,
                        "cosine": cosine,
                        "sign": sign,
                        "first_order_contribution": dot,
                        "observed_loss_change": observed,
                        "remainder": observed - dot,
                    }
                signs = [
                    ds[k]["sign"]
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
                ds["category"] = category
                ds["observed_change"] = observed
                ds["actual_first_order_contribution"] = float(np.dot(g, adam))
                ds["actual_remainder"] = (
                    observed - ds["actual_first_order_contribution"]
                )
                step_record["directions"][name] = ds
            theta[s] = step_record
    report = _compose_report(reg, theta, weighting)
    return report, outputs


def _predict(model, x, y, mask, batch):
    import torch
    import torch.nn.functional as F

    losses = []
    probs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            sl = slice(start, start + batch)
            logits, _ = model(torch.from_numpy(x[sl].copy()))
            legal = torch.from_numpy(mask[sl].copy())
            logp = F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
            target = torch.from_numpy(y[sl].copy())
            losses.append((-(target * logp).sum(1)).cpu().numpy().astype(np.float64))
            probs.append(logp.exp().cpu().numpy().astype(np.float32))
    return np.concatenate(losses), np.concatenate(probs)


def _compose_report(reg, steps, weighting):
    records = [steps[i] for i in range(16)]
    totals = {}
    for name in weighting:
        rs = [r["directions"][name] for r in records]
        cats = [r["category"] for r in rs]
        start = records[0]["losses"]["pre"][f"unseen_{name}_policy_ce"]
        end = records[-1]["losses"]["post"][f"unseen_{name}_policy_ce"]
        observed = float(sum(r["observed_change"] for r in rs))
        first = float(sum(r["actual_first_order_contribution"] for r in rs))
        remainder = float(sum(r["actual_remainder"] for r in rs))
        totals[name] = {
            "category_counts": {
                c: cats.count(c)
                for c in (
                    "minibatch_transport_failure",
                    "adam_coordinate_transport_failure",
                    "both_descent",
                    "mixed_or_ambiguous",
                )
            },
            "signed_direction_contribution_totals": {
                d: float(sum(r[d]["dot"] for r in rs))
                for d in ("full_negative", "minibatch_negative", "recorded_adam")
            },
            "observed_change_total": observed,
            "first_order_total": first,
            "remainder_total": remainder,
            "endpoint_change": end - start,
            "endpoint_reconciliation_residual": observed - (end - start),
        }
    tr_obs = float(sum(r["training_accounting"]["observed_change"] for r in records))
    tr_first = float(
        sum(r["training_accounting"]["first_order_contribution"] for r in records)
    )
    tr_start = records[0]["losses"]["pre"]["training_policy_ce"]
    tr_end = records[-1]["losses"]["post"]["training_policy_ce"]
    return {
        "schema": "seed476-direction-transport-supplement-evidence-v1",
        "post_execution": True,
        "forward_schedule": reg["forward_schedule"],
        "record_coverage": {
            "steps": 16,
            "required_steps": list(range(1, 17)),
            "unseen_occurrences": 859,
            "training_occurrences": 60015,
        },
        "steps": records,
        "totals": totals,
        "training_totals": {
            "observed_change_total": tr_obs,
            "first_order_total": tr_first,
            "remainder_total": tr_obs - tr_first,
            "endpoint_change": tr_end - tr_start,
            "endpoint_reconciliation_residual": tr_obs - (tr_end - tr_start),
        },
        "limitations": reg["limitation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.root)
    print("seed476 supplement independent verification passed")


if __name__ == "__main__":
    main()
