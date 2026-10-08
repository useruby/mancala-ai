"""Independent, read-only semantic verifier for seed445 publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.seed444_minibatch_denominator import _load_frozen_inputs

OUT = Path("docs/data/seed445-finite-step-attribution")
GROUPS = ("shared_trunk", "policy_head", "value_head")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: float, b: float, atol: float, rtol: float, label: str) -> None:
    if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
        raise ValueError(f"{label}_mismatch")


def _array_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    atol: float = 2e-7,
    rtol: float = 2e-6,
) -> None:
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, atol=atol, rtol=rtol
    ):
        raise ValueError(f"{label}_mismatch")


def _linked(previous: list[np.ndarray], current: list[np.ndarray], label: str) -> None:
    if len(previous) != len(current) or any(
        not np.array_equal(a, b) for a, b in zip(previous, current, strict=True)
    ):
        raise ValueError(f"{label}_invalid")


def _layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    entries, offset = [], 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = (
            "shared_trunk"
            if name.startswith(("input_layer.", "residual_layers."))
            else "policy_head"
            if name.startswith(("policy_hidden_layer.", "policy_head."))
            else "value_head"
            if name.startswith(("value_hidden_layer.", "value_head."))
            else None
        )
        if group is None:
            raise ValueError(f"unclassified_parameter:{name}")
        stop = offset + p.numel()
        entries.append(
            {
                "name": name,
                "shape": list(p.shape),
                "start": offset,
                "stop": stop,
                "group": group,
            }
        )
        offset = stop
    return entries


def _states(
    root: Path, arm: str, model: torch.nn.Module, initial: list[np.ndarray]
) -> list[list[np.ndarray]]:
    path = root / "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz"
    states = [[v.copy() for v in initial]]
    with np.load(path, allow_pickle=False) as archive:
        for step in range(16):
            pre = [
                archive[f"{arm}_{step:02d}_pre_{i:02d}"] for i in range(len(initial))
            ]
            post = [
                archive[f"{arm}_{step:02d}_post_{i:02d}"] for i in range(len(initial))
            ]
            _linked(states[-1], pre, f"trajectory_link:{arm}:{step + 1}")
            if any(
                a.dtype != np.float32 or not np.isfinite(a).all() for a in (*pre, *post)
            ):
                raise ValueError(f"trajectory_tensor_invalid:{arm}:{step + 1}")
            states.append([a.copy() for a in post])
    endpoint = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        endpoint, root / f"docs/data/seed442-kl-capped-adam-screen/{arm}-final.npz"
    )
    if any(
        not np.array_equal(a, b.detach().cpu().numpy())
        for a, b in zip(states[-1], endpoint.parameters(), strict=True)
    ):
        raise ValueError(f"trajectory_endpoint_invalid:{arm}")
    return states


def _metric_and_gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
    objective: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    parameters = tuple(q for q in model.parameters() if q.requires_grad)
    grads = [np.zeros(tuple(q.shape), dtype=np.float64) for q in parameters]
    predictions, raw_logits, raw_losses = [], [], []
    # Match publication prediction/metric batching exactly; gradient accumulation
    # below deliberately uses smaller chunks while preserving the full objective.
    with torch.no_grad():
        for start in range(0, len(ids), 512):
            ix = ids[start : start + 512]
            logits, pred = model(torch.from_numpy(x[ix]))
            predictions.extend(pred.reshape(-1).cpu().numpy().tolist())
            raw_logits.append(logits.cpu().numpy())
            if objective == "policy_ce":
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(x[ix])
                )
                per_row = train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[ix])
                )
            else:
                per_row = (
                    pred.reshape(-1).double()
                    - torch.from_numpy(v[ix]).reshape(-1).double()
                ).square()
            raw_losses.extend(per_row.double().cpu().numpy().tolist())
    for start in range(0, len(ids), 256):
        ix = ids[start : start + 256]
        w = weights[start : start + 256]
        logits, pred = model(torch.from_numpy(x[ix]))
        if objective == "policy_ce":
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[ix]))
            per_row = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[ix])
            )
        else:
            per_row = (
                pred.reshape(-1).double() - torch.from_numpy(v[ix]).reshape(-1).double()
            ).square()
        scale = float(np.sum(weights, dtype=np.float64))
        if objective == "policy_ce":
            loss = torch.sum(per_row.double() * torch.from_numpy(w).double()) / scale
        else:
            loss = torch.sum(per_row.double() * torch.from_numpy(w).double()) / scale
        derivs = torch.autograd.grad(loss, parameters, allow_unused=True)
        for j, derivative in enumerate(derivs):
            if derivative is not None:
                grads[j] += derivative.detach().cpu().numpy().astype(np.float64)
    values = np.asarray(raw_losses, dtype=np.float64)
    return (
        float(
            np.sum(values * weights, dtype=np.float64)
            / np.sum(weights, dtype=np.float64)
        ),
        np.concatenate(raw_logits),
        np.asarray(predictions, dtype=np.float32),
        np.concatenate([a.reshape(-1) for a in grads]),
    )


def _weights(identities: list[str]) -> dict[str, np.ndarray]:
    counts: dict[str, int] = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    unique = len(counts)
    return {
        "exposure_weighted": np.full(
            len(identities), 1 / len(identities), dtype=np.float64
        ),
        "equal_input": np.asarray(
            [1 / (unique * counts[i]) for i in identities], dtype=np.float64
        ),
    }


def _classify(totals: dict[str, dict[str, float]]) -> str:
    if all(row["D"] > 0 and row["S"] >= 0.75 * row["D"] for row in totals.values()):
        return "first_order_value_harm_dominant"
    if all(row["D"] > 0 and row["R"] >= 0.75 * row["D"] for row in totals.values()):
        return "finite_step_value_residual_dominant"
    return "mixed_or_no_common_value_dominance"


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    protocol = json.loads((out / "protocol-v2.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_new_gradients":
        raise ValueError("protocol_status_invalid")
    amendment_path = out / "verifier-amendment-1.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    bound_verifier = protocol["execution_source_sha256"][
        "ml/alphazero_lite/verify_seed445_finite_step_attribution.py"
    ]
    current_verifier = _sha(
        root / "ml/alphazero_lite/verify_seed445_finite_step_attribution.py"
    )
    if (
        amendment.get("schema") != "seed445-post-execution-verifier-amendment-v1"
        or amendment.get("protocol_sha256") != _sha(out / "protocol-v2.json")
        or amendment.get("original_verifier_sha256") != bound_verifier
        or amendment.get("corrected_verifier_sha256") != current_verifier
        or amendment.get("arithmetic_protocol_changed") is not False
    ):
        raise ValueError("verifier_amendment_invalid")
    for relative, expected in {
        **protocol["execution_source_sha256"],
        **protocol["authoritative_input_sha256"],
    }.items():
        actual = _sha(root / relative)
        if relative == "ml/alphazero_lite/verify_seed445_finite_step_attribution.py":
            if actual != expected and actual != amendment["corrected_verifier_sha256"]:
                raise ValueError(f"frozen_binding_mismatch:{relative}")
        elif actual != expected:
            raise ValueError(f"frozen_binding_mismatch:{relative}")
    result = json.loads((out / "step-ledger.json").read_text(encoding="utf-8"))
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != _sha(out / "protocol-v2.json"):
        raise ValueError("protocol_receipt_mismatch")
    for relative, expected in receipt["artifacts_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError(f"artifact_hash_mismatch:{relative}")
    for relative, expected in receipt["historical_seed442_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError(f"historical_seed442_changed:{relative}")

    source_registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text(encoding="utf-8")
    )
    arrays = _load_frozen_inputs(root, source_registration)
    x, p, v = arrays[:3]
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        members = [json.loads(line) for line in stream]
    selected = [
        r for r in members if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    ids = np.asarray([int(r["compact_row"]) for r in selected], dtype=np.int64)
    identities = [r["input_identity"] for r in selected]
    weights = _weights(identities)
    if (len(ids), len(set(identities))) != (2607, 1242):
        raise ValueError("population_invalid")
    if (
        result["row_identities"] != identities
        or result["row_compact_indices"] != ids.tolist()
    ):
        raise ValueError("row_identity_archive_mismatch")

    model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(
        model,
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    )
    layout = _layout(model)
    if (
        result["parameter_layout"] != layout
        or json.loads((out / "parameter-layout.json").read_text()) != layout
    ):
        raise ValueError("parameter_layout_mismatch")
    initial = [q.detach().cpu().numpy().copy() for q in model.parameters()]
    pred_archive = np.load(out / "predictions.npz", allow_pickle=False)
    grad_archive = np.load(out / "gradients.npz", allow_pickle=False)
    if not np.array_equal(pred_archive["row_identities"], np.asarray(identities)):
        raise ValueError("prediction_identity_mismatch")
    atol, rtol = protocol["tolerances"]["absolute"], protocol["tolerances"]["relative"]
    historical_metrics = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]
    group_indices = {
        g: np.concatenate(
            [np.arange(e["start"], e["stop"]) for e in layout if e["group"] == g]
        )
        for g in GROUPS
    }
    for arm in ("A", "B"):
        states = _states(root, arm, model, initial)
        metric_values: dict[tuple[str, str], list[float]] = {}
        metric_grads: dict[tuple[str, str, int], np.ndarray] = {}
        for objective in ("policy_ce", "value_mse"):
            for weighting, weight in weights.items():
                values = []
                for si, state in enumerate(states):
                    with torch.no_grad():
                        for param, value in zip(model.parameters(), state, strict=True):
                            param.copy_(torch.from_numpy(value.copy()))
                    loss, logits, pred, grad = _metric_and_gradient(
                        model, x, p, v, ids, weight, objective
                    )
                    # No archive gradient is expected for the endpoint-only state.
                    _array_close(
                        pred_archive[f"{arm}_{si:02d}_logits"],
                        logits,
                        f"altered_prediction_logits:{arm}:{si}:{weighting}:{objective}",
                        atol,
                        rtol,
                    )
                    _array_close(
                        pred_archive[f"{arm}_{si:02d}_values"],
                        pred,
                        f"altered_prediction_values:{arm}:{si}:{weighting}:{objective}",
                        atol,
                        rtol,
                    )
                    values.append(loss)
                    if si < 16:
                        metric_grads[(objective, weighting, si)] = grad
                metric_values[(objective, weighting)] = values
                _close(
                    values[-1],
                    historical_metrics[arm][weighting][objective],
                    atol,
                    rtol,
                    f"seed442_endpoint_metric:{arm}:{objective}:{weighting}",
                )
        ledger = result["arms"][arm]["step_ledger"]
        if len(ledger) != 64:
            raise ValueError(f"step_ledger_length_invalid:{arm}")
        totals = {}
        for objective in ("policy_ce", "value_mse"):
            for weighting in weights:
                sums = {"D": 0.0, "S": 0.0, "R": 0.0}
                values = metric_values[(objective, weighting)]
                for step in range(16):
                    rec = next(
                        (
                            row
                            for row in ledger
                            if row["step"] == step + 1
                            and row["objective"] == objective
                            and row["weighting"] == weighting
                        ),
                        None,
                    )
                    if rec is None:
                        raise ValueError(
                            f"ledger_row_missing:{arm}:{step}:{objective}:{weighting}"
                        )
                    delta = np.concatenate(
                        [
                            (b.astype(np.float64) - a.astype(np.float64)).reshape(-1)
                            for a, b in zip(states[step], states[step + 1], strict=True)
                        ]
                    )
                    grad = metric_grads[(objective, weighting, step)]
                    d = values[step + 1] - values[step]
                    s = float(np.dot(grad, delta))
                    r = d - s
                    for key, actual in (
                        ("pre", values[step]),
                        ("post", values[step + 1]),
                        ("d", d),
                        ("s", s),
                        ("r", r),
                    ):
                        _close(
                            float(rec[key]),
                            float(actual),
                            atol,
                            rtol,
                            f"ledger_{key}:{arm}:{step}:{objective}:{weighting}",
                        )
                    archived_grad = grad_archive[
                        f"{arm}_{objective}_{weighting}_{step:02d}"
                    ]
                    if not np.array_equal(archived_grad, grad):
                        raise ValueError(
                            f"altered_gradient:{arm}:{step}:{objective}:{weighting}"
                        )
                    group_dots = {
                        g: float(
                            np.dot(grad[group_indices[g]], delta[group_indices[g]])
                        )
                        for g in GROUPS
                    }
                    if rec["group_s"].keys() != group_dots.keys():
                        raise ValueError("group_dot_keys_invalid")
                    for g in GROUPS:
                        _close(
                            float(rec["group_s"][g]),
                            group_dots[g],
                            atol,
                            rtol,
                            f"group_dot:{g}",
                        )
                    _close(
                        sum(group_dots.values()), s, atol, rtol, "group_reconciliation"
                    )
                    sums["D"] += d
                    sums["S"] += s
                    sums["R"] += r
                _close(sums["D"], values[-1] - values[0], atol, rtol, "telescoping")
                _close(
                    sums["D"], sums["S"] + sums["R"], atol, rtol, "D_equals_S_plus_R"
                )
                totals[f"{objective}/{weighting}"] = sums
                for key in sums:
                    _close(
                        result["arms"][arm]["totals"][f"{objective}/{weighting}"][key],
                        sums[key],
                        atol,
                        rtol,
                        f"total_{key}",
                    )
    classification = _classify(
        {w: result["arms"]["B"]["totals"][f"value_mse/{w}"] for w in weights}
    )
    if result.get("classification") != classification:
        raise ValueError("classification_mismatch")
    return {
        "status": "valid",
        "classification": classification,
        "population": {"exposures": len(ids), "identities": len(set(identities))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
