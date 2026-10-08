"""Retrospective finite-step attribution for the frozen seed442 trajectories."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed445-finite-step-attribution")
ATOL = 2e-7
RTOL = 2e-6
GROUPS = ("shared_trunk", "policy_head", "value_head")


def classify(totals: dict[str, dict[str, float]]) -> str:
    """Apply the frozen signed B-value-MSE classifier to both weightings."""
    first = all(v["D"] > 0 and v["S"] >= 0.75 * v["D"] for v in totals.values())
    if first:
        return "first_order_value_harm_dominant"
    residual = all(v["D"] > 0 and v["R"] >= 0.75 * v["D"] for v in totals.values())
    if residual:
        return "finite_step_value_residual_dominant"
    return "mixed_or_no_common_value_dominance"


def decompose(
    before: float,
    after: float,
    gradient: np.ndarray,
    displacement: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Calculate signed finite-step attribution and reconciled group dots."""
    d = float(after - before)
    s = float(np.dot(gradient, displacement))
    group_s = {
        name: float(np.dot(gradient[ix], displacement[ix]))
        for name, ix in groups.items()
    }
    if not np.isclose(sum(group_s.values()), s, atol=ATOL, rtol=RTOL):
        raise ValueError("group_dot_reconciliation_failed")
    return {"d": d, "s": s, "r": d - s, "group_s": group_s}


def layout_for(model: torch.nn.Module) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    offset = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
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
        stop = offset + param.numel()
        result.append(
            {
                "name": name,
                "shape": list(param.shape),
                "start": offset,
                "stop": stop,
                "group": group,
            }
        )
        offset = stop
    return result


def shaped_gradient_components(
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> tuple[np.ndarray, ...]:
    """Preserve one float64 array per parameter, including unused slots."""
    if len(parameters) != len(gradients):
        raise ValueError("gradient_parameter_count_mismatch")
    return tuple(
        np.zeros(tuple(parameter.shape), dtype=np.float64)
        if gradient is None
        else gradient.detach().cpu().numpy().astype(np.float64)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    )


def weight_vectors(identities: list[str]) -> dict[str, np.ndarray]:
    counts: dict[str, int] = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    return {
        "exposure_weighted": np.full(
            len(identities), 1 / len(identities), dtype=np.float64
        ),
        "equal_input": np.asarray(
            [1 / (len(counts) * counts[i]) for i in identities], dtype=np.float64
        ),
    }


def _flat_grads(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
    objective: str,
    chunk: int = 256,
) -> np.ndarray:
    params = tuple(q for q in model.parameters() if q.requires_grad)
    accum = [np.zeros(tuple(q.shape), dtype=np.float64) for q in params]
    model.eval()
    # Chunk gradients are accumulated with their exact objective mass; chunking bounds memory.
    for begin in range(0, len(ids), chunk):
        rows, w = ids[begin : begin + chunk], weights[begin : begin + chunk]
        xx = torch.from_numpy(x[rows])
        logits, prediction = model(xx)
        if objective == "policy_ce":
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[rows])
            )
            losses = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[rows])
            )
            mass = float(np.sum(w, dtype=np.float64))
            if mass == 0:
                continue
            loss = torch.sum(losses.double() * torch.from_numpy(w).double()) / float(
                np.sum(weights, dtype=np.float64)
            )
        else:
            losses = (
                prediction.reshape(-1).double()
                - torch.from_numpy(v[rows]).reshape(-1).double()
            ).square()
            loss = torch.sum(losses.double() * torch.from_numpy(w).double()) / float(
                np.sum(weights, dtype=np.float64)
            )
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for i, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if grad is not None:
                accum[i] += grad.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([a.reshape(-1) for a in accum])


def _loss(
    model: torch.nn.Module,
    x: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    ids: np.ndarray,
    weights: np.ndarray,
    objective: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits_parts, value_parts, losses = [], [], []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(ids), 512):
            rows = ids[begin : begin + 512]
            logits, prediction = model(torch.from_numpy(x[rows]))
            logits_parts.append(logits.cpu().numpy())
            value_parts.append(prediction.reshape(-1).cpu().numpy())
            if objective == "policy_ce":
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(x[rows])
                )
                vals = (
                    train.compute_policy_cross_entropy(
                        logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(p[rows])
                    )
                    .double()
                    .cpu()
                    .numpy()
                )
            else:
                vals = (
                    (
                        prediction.reshape(-1).double()
                        - torch.from_numpy(v[rows]).reshape(-1).double()
                    )
                    .square()
                    .cpu()
                    .numpy()
                )
            losses.extend(vals.tolist())
    raw = np.asarray(losses, dtype=np.float64)
    return (
        float(
            np.sum(raw * weights, dtype=np.float64) / np.sum(weights, dtype=np.float64)
        ),
        np.concatenate(logits_parts),
        np.concatenate(value_parts),
    )


def _states(
    root: Path, arm: str, model: torch.nn.Module, initializer: list[np.ndarray]
) -> list[list[np.ndarray]]:
    archive_path = root / "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz"
    endpoint_path = root / f"docs/data/seed442-kl-capped-adam-screen/{arm}-final.npz"
    params = list(model.parameters())
    states = [[a.copy() for a in initializer]]
    with np.load(archive_path, allow_pickle=False) as archive:
        for step in range(16):
            pre = [archive[f"{arm}_{step:02d}_pre_{i:02d}"] for i in range(len(params))]
            post = [
                archive[f"{arm}_{step:02d}_post_{i:02d}"] for i in range(len(params))
            ]
            if any(
                not np.array_equal(a, b) for a, b in zip(states[-1], pre, strict=True)
            ):
                raise ValueError(f"trajectory_link_invalid:{arm}:{step + 1}")
            states.append([np.asarray(a).copy() for a in post])
    endpoint = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(endpoint, endpoint_path)
    if any(
        not np.array_equal(a, b.detach().cpu().numpy())
        for a, b in zip(states[-1], endpoint.parameters(), strict=True)
    ):
        raise ValueError(f"trajectory_endpoint_invalid:{arm}")
    return states


def freeze(root: Path) -> Path:
    """Write source/input-bound protocol v2 before any diagnostic gradient."""
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol-v2.json"
    if path.exists():
        raise ValueError("seed445_protocol_is_immutable")
    old_path = out / "protocol.json"
    old_sha = (
        hashlib.sha256(old_path.read_bytes()).hexdigest()
        if old_path.is_file()
        else None
    )
    execution_sources = (
        "ml/alphazero_lite/seed445_finite_step_attribution.py",
        "ml/alphazero_lite/verify_seed445_finite_step_attribution.py",
        "ml/alphazero_lite/test_seed445_finite_step_attribution.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed442_kl_capped_adam.py",
        "ml/alphazero_lite/seed444_minibatch_denominator.py",
        "ml/alphazero_lite/verify_seed443_seed442_trajectory.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
    )
    authoritative = (
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "docs/data/seed442-kl-capped-adam-screen/A-final.npz",
        "docs/data/seed442-kl-capped-adam-screen/B-final.npz",
        "docs/data/seed442-kl-capped-adam-screen/registration.json",
        "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
        "docs/data/seed442-kl-capped-adam-screen/evidence.json",
        "docs/data/seed442-kl-capped-adam-screen/receipt.json",
        "docs/data/seed443-seed442-trajectory/receipt.json",
        "docs/data/seed443-seed442-trajectory/verification-report.json",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed444-minibatch-denominator-audit/corrected-rerun-registration-v2.json",
        "docs/data/seed444-minibatch-denominator-audit/corrected-rerun-evidence.json",
        "docs/data/seed444-minibatch-denominator-audit/receipt.json",
    )
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    replay_files = [
        f"docs/data/seed426-canonical-overlap/sources/{r['name']}.jsonl.gz"
        for r in registration["replays"]
    ]
    authoritative = (*authoritative, *replay_files)
    for rel in (*execution_sources, *authoritative):
        if not (root / rel).is_file():
            raise ValueError(f"seed445_binding_file_missing:{rel}")
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    layout = layout_for(model)
    protocol = {
        "schema": "seed445-frozen-protocol-v2",
        "status": "frozen_before_new_gradients",
        "temporal_scope": "retrospective attribution; endpoints and failed gates were already known before protocol finalization; no prospective outcome claim",
        "preserved_invalid_draft": {
            "path": "docs/data/seed445-finite-step-attribution/protocol.json",
            "sha256": old_sha,
            "reason": "preparation edits after initial draft changed source bytes; draft binding invalidated before any diagnostic gradient; original bytes preserved",
        },
        "primary_arm": "B",
        "primary_objective": "unseen value MSE, unweighted by training coefficient",
        "reference": "A and policy CE descriptive only",
        "population": "unseen >32; 2,607 exposure rows and 1,242 exact-input identities",
        "weightings": {
            "exposure_weighted": "each exposure equal mass",
            "equal_input": "within-identity exposure mean then identities equal mass",
        },
        "step": "16 accepted steps, 17 states; exact initializer/pre/post state continuity; B uses accepted archived post tensors including selected scales; float64 difference of adjacent float32 states; d=L(post)-L(pre); s=gradient L(pre) dot displacement; r=d-s",
        "objectives": {
            "policy_ce": "production legal-masked policy CE",
            "value_mse": "mean squared prediction error; no training coefficient",
        },
        "classification": {
            "order": [
                "if D>0 and S>=0.75D for both weightings: first_order_value_harm_dominant",
                "else if D>0 and R>=0.75D for both weightings: finite_step_value_residual_dominant",
                "else: mixed_or_no_common_value_dominance",
            ],
            "signed": True,
        },
        "tolerances": {
            "absolute": ATOL,
            "relative": RTOL,
            "arithmetic_identity": "all scalar reconstruction checks use numpy.isclose with these tolerances; gradient archive comparisons exact after float64 serialization",
        },
        "parameter_layout": layout,
        "weighting_arithmetic": "exposure weights are uniform over all 2,607 selected exposure rows; equal-input weights assign each of 1,242 identities equal total mass, retaining all selected exposures within each identity",
        "gradient_arithmetic": "float32 autograd gradients converted to float64 and accumulated over deterministic chunks; retain every requires_grad parameter in named-parameter order, with correctly shaped float64 zeros for None gradients",
        "group_arithmetic": "shared_trunk=input_layer/residual_layers; policy_head=policy_hidden_layer/policy_head; value_head=value_hidden_layer/value_head; group first-order dots reconcile to full dot; residual is not grouped",
        "input_arithmetic": "reconstruct replay rows with portable compressed source snapshots and frozen registration/loader semantics; membership identities select exact input rows; no historical run-local paths",
        "execution_source_sha256": {
            rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
            for rel in execution_sources
        },
        "authoritative_input_sha256": {
            rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
            for rel in authoritative
        },
        "interpretation": "descriptive retrospective attribution, not causal interventions or prospective outcome evidence; residual includes nonlinear and numerical effects and is not pure curvature; authorizes no training, gate change, or strength claim",
    }
    path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def run(root: Path = ROOT, publish: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if publish:
        if not (root / OUT / "protocol-v2.json").is_file():
            freeze(root)
    protocol_path = root / OUT / "protocol-v2.json"
    if not protocol_path.is_file():
        raise ValueError("seed445_protocol_v2_missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _check_bindings(root, protocol)
    from ml.alphazero_lite.seed444_minibatch_denominator import _load_frozen_inputs

    reg416 = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text(encoding="utf-8")
    )
    arrays = _load_frozen_inputs(root, reg416)
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
    if (len(ids), len(set(identities))) != (2607, 1242):
        raise ValueError("selected_population_invalid")
    weightings = weight_vectors(identities)
    model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    init_path = (
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
    )
    train.load_checkpoint_into_model(model, init_path)
    initializer = [a.detach().cpu().numpy().copy() for a in model.parameters()]
    layout = layout_for(model)
    result: dict[str, Any] = {
        "schema": "seed445-finite-step-attribution-v1",
        "status": "retrospective_frozen_trajectory_attribution",
        "tolerances": {"absolute": ATOL, "relative": RTOL},
        "membership": {"exposures": len(ids), "identities": len(set(identities))},
        "parameter_layout": layout,
        "row_identities": identities,
        "row_compact_indices": ids.tolist(),
        "arms": {},
    }
    prediction_archive: dict[str, np.ndarray] = {
        "row_identities": np.asarray(identities)
    }
    gradient_archive: dict[str, np.ndarray] = {}
    _check_bindings(root, protocol)
    seed442_metrics = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]
    for arm in ("A", "B"):
        states = _states(root, arm, model, initializer)
        per_arm: dict[str, Any] = {"step_ledger": [], "totals": {}}
        for objective in ("policy_ce", "value_mse"):
            for weighting, weights in weightings.items():
                values = []
                for si, state in enumerate(states):
                    with torch.no_grad():
                        for parameter, value in zip(
                            model.parameters(), state, strict=True
                        ):
                            parameter.copy_(torch.from_numpy(value.copy()))
                    val, logits, pred = _loss(model, x, p, v, ids, weights, objective)
                    values.append(val)
                    prediction_archive[f"{arm}_{si:02d}_logits"] = logits
                    prediction_archive[f"{arm}_{si:02d}_values"] = pred
                records = []
                sums = {k: 0.0 for k in ("D", "S", "R")}
                for si in range(16):
                    with torch.no_grad():
                        for parameter, value in zip(
                            model.parameters(), states[si], strict=True
                        ):
                            parameter.copy_(torch.from_numpy(value.copy()))
                    grad = _flat_grads(model, x, p, v, ids, weights, objective)
                    delta = np.concatenate(
                        [
                            (b.astype(np.float64) - a.astype(np.float64)).reshape(-1)
                            for a, b in zip(states[si], states[si + 1], strict=True)
                        ]
                    )
                    if grad.shape != delta.shape:
                        raise ValueError("gradient_layout_mismatch")
                    diff = values[si + 1] - values[si]
                    group_indices = {
                        group: np.concatenate(
                            [
                                np.arange(e["start"], e["stop"])
                                for e in layout
                                if e["group"] == group
                            ]
                        )
                        for group in GROUPS
                    }
                    pieces = decompose(
                        values[si], values[si + 1], grad, delta, group_indices
                    )
                    dot, residual, group_dots = (
                        pieces["s"],
                        pieces["r"],
                        pieces["group_s"],
                    )
                    rec = {
                        "step": si + 1,
                        "objective": objective,
                        "weighting": weighting,
                        "pre": values[si],
                        "post": values[si + 1],
                        "d": pieces["d"],
                        "s": dot,
                        "r": residual,
                        "group_s": group_dots,
                    }
                    records.append(rec)
                    sums["D"] += diff
                    sums["S"] += dot
                    sums["R"] += residual
                    gradient_archive[f"{arm}_{objective}_{weighting}_{si:02d}"] = grad
                if not np.isclose(
                    sums["D"], values[-1] - values[0], atol=ATOL, rtol=RTOL
                ) or not np.isclose(
                    sums["D"], sums["S"] + sums["R"], atol=ATOL, rtol=RTOL
                ):
                    raise ValueError("telescoping_or_residual_identity_failed")
                per_arm["step_ledger"].extend(records)
                per_arm["totals"][f"{objective}/{weighting}"] = {
                    **sums,
                    "endpoint_difference": values[-1] - values[0],
                }
                expected = seed442_metrics[arm][weighting][objective]
                if not np.isclose(values[-1], expected, atol=ATOL, rtol=RTOL):
                    raise ValueError(
                        f"seed442_endpoint_metric_mismatch:{arm}:{objective}:{weighting}"
                    )
        per_arm["path_states"] = 17
        result["arms"][arm] = per_arm
    result["classification"] = classify(
        {w: result["arms"]["B"]["totals"][f"value_mse/{w}"] for w in weightings}
    )
    if publish:
        out = root / OUT
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "predictions.npz", **prediction_archive)  # type: ignore[arg-type]
        np.savez_compressed(out / "gradients.npz", **gradient_archive)  # type: ignore[arg-type]
        (out / "step-ledger.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        (out / "parameter-layout.json").write_text(
            json.dumps(layout, indent=2, sort_keys=True) + "\n"
        )
        (out / "row-identities.json").write_text(
            json.dumps(
                {"input_identity": identities, "compact_row": ids.tolist()},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        _write_analysis(out, result)
        _write_receipt(root, out)
    return result


def _write_analysis(out: Path, result: dict[str, Any]) -> None:
    b = result["arms"]["B"]["totals"]
    a = result["arms"]["A"]["totals"]
    lines = [
        "# seed445 — paired policy/value finite-step attribution",
        "",
        "Retrospective attribution of the already-known frozen seed442 paths. The endpoints and failed gates predate this diagnostic; this is not prospective outcome evidence.",
        "",
        f"**B unseen value-MSE classification: `{result['classification']}`.**",
        "",
        "## Primary B value MSE",
        "",
        "| Weighting | D | S | R | endpoint difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for w in ("exposure_weighted", "equal_input"):
        row = b[f"value_mse/{w}"]
        lines.append(
            f"| {w} | {row['D']:.12g} | {row['S']:.12g} | {row['R']:.12g} | {row['endpoint_difference']:.12g} |"
        )
    lines += [
        "",
        "## Descriptive A endpoint totals",
        "",
        f"A value MSE: `{json.dumps({w: a[f'value_mse/{w}'] for w in ('exposure_weighted', 'equal_input')}, sort_keys=True)}`",
        "",
        "Policy CE totals for A and B are in the complete step ledger.",
        "",
        "Group contributions decompose only first-order dots; residuals are not assigned to groups and are not pure curvature estimates. No classification authorizes training, changes historical gates, or establishes playing strength.",
        "",
    ]
    (out / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _write_receipt(root: Path, out: Path) -> None:
    artifacts = [p for p in out.iterdir() if p.is_file() and p.name != "receipt.json"]
    receipt = {
        "schema": "seed445-publication-receipt-v1",
        "status": "published_retrospective_attribution",
        "protocol_sha256": hashlib.sha256(
            (out / "protocol-v2.json").read_bytes()
        ).hexdigest(),
        "artifacts_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(artifacts)
        },
        "historical_seed442_sha256": {
            rel: protocol_hash(root, rel)
            for rel in (
                "docs/data/seed442-kl-capped-adam-screen/registration.json",
                "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
                "docs/data/seed442-kl-capped-adam-screen/A-final.npz",
                "docs/data/seed442-kl-capped-adam-screen/B-final.npz",
                "docs/data/seed442-kl-capped-adam-screen/receipt.json",
            )
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def protocol_hash(root: Path, relative: str) -> str:
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def _check_bindings(root: Path, protocol: dict[str, Any]) -> None:
    bindings = {
        **protocol["execution_source_sha256"],
        **protocol["authoritative_input_sha256"],
    }
    for relative, expected in bindings.items():
        if not (root / relative).is_file() or protocol_hash(root, relative) != expected:
            raise ValueError(f"seed445_frozen_binding_mismatch:{relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        print(freeze(args.root))
    else:
        print(json.dumps(run(args.root, args.publish), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
