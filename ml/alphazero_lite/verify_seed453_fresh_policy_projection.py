"""Independent, read-only reconstruction verifier for seed453."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import seed447_joint_output_cap as seed447
from ml.alphazero_lite import train

OUT = Path("docs/data/seed453-fresh-policy-projection")
SCALES = tuple(2.0**-i for i in range(8))


def sha(path: Path) -> str:
    """Hash file bytes without writing to the checkout."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_inputs(root: Path) -> tuple[np.ndarray, ...]:
    """Reconstruct registered lane-A derivatives without opening old absolutes."""
    seed416_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    seed416 = json.loads(seed416_path.read_text())
    scratch = root / ".tmp"
    scratch.mkdir(exist_ok=True)
    from ml.alphazero_lite.verify_seed450_seed449_correction import (
        _reconstruct_derivatives,
    )
    from ml.alphazero_lite import seed435_adam_direction as seed435

    with tempfile.TemporaryDirectory(
        prefix="seed453-verify-", dir=scratch
    ) as directory:
        derivatives = _reconstruct_derivatives(root, seed416, Path(directory))
        original = seed435.load_jsonl_replay

        def rooted_loader(paths: Any, weights: Any = None, **kwargs: Any) -> Any:
            return original(
                [derivatives[row["name"]] for row in seed416["replays"]],
                weights,
                **kwargs,
            )

        train.load_jsonl_replay = rooted_loader
        seed435.load_jsonl_replay = rooted_loader
        try:
            return seed442.load_inputs()
        finally:
            train.load_jsonl_replay = original
            seed435.load_jsonl_replay = original


def require_array(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    """Reject semantically changed numeric evidence."""
    if not np.array_equal(actual, expected):
        raise ValueError(f"semantic_tensor_mismatch:{label}")


def require_value(actual: Any, expected: Any, label: str) -> None:
    """Reject changed identities, scales, or fixed decision values."""
    if actual != expected:
        raise ValueError(f"semantic_value_mismatch:{label}")


def require_close(actual: float, expected: float, label: str) -> None:
    """Reject changed recorded numeric metrics under a fixed tolerance."""
    if not np.isclose(actual, expected, atol=2e-12, rtol=0):
        raise ValueError(f"semantic_metric_mismatch:{label}")


def _independent_fresh_gradient(
    model: torch.nn.Module,
    x: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
) -> list[np.ndarray]:
    params = tuple(p for p in model.parameters() if p.requires_grad)
    total = [np.zeros(tuple(p.shape), dtype=np.float64) for p in params]
    for start in range(0, len(rows), 512):
        selected = rows[start : start + 512]
        logits, _ = model(torch.from_numpy(x[selected]))
        legal = torch.from_numpy(
            train.legal_mask_matrix_for_encoded_states(x[selected])
        )
        ce = train.compute_policy_cross_entropy(
            logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(target[selected])
        )
        gradients = torch.autograd.grad(ce.sum() / len(rows), params, allow_unused=True)
        for index, gradient in enumerate(gradients):
            if gradient is not None:
                total[index] += gradient.detach().cpu().numpy().astype(np.float64)
    return [value.astype(np.float32) for value in total]


def _fresh_guard(root: Path, arrays: tuple[np.ndarray, ...]) -> list[dict[str, Any]]:
    x, _p, _v, replay, _coeff, train_pos, _val, sources = arrays
    training = set(map(int, replay[train_pos]))
    base = seed442.guard_cohort(root, arrays)
    unseen = [
        r
        for r in seed442.seed435.membership_rows()
        if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    exact_unseen = {r["input_identity"] for r in unseen}
    canonical_unseen = {r["canonical_identity"] for r in unseen}
    result = []
    seen: set[str] = set()
    for record in base:
        row = int(record["compact_row"])
        if sources[row].get("source") != "fresh":
            continue
        identity = x[row].astype("<f4", copy=False).tobytes().hex()
        if (
            row not in training
            or identity != record["exact_input_identity"]
            or identity in seen
        ):
            raise ValueError(f"fresh_guard_reconstruction_invalid:{row}")
        if identity in exact_unseen or record["canonical_identity"] in canonical_unseen:
            raise ValueError(f"fresh_guard_evaluation_overlap:{row}")
        seen.add(identity)
        result.append(record)
    return result


def _dot(delta: list[np.ndarray], gradient: list[np.ndarray]) -> float:
    return float(
        sum(
            np.sum(a.astype(np.float64) * b.astype(np.float64), dtype=np.float64)
            for a, b in zip(delta, gradient, strict=True)
        )
    )


def _fresh_endpoint(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    members: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Recompute fresh strict-unseen CE/MSE under both registered weightings."""
    x, policy, target_value, *_ = arrays
    rows = [
        r
        for r in members
        if r["subset"] == "unseen"
        and r["active_stones"] > 32
        and r.get("source") == "fresh"
    ]
    ids = np.asarray([int(r["compact_row"]) for r in rows], dtype=np.int64)
    losses = []
    squared = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(ids), 512):
            selected = ids[offset : offset + 512]
            logits, prediction = model(torch.from_numpy(x[selected]))
            legal = torch.from_numpy(
                train.legal_mask_matrix_for_encoded_states(x[selected])
            )
            ce = train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0, -1e9), torch.from_numpy(policy[selected])
            )
            losses.extend(ce.double().cpu().numpy().tolist())
            value = prediction.reshape(-1).double().cpu().numpy()
            squared.extend(
                np.square(value - target_value[selected, 0].astype(np.float64)).tolist()
            )
    losses_array = np.asarray(losses, dtype=np.float64)
    squared_array = np.asarray(squared, dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["input_identity"], []).append(index)
    return {
        "exposure_weighted": {
            "policy_ce": float(np.mean(losses_array)),
            "value_mse": float(np.mean(squared_array)),
        },
        "equal_input": {
            "policy_ce": float(
                np.mean([np.mean(losses_array[group]) for group in groups.values()])
            ),
            "value_mse": float(
                np.mean([np.mean(squared_array[group]) for group in groups.values()])
            ),
        },
    }


def verify(root: Path) -> dict[str, Any]:
    """Reconstruct frozen inputs, gradients, Adam, projections, trials, and outputs."""
    root = root.resolve()
    out = root / OUT
    seed442.configure_root(root)
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    amendment = json.loads((out / "verifier-correction.json").read_text())
    correction_receipt_path = out / "verifier-correction-receipt.json"
    correction_receipt = json.loads(correction_receipt_path.read_text())
    for rel, digest in correction_receipt["files_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"verifier_correction_receipt_mismatch:{rel}")
    for rel, digest in registration["source_sha256"].items():
        if rel in amendment.get("corrected_source_sha256", {}):
            snapshot = out / "source-snapshots" / Path(rel).name
            if (
                sha(snapshot) != digest
                or amendment["registered_source_sha256"].get(rel) != digest
                or amendment["corrected_source_sha256"].get(rel) != sha(root / rel)
            ):
                raise ValueError(f"source_correction_binding_invalid:{rel}")
            continue
        if (
            sha(root / rel) != digest
            or sha(out / "source-snapshots" / Path(rel).name) != digest
        ):
            raise ValueError(f"source_binding_invalid:{rel}")
    for rel, digest in registration["input_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"input_binding_invalid:{rel}")
    arrays = _portable_inputs(root)
    x, policy, value, replay, coeff, train_positions, _validation, _source = arrays
    guard = _fresh_guard(root, arrays)
    require_value(guard, registration["guard"], "guard_identities")
    if sha(out / "fresh-guard.json") != registration["guard_sha256"]:
        raise ValueError("guard_semantic_binding_invalid")
    membership = seed442.seed435.membership_rows()
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    if registration["unseen_population"] != [
        {
            key: r[key]
            for key in (
                "compact_row",
                "input_identity",
                "canonical_identity",
                "active_stones",
            )
        }
        for r in unseen
    ]:
        raise ValueError("strict_unseen_population_changed")
    guard_rows = np.asarray([r["compact_row"] for r in guard], dtype=np.int64)
    train_rows = replay[train_positions]
    batches = np.asarray(registration["seed442_first_16_positions"], dtype=np.int64)
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence["registration_sha256"] != sha(registration_path):
        raise ValueError("evidence_registration_binding_invalid")
    archive_contexts = {
        arm: np.load(out / f"step-tensors-{arm}.npz", allow_pickle=False)
        for arm in ("C", "B")
    }
    try:
        init = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(init, seed442.seed435.INIT)
        initial = [p.detach().cpu().numpy().copy() for p in init.parameters()]
        for arm in ("C", "B"):
            archive = archive_contexts[arm]
            model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(model, seed442.seed435.INIT)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
            )
            previous = [a.copy() for a in initial]
            previous_avg = [np.zeros_like(a) for a in initial]
            previous_sq = [np.zeros_like(a) for a in initial]
            records = evidence["arms"][arm]["steps"]
            if len(records) != 16:
                raise ValueError(f"proposal_count_invalid:{arm}")
            for step, record in enumerate(records):
                if (
                    record.get("step") != step + 1
                    or record.get("optimizer_step") != step + 1
                ):
                    raise ValueError(f"step_index_invalid:{arm}:{step}")
                rows = train_rows[batches[step * 512 : (step + 1) * 512]]
                if record["compact_rows"] != rows.tolist():
                    raise ValueError(f"batch_identity_invalid:{arm}:{step}")
                kinds = (
                    "pre",
                    "raw_proposal",
                    "projected_proposal",
                    "post",
                    "objective_gradient_unclipped",
                    "objective_gradient",
                    "fresh_gradient",
                    "exp_avg_pre",
                    "exp_avg_sq_pre",
                    "exp_avg",
                    "exp_avg_sq",
                )
                archived = {
                    kind: [
                        archive[f"{arm}_{step:02d}_{i:02d}_{kind}"].copy()
                        for i in range(len(params))
                    ]
                    for kind in kinds
                }
                pre = archived["pre"]
                for index, tensor in enumerate(pre):
                    require_array(
                        tensor, previous[index], f"trajectory:{arm}:{step}:{index}"
                    )
                    params[index].data.copy_(torch.from_numpy(tensor.copy()))
                    params[index].grad = None
                for i, param in enumerate(params):
                    state = optimizer.state[param]
                    if step:
                        param_state_avg = archived["exp_avg_pre"][i]
                        param_state_sq = archived["exp_avg_sq_pre"][i]
                        require_array(
                            param_state_avg,
                            previous_avg[i],
                            f"moment_continuity:{arm}:{step}:{i}",
                        )
                        require_array(
                            param_state_sq,
                            previous_sq[i],
                            f"moment_sq_continuity:{arm}:{step}:{i}",
                        )
                    else:
                        param_state_avg = np.zeros_like(pre[i])
                        param_state_sq = np.zeros_like(pre[i])
                    if (
                        not np.array_equal(param_state_avg, np.zeros_like(pre[i]))
                        and step == 0
                    ):
                        raise ValueError("nonzero_initial_moment")
                    state["step"] = torch.tensor(float(step), dtype=torch.float32)
                    state["exp_avg"] = torch.from_numpy(param_state_avg.copy())
                    state["exp_avg_sq"] = torch.from_numpy(param_state_sq.copy())
                logits, predicted = model(torch.from_numpy(x[rows]))
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(x[rows])
                )
                cb = torch.from_numpy(coeff[rows])
                pol = (
                    train.compute_policy_cross_entropy(
                        logits.masked_fill(legal <= 0, -1e9),
                        torch.from_numpy(policy[rows]),
                    )
                    * cb
                ).sum() / cb.sum()
                val = train.compute_value_loss_vector(
                    predicted,
                    torch.from_numpy(value[rows]),
                    value_loss="huber",
                    huber_delta=1.0,
                ).mean()
                objective = pol + 0.3 * val
                objective.backward()
                unclipped = [p.grad.detach().clone() for p in params]
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                clipped = [p.grad.detach().clone() for p in params]
                for i in range(len(params)):
                    require_array(
                        unclipped[i].cpu().numpy(),
                        archived["objective_gradient_unclipped"][i],
                        f"objective_gradient:{arm}:{step}:{i}",
                    )
                    require_array(
                        clipped[i].cpu().numpy(),
                        archived["objective_gradient"][i],
                        f"clipped_gradient:{arm}:{step}:{i}",
                    )
                fresh = (
                    _independent_fresh_gradient(model, x, policy, guard_rows)
                    if arm == "B"
                    else [np.zeros_like(p) for p in pre]
                )
                if any(
                    not torch.equal(param.grad, clipped_grad)
                    for param, clipped_grad in zip(params, clipped, strict=True)
                ):
                    raise ValueError("fresh_gradient_overwrote_objective_gradient")
                for i in range(len(params)):
                    require_array(
                        fresh[i],
                        archived["fresh_gradient"][i],
                        f"fresh_gradient:{arm}:{step}:{i}",
                    )
                optimizer.step()
                raw = [p.detach().cpu().numpy().copy() for p in params]
                proposal_norm = float(
                    np.sqrt(
                        sum(
                            np.sum((b.astype(np.float64) - a.astype(np.float64)) ** 2)
                            for a, b in zip(pre, raw, strict=True)
                        )
                    )
                )
                if not np.isclose(
                    proposal_norm, record["proposal_norm"], atol=1e-12, rtol=1e-12
                ):
                    raise ValueError(f"proposal_norm_tampered:{arm}:{step}")
                for i in range(len(params)):
                    require_array(
                        raw[i],
                        archived["raw_proposal"][i],
                        f"adam_proposal:{arm}:{step}:{i}",
                    )
                    require_array(
                        optimizer.state[params[i]]["exp_avg"].cpu().numpy(),
                        archived["exp_avg"][i],
                        f"moment:{arm}:{step}:{i}",
                    )
                    require_array(
                        optimizer.state[params[i]]["exp_avg_sq"].cpu().numpy(),
                        archived["exp_avg_sq"][i],
                        f"moment_sq:{arm}:{step}:{i}",
                    )
                    previous_avg[i] = archived["exp_avg"][i].copy()
                    previous_sq[i] = archived["exp_avg_sq"][i].copy()
                if arm == "B":
                    displacement = [
                        b.astype(np.float64) - a.astype(np.float64)
                        for a, b in zip(pre, raw, strict=True)
                    ]
                    g64 = [g.astype(np.float64) for g in fresh]
                    dot = _dot(displacement, g64)
                    norm_sq = sum(np.sum(g * g, dtype=np.float64) for g in g64)
                    coefficient = max(0.0, dot) / norm_sq if norm_sq > 0 else 0.0
                    projection = [
                        np.asarray(
                            a.astype(np.float64) + d - coefficient * g, dtype=np.float32
                        )
                        for a, d, g in zip(pre, displacement, g64, strict=True)
                    ]
                    registered_projection = record["projection"]
                    computed_projection = {
                        "raw_dot": dot,
                        "gradient_norm": float(np.sqrt(norm_sq)),
                        "coefficient": coefficient,
                    }
                    for key, calculated in computed_projection.items():
                        if not np.isclose(
                            calculated,
                            registered_projection[key],
                            atol=1e-12,
                            rtol=1e-12,
                        ):
                            raise ValueError(
                                f"projection_report_tampered:{arm}:{step}:{key}"
                            )
                else:
                    projection = raw
                for i in range(len(params)):
                    require_array(
                        projection[i],
                        archived["projected_proposal"][i],
                        f"projection:{arm}:{step}:{i}",
                    )
                with torch.no_grad():
                    for parameter, before in zip(params, pre, strict=True):
                        parameter.copy_(torch.from_numpy(before.copy()))
                pre_policy = seed442._pre_policy(model, x, rows)
                guard_policy = seed442._pre_policy(model, x, guard_rows)
                pre_batch_values = seed447._predict_values(model, x, rows)
                pre_guard_values = seed447._predict_values(model, x, guard_rows)
                trial_rows = []
                for trial, scale in zip(record["trials"], SCALES, strict=True):
                    require_value(trial["scale"], scale, f"scale:{arm}:{step}")
                    with torch.no_grad():
                        for param, before, proposal in zip(
                            params, pre, projection, strict=True
                        ):
                            if scale == 1:
                                param.copy_(torch.from_numpy(proposal.copy()))
                            else:
                                param.copy_(
                                    torch.from_numpy(
                                        np.asarray(
                                            before + (proposal - before) * scale,
                                            dtype=np.float32,
                                        )
                                    )
                                )
                    bkl = seed442._kl(
                        model, x, rows, np.full(len(rows), 1 / len(rows)), pre_policy
                    )
                    gkl = seed442._kl(
                        model,
                        x,
                        guard_rows,
                        np.full(len(guard_rows), 1 / len(guard_rows)),
                        guard_policy,
                    )
                    bm = seed447.value_movement(
                        pre_batch_values,
                        seed447._predict_values(model, x, rows),
                        np.full(len(rows), 1 / len(rows)),
                    )
                    gm = seed447.value_movement(
                        pre_guard_values,
                        seed447._predict_values(model, x, guard_rows),
                        np.full(len(guard_rows), 1 / len(guard_rows)),
                    )
                    realized = [p.detach().cpu().numpy().copy() for p in params]
                    for i, trial_state in enumerate(realized):
                        require_array(
                            trial_state,
                            archive[f"{arm}_{step:02d}_{i:02d}_trial_{scale:g}"],
                            f"trial_parameters:{arm}:{step}:{scale}:{i}",
                        )
                    delta = [
                        b.astype(np.float64) - a.astype(np.float64)
                        for a, b in zip(pre, realized, strict=True)
                    ]
                    fresh_dot = _dot(delta, fresh) if arm == "B" else 0.0
                    values = {
                        "batch_kl": bkl,
                        "guard_kl": gkl,
                        "batch_value_movement": bm,
                        "guard_value_movement": gm,
                        "realized_fresh_dot": fresh_dot,
                    }
                    for key, actual in values.items():
                        require_close(
                            actual, trial[key], f"trial:{arm}:{step}:{scale}:{key}"
                        )
                    feasible = (
                        bkl <= 0.005 + 1e-10
                        and gkl <= 0.005 + 1e-10
                        and bm <= 0.0001 + 1e-12
                        and gm <= 0.0001 + 1e-12
                        and (arm == "C" or fresh_dot <= 1e-7)
                    )
                    if trial["feasible"] != feasible:
                        raise ValueError(f"feasibility_tampered:{arm}:{step}:{scale}")
                    trial_rows.append({"scale": scale, "feasible": feasible})
                selected = next((t["scale"] for t in trial_rows if t["feasible"]), None)
                if selected != record["selected_scale"] or record["rejected"] != (
                    selected is None
                ):
                    raise ValueError(f"selected_scale_tampered:{arm}:{step}")
                expected_post = (
                    pre
                    if selected is None
                    else [
                        b.copy()
                        if selected == 1
                        else np.asarray(a + (b - a) * selected, dtype=np.float32)
                        for a, b in zip(pre, projection, strict=True)
                    ]
                )
                for i, param in enumerate(params):
                    require_array(
                        expected_post[i], archived["post"][i], f"post:{arm}:{step}:{i}"
                    )
                    previous[i] = expected_post[i].copy()
                    param.data.copy_(torch.from_numpy(expected_post[i].copy()))
                changed = any(
                    not np.array_equal(a, b)
                    for a, b in zip(raw, projection, strict=True)
                )
                activated = (
                    arm == "B"
                    and record["projection"]["raw_dot"] > 0
                    and changed
                    and selected is not None
                    and any(
                        not np.array_equal(a, b)
                        for a, b in zip(raw, expected_post, strict=True)
                    )
                )
                if (
                    record["projection_changed"] != changed
                    or record["activation"] != activated
                ):
                    raise ValueError(f"projection_activation_tampered:{arm}:{step}")
            final = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(final, out / f"{arm}-final.npz")
            if any(
                not np.array_equal(p.detach().cpu().numpy(), state)
                for p, state in zip(final.parameters(), previous, strict=True)
            ):
                raise ValueError(f"endpoint_mismatch:{arm}")
            if (
                sha(out / f"{arm}-final.npz")
                != evidence["arms"][arm]["checkpoint_sha256"]
            ):
                raise ValueError(f"checkpoint_hash_invalid:{arm}")
            if arm == "C":
                with np.load(
                    root / "docs/data/seed447-joint-output-cap/step-tensors.npz",
                    allow_pickle=False,
                ) as old:
                    old_ev = json.loads(
                        (
                            root / "docs/data/seed447-joint-output-cap/evidence.json"
                        ).read_text()
                    )
                    if [r["selected_scale"] for r in records] != [
                        r["selected_scale"] for r in old_ev["arms"]["T"]["steps"]
                    ]:
                        raise ValueError("control_scale_mismatch_seed447_T")
                    for step in range(16):
                        for i in range(len(params)):
                            require_array(
                                previous[i]
                                if step == 15
                                else archive[f"C_{step:02d}_{i:02d}_post"],
                                old[f"T_{step:02d}_post_{i:02d}"],
                                f"control_post:{step}:{i}",
                            )
    finally:
        for archive_context in archive_contexts.values():
            archive_context.close()
    # Endpoint metrics and saved prediction tensors are recomputed using the historic
    # read-only metric implementation plus explicit seed451 source masking.
    from ml.alphazero_lite.verify_seed447_joint_output_cap import (
        _metrics,
        _verify_predictions,
    )

    seed442_evidence = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
    )
    if evidence["metrics"]["seed442_A"] != seed442_evidence["metrics"]["A"]:
        raise ValueError("ordinary_adam_comparator_tampered")

    for label in ("initializer", "C", "B"):
        model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
        train.load_checkpoint_into_model(
            model,
            seed442.seed435.INIT
            if label == "initializer"
            else out / f"{label}-final.npz",
        )
        actual = _metrics(model, arrays, membership)
        recorded = evidence["metrics"][label]
        seed447_verifier_metrics = actual
        for weight in ("exposure_weighted", "equal_input"):
            for metric in ("policy_ce", "value_mse"):
                if not np.isclose(
                    seed447_verifier_metrics[weight][metric],
                    recorded[weight][metric],
                    atol=2e-6,
                    rtol=2e-6,
                ):
                    raise ValueError(
                        f"endpoint_metric_mismatch:{label}:{weight}:{metric}"
                    )
        if not np.isclose(
            actual["full_training_objective"],
            recorded["full_training_objective"],
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError(f"training_objective_mismatch:{label}")
        expected_hash = seed442.vector_hash(
            [parameter for parameter in model.parameters()]
        )
        if (
            label != "initializer"
            and expected_hash != evidence["arms"][label]["final_parameter_sha256"]
        ):
            raise ValueError(f"final_parameter_hash_invalid:{label}")
        fresh_actual = _fresh_endpoint(model, arrays, membership)
        for weight in ("exposure_weighted", "equal_input"):
            for metric in ("policy_ce", "value_mse"):
                if not np.isclose(
                    fresh_actual[weight][metric],
                    recorded["fresh"][weight][metric],
                    atol=2e-6,
                    rtol=2e-6,
                ):
                    raise ValueError(
                        f"fresh_endpoint_metric_mismatch:{label}:{weight}:{metric}"
                    )
        _verify_predictions(out, label, model, x, unseen)
    result = evidence["decision"]["classification"]
    recalculated = _decision(evidence["metrics"], evidence["arms"]["B"]["steps"])
    require_value(result, recalculated, "decision")
    expected_effect = {
        "full_training_objective": evidence["metrics"]["B"]["full_training_objective"]
        - evidence["metrics"]["C"]["full_training_objective"]
    }
    for weighting in ("exposure_weighted", "equal_input"):
        expected_effect[weighting] = {
            metric: evidence["metrics"]["B"][weighting][metric]
            - evidence["metrics"]["C"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
    if evidence["metrics"]["B_minus_C"] != expected_effect:
        raise ValueError("B_minus_C_effect_tampered")
    receipt = json.loads((out / "receipt.json").read_text())
    for rel, digest in receipt["files_sha256"].items():
        if sha(root / rel) != digest:
            raise ValueError(f"receipt_mismatch:{rel}")
    return {
        "status": "valid",
        "steps_per_arm": 16,
        "fresh_guard_rows": len(guard),
        "classification": result,
    }


def _decision(metrics: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    checks = []
    for weighting in ("exposure_weighted", "equal_input"):
        b = metrics["B"][weighting]
        init = metrics["initializer"][weighting]
        a = metrics["seed442_A"][weighting]
        checks.extend(
            (
                b["policy_ce"] <= a["policy_ce"] - 0.01,
                b["policy_ce"] <= init["policy_ce"] - 0.005,
                b["value_mse"] <= init["value_mse"] + 0.002,
                metrics["B"]["fresh"][weighting]["policy_ce"]
                <= metrics["initializer"]["fresh"][weighting]["policy_ce"] + 2e-6,
            )
        )
    checks.append(
        metrics["B"]["full_training_objective"]
        < metrics["initializer"]["full_training_objective"]
    )
    active = any(s["activation"] and not s["rejected"] for s in steps)
    checks.append(active)
    checks.append(not any(s["proposal_norm"] > 0 and s["rejected"] for s in steps))
    if not active:
        return "fresh_projection_inactive_no_followup"
    return (
        "advance_to_separately_preregistered_strength_experiment"
        if all(checks)
        else "close_fresh_policy_projection_branch"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
