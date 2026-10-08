"""Independent read-only replay verifier for prospective seed442 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed441_receipt_hashes import (
    verify as verify_seed441_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_receipt(root: Path) -> dict[str, Any]:
    """Bind the finalized training evidence and post-execution verifier sources."""
    root = root.resolve()
    seed442.configure_root(root)
    out = seed442.OUT
    sources = (
        "ml/alphazero_lite/seed442_kl_capped_adam.py",
        "ml/alphazero_lite/seed442_analysis.py",
        "ml/alphazero_lite/verify_seed442_kl_capped_adam.py",
        "ml/alphazero_lite/test_seed442_kl_capped_adam.py",
        "ml/alphazero_lite/verify_seed442_publication.py",
        "ml/alphazero_lite/test_seed442_publication.py",
    )
    files = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "receipt.json"
    }
    receipt = {
        "schema": "seed442-publication-receipt-v1",
        "status": "post-execution-verification-binding",
        "registration_sha256": _sha(out / "registration.json"),
        "evidence_files_sha256": files,
        "source_sha256": {relative: _sha(root / relative) for relative in sources},
        "interpretation": "retrospective validation screen; not evidence of playing strength",
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def publish_optimizer_state_reconstruction(root: Path) -> dict[str, Any]:
    """Recover post-step Adam tensors from the archived clipped gradients."""
    root = root.resolve()
    seed442.configure_root(root)
    out = seed442.OUT
    evidence = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    reconstructed: dict[str, np.ndarray] = {}
    with np.load(out / "step-tensors.npz", allow_pickle=False) as archive:
        for arm in ("A", "B"):
            count = sum(key.startswith(f"{arm}_00_pre_") for key in archive.files)
            moments = [
                torch.zeros_like(
                    torch.from_numpy(archive[f"{arm}_00_pre_{i:02d}"].copy())
                )
                for i in range(count)
            ]
            variances = [torch.zeros_like(value) for value in moments]
            for step, log in enumerate(evidence["arms"][arm]["steps"]):
                for index in range(count):
                    gradient = torch.from_numpy(
                        archive[f"{arm}_{step:02d}_gradient_{index:02d}"].copy()
                    )
                    moments[index].lerp_(gradient, 0.1)
                    variances[index].mul_(0.999).addcmul_(
                        gradient, gradient, value=0.001
                    )
                    reconstructed[f"{arm}_{step:02d}_exp_avg_post_{index:02d}"] = (
                        moments[index].numpy().copy()
                    )
                    reconstructed[f"{arm}_{step:02d}_exp_avg_sq_post_{index:02d}"] = (
                        variances[index].numpy().copy()
                    )
                state_arrays = [
                    array
                    for index in range(count)
                    for array in (moments[index].numpy(), variances[index].numpy())
                ]
                if _flat_hash(state_arrays) != log["moment_post_hash"]:
                    raise ValueError(
                        f"optimizer_state_reconstruction_mismatch:{arm}:{step + 1}"
                    )
    path = out / "optimizer-state-reconstruction.npz"
    np.savez_compressed(path, **reconstructed)
    return {"status": "valid", "states_per_arm": 16, "tensor_count": len(reconstructed)}


def _verify_receipt(root: Path) -> dict[str, Any]:
    out = root / "docs/data/seed442-kl-capped-adam-screen"
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for relative, expected in receipt["evidence_files_sha256"].items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"evidence_file_hash_mismatch:{relative}")
    for relative, expected in receipt["source_sha256"].items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"publication_source_hash_mismatch:{relative}")
    if receipt["registration_sha256"] != _sha(out / "registration.json"):
        raise ValueError("receipt_registration_mismatch")
    return receipt


def _flat_hash(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f4").tobytes())
    return digest.hexdigest()


def _load_model(checkpoint: Path, input_size: int) -> torch.nn.Module:
    model = train.PolicyValueNet((96, 3), "residual_v3", input_size)
    train.load_checkpoint_into_model(model, checkpoint)
    return model


def _metrics_from_model(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
) -> dict[str, Any]:
    x, policy, value, replay, coefficients, train_positions, _validation, _sources = (
        arrays
    )
    policy_losses = np.empty(len(x), dtype=np.float64)
    mse = np.empty(len(x), dtype=np.float64)
    huber = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            stop = min(start + 512, len(x))
            states = x[start:stop]
            logits, prediction = model(torch.from_numpy(states))
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(states))
            masked = logits.masked_fill(legal <= 0, -1e9)
            policy_losses[start:stop] = (
                train.compute_policy_cross_entropy(
                    masked, torch.from_numpy(policy[start:stop])
                )
                .double()
                .numpy()
            )
            prediction64 = prediction.reshape(-1).double().numpy()
            targets64 = value[start:stop, 0].astype(np.float64)
            mse[start:stop] = np.square(prediction64 - targets64)
            huber[start:stop] = (
                train.compute_value_loss_vector(
                    prediction,
                    torch.from_numpy(value[start:stop]),
                    value_loss="huber",
                    huber_delta=1.0,
                )
                .double()
                .numpy()
            )
    train_rows = replay[train_positions]
    full_objective = float(
        np.average(policy_losses[train_rows], weights=coefficients[train_rows])
        + 0.3 * np.mean(huber[train_rows])
    )
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    rows = [int(r["compact_row"]) for r in unseen]
    groups: dict[str, list[int]] = {}
    for row in unseen:
        groups.setdefault(row["input_identity"], []).append(int(row["compact_row"]))
    return {
        "full_training_objective": full_objective,
        "exposure_weighted": {
            "policy_ce": float(np.mean(policy_losses[rows])),
            "value_mse": float(np.mean(mse[rows])),
        },
        "equal_input": {
            "policy_ce": float(
                np.mean([np.mean(policy_losses[g]) for g in groups.values()])
            ),
            "value_mse": float(np.mean([np.mean(mse[g]) for g in groups.values()])),
        },
    }


def _verify_arm(
    arm: str,
    model: torch.nn.Module,
    evidence: dict[str, Any],
    archive: Any,
    optimizer_states: Any,
    arrays: tuple[np.ndarray, ...],
    guard_rows: np.ndarray,
    registration: dict[str, Any],
) -> None:
    x, target_policy, target_value, replay, coefficients, train_positions, *_ = arrays
    train_replay = replay[train_positions]
    parameters = [p for p in model.parameters() if p.requires_grad]
    logs = evidence["arms"][arm]["steps"]
    previous_m = [np.zeros_like(p.detach().numpy()) for p in parameters]
    previous_v = [np.zeros_like(p.detach().numpy()) for p in parameters]
    for step, log in enumerate(logs):
        if log["step"] != step + 1 or log["optimizer_step"] != step + 1:
            raise ValueError(f"optimizer_step_counter:{arm}:{step + 1}")
        positions = np.asarray(log["expanded_positions"], dtype=np.int64)
        if not np.array_equal(
            positions,
            np.asarray(registration["first_16_minibatch_positions"])[
                step * 512 : (step + 1) * 512
            ],
        ):
            raise ValueError(f"batch_prefix_mismatch:{arm}:{step + 1}")
        rows = train_replay[positions]
        if not np.array_equal(rows, np.asarray(log["compact_rows"], dtype=np.int64)):
            raise ValueError(f"batch_source_rows_mismatch:{arm}:{step + 1}")
        pre = [archive[f"{arm}_{step:02d}_pre_{i:02d}"] for i in range(len(parameters))]
        proposal = [
            archive[f"{arm}_{step:02d}_proposal_{i:02d}"]
            for i in range(len(parameters))
        ]
        post = [
            archive[f"{arm}_{step:02d}_post_{i:02d}"] for i in range(len(parameters))
        ]
        with torch.no_grad():
            for parameter, values in zip(parameters, pre, strict=True):
                parameter.copy_(torch.from_numpy(np.asarray(values, dtype=np.float32)))
        for parameter in parameters:
            parameter.grad = None
        model.train()
        batch_x = torch.from_numpy(x[rows])
        batch_legal = torch.from_numpy(
            train.legal_mask_matrix_for_encoded_states(x[rows])
        )
        batch_coeff = torch.from_numpy(coefficients[rows])
        logits, predicted = model(batch_x)
        policy_loss = (
            train.compute_policy_cross_entropy(
                logits.masked_fill(batch_legal <= 0, -1e9),
                torch.from_numpy(target_policy[rows]),
            )
            * batch_coeff
        ).sum() / batch_coeff.sum()
        value_loss = train.compute_value_loss_vector(
            predicted,
            torch.from_numpy(target_value[rows]),
            value_loss="huber",
            huber_delta=1.0,
        ).mean()
        (policy_loss + 0.3 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        grads = [p.grad.detach().clone() for p in parameters]
        for index, (parameter, gradient) in enumerate(
            zip(parameters, grads, strict=True)
        ):
            recorded_gradient = archive[f"{arm}_{step:02d}_gradient_{index:02d}"]
            if not np.allclose(
                recorded_gradient, gradient.cpu().numpy(), atol=2e-7, rtol=2e-6
            ):
                raise ValueError(f"clipped_gradient_mismatch:{arm}:{step + 1}:{index}")
        optimizer = torch.optim.Adam(
            parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        for index, parameter in enumerate(parameters):
            optimizer.state[parameter]["step"] = torch.tensor(
                float(step), dtype=torch.float32
            )
            optimizer.state[parameter]["exp_avg"] = torch.from_numpy(
                previous_m[index].copy()
            )
            optimizer.state[parameter]["exp_avg_sq"] = torch.from_numpy(
                previous_v[index].copy()
            )
        optimizer.step()
        reproduced_proposal = [p.detach().cpu().numpy().copy() for p in parameters]
        if any(
            not np.array_equal(a, b)
            for a, b in zip(reproduced_proposal, proposal, strict=True)
        ):
            raise ValueError(f"adam_proposal_mismatch:{arm}:{step + 1}")
        next_m, next_v = [], []
        for index, (parameter, grad) in enumerate(zip(parameters, grads, strict=True)):
            state = optimizer.state[parameter]
            next_m.append(state["exp_avg"].cpu().numpy().copy())
            next_v.append(state["exp_avg_sq"].cpu().numpy().copy())
            m_before = archive[f"{arm}_{step:02d}_exp_avg_pre_{index:02d}"]
            v_before = archive[f"{arm}_{step:02d}_exp_avg_sq_pre_{index:02d}"]
            if not np.array_equal(m_before, previous_m[index]) or not np.array_equal(
                v_before, previous_v[index]
            ):
                raise ValueError(f"adam_moment_pre_mismatch:{arm}:{step + 1}:{index}")
        moment_hash = _flat_hash(
            [a for pair in zip(next_m, next_v, strict=True) for a in pair]
        )
        if moment_hash != log["moment_post_hash"]:
            raise ValueError(f"adam_moment_identity_mismatch:{arm}:{step + 1}")
        for index in range(len(parameters)):
            if not np.array_equal(
                optimizer_states[f"{arm}_{step:02d}_exp_avg_post_{index:02d}"],
                next_m[index],
            ) or not np.array_equal(
                optimizer_states[f"{arm}_{step:02d}_exp_avg_sq_post_{index:02d}"],
                next_v[index],
            ):
                raise ValueError(
                    f"adam_moment_archive_mismatch:{arm}:{step + 1}:{index}"
                )
        previous_m, previous_v = next_m, next_v

        if arm == "A":
            if any(
                not np.array_equal(a, b) for a, b in zip(proposal, post, strict=True)
            ):
                raise ValueError(f"ordinary_adam_step_mismatch:{step + 1}")
            continue

        pre_model = _load_model(
            registration["frozen_input_sha256"] and seed442.seed435.INIT, x.shape[1]
        )
        with torch.no_grad():
            for parameter, values in zip(pre_model.parameters(), pre, strict=True):
                parameter.copy_(torch.from_numpy(np.asarray(values, dtype=np.float32)))
        batch_pre = seed442._pre_policy(pre_model, x, rows)
        guard_pre = seed442._pre_policy(pre_model, x, guard_rows)
        trials = log["trials"]
        scales = [trial["scale"] for trial in trials]
        if scales != list(seed442.SCALES[: len(scales)]):
            raise ValueError(f"trial_order_invalid:{step + 1}")
        seen_accept = False
        for trial in trials:
            scale = float(trial["scale"])
            candidate = [
                np.asarray(b, dtype=np.float32)
                if scale == 1.0
                else np.asarray(a + (b - a) * scale, dtype=np.float32)
                for a, b in zip(pre, proposal, strict=True)
            ]
            with torch.no_grad():
                for parameter, values in zip(parameters, candidate, strict=True):
                    parameter.copy_(torch.from_numpy(values))
            batch_kl = seed442._kl(
                model, x, rows, np.full(len(rows), 1 / len(rows)), batch_pre
            )
            guard_kl = seed442._kl(
                model,
                x,
                guard_rows,
                np.full(len(guard_rows), 1 / len(guard_rows)),
                guard_pre,
            )
            tolerance = registration["kl"]["acceptance_tolerance"]
            if (
                abs(batch_kl - trial["batch_kl"]) > tolerance
                or abs(guard_kl - trial["guard_kl"]) > tolerance
            ):
                raise ValueError(f"trial_kl_mismatch:{step + 1}:{scale}")
            accepted = (
                batch_kl <= registration["kl"]["cap"] + tolerance
                and guard_kl <= registration["kl"]["cap"] + tolerance
            )
            if bool(trial["accepted"]) != accepted or (seen_accept and accepted):
                raise ValueError(f"trial_acceptance_invalid:{step + 1}:{scale}")
            seen_accept = accepted
        selected = log["selected_scale"]
        if selected is None:
            expected_post = pre
        else:
            if (
                not trials
                or not trials[-1]["accepted"]
                or trials[-1]["scale"] != selected
            ):
                raise ValueError(f"selected_scale_invalid:{step + 1}")
            expected_post = [
                np.asarray(b, dtype=np.float32)
                if selected == 1.0
                else np.asarray(a + (b - a) * selected, dtype=np.float32)
                for a, b in zip(pre, proposal, strict=True)
            ]
        if any(
            not np.array_equal(a, b) for a, b in zip(expected_post, post, strict=True)
        ):
            raise ValueError(f"parameter_restore_mismatch:{step + 1}")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    seed442.configure_root(root)
    out = seed442.OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    seed442.validate_registration_bindings(registration)
    _verify_receipt(root)
    if _sha(seed442.seed435.INIT) != seed442.INIT_SHA:
        raise ValueError("initializer_binding_invalid")
    if verify_seed441_receipt(root)["status"] != "valid":
        raise ValueError("seed441_receipt_invalid")
    cohort_path = out / "guard-cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if (
        len(cohort) != 2048
        or _sha(cohort_path) != registration["guard"]["cohort_sha256"]
    ):
        raise ValueError("guard_cohort_binding_invalid")
    arrays = seed442.load_inputs()
    expected_cohort = seed442.guard_cohort(root, arrays)
    if expected_cohort != cohort:
        raise ValueError("guard_cohort_reconstruction_mismatch")
    x = arrays[0]
    guard_rows = np.asarray([item["compact_row"] for item in cohort], dtype=np.int64)
    membership = seed442.seed435.membership_rows()
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["registration_sha256"] != _sha(registration_path):
        raise ValueError("evidence_registration_binding_invalid")
    with (
        np.load(out / "step-tensors.npz", allow_pickle=False) as archive,
        np.load(
            out / "optimizer-state-reconstruction.npz", allow_pickle=False
        ) as optimizer_states,
    ):
        for arm in ("A", "B"):
            _verify_arm(
                arm,
                _load_model(seed442.seed435.INIT, x.shape[1]),
                evidence,
                archive,
                optimizer_states,
                arrays,
                guard_rows,
                registration,
            )
            final = _load_model(out / f"{arm}-final.npz", x.shape[1])
            final_hash = seed442.vector_hash(
                [
                    parameter
                    for parameter in final.parameters()
                    if parameter.requires_grad
                ]
            )
            if final_hash != evidence["arms"][arm]["final_parameter_sha256"]:
                raise ValueError(f"endpoint_parameter_hash_mismatch:{arm}")
            if arm == "A":
                baseline = _load_model(seed442.seed435.DATA / "A-final.npz", x.shape[1])
                if any(
                    not torch.equal(left, right)
                    for left, right in zip(
                        final.parameters(), baseline.parameters(), strict=True
                    )
                ):
                    raise ValueError("ordinary_adam_endpoint_differs_from_seed435")
            last_post = [
                archive[f"{arm}_15_post_{index:02d}"]
                for index, _parameter in enumerate(final.parameters())
            ]
            if any(
                not np.array_equal(parameter.detach().cpu().numpy(), values)
                for parameter, values in zip(final.parameters(), last_post, strict=True)
            ):
                raise ValueError(f"endpoint_parameter_mismatch:{arm}")
    metrics: dict[str, Any] = {}
    for label in ("initializer", "A", "B"):
        checkpoint = (
            seed442.seed435.INIT
            if label == "initializer"
            else out / f"{label}-final.npz"
        )
        model = _load_model(checkpoint, x.shape[1])
        metrics[label] = _metrics_from_model(model, arrays, membership)
        pred_path = out / f"{label}-predictions.npz"
        rows = [
            r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
        ]
        ids = np.asarray([int(r["compact_row"]) for r in rows], dtype=np.int64)
        with np.load(pred_path, allow_pickle=False) as predictions:
            if not np.array_equal(predictions["compact_rows"], ids):
                raise ValueError(f"prediction_membership_invalid:{label}")
            if not np.array_equal(
                predictions["input_identity"],
                np.asarray([r["input_identity"] for r in rows]),
            ):
                raise ValueError(f"prediction_identity_invalid:{label}")
            logits_parts, value_parts = [], []
            with torch.inference_mode():
                for offset in range(0, len(ids), 512):
                    part = ids[offset : offset + 512]
                    logits_part, value_part = model(torch.from_numpy(x[part]))
                    logits_parts.append(logits_part.numpy())
                    value_parts.append(value_part.reshape(-1).numpy())
            logits = np.concatenate(logits_parts)
            values = np.concatenate(value_parts)
            if not np.allclose(predictions["policy_logits"], logits, atol=2e-7, rtol=0):
                raise ValueError(f"prediction_logits_tampered:{label}")
            if not np.allclose(
                predictions["value_predictions"],
                values,
                atol=2e-7,
                rtol=0,
            ):
                raise ValueError(f"prediction_values_tampered:{label}")
        if metrics[label] != evidence["metrics"][label]:
            for weighting in ("exposure_weighted", "equal_input"):
                for metric in ("policy_ce", "value_mse"):
                    if not np.isclose(
                        metrics[label][weighting][metric],
                        evidence["metrics"][label][weighting][metric],
                        atol=2e-6,
                        rtol=2e-6,
                    ):
                        raise ValueError(
                            f"endpoint_metric_mismatch:{label}:{weighting}:{metric}"
                        )
            if not np.isclose(
                metrics[label]["full_training_objective"],
                evidence["metrics"][label]["full_training_objective"],
                atol=2e-6,
                rtol=2e-6,
            ):
                raise ValueError(f"full_training_objective_mismatch:{label}")
    independently_decided = dict(evidence)
    independently_decided["metrics"] = metrics
    decision = seed442.decide(independently_decided)
    if decision != evidence["decision"]:
        raise ValueError("decision_reproduction_mismatch")
    return {
        "status": "valid",
        "classification": decision["classification"],
        "proposals_per_arm": 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
