"""Independent, read-only reconstruction verifier for seed447 evidence."""

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
from ml.alphazero_lite.seed447_joint_output_cap import (
    FORWARD_BATCH_SIZE,
    OUT_NAME,
    VALUE_CAP,
    VALUE_TOLERANCE,
    choose_largest_feasible,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _equal_hash(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f4").tobytes())
    return digest.hexdigest()


def _values(model: torch.nn.Module, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), FORWARD_BATCH_SIZE):
            _logits, predicted = model(
                torch.from_numpy(x[rows[start : start + FORWARD_BATCH_SIZE]])
            )
            batches.append(predicted.reshape(-1).detach().cpu().numpy().copy())
    return np.concatenate(batches).astype(np.float32, copy=False)


def _movement(before: np.ndarray, after: np.ndarray) -> float:
    delta = after.astype(np.float64) - before.astype(np.float64)
    return float(np.mean(delta * delta, dtype=np.float64))


def _metrics(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    membership: list[dict[str, Any]],
) -> dict[str, Any]:
    x, target_policy, target_value, replay, coeff, train_pos, *_ = arrays
    policy_losses = np.empty(len(x), dtype=np.float64)
    mse = np.empty(len(x), dtype=np.float64)
    huber = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), FORWARD_BATCH_SIZE):
            stop = min(start + FORWARD_BATCH_SIZE, len(x))
            states = x[start:stop]
            logits, predicted = model(torch.from_numpy(states))
            legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(states))
            policy_losses[start:stop] = (
                train.compute_policy_cross_entropy(
                    logits.masked_fill(legal <= 0, -1e9),
                    torch.from_numpy(target_policy[start:stop]),
                )
                .double()
                .cpu()
                .numpy()
            )
            pred64 = predicted.reshape(-1).double().cpu().numpy()
            targets64 = target_value[start:stop, 0].astype(np.float64)
            mse[start:stop] = (pred64 - targets64) ** 2
            huber[start:stop] = (
                train.compute_value_loss_vector(
                    predicted,
                    torch.from_numpy(target_value[start:stop]),
                    value_loss="huber",
                    huber_delta=1.0,
                )
                .double()
                .cpu()
                .numpy()
            )
    repeated = replay[train_pos]
    objective = float(
        np.average(policy_losses[repeated], weights=coeff[repeated])
        + 0.3 * np.mean(huber[repeated])
    )
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    by_identity: dict[str, list[int]] = {}
    rows = []
    for row in unseen:
        compact = int(row["compact_row"])
        rows.append(compact)
        by_identity.setdefault(row["input_identity"], []).append(compact)
    return {
        "full_training_objective": objective,
        "exposure_weighted": {
            "policy_ce": float(np.mean(policy_losses[rows])),
            "value_mse": float(np.mean(mse[rows])),
        },
        "equal_input": {
            "policy_ce": float(
                np.mean([np.mean(policy_losses[g]) for g in by_identity.values()])
            ),
            "value_mse": float(
                np.mean([np.mean(mse[g]) for g in by_identity.values()])
            ),
        },
    }


def _same_metrics(actual: dict[str, Any], recorded: dict[str, Any], label: str) -> None:
    for weighting in ("exposure_weighted", "equal_input"):
        for metric in ("policy_ce", "value_mse"):
            if not np.isclose(
                actual[weighting][metric],
                recorded[weighting][metric],
                atol=2e-6,
                rtol=2e-6,
            ):
                raise ValueError(f"metric_mismatch:{label}:{weighting}:{metric}")
    if not np.isclose(
        actual["full_training_objective"],
        recorded["full_training_objective"],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise ValueError(f"training_objective_mismatch:{label}")


def verify(root: Path) -> dict[str, Any]:
    """Reconstruct training, constraints, endpoints, predictions, and decision."""
    root = root.resolve()
    out = root / "docs/data" / OUT_NAME
    seed442.configure_root(root)
    registration = json.loads((out / "registration.json").read_text())
    for rel, expected in registration["source_sha256"].items():
        if _sha(root / rel) != expected:
            raise ValueError(f"source_snapshot_mismatch:{rel}")
    base = root / "docs/data/seed442-kl-capped-adam-screen"
    if _sha(base / "registration.json") != registration["seed442_registration_sha256"]:
        raise ValueError("seed442_registration_changed")
    if _sha(base / "evidence.json") != registration["seed442_evidence_sha256"]:
        raise ValueError("seed442_evidence_changed")
    if (
        _sha(base / "B-final.npz")
        != "4302216db4e359d3045ca00a01ab81f04b9ea85195432ca8cd6aaabd50cad2db"
    ):
        raise ValueError("seed442_B_checkpoint_identity_invalid")
    for relative, digest in registration["source_sha256"].items():
        snapshot = out / "source-snapshots" / Path(relative).name
        if _sha(snapshot) != digest:
            raise ValueError(f"source_snapshot_tampered:{relative}")
    protocol = registration["seed442_registration"]
    for relative, expected in registration["bound_input_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError(f"frozen_input_changed:{relative}")
    arrays = seed442.load_inputs()
    x, target_policy, target_value, replay, coefficients, train_pos, _val, _sources = (
        arrays
    )
    train_rows = replay[train_pos]
    guard = json.loads((base / "guard-cohort.json").read_text())
    if guard != registration["guard_input_identities"]:
        raise ValueError("registered_guard_identities_changed")
    if seed442.guard_cohort(root, arrays) != guard or len(guard) != 2048:
        raise ValueError("guard_reconstruction_mismatch")
    membership = seed442.seed435.membership_rows()
    unseen = [
        r for r in membership if r["subset"] == "unseen" and r["active_stones"] > 32
    ]
    exact_unseen = {r["input_identity"] for r in unseen}
    canonical_unseen = {r["canonical_identity"] for r in unseen}
    exact_guard = [r["exact_input_identity"] for r in guard]
    canonical_guard = [r["canonical_identity"] for r in guard]
    if len(set(exact_guard)) != 2048:
        raise ValueError("guard_identity_not_unique")
    if set(exact_guard) & exact_unseen or set(canonical_guard) & canonical_unseen:
        raise ValueError("guard_evaluation_overlap")
    if len(unseen) != 2607 or len(exact_unseen) != 1242:
        raise ValueError("strict_unseen_population_invalid")
    registered_unseen = registration["unseen_population"]
    if registered_unseen != [
        {
            "compact_row": int(r["compact_row"]),
            "input_identity": r["input_identity"],
            "canonical_identity": r["canonical_identity"],
            "active_stones": r["active_stones"],
        }
        for r in unseen
    ]:
        raise ValueError("registered_unseen_population_changed")
    guard_rows = np.asarray([r["compact_row"] for r in guard], dtype=np.int64)
    evidence_path = out / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence["registration_sha256"] != _sha(out / "registration.json"):
        raise ValueError("evidence_registration_hash_invalid")
    batch_positions = np.asarray(
        protocol["first_16_minibatch_positions"], dtype=np.int64
    )
    initial_model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
    train.load_checkpoint_into_model(initial_model, seed442.seed435.INIT)
    initial_params = [
        p.detach().cpu().numpy().copy() for p in initial_model.parameters()
    ]
    with np.load(out / "step-tensors.npz", allow_pickle=False) as archive:
        for arm in ("C", "T"):
            model = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(model, seed442.seed435.INIT)
            params = [p for p in model.parameters() if p.requires_grad]
            seed442._CURRENT_PARAMETERS[:] = params
            m = [np.zeros_like(a) for a in initial_params]
            v = [np.zeros_like(a) for a in initial_params]
            previous_post = [a.copy() for a in initial_params]
            steps = evidence["arms"][arm]["steps"]
            if len(steps) != 16:
                raise ValueError(f"proposal_count_invalid:{arm}")
            for k, record in enumerate(steps):
                if record["step"] != k + 1 or record["optimizer_step"] != k + 1:
                    raise ValueError(f"step_index_invalid:{arm}:{k}")
                positions = batch_positions[k * 512 : (k + 1) * 512]
                if record["expanded_positions"] != positions.tolist():
                    raise ValueError(f"batch_order_invalid:{arm}:{k}")
                rows = train_rows[positions]
                if record["compact_rows"] != rows.tolist():
                    raise ValueError(f"batch_rows_invalid:{arm}:{k}")
                prefix = f"{arm}_{k:02d}"
                arrays_by_kind = {
                    kind: [
                        archive[f"{prefix}_{kind}_{i:02d}"].copy()
                        for i in range(len(params))
                    ]
                    for kind in (
                        "pre",
                        "proposal",
                        "post",
                        "gradient_unclipped",
                        "gradient",
                        "exp_avg_pre",
                        "exp_avg_sq_pre",
                        "exp_avg",
                        "exp_avg_sq",
                    )
                }
                pre, proposal, post = (
                    arrays_by_kind[n] for n in ("pre", "proposal", "post")
                )
                if any(
                    not np.array_equal(a, b)
                    for a, b in zip(
                        pre, initial_params if k == 0 else previous_post, strict=True
                    )
                ):
                    raise ValueError(f"parameter_trajectory_discontinuity:{arm}:{k}")
                for i, p in enumerate(params):
                    p.data.copy_(torch.from_numpy(pre[i].copy()))
                    p.grad = None
                model.eval()
                logits, predicted = model(torch.from_numpy(x[rows]))
                legal = torch.from_numpy(
                    train.legal_mask_matrix_for_encoded_states(x[rows])
                )
                coeff = torch.from_numpy(coefficients[rows])
                pl = (
                    train.compute_policy_cross_entropy(
                        logits.masked_fill(legal <= 0, -1e9),
                        torch.from_numpy(target_policy[rows]),
                    )
                    * coeff
                ).sum() / coeff.sum()
                vl = train.compute_value_loss_vector(
                    predicted,
                    torch.from_numpy(target_value[rows]),
                    value_loss="huber",
                    huber_delta=1.0,
                ).mean()
                (pl + 0.3 * vl).backward()
                unclipped = [p.grad.detach().clone() for p in params]
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                clipped = [p.grad.detach().clone() for p in params]
                for i, (a, b) in enumerate(
                    zip(unclipped, arrays_by_kind["gradient_unclipped"], strict=True)
                ):
                    if not np.allclose(a.cpu().numpy(), b, atol=1e-7, rtol=2e-6):
                        raise ValueError(f"unclipped_gradient_mismatch:{arm}:{k}:{i}")
                for i, (a, b) in enumerate(
                    zip(clipped, arrays_by_kind["gradient"], strict=True)
                ):
                    if not np.allclose(a.cpu().numpy(), b, atol=2e-7, rtol=2e-6):
                        raise ValueError(f"clipped_gradient_mismatch:{arm}:{k}:{i}")
                if any(
                    not np.array_equal(arrays_by_kind["exp_avg_pre"][i], m[i])
                    or not np.array_equal(arrays_by_kind["exp_avg_sq_pre"][i], v[i])
                    for i in range(len(params))
                ):
                    raise ValueError(f"moment_continuity_invalid:{arm}:{k}")
                optimizer = torch.optim.Adam(
                    params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
                )
                for i, p in enumerate(params):
                    optimizer.state[p]["step"] = torch.tensor(
                        float(k), dtype=torch.float32
                    )
                    optimizer.state[p]["exp_avg"] = torch.from_numpy(m[i].copy())
                    optimizer.state[p]["exp_avg_sq"] = torch.from_numpy(v[i].copy())
                optimizer.step()
                reproduced = [p.detach().cpu().numpy().copy() for p in params]
                if any(
                    not np.array_equal(a, b)
                    for a, b in zip(reproduced, proposal, strict=True)
                ):
                    raise ValueError(f"adam_proposal_mismatch:{arm}:{k}")
                m_next = [
                    optimizer.state[p]["exp_avg"].cpu().numpy().copy() for p in params
                ]
                v_next = [
                    optimizer.state[p]["exp_avg_sq"].cpu().numpy().copy()
                    for p in params
                ]
                if any(
                    not np.array_equal(a, b)
                    for a, b in zip(m_next, arrays_by_kind["exp_avg"], strict=True)
                ) or any(
                    not np.array_equal(a, b)
                    for a, b in zip(v_next, arrays_by_kind["exp_avg_sq"], strict=True)
                ):
                    raise ValueError(f"moment_post_mismatch:{arm}:{k}")
                m, v = m_next, v_next
                # All eight trial states and both population constraints.
                with torch.no_grad():
                    for parameter, old in zip(params, pre, strict=True):
                        parameter.copy_(torch.from_numpy(old.copy()))
                batch_pre_policy = seed442._pre_policy(model, x, rows)
                guard_pre_policy = seed442._pre_policy(model, x, guard_rows)
                batch_pre_values = _values(model, x, rows)
                guard_pre_values = _values(model, x, guard_rows)
                rebuilt_trials = []
                for trial, scale in zip(record["trials"], seed442.SCALES, strict=True):
                    if trial["scale"] != scale:
                        raise ValueError(f"trial_scale_tampered:{arm}:{k}")
                    seed442._set_trial(pre, proposal, scale)
                    bkl = seed442._kl(
                        model,
                        x,
                        rows,
                        np.full(len(rows), 1.0 / len(rows)),
                        batch_pre_policy,
                    )
                    gkl = seed442._kl(
                        model,
                        x,
                        guard_rows,
                        np.full(len(guard_rows), 1.0 / len(guard_rows)),
                        guard_pre_policy,
                    )
                    bv, gv = _values(model, x, rows), _values(model, x, guard_rows)
                    bm, gm = (
                        _movement(batch_pre_values, bv),
                        _movement(guard_pre_values, gv),
                    )
                    for key, actual in (
                        ("batch_kl", bkl),
                        ("guard_kl", gkl),
                        ("batch_value_movement", bm),
                        ("guard_value_movement", gm),
                    ):
                        if not np.isclose(actual, trial[key], atol=2e-12, rtol=0):
                            raise ValueError(
                                f"trial_measurement_mismatch:{arm}:{k}:{scale}:{key}"
                            )
                    policy_ok = (
                        bkl <= seed442.CAP + seed442.KL_TOLERANCE
                        and gkl <= seed442.CAP + seed442.KL_TOLERANCE
                    )
                    joint_ok = (
                        policy_ok
                        and bm <= VALUE_CAP + VALUE_TOLERANCE
                        and gm <= VALUE_CAP + VALUE_TOLERANCE
                    )
                    if (
                        trial["policy_feasible"] != policy_ok
                        or trial["joint_feasible"] != joint_ok
                    ):
                        raise ValueError(
                            f"trial_feasibility_tampered:{arm}:{k}:{scale}"
                        )
                    if trial.get("accepted") != (policy_ok if arm == "C" else joint_ok):
                        raise ValueError(f"trial_acceptance_tampered:{arm}:{k}:{scale}")
                    rebuilt_trials.append(
                        {
                            **trial,
                            "policy_feasible": policy_ok,
                            "joint_feasible": joint_ok,
                        }
                    )
                policy_scale = next(
                    (t["scale"] for t in rebuilt_trials if t["policy_feasible"]), None
                )
                selected = (
                    policy_scale
                    if arm == "C"
                    else choose_largest_feasible(
                        rebuilt_trials, cap=VALUE_CAP, tolerance=VALUE_TOLERANCE
                    )
                )
                if (
                    selected != record["selected_scale"]
                    or policy_scale != record["policy_only_scale"]
                ):
                    raise ValueError(f"largest_feasible_scale_invalid:{arm}:{k}")
                activated = (
                    arm == "T"
                    and policy_scale is not None
                    and selected is not None
                    and selected < policy_scale
                )
                if record["value_cap_activated"] != activated or record["rejected"] != (
                    selected is None
                ):
                    raise ValueError(f"cap_or_rejection_flag_invalid:{arm}:{k}")
                expected_post = (
                    pre
                    if selected is None
                    else [
                        b
                        if selected == 1.0
                        else np.asarray(a + (b - a) * selected, dtype=np.float32)
                        for a, b in zip(pre, proposal, strict=True)
                    ]
                )
                if any(
                    not np.array_equal(a, b)
                    for a, b in zip(expected_post, post, strict=True)
                ):
                    raise ValueError(f"post_parameters_invalid:{arm}:{k}")
                if arm == "C":
                    with np.load(
                        base / "step-tensors.npz", allow_pickle=False
                    ) as old_archive:
                        old_post = [
                            old_archive[f"B_{k:02d}_post_{i:02d}"]
                            for i in range(len(params))
                        ]
                    if any(
                        not np.array_equal(a, b)
                        for a, b in zip(post, old_post, strict=True)
                    ):
                        raise ValueError(f"control_step_tensor_mismatch_seed442_B:{k}")
                previous_post = [a.copy() for a in post]
            final = train.PolicyValueNet((96, 3), "residual_v3", x.shape[1])
            train.load_checkpoint_into_model(final, out / f"{arm}-final.npz")
            if any(
                not np.array_equal(p.detach().cpu().numpy(), a)
                for p, a in zip(final.parameters(), previous_post, strict=True)
            ):
                raise ValueError(f"endpoint_mismatch:{arm}")
            if (
                seed442.vector_hash([p for p in final.parameters()])
                != evidence["arms"][arm]["final_parameter_sha256"]
            ):
                raise ValueError(f"endpoint_hash_invalid:{arm}")
            _same_metrics(
                _metrics(final, arrays, membership), evidence["metrics"][arm], arm
            )
            _verify_predictions(out, arm, final, x, unseen)
    target_path = out / "ordered-targets.npz"
    with np.load(target_path, allow_pickle=False) as targets:
        ids = np.asarray([r["compact_row"] for r in unseen], dtype=np.int64)
        if not np.array_equal(targets["compact_rows"], ids) or not np.array_equal(
            targets["input_identity"], np.asarray([r["input_identity"] for r in unseen])
        ):
            raise ValueError("target_population_identity_invalid")
        if not np.array_equal(
            targets["policy_targets"], target_policy[ids]
        ) or not np.array_equal(targets["value_targets"], target_value[ids]):
            raise ValueError("targets_not_source_reconstructed")
    init_metrics = _metrics(initial_model, arrays, membership)
    _same_metrics(init_metrics, evidence["metrics"]["initializer"], "initializer")
    _verify_predictions(out, "initializer", initial_model, x, unseen)
    evidence_metrics = dict(evidence["metrics"])
    from ml.alphazero_lite.seed447_joint_output_cap import decide

    decision = decide({"metrics": evidence_metrics, "arms": evidence["arms"]})
    if decision != evidence["decision"]:
        raise ValueError("decision_classification_tampered")
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt["registration_sha256"] != _sha(out / "registration.json"):
        raise ValueError("receipt_registration_mismatch")
    for relative, expected in receipt["evidence_files_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError(f"publication_receipt_hash_mismatch:{relative}")
    return {
        "status": "valid",
        "population": {"exposures": 2607, "unique_inputs": 1242},
        "steps_per_arm": 16,
        "classification": decision["classification"],
    }


def _verify_predictions(
    out: Path,
    label: str,
    model: torch.nn.Module,
    x: np.ndarray,
    unseen: list[dict[str, Any]],
) -> None:
    ids = np.asarray([r["compact_row"] for r in unseen], dtype=np.int64)
    with np.load(out / f"{label}-predictions.npz", allow_pickle=False) as pred:
        if not np.array_equal(pred["compact_rows"], ids):
            raise ValueError(f"prediction_population_identity_invalid:{label}")
        if not np.array_equal(
            pred["input_identity"], np.asarray([r["input_identity"] for r in unseen])
        ):
            raise ValueError(f"prediction_input_identity_invalid:{label}")
        logits_ref, values_ref = [], []
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(ids), FORWARD_BATCH_SIZE):
                logits, val = model(
                    torch.from_numpy(x[ids[offset : offset + FORWARD_BATCH_SIZE]])
                )
                logits_ref.append(logits.cpu().numpy().copy())
                values_ref.append(val.reshape(-1).cpu().numpy().copy())
        if not np.array_equal(
            pred["policy_logits"], np.concatenate(logits_ref)
        ) or not np.array_equal(pred["value_predictions"], np.concatenate(values_ref)):
            raise ValueError(f"prediction_tampered:{label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
