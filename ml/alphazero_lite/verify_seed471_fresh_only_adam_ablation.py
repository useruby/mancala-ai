"""Independent read-only seed471 reconstruction and receipt verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)

OUT = Path("docs/data/seed471-fresh-only-adam-ablation")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
INPUTS = {
    "manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "training": "docs/data/seed466-replay-input-bundle/training.npz",
    "evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": str(INIT),
    "seed469_registration": "docs/data/seed469-optimizer-direction-correction/registration.json",
    "seed469_receipt": "docs/data/seed469-optimizer-direction-correction/receipt.json",
    "seed470_registration": "docs/data/seed470-finite-step-transport/registration.json",
    "seed470_receipt": "docs/data/seed470-finite-step-transport/receipt.json",
}
SOURCES = (
    "ml/alphazero_lite/seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/publish_seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/verify_seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/test_seed471_fresh_only_adam_ablation.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/verify_seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
)
RTOL, ATOL = 2e-6, 2e-8
INIT_SHA256 = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_occurrences(
    ce: np.ndarray, mse: np.ndarray, identities: list[str]
) -> dict[str, dict[str, float]]:
    """Aggregate rows without collapsing occurrence-specific targets."""
    if ce.shape != mse.shape or len(identities) != len(ce) or not len(ce):
        raise ValueError("occurrence_metric_shape_mismatch")
    grouped: dict[str, list[int]] = {}
    for i, identity in enumerate(identities):
        grouped.setdefault(identity, []).append(i)
    return {
        "exposure_weighted": {
            "policy_ce": float(np.mean(ce)),
            "value_mse": float(np.mean(mse)),
        },
        "equal_input": {
            "policy_ce": float(np.mean([np.mean(ce[ix]) for ix in grouped.values()])),
            "value_mse": float(np.mean([np.mean(mse[ix]) for ix in grouped.values()])),
        },
    }


def reconstruct_gates(arms: dict[str, Any]) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        t, c, init = (arms[arm] for arm in ("T", "C", "initializer"))
        gates[f"fresh_ce_vs_initializer_{weighting}"] = (
            t["fresh"][weighting]["policy_ce"]
            <= init["fresh"][weighting]["policy_ce"] - 0.001
        )
        gates[f"fresh_ce_vs_C_{weighting}"] = (
            t["fresh"][weighting]["policy_ce"]
            <= c["fresh"][weighting]["policy_ce"] - 0.001
        )
        gates[f"pooled_ce_vs_initializer_{weighting}"] = (
            t["pooled"][weighting]["policy_ce"]
            <= init["pooled"][weighting]["policy_ce"] + 0.005
        )
        gates[f"pooled_mse_vs_initializer_{weighting}"] = (
            t["pooled"][weighting]["value_mse"]
            <= init["pooled"][weighting]["value_mse"] + 0.01
        )
    gates["training_J_decreases"] = (
        arms["T"]["training_J"]["total"] < arms["initializer"]["training_J"]["total"]
    )
    return gates


def decision_from_gates(gates: dict[str, bool]) -> str:
    return (
        "advance_to_fresh_confirmation"
        if all(gates.values())
        else "close_small_step_fresh_only_branch"
    )


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
            reg.get("schema") != "seed471-fresh-only-adam-ablation-registration-v1"
            or reg.get("status") != "frozen_before_execution"
        ):
            raise ValueError("registration_schema_or_state_mismatch")
        if set(reg.get("inputs_sha256", {})) != set(INPUTS) or set(
            reg.get("source_closure_sha256", {})
        ) != set(SOURCES):
            raise ValueError("registered_closure_inventory_mismatch")
        if reg.get("dependencies") != {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }:
            raise ValueError("runtime_dependency_identity_mismatch")
        if reg.get("arms") != {"C": {"lr": 0.001}, "T": {"lr": 0.0000625}}:
            raise ValueError("registered_learning_rates_mismatch")
        if reg.get("optimizer") != {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "gradient_clip_norm": 1.0,
            "initial_moments": "zero",
            "sequential_state_per_arm": True,
        }:
            raise ValueError("registered_optimizer_mismatch")
        if (
            reg.get("minibatches", {}).get("updates") != 16
            or reg["minibatches"].get("batch_size") != 512
        ):
            raise ValueError("registered_update_schedule_mismatch")
        expected_objective = {
            "J": "sum(c_i*fresh_i*legal_CE_i)/sum(c_i*fresh_i) + 0.3*sum(fresh_i*Huber_i)/512",
            "huber_delta": 1.0,
            "value_denominator": 512,
            "historical_rows": "contribute neither policy nor value loss",
            "training_J_population": "all 134502 expanded occurrences at frozen train_positions, in stored order with multiplicity",
            "training_J_aggregation": "policy numerator sum(c*fresh*CE), policy denominator sum(c*fresh); value numerator sum(fresh*Huber), value denominator 134502; J=policy numerator/policy denominator+0.3*value numerator/value denominator",
        }
        if reg.get("objective") != expected_objective:
            raise ValueError("registered_objective_mismatch")
        if digest(root / INIT) != INIT_SHA256:
            raise ValueError("initializer_identity_mismatch")
        if reg.get("decision_rules") != {
            "fresh_ce_improvement_vs_initializer": 0.001,
            "fresh_ce_improvement_vs_C": 0.001,
            "pooled_ce_max_regression_vs_initializer": 0.005,
            "pooled_mse_max_regression_vs_initializer": 0.01,
            "training_J": "T endpoint objective strictly below initializer",
            "decision_pass": "advance_to_fresh_confirmation",
            "decision_fail": "close_small_step_fresh_only_branch",
            "authorization": "separately designed fresh confirmation only; no arena execution or promotion",
        }:
            raise ValueError("registered_decision_rules_mismatch")
        receipt = json.loads((out / "receipt.json").read_text())
        if receipt.get("status") != "post_execution_bound_receipt":
            raise ValueError("receipt_status_mismatch")
        for name, expected in receipt.items():
            if name != "status" and digest(out / name) != expected:
                raise ValueError(f"receipt_binding_mismatch:{name}")
        required_publication = {
            "registration.json",
            "endpoint-evidence.json",
            "results.md",
            "C-steps.npz",
            "T-steps.npz",
            "C-trajectory.json",
            "T-trajectory.json",
            *(
                f"{arm}-{cohort}-predictions.npz"
                for arm in ("initializer", "C", "T")
                for cohort in ("fresh", "pooled")
            ),
        }
        if set(receipt) != required_publication | {"status"}:
            raise ValueError("publication_inventory_mismatch")
        for rel, expected in reg["source_closure_sha256"].items():
            if digest(root / rel) != expected:
                raise ValueError(f"source_closure_mismatch:{rel}")
        for key, expected in reg["inputs_sha256"].items():
            rel = INPUTS[key]
            if digest(root / rel) != expected:
                raise ValueError(f"input_mismatch:{key}")
        arrays = read_authenticated_bundle(
            root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
        )
        tr = arrays["training"]
        init_model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(init_model, root / INIT)
        initial_model_state = {
            key: value.detach().clone()
            for key, value in init_model.state_dict().items()
        }
        positions = np.asarray(reg["minibatches"]["positions"], dtype=np.int64)
        if not np.array_equal(positions, tr["first16_minibatch_positions"][:8192]):
            raise ValueError("registered_minibatch_positions_mismatch")
        compact = tr["expanded_replay_indices"][tr["train_positions"][positions]]
        x, targets, masks = (
            tr["inputs"][compact],
            tr["policy_targets"][compact],
            tr["legal_masks"][compact],
        )
        coeff = tr["policy_coefficients"][compact]
        fresh = tr["fresh_source_mask"][compact].astype(np.float32)
        values = tr["value_targets"][compact].reshape(-1)
        torch.set_num_threads(1)
        endpoint_models: dict[str, torch.nn.Module] = {"initializer": init_model}
        for arm, lr in (("C", 0.001), ("T", 0.0000625)):
            model = train.PolicyValueNet((96, 3), "residual_v3", 27)
            model.load_state_dict(init_model.state_dict())
            params = list(model.parameters())
            optimizer = torch.optim.Adam(
                params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            with np.load(out / f"{arm}-steps.npz", allow_pickle=False) as archive:
                found = {key: archive[key] for key in archive.files}
            expected_keys = set()
            expected_steps: list[dict[str, Any]] = []
            for step in range(16):
                sl = slice(step * 512, (step + 1) * 512)
                expected_keys.update(
                    f"{step:02d}_{kind}_{i:02d}"
                    for i in range(len(params))
                    for kind in (
                        "pre",
                        "gradient",
                        "post",
                        "exp_avg",
                        "exp_avg_sq",
                        "displacement",
                    )
                )
                for i, p in enumerate(params):
                    archived_pre = found[f"{step:02d}_pre_{i:02d}"]
                    reconstructed_pre = p.detach().cpu().numpy()
                    if not np.array_equal(archived_pre, reconstructed_pre):
                        raise ValueError(f"pre_step_state_mismatch:{arm}:{step}:{i}")
                optimizer.zero_grad(set_to_none=True)
                logits, prediction = model(torch.from_numpy(x[sl].copy()))
                logp = F.log_softmax(
                    logits.masked_fill(torch.from_numpy(masks[sl].copy()) == 0, -1e9),
                    dim=1,
                )
                ce = -(torch.from_numpy(targets[sl].copy()) * logp).sum(1)
                w = torch.from_numpy((coeff[sl] * fresh[sl]).copy())
                policy = (ce * w).sum() / w.sum()
                huber = F.smooth_l1_loss(
                    prediction.reshape(-1),
                    torch.from_numpy(values[sl].copy()),
                    reduction="none",
                )
                objective = (
                    policy
                    + 0.3 * (huber * torch.from_numpy(fresh[sl].copy())).sum() / 512
                )
                objective.backward()
                preclip_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                grad_norm_float = float(preclip_norm)
                gradients = [p.grad.detach().clone() for p in params]
                optimizer.step()
                expected_steps.append(
                    {
                        "step": step + 1,
                        "objective": float(objective.detach()),
                        "policy_loss": float(policy.detach()),
                        "value_loss": float(
                            (
                                (huber * torch.from_numpy(fresh[sl].copy())).sum() / 512
                            ).detach()
                        ),
                        "policy_mass": float(w.sum()),
                        "preclip_norm": grad_norm_float,
                        "clipped": grad_norm_float > 1.0,
                        "occurrence_indices": compact[sl].tolist(),
                        "occurrence_identities": [
                            f"source={int(tr['source_ids'][row])};raw_line={int(tr['raw_lines'][row])};compact_row={int(row)}"
                            for row in compact[sl]
                        ],
                    }
                )
                for i, (p, grad) in enumerate(zip(params, gradients, strict=True)):
                    states = optimizer.state[p]
                    pairs = {
                        "gradient": grad,
                        "post": p.detach(),
                        "exp_avg": states["exp_avg"],
                        "exp_avg_sq": states["exp_avg_sq"],
                        "displacement": p.detach()
                        - torch.from_numpy(found[f"{step:02d}_pre_{i:02d}"].copy()),
                    }
                    for kind, tensor in pairs.items():
                        actual = found[f"{step:02d}_{kind}_{i:02d}"]
                        expected = tensor.detach().cpu().numpy()
                        if actual.dtype != expected.dtype or not np.allclose(
                            actual, expected, rtol=RTOL, atol=ATOL
                        ):
                            raise ValueError(
                                f"update_reconstruction_mismatch:{arm}:{step}:{kind}:{i}"
                            )
            if set(found) != expected_keys:
                raise ValueError(f"archive_key_set_mismatch:{arm}")
            trajectory = json.loads((out / f"{arm}-trajectory.json").read_text())
            if (
                trajectory.get("arm") != arm
                or trajectory.get("steps") != expected_steps
            ):
                raise ValueError(f"trajectory_reconstruction_mismatch:{arm}")
            endpoint_models[arm] = model
        report = json.loads((out / "endpoint-evidence.json").read_text())
        ev = arrays["evaluation"]
        fresh_occ = np.flatnonzero(ev["source_ids"] == 0)
        all_occ = np.arange(len(ev["source_ids"]))
        compact_eval = ev["compact_rows"]
        eval_x = tr["inputs"][compact_eval]
        eval_masks = tr["legal_masks"][compact_eval]
        identities = ev["input_identity"].tolist()
        eval_targets = ev["policy_targets"]
        eval_values = ev["value_targets"].reshape(-1)
        expected_arms: dict[str, dict[str, Any]] = {}
        for arm, model in endpoint_models.items():
            model.eval()
            with torch.inference_mode():
                logits, val_out = model(torch.from_numpy(eval_x.copy()))
                log_probs = (
                    F.log_softmax(
                        logits.masked_fill(
                            torch.from_numpy(eval_masks.copy()) == 0, -torch.inf
                        ),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
                val_pred = val_out.reshape(-1).cpu().numpy()
            expected_arms[arm] = {}
            for cohort, occurrence_indices in (
                ("fresh", fresh_occ),
                ("pooled", all_occ),
            ):
                group_positions: dict[str, list[int]] = {}
                for local, occurrence in enumerate(occurrence_indices):
                    group_positions.setdefault(identities[int(occurrence)], []).append(
                        local
                    )
                ce = (
                    -(
                        eval_targets[occurrence_indices]
                        * np.where(
                            np.isfinite(log_probs[occurrence_indices]),
                            log_probs[occurrence_indices],
                            0,
                        )
                    )
                    .sum(axis=1)
                    .astype(np.float64)
                )
                mse = np.square(
                    val_pred[occurrence_indices].astype(np.float64)
                    - eval_values[occurrence_indices].astype(np.float64)
                )
                summary = {
                    "occurrences": len(occurrence_indices),
                    "identities": len(group_positions),
                    **summarize_occurrences(
                        ce, mse, [identities[int(i)] for i in occurrence_indices]
                    ),
                }
                expected_arms[arm][cohort] = summary
                evidence = out / f"{arm}-{cohort}-predictions.npz"
                with np.load(evidence, allow_pickle=False) as z:
                    wanted = {
                        "prediction": log_probs[occurrence_indices].astype(np.float32),
                        "value": val_pred[occurrence_indices],
                        "policy_target": eval_targets[occurrence_indices],
                        "value_target": eval_values[occurrence_indices],
                        "input_identity": np.asarray(
                            [identities[int(i)] for i in occurrence_indices],
                            dtype="U256",
                        ),
                        "legal_mask": eval_masks[occurrence_indices],
                        "exposure_weight": np.full(
                            len(occurrence_indices),
                            1.0 / len(occurrence_indices),
                            dtype=np.float64,
                        ),
                        "equal_input_weight": np.asarray(
                            [
                                1.0
                                / (
                                    len(group_positions)
                                    * len(group_positions[identity])
                                )
                                for identity in [
                                    identities[int(i)] for i in occurrence_indices
                                ]
                            ],
                            dtype=np.float64,
                        ),
                    }
                    if set(z.files) != set(wanted):
                        raise ValueError(
                            f"endpoint_prediction_keys_mismatch:{arm}:{cohort}"
                        )
                    for key, expected in wanted.items():
                        actual = z[key]
                        agrees = (
                            np.array_equal(actual, expected)
                            if expected.dtype.kind in "US"
                            else np.allclose(
                                actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True
                            )
                        )
                        if (
                            actual.shape != expected.shape
                            or actual.dtype != expected.dtype
                            or not agrees
                        ):
                            raise ValueError(
                                f"endpoint_prediction_mismatch:{arm}:{cohort}:{key}"
                            )

            # Objective population is every occurrence in frozen train_positions expanded
            # through source multiplicity. Its policy denominator is sum(c*fresh),
            # and its value denominator is the full occurrence count, per registered J.
            train_rows = tr["expanded_replay_indices"][tr["train_positions"]]
            policy_ce_parts: list[np.ndarray] = []
            value_parts: list[np.ndarray] = []
            for start in range(0, len(train_rows), 1024):
                rows = train_rows[start : start + 1024]
                with torch.inference_mode():
                    lg, pv = model(torch.from_numpy(tr["inputs"][rows].copy()))
                    legal = torch.from_numpy(tr["legal_masks"][rows].copy())
                    lp = F.log_softmax(lg.masked_fill(legal == 0, -1e9), dim=1)
                    ce_rows = -(
                        torch.from_numpy(tr["policy_targets"][rows].copy()) * lp
                    ).sum(1)
                policy_ce_parts.append(ce_rows.cpu().numpy().astype(np.float64))
                value_parts.append(pv.reshape(-1).cpu().numpy().astype(np.float64))
            ce_train = np.concatenate(policy_ce_parts)
            pred_train = np.concatenate(value_parts)
            train_fresh = tr["fresh_source_mask"][train_rows].astype(np.float64)
            train_coeff = tr["policy_coefficients"][train_rows].astype(np.float64)
            err = np.abs(
                pred_train
                - tr["value_targets"][train_rows].reshape(-1).astype(np.float64)
            )
            train_huber = np.where(err < 1.0, 0.5 * err * err, err - 0.5)
            pden = float(np.sum(train_coeff * train_fresh, dtype=np.float64))
            pnum = float(np.sum(ce_train * train_coeff * train_fresh, dtype=np.float64))
            vnum = float(np.sum(train_huber * train_fresh, dtype=np.float64))
            expected_arms[arm]["training_J"] = {
                "total": pnum / pden + 0.3 * vnum / len(train_rows),
                "policy_numerator": pnum,
                "policy_denominator": pden,
                "value_numerator": vnum,
                "value_denominator": len(train_rows),
                "population_occurrences": len(train_rows),
            }

        if set(report["arms"]) != set(expected_arms):
            raise ValueError("endpoint_arm_set_mismatch")
        for arm, cohorts in expected_arms.items():
            for cohort, metrics in cohorts.items():
                for name, expected in metrics.items():
                    actual = report["arms"][arm][cohort][name]
                    if isinstance(expected, dict):
                        for metric, value in expected.items():
                            if not np.isclose(
                                actual[metric], value, rtol=RTOL, atol=ATOL
                            ):
                                raise ValueError(
                                    f"endpoint_metric_mismatch:{arm}:{cohort}:{name}:{metric}"
                                )
                    elif actual != expected:
                        raise ValueError(
                            f"endpoint_count_mismatch:{arm}:{cohort}:{name}"
                        )
        effects: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for cohort in ("fresh", "pooled"):
            effects[cohort] = {}
            for weighting in ("exposure_weighted", "equal_input"):
                effects[cohort][weighting] = {}
                for arm in ("C", "T"):
                    effects[cohort][weighting][arm] = {
                        metric: expected_arms[arm][cohort][weighting][metric]
                        - expected_arms["initializer"][cohort][weighting][metric]
                        for metric in ("policy_ce", "value_mse")
                    }
                effects[cohort][weighting]["T_minus_C"] = {
                    metric: expected_arms["T"][cohort][weighting][metric]
                    - expected_arms["C"][cohort][weighting][metric]
                    for metric in ("policy_ce", "value_mse")
                }
        for cohort, by_weight in effects.items():
            for weighting, by_arm in by_weight.items():
                for arm, metrics in by_arm.items():
                    for metric, expected in metrics.items():
                        actual = report["effects"][cohort][weighting][arm][metric]
                        if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL):
                            raise ValueError(
                                f"effect_mismatch:{cohort}:{weighting}:{arm}:{metric}"
                            )
        gates = reconstruct_gates(expected_arms)
        if report["gates"] != gates:
            raise ValueError("gate_reconstruction_mismatch")
        decision = decision_from_gates(gates)
        if report.get("decision") != decision:
            raise ValueError("decision_mismatch")
        if report.get("classification") != decision:
            raise ValueError("classification_mismatch")
        if any(
            not torch.equal(init_model.state_dict()[key], value)
            for key, value in initial_model_state.items()
        ):
            raise ValueError("initializer_or_buffer_mutated")
    finally:
        after = {
            p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in out.iterdir()
            if p.is_file()
        }
        if before != after:
            raise ValueError("verification_mutated_publication")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    verify(parser.parse_args().root)
    print("seed471 independent verification passed")


if __name__ == "__main__":
    main()
