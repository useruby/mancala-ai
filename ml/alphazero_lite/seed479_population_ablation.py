"""Seed479 sequential minibatch/full-population normalized-gradient ablation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

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
OUT = Path("docs/data/seed479-population-ablation")
INIT = Path(
    "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
HIST = Path("docs/data/seed474-fresh-value-removal")
STEP_NORM = 0.00375
SOURCE_PATHS = (
    "ml/alphazero_lite/seed479_population_ablation.py",
    "ml/alphazero_lite/publish_seed479_population_ablation.py",
    "ml/alphazero_lite/verify_seed479_population_ablation.py",
    "ml/alphazero_lite/test_seed479_population_ablation.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
)
INPUT_PATHS = {
    "seed466_manifest": Path("docs/data/seed466-replay-input-bundle/manifest.json"),
    "seed466_training": Path("docs/data/seed466-replay-input-bundle/training.npz"),
    "seed466_evaluation": Path("docs/data/seed466-replay-input-bundle/evaluation.npz"),
    "initializer": INIT,
    "seed474_registration": HIST / "registration.json",
    "seed474_receipt": HIST / "receipt.json",
    "seed474_P_steps": HIST / "P-steps.npz",
    "seed474_P_trajectory": HIST / "P-trajectory.json",
    "seed475_registration": Path(
        "docs/data/seed475-policy-population-attribution/registration.json"
    ),
    "seed475_receipt": Path(
        "docs/data/seed475-policy-population-attribution/receipt.json"
    ),
    "seed475_gradients": Path(
        "docs/data/seed475-policy-population-attribution/gradients.npz"
    ),
    "seed478_registration": Path(
        "docs/data/seed478-finite-step-evaluation/registration.json"
    ),
    "seed478_receipt": Path("docs/data/seed478-finite-step-evaluation/receipt.json"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat(parameters: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in parameters]
    )


def normalized_proposal(negative_gradient: np.ndarray) -> np.ndarray:
    """Return an explicit float32 proposal with fixed float64-measured norm."""
    vector = np.asarray(negative_gradient, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return np.zeros(vector.shape, dtype=np.float32)
    return (vector * (STEP_NORM / norm)).astype(np.float32)


def apply_proposal(parameters: list[torch.Tensor], proposal: np.ndarray) -> np.ndarray:
    """Apply flattened float32 proposal parameterwise and return float64 displacement."""
    before = flat(parameters)
    cursor = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            part = (
                proposal[cursor : cursor + count].reshape(tuple(parameter.shape)).copy()
            )
            parameter.add_(torch.from_numpy(part))
            cursor += count
    if cursor != proposal.size:
        raise ValueError("proposal_parameter_count_mismatch")
    return flat(parameters) - before


def freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    occ = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh = tr["fresh_source_mask"][occ].astype(bool)
    rows = occ[fresh]
    mass = float(
        tr["policy_coefficients"][rows].astype(np.float64).sum(dtype=np.float64)
    )
    batches = tr["first16_minibatch_positions"][:8192].reshape(16, 512)
    eval_rows = ev["compact_rows"]
    stones = np.asarray(
        [active_pit_stones_from_encoded_state(x) for x in tr["inputs"][eval_rows]]
    )
    unseen_gt32 = (ev["source_ids"] == 0) & (stones > 32)
    identities = ev["input_identity"][unseen_gt32].tolist()
    registration = {
        "schema": "seed479-population-ablation-registration-v1",
        "status": "frozen_before_execution",
        "interpretation": "exploratory; existing cohort and seed478 findings informed protocol; not fresh confirmation",
        "population_difference": "M divides each registered minibatch fresh-policy loss numerator by that minibatch's fresh policy mass. F divides the full fresh training-policy loss numerator by the complete fresh training policy mass.",
        "inputs_sha256": {key: sha(root / path) for key, path in INPUT_PATHS.items()},
        "sources_sha256": {path: sha(root / path) for path in SOURCE_PATHS},
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "initializer_sha256": sha(root / INIT),
        "population": {
            "training_occurrences": len(occ),
            "fresh_training_occurrences": len(rows),
            "full_fresh_policy_mass": mass,
            "full_fresh_occurrence_indices_sha256": hashlib.sha256(
                rows.astype("<i8").tobytes()
            ).hexdigest(),
            "full_fresh_occurrence_coefficients_sha256": hashlib.sha256(
                tr["policy_coefficients"][rows].astype("<f4").tobytes()
            ).hexdigest(),
            "minibatch_positions": batches.tolist(),
            "minibatch_positions_sha256": hashlib.sha256(
                batches.astype("<i8").tobytes()
            ).hexdigest(),
            "unseen_gt32_occurrences": len(identities),
            "unseen_gt32_exact_inputs": len(set(identities)),
        },
        "updates": 16,
        "objective": "fresh-policy cross-entropy only; occurrence-specific targets, legal masks, coefficients, and multiplicity retained",
        "gradient_accumulation": "float32 autograd in occurrence order, contiguous chunks of 512; F uses all 60,015 fresh training occurrences, M uses fresh rows of the corresponding exact 512-position minibatch",
        "proposal": {
            "norm": STEP_NORM,
            "normalization": "raw full-parameter negative-gradient L2 norm in float64; zero norm produces zero proposal",
            "writeback": "cast each proposal parameter to float32 and add to float32 state",
            "optimizer": "none; no clipping, momentum, adaptive scale, or line search",
        },
        "cohorts": {
            "fresh_unseen_gt32": "evaluation source_ids==0 and active pit stones>32; occurrence order, duplicates, and each occurrence's target retained",
            "pooled_unseen_gt32": "all evaluation occurrences with active pit stones>32; occurrence order, duplicates, and each occurrence's target retained",
        },
        "weightings": {
            "exposure": "each occurrence has equal mass",
            "equal_input": "each exact input identity has equal total mass, divided equally among its occurrences",
        },
        "gates": {
            "fresh CE F vs initializer improvement >= 0.001 under both weightings": True,
            "fresh CE F vs M improvement >= 0.001 under both weightings": True,
            "pooled CE F regression vs initializer <= 0.005 under both weightings": True,
            "pooled MSE F regression vs initializer <= 0.01 under both weightings": True,
            "full training policy CE F decreases": True,
        },
        "decision_rules": {
            "pass": "advance_to_fresh_confirmation",
            "fail": "close_full_population_direction_branch",
            "authorization": "passing authorizes only a separately designed fresh confirmation",
        },
        "comparison": "validate initializer F gradient against seed475 full fresh-training gradient, reconstructed with seed475 ordered 512-row accumulation semantics",
        "tolerances": {"rtol": 2e-6, "atol": 2e-8},
    }
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed479_output_exists")
    out.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    return registration


def calculate(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute separate sequential M/F trajectories and archive each numerical stage."""
    root = root.resolve()
    torch.set_num_threads(1)
    bundle = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = bundle["training"], bundle["evaluation"]
    occurrences = tr["expanded_replay_indices"][tr["train_positions"]]
    fresh_mask = tr["fresh_source_mask"][occurrences].astype(bool)
    fresh_rows = occurrences[fresh_mask]
    full_coeff = tr["policy_coefficients"][fresh_rows].astype(np.float64)
    full_mass = float(full_coeff.sum(dtype=np.float64))
    batch_positions = tr["first16_minibatch_positions"][:8192].reshape(16, 512)
    eval_rows = ev["compact_rows"]
    stones = np.asarray(
        [active_pit_stones_from_encoded_state(x) for x in tr["inputs"][eval_rows]]
    )
    fresh_eval = np.flatnonzero((ev["source_ids"] == 0) & (stones > 32))
    pooled_eval = np.flatnonzero(stones > 32)
    cohorts = {"fresh": fresh_eval, "pooled": pooled_eval}
    weights: dict[str, dict[str, np.ndarray]] = {}
    for cohort, selected in cohorts.items():
        ids = ev["input_identity"][selected].tolist()
        counts = {key: ids.count(key) for key in set(ids)}
        weights[cohort] = {
            "exposure": np.full(len(selected), 1 / len(selected), dtype=np.float64),
            "equal_input": np.asarray(
                [1 / (len(counts) * counts[key]) for key in ids], dtype=np.float64
            ),
        }
    targets = {
        cohort: ev["policy_targets"][selected].copy()
        for cohort, selected in cohorts.items()
    }
    inputs = {
        cohort: tr["inputs"][ev["compact_rows"][selected]]
        for cohort, selected in cohorts.items()
    }
    masks = {
        cohort: tr["legal_masks"][ev["compact_rows"][selected]]
        for cohort, selected in cohorts.items()
    }
    arrays: dict[str, np.ndarray] = {
        f"{cohort}_targets": targets[cohort] for cohort in cohorts
    }
    arrays.update(
        {
            f"{cohort}_{weight}_weights": values
            for cohort, mapping in weights.items()
            for weight, values in mapping.items()
        }
    )
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INIT)
    model.eval()
    parameters = list(model.parameters())

    def predictions() -> dict[str, np.ndarray]:
        result = {}
        model.eval()
        with torch.inference_mode():
            for cohort in cohorts:
                batches = []
                for start in range(0, len(inputs[cohort]), 512):
                    sl = slice(start, start + 512)
                    logits, _ = model(torch.from_numpy(inputs[cohort][sl].copy()))
                    legal = torch.from_numpy(masks[cohort][sl].copy())
                    batches.append(
                        F.softmax(logits.masked_fill(legal == 0, -1e9), 1).cpu().numpy()
                    )
                result[cohort] = np.concatenate(batches)
        return result

    def store_state(key: str) -> None:
        for i, p in enumerate(parameters):
            arrays[f"{key}_parameter_{i:02d}"] = p.detach().cpu().numpy().copy()

    def losses(prob: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.maximum(prob.astype(np.float64), 1e-300)
        ce = -(target.astype(np.float64) * np.log(p)).sum(axis=1)
        mse = ((prob.astype(np.float64) - target.astype(np.float64)) ** 2).sum(axis=1)
        return ce, mse

    initial_pred = predictions()
    for cohort, values in initial_pred.items():
        arrays[f"initializer_{cohort}_predictions"] = values
    records_by_step = [{"step": step + 1, "arms": {}} for step in range(16)]
    endpoint_pred: dict[str, dict[str, np.ndarray]] = {}
    seed475_comparison: dict[str, float] | None = None
    for arm in ("M", "F"):
        train.load_checkpoint_into_model(model, root / INIT)
        model.eval()
        endpoint_pred[arm] = {}
        for step in range(16):
            prefix = f"{arm}_step_{step + 1:02d}"
            pre = [p.detach().clone() for p in parameters]
            preflat = flat(pre)
            store_state(f"{prefix}_pre")
            if arm == "F":
                selected_rows, coefficient, mass = fresh_rows, full_coeff, full_mass
            else:
                batch_rows = occurrences[batch_positions[step]]
                isfresh = tr["fresh_source_mask"][batch_rows].astype(bool)
                selected_rows = batch_rows[isfresh]
                coefficient = tr["policy_coefficients"][selected_rows].astype(
                    np.float64
                )
                mass = float(coefficient.sum(dtype=np.float64))
            accum = [torch.zeros_like(p) for p in parameters]
            numerator = 0.0
            for start in range(0, len(selected_rows), 512):
                sl = slice(start, start + 512)
                rr = selected_rows[sl]
                logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                target = torch.from_numpy(tr["policy_targets"][rr].copy())
                ce = -(
                    target * F.log_softmax(logits.masked_fill(legal == 0, -1e9), dim=1)
                ).sum(1)
                numerator += float(
                    (ce.detach().double() * torch.from_numpy(coefficient[sl])).sum()
                )
                objective = (
                    ce * torch.from_numpy((coefficient[sl] / mass).astype(np.float32))
                ).sum()
                grads = torch.autograd.grad(objective, parameters, allow_unused=True)
                for dst, grad in zip(accum, grads, strict=True):
                    if grad is not None:
                        dst.add_(grad.detach())
            gradient_parts = [
                torch.zeros_like(p) if grad is None else grad
                for p, grad in zip(parameters, accum, strict=True)
            ]
            raw_gradient = flat(gradient_parts)
            if arm == "F" and step == 0:
                with np.load(
                    root / INPUT_PATHS["seed475_gradients"], allow_pickle=False
                ) as reference_archive:
                    reference = (
                        reference_archive["initializer_full_gradient"]
                        .reshape(-1)
                        .astype(np.float64)
                    )
                difference = raw_gradient - reference
                seed475_comparison = {
                    "max_abs_difference": float(np.max(np.abs(difference))),
                    "l2_difference": float(np.linalg.norm(difference)),
                }
            raw_norm = float(np.linalg.norm(raw_gradient))
            proposal = normalized_proposal(-raw_gradient)
            arrays[f"{prefix}_gradient"] = raw_gradient.copy()
            arrays[f"{prefix}_proposal"] = proposal.copy()
            displacement = apply_proposal(parameters, proposal)
            postflat = flat(parameters)
            arrays[f"{prefix}_displacement"] = displacement.copy()
            store_state(f"{prefix}_post")
            post_pred = predictions()
            for cohort, values in post_pred.items():
                arrays[f"{prefix}_{cohort}_predictions"] = values
                endpoint_pred[arm][cohort] = values
            records_by_step[step]["arms"][arm] = {
                "fresh_occurrences": len(selected_rows),
                "fresh_policy_mass": mass,
                "objective_numerator": numerator,
                "objective_denominator": mass,
                "objective": numerator / mass,
                "raw_gradient_norm": raw_norm,
                "raw_gradient_sha256": hashlib.sha256(
                    raw_gradient.astype("<f8").tobytes()
                ).hexdigest(),
                "proposal_norm_float64": float(
                    np.linalg.norm(proposal.astype(np.float64))
                ),
                "realized_displacement_norm_float64": float(
                    np.linalg.norm(displacement)
                ),
                "pre_state_sha256": hashlib.sha256(
                    preflat.astype("<f8").tobytes()
                ).hexdigest(),
                "post_state_sha256": hashlib.sha256(
                    postflat.astype("<f8").tobytes()
                ).hexdigest(),
            }
    endpoint_metrics: dict[str, Any] = {}
    initial_metrics: dict[str, Any] = {}
    for cohort in cohorts:
        initial_ce, initial_mse = losses(initial_pred[cohort], targets[cohort])
        initial_metrics[cohort] = {"ce": {}, "mse": {}}
        for weight_name, weight in weights[cohort].items():
            initial_metrics[cohort]["ce"][weight_name] = float(
                np.dot(initial_ce, weight)
            )
            initial_metrics[cohort]["mse"][weight_name] = float(
                np.dot(initial_mse, weight)
            )
        for arm in ("M", "F"):
            ce, mse = losses(endpoint_pred[arm][cohort], targets[cohort])
            endpoint_metrics.setdefault(arm, {})[cohort] = {"ce": {}, "mse": {}}
            for weight_name, weight in weights[cohort].items():
                endpoint_metrics[arm][cohort]["ce"][weight_name] = float(
                    np.dot(ce, weight)
                )
                endpoint_metrics[arm][cohort]["mse"][weight_name] = float(
                    np.dot(mse, weight)
                )
    # Also score the full fresh training policy population at each endpoint.
    train_metrics: dict[str, float] = {}
    training_weights = full_coeff / full_mass
    for arm in ("initializer", "M", "F"):
        if arm == "initializer":
            train.load_checkpoint_into_model(model, root / INIT)
        else:
            for i, parameter in enumerate(parameters):
                parameter.data.copy_(
                    torch.from_numpy(
                        arrays[f"{arm}_step_16_post_parameter_{i:02d}"].copy()
                    )
                )
        model.eval()
        ces: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(fresh_rows), 512):
                sl = slice(start, start + 512)
                rr = fresh_rows[sl]
                logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                lp = F.log_softmax(logits.masked_fill(legal == 0, -1e9), 1)
                target = torch.from_numpy(tr["policy_targets"][rr].copy())
                ces.append((-(target * lp).sum(1)).cpu().numpy().astype(np.float64))
        train_metrics[arm] = float(np.dot(np.concatenate(ces), training_weights))
    endpoint_metrics["initializer"] = initial_metrics
    effects: dict[str, Any] = {}
    for arm in ("M", "F"):
        effects[arm] = {}
        for cohort in cohorts:
            effects[arm][cohort] = {}
            for metric in ("ce", "mse"):
                effects[arm][cohort][metric] = {
                    weighting: endpoint_metrics[arm][cohort][metric][weighting]
                    - initial_metrics[cohort][metric][weighting]
                    for weighting in weights[cohort]
                }
    effects["F_minus_M"] = {
        cohort: {
            metric: {
                weighting: endpoint_metrics["F"][cohort][metric][weighting]
                - endpoint_metrics["M"][cohort][metric][weighting]
                for weighting in weights[cohort]
            }
            for metric in ("ce", "mse")
        }
        for cohort in cohorts
    }
    report: dict[str, Any] = {
        "schema": "seed479-population-ablation-evidence-v1",
        "steps": records_by_step,
        "initializer_metrics": initial_metrics,
        "endpoint_metrics": endpoint_metrics,
        "effects": effects,
        "full_training_policy_ce": train_metrics,
        "seed475_initializer_gradient_comparison": seed475_comparison,
    }
    gates: dict[str, bool] = {}
    for weighting in ("exposure", "equal_input"):
        fce = endpoint_metrics["F"]["fresh"]["ce"][weighting]
        mce = endpoint_metrics["M"]["fresh"]["ce"][weighting]
        ice = initial_metrics["fresh"]["ce"][weighting]
        gates[f"F_fresh_CE_improves_initializer_{weighting}_by_0.001"] = (
            ice - fce >= 0.001
        )
        gates[f"F_fresh_CE_improves_M_{weighting}_by_0.001"] = mce - fce >= 0.001
        gates[f"F_pooled_CE_regression_{weighting}_at_most_0.005"] = (
            endpoint_metrics["F"]["pooled"]["ce"][weighting]
            - initial_metrics["pooled"]["ce"][weighting]
            <= 0.005
        )
        gates[f"F_pooled_MSE_regression_{weighting}_at_most_0.01"] = (
            endpoint_metrics["F"]["pooled"]["mse"][weighting]
            - initial_metrics["pooled"]["mse"][weighting]
            <= 0.01
        )
    gates["F_full_training_policy_CE_decreases"] = (
        train_metrics["F"] < train_metrics["initializer"]
    )
    report["gates"] = gates
    report["decision"] = (
        "advance_to_fresh_confirmation"
        if all(gates.values())
        else "close_full_population_direction_branch"
    )
    # Verify independent registered seed475 initializer gradient semantics separately during verification.
    return report, arrays
