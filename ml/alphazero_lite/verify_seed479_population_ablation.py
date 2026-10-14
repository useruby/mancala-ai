"""Independent read-only reconstruction and verification of seed479."""

from __future__ import annotations

import argparse
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
from ml.alphazero_lite.seed479_population_ablation import (
    INPUT_PATHS,
    OUT,
    SOURCE_PATHS,
    STEP_NORM,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flat(values: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [x.detach().cpu().numpy().reshape(-1).astype(np.float64) for x in values]
    )


def _assert_close(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"inventory_mismatch:{name}")
        for key, value in expected.items():
            _assert_close(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=2e-6, atol=2e-8):
            raise ValueError(f"value_mismatch:{name}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"length_mismatch:{name}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(a, e, f"{name}[{i}]")
    elif actual != expected:
        raise ValueError(f"value_mismatch:{name}")


def _predict(model, x: np.ndarray, legal: np.ndarray) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            sl = slice(start, start + 512)
            logits, _ = model(torch.from_numpy(x[sl].copy()))
            mask = torch.from_numpy(legal[sl].copy())
            outputs.append(
                F.softmax(logits.masked_fill(mask == 0, -1e9), 1).cpu().numpy()
            )
    return np.concatenate(outputs)


def _ce_mse(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = np.maximum(prediction.astype(np.float64), 1e-300)
    y = target.astype(np.float64)
    return -(y * np.log(p)).sum(1), ((prediction.astype(np.float64) - y) ** 2).sum(1)


def verify(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    original = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        receipt = json.loads((out / "receipt.json").read_text())
        expected_receipt = {
            "status",
            "registration.json",
            "evidence.json",
            "trajectory-evidence.npz",
            "results.md",
        }
        if (
            set(receipt) != expected_receipt
            or receipt.get("status") != "seed479_population_ablation_bound"
        ):
            raise ValueError("receipt_inventory_or_status")
        for name in expected_receipt - {"status"}:
            if _sha(out / name) != receipt[name]:
                raise ValueError(f"receipt_mismatch:{name}")
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed479-population-ablation-registration-v1"
            or reg.get("status") != "frozen_before_execution"
        ):
            raise ValueError("registration_identity")
        if reg.get("dependencies") != {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }:
            raise ValueError("dependency_identity")
        if set(reg.get("sources_sha256", {})) != set(SOURCE_PATHS):
            raise ValueError("source_inventory")
        for relative, digest in reg["sources_sha256"].items():
            if _sha(root / relative) != digest:
                raise ValueError(f"source_binding:{relative}")
        if set(reg.get("inputs_sha256", {})) != set(INPUT_PATHS):
            raise ValueError("input_inventory")
        for key, relative in INPUT_PATHS.items():
            if _sha(root / relative) != reg["inputs_sha256"][key]:
                raise ValueError(f"input_binding:{key}")
        data = read_authenticated_bundle(
            root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
        )
        tr, ev = data["training"], data["evaluation"]
        occurrences = tr["expanded_replay_indices"][tr["train_positions"]]
        fresh_rows = occurrences[tr["fresh_source_mask"][occurrences].astype(bool)]
        coefficient_full = tr["policy_coefficients"][fresh_rows].astype(np.float64)
        full_mass = float(coefficient_full.sum(dtype=np.float64))
        positions = tr["first16_minibatch_positions"][:8192].reshape(16, 512)
        if not np.array_equal(
            np.asarray(reg["population"]["minibatch_positions"]), positions
        ):
            raise ValueError("minibatch_order_mismatch")
        if len(fresh_rows) != 60015 or len(occurrences) != 134502:
            raise ValueError("training_population_mismatch")
        if reg["population"].get("fresh_training_occurrences") != len(fresh_rows):
            raise ValueError("registered_fresh_population_count_mismatch")
        if reg["population"].get("training_occurrences") != len(occurrences):
            raise ValueError("registered_training_population_count_mismatch")
        if hashlib.sha256(fresh_rows.astype("<i8").tobytes()).hexdigest() != reg[
            "population"
        ].get("full_fresh_occurrence_indices_sha256"):
            raise ValueError("registered_full_fresh_population_identity_mismatch")
        if hashlib.sha256(
            tr["policy_coefficients"][fresh_rows].astype("<f4").tobytes()
        ).hexdigest() != reg["population"].get(
            "full_fresh_occurrence_coefficients_sha256"
        ):
            raise ValueError("registered_full_fresh_coefficients_identity_mismatch")
        if not np.isclose(
            full_mass, reg["population"]["full_fresh_policy_mass"], rtol=0, atol=1e-10
        ):
            raise ValueError("full_denominator_mismatch")
        eval_rows = ev["compact_rows"]
        stones = np.asarray(
            [active_pit_stones_from_encoded_state(x) for x in tr["inputs"][eval_rows]]
        )
        fresh_sel = np.flatnonzero((ev["source_ids"] == 0) & (stones > 32))
        pooled_sel = np.flatnonzero(stones > 32)
        if len(fresh_sel) != reg["population"]["unseen_gt32_occurrences"]:
            raise ValueError("fresh_unseen_population_mismatch")
        cohort_rows = {"fresh": fresh_sel, "pooled": pooled_sel}
        targets = {c: ev["policy_targets"][s] for c, s in cohort_rows.items()}
        xs = {c: tr["inputs"][ev["compact_rows"][s]] for c, s in cohort_rows.items()}
        masks = {
            c: tr["legal_masks"][ev["compact_rows"][s]] for c, s in cohort_rows.items()
        }
        weights = {}
        for cohort, selected in cohort_rows.items():
            ids = ev["input_identity"][selected].tolist()
            counts = {x: ids.count(x) for x in set(ids)}
            weights[cohort] = {
                "exposure": np.full(len(ids), 1 / len(ids)),
                "equal_input": np.asarray(
                    [1 / (len(counts) * counts[x]) for x in ids], dtype=np.float64
                ),
            }
        archive = np.load(out / "trajectory-evidence.npz", allow_pickle=False)
        for cohort in cohort_rows:
            if not np.array_equal(archive[f"{cohort}_targets"], targets[cohort]):
                raise ValueError(f"occurrence_targets_mismatch:{cohort}")
            for weighting, value in weights[cohort].items():
                if not np.array_equal(archive[f"{cohort}_{weighting}_weights"], value):
                    raise ValueError(
                        f"occurrence_weights_mismatch:{cohort}:{weighting}"
                    )
        evidence = json.loads((out / "evidence.json").read_text())
        if len(evidence.get("steps", [])) != 16:
            raise ValueError("step_coverage")
        torch.set_num_threads(1)
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        init_path = root / INPUT_PATHS["initializer"]
        train.load_checkpoint_into_model(model, init_path)
        params = list(model.parameters())
        seed475 = np.load(root / INPUT_PATHS["seed475_gradients"], allow_pickle=False)
        reference_key = "initializer_full_gradient"
        if reference_key not in seed475.files:
            raise ValueError("seed475_initializer_gradient_missing")
        reference = seed475[reference_key].reshape(-1).astype(np.float64)
        full_initial: np.ndarray | None = None
        seed475_comparison: dict[str, float] | None = None
        initial_predictions: dict[str, np.ndarray] | None = None
        endpoint: dict[str, dict[str, np.ndarray]] = {}
        expected_records = [{"step": i + 1, "arms": {}} for i in range(16)]
        for arm in ("M", "F"):
            train.load_checkpoint_into_model(model, init_path)
            endpoint[arm] = {}
            for step in range(16):
                prefix = f"{arm}_step_{step + 1:02d}"
                preparts = [p.detach().clone() for p in params]
                preflat = _flat(preparts)
                for i, part in enumerate(preparts):
                    key = f"{prefix}_pre_parameter_{i:02d}"
                    if key not in archive.files or not np.array_equal(
                        archive[key], part.numpy()
                    ):
                        raise ValueError(f"pre_state_mismatch:{arm}:{step + 1}:{i}")
                if arm == "F":
                    rr, coeff, mass = fresh_rows, coefficient_full, full_mass
                else:
                    allrows = occurrences[positions[step]]
                    rr = allrows[tr["fresh_source_mask"][allrows].astype(bool)]
                    coeff = tr["policy_coefficients"][rr].astype(np.float64)
                    mass = float(coeff.sum(dtype=np.float64))
                grad_acc = [torch.zeros_like(p) for p in params]
                numerator = 0.0
                for start in range(0, len(rr), 512):
                    sl = slice(start, start + 512)
                    row = rr[sl]
                    logits, _ = model(torch.from_numpy(tr["inputs"][row].copy()))
                    legal = torch.from_numpy(tr["legal_masks"][row].copy())
                    target = torch.from_numpy(tr["policy_targets"][row].copy())
                    ce = -(
                        target * F.log_softmax(logits.masked_fill(legal == 0, -1e9), 1)
                    ).sum(1)
                    c = torch.from_numpy(coeff[sl])
                    numerator += float((ce.detach().double() * c).sum())
                    objective = (
                        ce * torch.from_numpy((coeff[sl] / mass).astype(np.float32))
                    ).sum()
                    gradients = torch.autograd.grad(
                        objective, params, allow_unused=True
                    )
                    for destination, grad in zip(grad_acc, gradients, strict=True):
                        if grad is not None:
                            destination.add_(grad.detach())
                parts = [
                    torch.zeros_like(p) if g is None else g
                    for p, g in zip(params, grad_acc, strict=True)
                ]
                grad = _flat(parts)
                if arm == "F" and step == 0:
                    full_initial = grad.copy()
                    if not np.allclose(grad, reference, rtol=2e-6, atol=2e-8):
                        raise ValueError("seed475_initializer_gradient_discrepancy")
                    difference = grad - reference
                    seed475_comparison = {
                        "max_abs_difference": float(np.max(np.abs(difference))),
                        "l2_difference": float(np.linalg.norm(difference)),
                    }
                if not np.allclose(
                    archive[f"{prefix}_gradient"], grad, rtol=2e-6, atol=2e-8
                ):
                    raise ValueError(f"gradient_mismatch:{arm}:{step + 1}")
                norm = float(np.linalg.norm(grad))
                prop64 = (
                    np.zeros_like(grad) if norm == 0 else -grad * (STEP_NORM / norm)
                )
                proposal = prop64.astype(np.float32)
                if not np.array_equal(archive[f"{prefix}_proposal"], proposal):
                    raise ValueError(f"proposal_mismatch:{arm}:{step + 1}")
                cursor = 0
                with torch.no_grad():
                    for parameter in params:
                        n = parameter.numel()
                        parameter.add_(
                            torch.from_numpy(
                                proposal[cursor : cursor + n]
                                .reshape(tuple(parameter.shape))
                                .copy()
                            )
                        )
                        cursor += n
                postflat = _flat(params)
                displacement = postflat - preflat
                if not np.allclose(
                    archive[f"{prefix}_displacement"], displacement, rtol=0, atol=0
                ):
                    raise ValueError(f"displacement_mismatch:{arm}:{step + 1}")
                for i, parameter in enumerate(params):
                    if not np.array_equal(
                        archive[f"{prefix}_post_parameter_{i:02d}"],
                        parameter.detach().numpy(),
                    ):
                        raise ValueError(f"post_state_mismatch:{arm}:{step + 1}:{i}")
                prediction = {c: _predict(model, xs[c], masks[c]) for c in cohort_rows}
                for cohort, values in prediction.items():
                    key = f"{prefix}_{cohort}_predictions"
                    if not np.allclose(archive[key], values, rtol=2e-6, atol=2e-8):
                        raise ValueError(
                            f"prediction_mismatch:{arm}:{step + 1}:{cohort}"
                        )
                    endpoint[arm][cohort] = values
                expected_records[step]["arms"][arm] = {
                    "fresh_occurrences": len(rr),
                    "fresh_policy_mass": mass,
                    "objective_numerator": numerator,
                    "objective_denominator": mass,
                    "objective": numerator / mass,
                    "raw_gradient_norm": norm,
                    "raw_gradient_sha256": hashlib.sha256(
                        grad.astype("<f8").tobytes()
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
                if step == 0 and arm == "M":
                    initial_predictions = {
                        c: _predict(model, xs[c], masks[c]) for c in cohort_rows
                    }
                    # The initializer predictions are reconstructed independently below from a reset model.
            # Each endpoint is model's complete post-step state; reset on next arm.
        train.load_checkpoint_into_model(model, init_path)
        initial_predictions = {c: _predict(model, xs[c], masks[c]) for c in cohort_rows}
        for cohort, values in initial_predictions.items():
            if not np.allclose(
                archive[f"initializer_{cohort}_predictions"],
                values,
                rtol=2e-6,
                atol=2e-8,
            ):
                raise ValueError(f"initializer_prediction_mismatch:{cohort}")
        # Compare archive metadata and all scalar records to independent values.
        for i, expected in enumerate(expected_records):
            _assert_close(evidence["steps"][i], expected, f"steps[{i}]")
        initial_metrics: dict[str, Any] = {}
        endpoint_metrics: dict[str, Any] = {}
        for cohort in cohort_rows:
            initial_metrics[cohort] = {"ce": {}, "mse": {}}
            ice, imse = _ce_mse(initial_predictions[cohort], targets[cohort])
            for weighting, w in weights[cohort].items():
                initial_metrics[cohort]["ce"][weighting] = float(np.dot(ice, w))
                initial_metrics[cohort]["mse"][weighting] = float(np.dot(imse, w))
            for arm in ("M", "F"):
                endpoint_metrics.setdefault(arm, {})[cohort] = {"ce": {}, "mse": {}}
                ce, mse = _ce_mse(endpoint[arm][cohort], targets[cohort])
                for weighting, w in weights[cohort].items():
                    endpoint_metrics[arm][cohort]["ce"][weighting] = float(
                        np.dot(ce, w)
                    )
                    endpoint_metrics[arm][cohort]["mse"][weighting] = float(
                        np.dot(mse, w)
                    )
        # Independently evaluate complete fresh training policy CE at initializer and both endpoints.
        train_w = coefficient_full / full_mass
        train_ce = {}
        for arm in ("initializer", "M", "F"):
            train.load_checkpoint_into_model(model, init_path)
            if arm != "initializer":
                for i, parameter in enumerate(params):
                    parameter.data.copy_(
                        torch.from_numpy(
                            archive[f"{arm}_step_16_post_parameter_{i:02d}"].copy()
                        )
                    )
            losses = []
            with torch.inference_mode():
                for start in range(0, len(fresh_rows), 512):
                    sl = slice(start, start + 512)
                    rr = fresh_rows[sl]
                    logits, _ = model(torch.from_numpy(tr["inputs"][rr].copy()))
                    legal = torch.from_numpy(tr["legal_masks"][rr].copy())
                    target = torch.from_numpy(tr["policy_targets"][rr].copy())
                    losses.append(
                        (
                            -(
                                target
                                * F.log_softmax(logits.masked_fill(legal == 0, -1e9), 1)
                            ).sum(1)
                        )
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
            train_ce[arm] = float(np.dot(np.concatenate(losses), train_w))
        endpoint_metrics["initializer"] = initial_metrics
        effects: dict[str, Any] = {}
        for arm in ("M", "F"):
            effects[arm] = {
                cohort: {
                    metric: {
                        weighting: endpoint_metrics[arm][cohort][metric][weighting]
                        - initial_metrics[cohort][metric][weighting]
                        for weighting in weights[cohort]
                    }
                    for metric in ("ce", "mse")
                }
                for cohort in cohort_rows
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
            for cohort in cohort_rows
        }
        gates = {}
        for weighting in ("exposure", "equal_input"):
            gates[f"F_fresh_CE_improves_initializer_{weighting}_by_0.001"] = (
                initial_metrics["fresh"]["ce"][weighting]
                - endpoint_metrics["F"]["fresh"]["ce"][weighting]
                >= 0.001
            )
            gates[f"F_fresh_CE_improves_M_{weighting}_by_0.001"] = (
                endpoint_metrics["M"]["fresh"]["ce"][weighting]
                - endpoint_metrics["F"]["fresh"]["ce"][weighting]
                >= 0.001
            )
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
            train_ce["F"] < train_ce["initializer"]
        )
        reconstructed = {
            "schema": "seed479-population-ablation-evidence-v1",
            "steps": expected_records,
            "initializer_metrics": initial_metrics,
            "endpoint_metrics": endpoint_metrics,
            "effects": effects,
            "full_training_policy_ce": train_ce,
            "seed475_initializer_gradient_comparison": seed475_comparison,
            "gates": gates,
            "decision": "advance_to_fresh_confirmation"
            if all(gates.values())
            else "close_full_population_direction_branch",
        }
        _assert_close(evidence, reconstructed, "evidence")
        # Seed475 uses the same ordered 512-occurrence fresh policy population and mass.
        if full_initial is None:
            raise ValueError("initializer_full_gradient_not_reconstructed")
    finally:
        for name, (content, mtime) in original.items():
            path = out / name
            if (
                not path.exists()
                or path.read_bytes() != content
                or path.stat().st_mtime_ns != mtime
            ):
                raise ValueError(f"publication_modified_during_verification:{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
    print("seed479 independent verification passed")


if __name__ == "__main__":
    main()
