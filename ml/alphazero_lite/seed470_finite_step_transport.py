"""Publish the seed470 fixed-initializer finite-step transport diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed469_correction import (
    reconstruct as seed469_reconstruct,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed470-finite-step-transport"
INIT = (
    ROOT
    / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
)
SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
INPUTS = {
    "seed466_manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "seed466_training": "docs/data/seed466-replay-input-bundle/training.npz",
    "seed466_evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    "seed469_registration": "docs/data/seed469-optimizer-direction-correction/registration.json",
    "seed469_receipt": "docs/data/seed469-optimizer-direction-correction/receipt.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(ids: list[str], equal: bool) -> np.ndarray:
    counts = {item: ids.count(item) for item in set(ids)}
    if equal:
        return np.asarray(
            [1.0 / (len(counts) * counts[item]) for item in ids], dtype=np.float64
        )
    return np.full(len(ids), 1.0 / len(ids), dtype=np.float64)


def set_parameters(model: torch.nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(
                torch.from_numpy(flat[offset : offset + size].reshape(parameter.shape))
            )
            offset += size
    if offset != flat.size:
        raise ValueError("parameter_vector_size_mismatch")


def _eval(
    model: torch.nn.Module, x: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predictions, losses = [], []
    for start in range(0, len(x), 128):
        xx = torch.from_numpy(x[start : start + 128].copy())
        tt = torch.from_numpy(target[start : start + 128].copy())
        mm = torch.from_numpy(mask[start : start + 128].copy())
        logits, _ = model(xx)
        logp = F.log_softmax(logits.masked_fill(mm == 0, -1e9), dim=1)
        predictions.append(logp.exp().detach().cpu().numpy())
        losses.append((-(tt * logp).sum(dim=1)).detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(losses)


def classify(curves: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for objective in ("P", "J"):
        for direction in ("adam", "negative_gradient"):
            for weighting in ("exposure_weighted", "equal_input"):
                rows = [
                    r
                    for r in curves
                    if (r["objective"], r["direction"], r["weighting"])
                    == (objective, direction, weighting)
                ]
                by_batch = {}
                for item in rows:
                    by_batch.setdefault(item["batch"], {})[item["scale_index"]] = item[
                        "ce_change"
                    ]
                full_down = sum(
                    r["scale_index"] == 0 and r["ce_change"] < -1e-6 for r in rows
                )
                overshoots = sum(
                    r["scale_index"] == 0
                    and r["predicted_descent_at_1"]
                    and r["ce_change"] > 1e-6
                    and any(
                        by_batch[r["batch"]].get(index, 0.0) < -1e-6
                        for index in range(3, 8)
                    )
                    for r in rows
                )
                if overshoots >= 12:
                    label = "finite_step_overshoot_consistent"
                elif full_down >= 12:
                    label = "finite_step_improvement_consistent"
                else:
                    label = "mixed_or_inconclusive_finite_step"
                result[f"{objective}/{direction}/{weighting}"] = label
    return result


def calculate(
    root: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    torch.set_num_threads(1)
    vectors, _ = seed469_reconstruct(root)
    bundle = root / "docs/data/seed466-replay-input-bundle"
    with np.load(bundle / "training.npz", allow_pickle=False) as z:
        tr = {key: z[key] for key in z.files}
    with np.load(bundle / "evaluation.npz", allow_pickle=False) as z:
        ev = {key: z[key] for key in z.files}
    select = np.flatnonzero(ev["source_ids"] == 0)
    ids = ev["input_identity"][select].tolist()
    if len(select) != 859 or len(set(ids)) != 851:
        raise ValueError("fresh_unseen_cohort_mismatch")
    compact = ev["compact_rows"][select]
    x = ev["inputs"][select] if "inputs" in ev else tr["inputs"][compact]
    target, mask = ev["policy_targets"][select], tr["legal_masks"][compact]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INPUTS["initializer"])
    state = {key: val.detach().clone() for key, val in model.state_dict().items()}
    initial = np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in model.parameters()]
    ).astype(np.float32)
    output: dict[str, np.ndarray] = {}
    output.update({f"seed469_{key}": value for key, value in vectors.items()})
    output["cohort_identities"] = np.asarray(ids, dtype="U256")
    output["cohort_targets"] = target.copy()
    output["cohort_legal_masks"] = mask.copy()
    output["cohort_weights_exposure_weighted"] = weights(ids, False)
    output["cohort_weights_equal_input"] = weights(ids, True)
    curves: list[dict[str, Any]] = []
    pred0, loss0 = _eval(model, x, target, mask)
    for weighting, equal in (("exposure_weighted", False), ("equal_input", True)):
        output[f"baseline_{weighting}_predictions"] = pred0
        output[f"baseline_{weighting}_losses"] = loss0
        w = weights(ids, equal)
        for batch in range(16):
            for objective in ("P", "J"):
                for direction in ("adam", "negative_gradient"):
                    stem = f"batch{batch:02d}_{objective}_{direction}"
                    proposal = vectors[f"{stem}_proposal"].astype(np.float32)
                    for si, scale in enumerate(SCALES):
                        # Explicit float32 multiply then float32 add, each materialized.
                        scaled = np.multiply(
                            proposal, np.float32(scale), dtype=np.float32
                        )
                        trial = np.add(initial, scaled, dtype=np.float32).astype(
                            np.float32
                        )
                        displacement = trial.astype(np.float64) - initial.astype(
                            np.float64
                        )
                        output[f"{stem}_scale{si}_trial_parameters"] = trial.copy()
                        set_parameters(model, trial)
                        pred, losses = _eval(model, x, target, mask)
                        output[f"{stem}_scale{si}_predictions"] = pred
                        output[f"{stem}_scale{si}_losses"] = losses
                        change = float(
                            np.dot(
                                losses.astype(np.float64) - loss0.astype(np.float64), w
                            )
                        )
                        eg = vectors[f"unseen_gradient_{weighting}"].astype(np.float64)
                        first = float(np.dot(eg, displacement))
                        curves.append(
                            {
                                "batch": batch,
                                "objective": objective,
                                "direction": direction,
                                "weighting": weighting,
                                "scale_index": si,
                                "scale": scale,
                                "displacement_norm": float(
                                    np.linalg.norm(displacement)
                                ),
                                "ce_change": change,
                                "first_order_prediction": first,
                                "remainder": change - first,
                                "predicted_descent": first < 0,
                                "predicted_descent_at_1": bool(
                                    np.dot(
                                        eg,
                                        (
                                            np.add(
                                                initial, proposal, dtype=np.float32
                                            ).astype(np.float64)
                                            - initial.astype(np.float64)
                                        ),
                                    )
                                    < 0
                                ),
                                "equal_input_change": None,
                            }
                        )
                    set_parameters(model, initial)
        # Equal-input and exposure metrics are joined by trial after both loops below.
    curve_map = {
        (
            r["batch"],
            r["objective"],
            r["direction"],
            r["weighting"],
            r["scale_index"],
        ): r
        for r in curves
    }
    # Include both aggregate effects per record; distinct weighting-specific first-order terms remain explicit.
    for r in curves:
        other = (
            "equal_input"
            if r["weighting"] == "exposure_weighted"
            else "exposure_weighted"
        )
        # paired output is retained in its own curve; the corresponding CE effect is named for clarity.
        r["other_weighting_ce_change"] = curve_map[
            (r["batch"], r["objective"], r["direction"], other, r["scale_index"])
        ]["ce_change"]
        r.pop("equal_input_change", None)
    if any(not torch.equal(model.state_dict()[k], v) for k, v in state.items()):
        raise ValueError("initializer_or_buffer_mutated")
    return (
        output,
        curves,
        {
            "identities": ids,
            "selection_indices": select.tolist(),
            "targets": target,
            "legal_masks": mask,
            "input_x": x,
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / OUT.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    from ml.alphazero_lite.seed466_runtime_bundle import (
        MANIFEST_SHA256,
        read_authenticated_bundle,
    )
    from ml.alphazero_lite.verify_seed466_replay_bundle import verify as verify_bundle

    bundle = root / "docs/data/seed466-replay-input-bundle"
    verify_bundle(bundle, root)
    read_authenticated_bundle(bundle, MANIFEST_SHA256)
    with np.load(bundle / "evaluation.npz", allow_pickle=False) as archive:
        evaluation = {key: archive[key] for key in archive.files}
    selected = np.flatnonzero(evaluation["source_ids"] == 0)
    identities = evaluation["input_identity"][selected].tolist()
    if (len(selected), len(set(identities))) != (859, 851):
        raise ValueError("fresh_unseen_cohort_mismatch")
    dependencies = (
        "ml/alphazero_lite/seed470_finite_step_transport.py",
        "ml/alphazero_lite/verify_seed470_finite_step_transport.py",
        "ml/alphazero_lite/test_seed470_finite_step_transport.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed469_correction.py",
        "ml/alphazero_lite/verify_seed469_correction.py",
        "ml/alphazero_lite/seed468_transport.py",
        "ml/alphazero_lite/seed466_runtime_bundle.py",
        "ml/alphazero_lite/verify_seed466_replay_bundle.py",
        "ml/alphazero_lite/seed466_replay_bundle.py",
        "ml/alphazero_lite/checkpoint_phase_selection.py",
        "ml/alphazero_lite/input_encodings.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/exact_root_policy_targets.py",
    )
    registration = {
        "schema": "seed470-finite-step-transport-registration-v1",
        "input_sha256": {key: sha(root / value) for key, value in INPUTS.items()},
        "seed469_fresh_cohort": {
            "definition": "seed469 fresh strict-unseen subset with active_stones>32; source_ids==0; preserve evaluation order and all occurrences",
            "occurrences": 859,
            "identities": 851,
            "selection_indices": selected.tolist(),
            "ordered_identity_sha256": hashlib.sha256(
                "\n".join(identities).encode()
            ).hexdigest(),
            "weighting": {
                "exposure_weighted": "1/859 per occurrence",
                "equal_input": "1/(851*k_i) per occurrence",
            },
        },
        "scales": SCALES,
        "protocol": {
            "trial_arithmetic": "scaled=float32(proposal*float32(scale)); trial=float32(initial+scaled); materialize each operation",
            "displacement": "float64(trial)-float64(initial), after float32 writeback",
            "forward_batching": "contiguous stored occurrence order, chunks of 128, final chunk 91; baseline once per weighting",
            "reduction": "float64 weighted sum in occurrence order; float64 G dot d",
            "dtypes": {
                "parameter": "float32",
                "proposal": "float32",
                "trial": "float32",
                "displacement": "float64",
                "weights": "float64",
                "aggregate": "float64",
            },
            "threads": 1,
            "tolerances": {"rtol": 2e-6, "atol": 2e-8, "decision_threshold": 1e-6},
            "decision_rules": {
                "overshoot": "12/16 scale-one predicted descent and CE increase >1e-6, same batches CE decrease <-1e-6 at a registered scale <=1/8",
                "improvement": "12/16 scale-one CE decrease <-1e-6, assessed after overshoot",
                "otherwise": "mixed_or_inconclusive_finite_step",
            },
        },
        "source_dependency_closure_sha256": {
            name: sha(root / name) for name in dependencies
        },
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": __import__("platform").platform(),
        },
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    output, curves, cohort = calculate(root)
    # Bind cohort occurrence data alongside every prediction/loss array.
    trial_states = {
        key: value for key, value in output.items() if "_trial_parameters" in key
    }
    core_evidence = {
        key: value for key, value in output.items() if "_trial_parameters" not in key
    }
    np.savez_compressed(out / "evidence.npz", **core_evidence)
    shards = (
        ("trial-states-00-07.npz", range(8)),
        ("trial-states-08-15.npz", range(8, 16)),
    )
    for name, batches in shards:
        prefixes = {f"batch{batch:02d}_" for batch in batches}
        selected = {
            key: value
            for key, value in trial_states.items()
            if any(key.startswith(prefix) for prefix in prefixes)
        }
        np.savez_compressed(out / name, **selected)
    labels = classify(curves)
    diagnostic = {
        "schema": "seed470-fixed-initializer-finite-step-transport-v1",
        "scales": SCALES,
        "curve_records": curves,
        "classifications": labels,
        "counts": {
            key: {
                "predicted_descent_observed_ascent_at_scale_1": sum(
                    r["predicted_descent_at_1"]
                    and r["scale_index"] == 0
                    and r["ce_change"] > 1e-6
                    for r in curves
                    if f"{r['objective']}/{r['direction']}/{r['weighting']}" == key
                ),
                "predicted_descent_observed_ascent_by_scale": {
                    str(scale): sum(
                        r["predicted_descent"] and r["ce_change"] > 1e-6
                        for r in curves
                        if f"{r['objective']}/{r['direction']}/{r['weighting']}" == key
                        and r["scale_index"] == index
                    )
                    for index, scale in enumerate(SCALES)
                },
                "scale_1_descent": sum(
                    r["scale_index"] == 0 and r["ce_change"] < -1e-6
                    for r in curves
                    if f"{r['objective']}/{r['direction']}/{r['weighting']}" == key
                ),
            }
            for key in labels
        },
        "cohort": {
            "definition": "seed469 fresh strict-unseen active_stones>32; source_ids==0; preserve evaluation order and all occurrences",
            "occurrences": 859,
            "identities": 851,
            "selection_indices": cohort["selection_indices"],
            "identity_sha256": hashlib.sha256(
                "\n".join(cohort["identities"]).encode()
            ).hexdigest(),
        },
    }
    (out / "diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Seed470 fixed-initializer finite-step transport",
        "",
        "Scale-one directions are first-order predictions; each scale is independently written from the same float32 initializer.",
        "",
        "| Objective / direction / weighting | Classification | Predicted descent, observed ascent at 1 |",
        "|---|---|---:|",
    ]
    lines += [
        f"| {key} | `{label}` | {diagnostic['counts'][key]['predicted_descent_observed_ascent_at_scale_1']} |"
        for key, label in labels.items()
    ]
    lines += [
        "",
        "Descriptive fixed-initializer finite-step diagnostic; not independent confirmation, a causal explanation of a historical trajectory, a learning-rate choice, or a model-strength claim. No training trajectory was run.",
    ]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    receipt = {
        name: sha(out / name)
        for name in (
            "registration.json",
            "diagnostic.json",
            "evidence.npz",
            "trial-states-00-07.npz",
            "trial-states-08-15.npz",
            "results.md",
        )
    }
    receipt["status"] = "post_execution_bound_receipt"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
