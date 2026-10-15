"""Publish the post-execution seed479 value-head MSE correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from ml.alphazero_lite import train
from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
)
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)
from ml.alphazero_lite.seed479_population_ablation import INPUT_PATHS
from ml.alphazero_lite.seed480_value_mse_correction import BASE, OUT, SOURCE_PATHS

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root: Path, *, replace: bool = False) -> None:
    root = root.resolve()
    out = root / OUT
    if replace and out.exists():
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed480_output_exists")
    base = root / BASE
    # Authenticate the historical receipt before using the frozen trajectory.
    from ml.alphazero_lite.verify_seed479_population_ablation import verify

    verify(root)
    base_receipt = json.loads((base / "receipt.json").read_text())
    data = read_authenticated_bundle(
        root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
    )
    tr, ev = data["training"], data["evaluation"]
    rows = ev["compact_rows"]
    stones = np.asarray(
        [active_pit_stones_from_encoded_state(x) for x in tr["inputs"][rows]]
    )
    cohorts = {
        "fresh": np.flatnonzero((ev["source_ids"] == 0) & (stones > 32)),
        "pooled": np.flatnonzero(stones > 32),
    }
    trajectory = np.load(base / "trajectory-evidence.npz", allow_pickle=False)
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    torch.set_num_threads(1)

    def predict(state: str, inputs: np.ndarray) -> np.ndarray:
        train.load_checkpoint_into_model(model, root / INPUT_PATHS["initializer"])
        if state != "initializer":
            arm = state[0]
            for i, parameter in enumerate(model.parameters()):
                parameter.data.copy_(
                    torch.from_numpy(
                        trajectory[f"{arm}_step_16_post_parameter_{i:02d}"].copy()
                    )
                )
        model.eval()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(inputs), 512):
                _, value = model(torch.from_numpy(inputs[start : start + 512].copy()))
                parts.append(value.reshape(-1).cpu().numpy().astype(np.float64))
        return np.concatenate(parts)

    identities: dict[str, list[str]] = {}
    targets: dict[str, np.ndarray] = {}
    weights: dict[str, dict[str, np.ndarray]] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for cohort, selected in cohorts.items():
        ids = ev["input_identity"][selected].astype(str).tolist()
        identities[cohort] = ids
        targets[cohort] = ev["value_targets"].reshape(-1)[selected].astype(np.float64)
        counts = {key: ids.count(key) for key in set(ids)}
        weights[cohort] = {
            "exposure": np.full(len(ids), 1 / len(ids), dtype=np.float64),
            "equal_input": np.asarray([1 / (len(counts) * counts[key]) for key in ids]),
        }
        x = tr["inputs"][rows[selected]]
        predictions[cohort] = {
            state: predict(state, x) for state in ("initializer", "M", "F")
        }

    metrics = {}
    effects = {}
    for cohort in cohorts:
        metrics[cohort] = {}
        for state in ("initializer", "M", "F"):
            err = np.square(predictions[cohort][state] - targets[cohort])
            metrics[cohort][state] = {
                w: float(np.dot(err, weight)) for w, weight in weights[cohort].items()
            }
        effects[cohort] = {
            contrast: {
                w: metrics[cohort][right][w] - metrics[cohort][left][w]
                for w in weights[cohort]
            }
            for contrast, left, right in (
                ("M_minus_initializer", "initializer", "M"),
                ("F_minus_initializer", "initializer", "F"),
                ("F_minus_M", "M", "F"),
            )
        }
    gates = {
        f"pooled_value_MSE_regression_{w}_at_most_0.01": effects["pooled"][
            "F_minus_initializer"
        ][w]
        <= 0.01
        for w in ("exposure", "equal_input")
    }
    old = json.loads((base / "evidence.json").read_text())
    gates.update({k: bool(v) for k, v in old["gates"].items() if "pooled_MSE" not in k})
    # Historical policy-vector MSE is retained with its proper explicit label.
    historical = {
        "label": "policy_vector_squared_error",
        "metrics": {
            c: {
                s: old["endpoint_metrics"][s][c]["mse"]
                for s in ("initializer", "M", "F")
            }
            for c in ("fresh", "pooled")
        },
    }
    evidence = {
        "schema": "seed480-value-mse-correction-v1",
        "correction_type": "post_execution_estimand_correction_not_registration_modification",
        "occurrences": {
            c: [
                {
                    "ordinal": i,
                    "evaluation_row": int(r),
                    "input_identity": identities[c][i],
                    "prediction": {
                        s: float(predictions[c][s][i]) for s in predictions[c]
                    },
                    "value_target": float(targets[c][i]),
                    "weights": {w: float(weights[c][w][i]) for w in weights[c]},
                }
                for i, r in enumerate(sel)
            ]
            for c, sel in cohorts.items()
        },
        "value_mse": metrics,
        "effects": effects,
        "historical_policy_metric": historical,
        "historical_ce_metrics": {
            "initializer": old["initializer_metrics"],
            "endpoints": old["endpoint_metrics"],
            "full_training_policy_ce": old["full_training_policy_ce"],
        },
        "gates": gates,
        "decision": "advance_to_fresh_confirmation"
        if all(gates.values())
        else "close_full_population_direction_branch",
    }
    registration = {
        "schema": "seed480-value-mse-correction-registration-v1",
        "status": "post_execution_correction",
        "interpretation": "Corrects the requested value-head MSE estimand after seed479 execution; seed479 registration and evidence remain immutable.",
        "dependencies": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "sources_sha256": {p: sha(root / p) for p in SOURCE_PATHS},
        "inputs_sha256": {
            "seed466_manifest": sha(
                root / "docs/data/seed466-replay-input-bundle/manifest.json"
            ),
            "seed466_training": sha(
                root / "docs/data/seed466-replay-input-bundle/training.npz"
            ),
            "seed466_evaluation": sha(
                root / "docs/data/seed466-replay-input-bundle/evaluation.npz"
            ),
            "initializer": sha(root / INPUT_PATHS["initializer"]),
        },
        "seed479_receipt_sha256": sha(base / "receipt.json"),
        "seed479_registration_sha256": sha(base / "registration.json"),
        "seed479_trajectory_sha256": sha(base / "trajectory-evidence.npz"),
        "seed479_evidence_sha256": sha(base / "evidence.json"),
        "forward_schedule": "initializer, then verified M and F step-16 post-parameter states; fixed batch size 512, model residual_v3 (96,3), value output as returned by PolicyValueNet.forward",
        "weighting": {
            "exposure": "equal mass per occurrence",
            "equal_input": "equal mass per exact input identity, divided equally among repeated occurrences",
        },
        "gate_definition": "F pooled value-head squared error versus occurrence-specific authenticated evaluation value targets, regression from initializer <= 0.01 under both exposure and equal_input weighting; all seed479 gates except the incorrectly labelled pooled policy-vector MSE gate retained.",
        "receipt_parent": base_receipt,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    (out / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        out / "occurrences.npz",
        **{
            f"{c}_{s}_predictions": predictions[c][s]
            for c in cohorts
            for s in predictions[c]
        },
        **{f"{c}_targets": targets[c] for c in cohorts},
        **{f"{c}_{w}_weights": weights[c][w] for c in cohorts for w in weights[c]},
    )
    (out / "results.md").write_text(
        f"# Seed480 value-MSE correction\n\nPost-execution correction of the requested estimand; seed479’s frozen registration and evidence are unchanged.\n\nDecision: `{evidence['decision']}`.\n\nThe historical policy metric is explicitly `policy_vector_squared_error`.\n"
    )
    receipt = {
        name: sha(out / name)
        for name in (
            "registration.json",
            "evidence.json",
            "occurrences.npz",
            "results.md",
        )
    }
    receipt["status"] = "seed480_value_mse_correction_bound"
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--replace-development-output", action="store_true")
    args = parser.parse_args()
    publish(args.root, replace=args.replace_development_output)
    print("seed480 value-MSE correction published")


if __name__ == "__main__":
    main()
