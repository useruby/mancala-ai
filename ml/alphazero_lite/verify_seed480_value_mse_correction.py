"""Independent, read-only verification of seed480 value-MSE correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from ml.alphazero_lite.verify_seed479_population_ablation import (
    verify as verify_seed479,
)
from ml.alphazero_lite.seed480_value_mse_correction import BASE, OUT, SOURCE_PATHS

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _near(a, b, name: str) -> None:
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            raise ValueError(f"inventory:{name}")
        for k in b:
            _near(a[k], b[k], f"{name}.{k}")
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            raise ValueError(f"length:{name}")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _near(x, y, f"{name}[{i}]")
    elif isinstance(b, float):
        if not np.isclose(a, b, rtol=2e-6, atol=2e-8):
            raise ValueError(f"number:{name}")
    elif a != b:
        raise ValueError(f"value:{name}")


def verify(root: Path) -> None:
    root = root.resolve()
    out, base = root / OUT, root / BASE
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    historical = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.iterdir() if p.is_file()
    }
    try:
        receipt = json.loads((out / "receipt.json").read_text())
        if receipt.get("status") != "seed480_value_mse_correction_bound" or set(
            receipt
        ) != {
            "status",
            "registration.json",
            "evidence.json",
            "occurrences.npz",
            "results.md",
        }:
            raise ValueError("receipt_inventory")
        for name in set(receipt) - {"status"}:
            if _sha(out / name) != receipt[name]:
                raise ValueError(f"receipt:{name}")
        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed480-value-mse-correction-registration-v1"
            or reg.get("status") != "post_execution_correction"
        ):
            raise ValueError("registration_identity")
        if reg.get("gate_definition") != (
            "F pooled value-head squared error versus occurrence-specific authenticated "
            "evaluation value targets, regression from initializer <= 0.01 under both "
            "exposure and equal_input weighting; all seed479 gates except the "
            "incorrectly labelled pooled policy-vector MSE gate retained."
        ):
            raise ValueError("gate_definition")
        expected_dependencies = {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        _near(reg.get("dependencies"), expected_dependencies, "dependencies")
        input_paths = {
            "seed466_manifest": root
            / "docs/data/seed466-replay-input-bundle/manifest.json",
            "seed466_training": root
            / "docs/data/seed466-replay-input-bundle/training.npz",
            "seed466_evaluation": root
            / "docs/data/seed466-replay-input-bundle/evaluation.npz",
            "initializer": root / INPUT_PATHS["initializer"],
        }
        if set(reg.get("inputs_sha256", {})) != set(input_paths):
            raise ValueError("input_inventory")
        for key, path in input_paths.items():
            if _sha(path) != reg["inputs_sha256"][key]:
                raise ValueError(f"input_binding:{key}")
        if set(reg.get("sources_sha256", {})) != set(SOURCE_PATHS):
            raise ValueError("source_inventory")
        for p in SOURCE_PATHS:
            if _sha(root / p) != reg["sources_sha256"][p]:
                raise ValueError(f"source_binding:{p}")
        for field, path in (
            ("seed479_receipt_sha256", "receipt.json"),
            ("seed479_registration_sha256", "registration.json"),
            ("seed479_trajectory_sha256", "trajectory-evidence.npz"),
            ("seed479_evidence_sha256", "evidence.json"),
        ):
            if _sha(base / path) != reg.get(field):
                raise ValueError(f"historical_binding:{field}")
        for p, (content, _) in historical.items():
            if (base / p).read_bytes() != content:
                raise ValueError(f"historical_changed:{p}")
        # This checks all archived step states, gradients, proposals, and predictions
        # before they are treated as the authenticated trajectory for endpoint loading.
        verify_seed479(root)
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
        archive = np.load(out / "occurrences.npz", allow_pickle=False)
        evidence = json.loads((out / "evidence.json").read_text())
        if (
            evidence.get("schema") != "seed480-value-mse-correction-v1"
            or evidence.get("correction_type")
            != "post_execution_estimand_correction_not_registration_modification"
        ):
            raise ValueError("evidence_identity")
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        torch.set_num_threads(1)
        trajectory = np.load(base / "trajectory-evidence.npz", allow_pickle=False)
        targets: dict[str, np.ndarray] = {}
        weights: dict[str, dict[str, np.ndarray]] = {}
        predictions: dict[str, dict[str, np.ndarray]] = {}
        occurrences = {}
        for cohort, selected in cohorts.items():
            identities = ev["input_identity"][selected].astype(str).tolist()
            targets[cohort] = (
                ev["value_targets"].reshape(-1)[selected].astype(np.float64)
            )
            counts = {item: identities.count(item) for item in set(identities)}
            weights[cohort] = {
                "exposure": np.full(len(selected), 1 / len(selected)),
                "equal_input": np.asarray(
                    [1 / (len(counts) * counts[item]) for item in identities],
                    dtype=np.float64,
                ),
            }
            x = tr["inputs"][rows[selected]]
            predictions[cohort] = {}
            for state in ("initializer", "M", "F"):
                train.load_checkpoint_into_model(
                    model, root / INPUT_PATHS["initializer"]
                )
                if state != "initializer":
                    for i, parameter in enumerate(model.parameters()):
                        parameter.data.copy_(
                            torch.from_numpy(
                                trajectory[
                                    f"{state[0]}_step_16_post_parameter_{i:02d}"
                                ].copy()
                            )
                        )
                model.eval()
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(x), 512):
                        _, value = model(
                            torch.from_numpy(x[start : start + 512].copy())
                        )
                        chunks.append(
                            value.reshape(-1).cpu().numpy().astype(np.float64)
                        )
                predictions[cohort][state] = np.concatenate(chunks)
                if not np.allclose(
                    archive[f"{cohort}_{state}_predictions"],
                    predictions[cohort][state],
                    rtol=2e-6,
                    atol=2e-8,
                ):
                    raise ValueError(f"value_prediction:{cohort}:{state}")
            for name, expected in (
                (f"{cohort}_targets", targets[cohort]),
                *((f"{cohort}_{w}_weights", x) for w, x in weights[cohort].items()),
            ):
                if not np.array_equal(archive[name], expected):
                    raise ValueError(f"occurrence_array:{name}")
            occurrences[cohort] = [
                {
                    "ordinal": i,
                    "evaluation_row": int(row),
                    "input_identity": identities[i],
                    "prediction": {
                        s: float(predictions[cohort][s][i]) for s in predictions[cohort]
                    },
                    "value_target": float(targets[cohort][i]),
                    "weights": {
                        w: float(weights[cohort][w][i]) for w in weights[cohort]
                    },
                }
                for i, row in enumerate(selected)
            ]
        metrics = {}
        effects = {}
        for cohort in cohorts:
            metrics[cohort] = {}
            for state in ("initializer", "M", "F"):
                squared = np.square(predictions[cohort][state] - targets[cohort])
                metrics[cohort][state] = {
                    w: float(np.dot(squared, weight))
                    for w, weight in weights[cohort].items()
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
        gates.update(
            {k: bool(v) for k, v in old["gates"].items() if "pooled_MSE" not in k}
        )
        historical_metric = {
            "label": "policy_vector_squared_error",
            "metrics": {
                c: {
                    s: old["endpoint_metrics"][s][c]["mse"]
                    for s in ("initializer", "M", "F")
                }
                for c in ("fresh", "pooled")
            },
        }
        expected = {
            "schema": "seed480-value-mse-correction-v1",
            "correction_type": "post_execution_estimand_correction_not_registration_modification",
            "occurrences": occurrences,
            "value_mse": metrics,
            "effects": effects,
            "historical_policy_metric": historical_metric,
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
        _near(evidence, expected, "evidence")
    finally:
        for directory, snapshot in ((out, before), (base, historical)):
            for name, (content, mtime) in snapshot.items():
                p = directory / name
                if (
                    not p.exists()
                    or p.read_bytes() != content
                    or p.stat().st_mtime_ns != mtime
                ):
                    raise ValueError(f"publication_modified_during_verification:{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
    print("seed480 independent verification passed")


if __name__ == "__main__":
    main()
