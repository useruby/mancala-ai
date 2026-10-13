"""Append-only seed469 correction for seed468 gradient and writeback checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed468_transport as original

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/seed469-optimizer-direction-correction"


def realize(initial: np.ndarray, update: np.ndarray) -> np.ndarray:
    """Apply a float32 proposal and return its actual float64 parameter delta."""
    initial32 = np.asarray(initial, dtype=np.float32)
    proposed = (initial32 + np.asarray(update, dtype=np.float32)).astype(np.float32)
    return proposed.astype(np.float64) - initial32.astype(np.float64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    records, old_vectors = original.calculate(root)
    registration = json.loads(
        (
            root
            / "docs/data/seed468-first-step-optimizer-direction-transport/registration.json"
        ).read_text()
    )
    init_path = root / registration["inputs"]["initializer_path"]
    gradients = {
        key: value.copy()
        for key, value in old_vectors.items()
        if key.startswith("unseen_gradient_")
        or key.endswith("_P_gradient")
        or key.endswith("_J_gradient")
    }
    # Load the actual parameter vector in registration order, preserving unused parameters.
    model = original.train.PolicyValueNet((96, 3), "residual_v3", 27)
    original.train.load_checkpoint_into_model(model, init_path)
    initial = original.flatten([p.detach() for p in model.parameters()])
    updated: list[dict[str, Any]] = []
    vectors = dict(gradients)
    for record in records:
        batch, objective, direction = (
            record["batch"],
            record["objective"],
            record["direction"],
        )
        gradient = gradients[f"batch{batch:02d}_{objective}_gradient"]
        if direction == "adam":
            update, _, _ = original.adam_displacement(gradient)
        else:
            adam_update, _, _ = original.adam_displacement(gradient)
            adam_realized = realize(initial, adam_update)
            update = np.zeros_like(gradient)
            gn = float(np.linalg.norm(gradient.astype(np.float64)))
            if gn:
                update = (
                    -gradient * np.float32(np.linalg.norm(adam_realized) / gn)
                ).astype(np.float32)
        delta = realize(initial, update)
        stem = f"batch{batch:02d}_{objective}_{direction}"
        vectors[f"{stem}_proposal"] = update
        vectors[f"{stem}_realized"] = delta
        weighting = record["weighting"]
        eval_gradient = vectors[f"unseen_gradient_{weighting}"]
        geometry = original.geometry(eval_gradient, delta)
        updated.append(
            {
                **{k: v for k, v in record.items() if k not in geometry},
                **geometry,
                "proposal_norm": float(np.linalg.norm(update.astype(np.float64))),
                "adam_realized_norm": float(
                    np.linalg.norm(
                        realize(initial, original.adam_displacement(gradient)[0])
                    )
                ),
                "writeback_norm_discrepancy": float(
                    np.linalg.norm(delta) - np.linalg.norm(update.astype(np.float64))
                ),
            }
        )
    if not np.array_equal(
        initial, original.flatten([p.detach() for p in model.parameters()])
    ):
        raise ValueError("initializer_mutated")
    return vectors, updated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / OUT
    output.mkdir(parents=True, exist_ok=True)
    vectors, comparisons = calculate(root)
    np.savez_compressed(output / "vectors.npz", **vectors)
    historical = json.loads(
        (
            root
            / "docs/data/seed468-first-step-optimizer-direction-transport/diagnostic.json"
        ).read_text()
    )
    counts: dict[str, dict[str, int]] = {}
    for objective in ("P", "J"):
        for direction in ("adam", "negative_gradient"):
            for weighting in ("exposure_weighted", "equal_input"):
                rows = [
                    r
                    for r in comparisons
                    if (r["objective"], r["direction"], r["weighting"])
                    == (objective, direction, weighting)
                ]
                cosines = [r["cosine"] for r in rows]
                counts[f"{objective}/{direction}/{weighting}"] = {
                    "descent": sum(c is not None and c <= -1e-6 for c in cosines),
                    "ascent": sum(c is not None and c >= 1e-6 for c in cosines),
                    "ambiguous_or_zero_norm": sum(
                        c is None or -1e-6 < c < 1e-6 for c in cosines
                    ),
                }
    report = {
        "schema": "seed469-optimizer-direction-correction-v1",
        "comparisons": comparisons,
        "classification_counts": counts,
        "classification": original.classify(comparisons),
        "historical_seed468_classification": historical["classification"],
        "classification_changed": original.classify(comparisons)
        != historical["classification"],
        "writeback_summary": {
            "proposal_realized_vector_differences": sum(
                not np.array_equal(
                    vectors[f"batch{b:02d}_{o}_{d}_proposal"].astype(np.float64),
                    vectors[f"batch{b:02d}_{o}_{d}_realized"],
                )
                for b in range(16)
                for o in ("P", "J")
                for d in ("adam", "negative_gradient")
            ),
            "maximum_absolute_norm_discrepancy": max(
                abs(row["writeback_norm_discrepancy"]) for row in comparisons
            ),
        },
        "interpretation": "Corrected diagnostic only; seed468 remains the historical publication.",
    }
    (output / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    registration = {
        "schema": "seed469-optimizer-direction-correction-registration-v1",
        "seed468_registration_sha256": sha(
            root
            / "docs/data/seed468-first-step-optimizer-direction-transport/registration.json"
        ),
        "seed468_receipt_sha256": sha(
            root
            / "docs/data/seed468-first-step-optimizer-direction-transport/receipt.json"
        ),
        "input_bundle_manifest_sha256": original.MANIFEST_SHA,
        "initializer_sha256": original.INIT_SHA,
        "input_sha256": {
            "seed466_manifest": sha(
                root / "docs/data/seed466-replay-input-bundle/manifest.json"
            ),
            "seed466_training": sha(
                root / "docs/data/seed466-replay-input-bundle/training.npz"
            ),
            "seed466_evaluation": sha(
                root / "docs/data/seed466-replay-input-bundle/evaluation.npz"
            ),
            "initializer": sha(
                root
                / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
            ),
        },
        "correction_sources_sha256": {
            "ml/alphazero_lite/seed469_correction.py": sha(
                root / "ml/alphazero_lite/seed469_correction.py"
            ),
            "ml/alphazero_lite/verify_seed469_correction.py": sha(
                root / "ml/alphazero_lite/verify_seed469_correction.py"
            ),
            "ml/alphazero_lite/test_seed469_correction.py": sha(
                root / "ml/alphazero_lite/test_seed469_correction.py"
            ),
        },
        "historical_classification": historical["classification"],
        "comparison_tolerances": {
            "rtol": 2e-6,
            "atol": 2e-8,
            "explanation": "Float32 PyTorch gradient reconstruction may differ by reduction-order rounding; tolerances apply only to independently reconstructed numeric vectors and float comparison fields. Keys, identities, shapes, dtypes, finite status, counts, classifications, and receipt hashes are exact.",
        },
        "protocol": "float32 proposal; theta_prime=float32(theta+u); realized=float64(theta_prime)-float64(theta); geometry on realized displacement",
    }
    (output / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    summary = [
        "# Seed469 post-execution correction",
        "",
        "This is an append-only, local first-order diagnostic correction. Seed468 remains the historical publication.",
        "",
        f"Historical classification: `{historical['classification']}`.",
        f"Corrected classification: `{report['classification']}`.",
        f"Classification changed: `{report['classification_changed']}`.",
        "",
        "Realized displacement is computed from float32 parameter writeback; comparison geometry uses float64 realized displacement. Each comparison records raw proposal norm, realized Adam norm, and post-writeback norm discrepancy.",
        f"Across the 64 proposals, {report['writeback_summary']['proposal_realized_vector_differences']} proposal vectors differ from their realized writeback vectors; maximum absolute proposal-vs-realized norm discrepancy is {report['writeback_summary']['maximum_absolute_norm_discrepancy']:.9g}.",
        "Numerical reconstruction tolerances are bound in registration.json: rtol=2e-6 and atol=2e-8 for floating-point comparisons only; structural and classification checks are exact.",
        "",
        "No training trajectory, searches, games, runtime export, promotion, or strength claim.",
    ]
    (output / "results.md").write_text("\n".join(summary) + "\n")
    receipt = {
        name: sha(output / name)
        for name in (
            "registration.json",
            "diagnostic.json",
            "vectors.npz",
            "results.md",
        )
    }
    receipt["status"] = "post_execution_correction_receipt"
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
