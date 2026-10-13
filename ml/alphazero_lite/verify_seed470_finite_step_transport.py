"""Independent read-only verifier for seed470 finite-step evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ml.alphazero_lite import train
from ml.alphazero_lite.seed466_runtime_bundle import (
    MANIFEST_SHA256,
    read_authenticated_bundle,
)
from ml.alphazero_lite.verify_seed466_replay_bundle import verify as verify_bundle
from ml.alphazero_lite.verify_seed469_correction import (
    INIT_SHA,
    reconstruct as reconstruct_seed469,
)

SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
OUT = Path("docs/data/seed470-finite-step-transport")
INPUTS = {
    "seed466_manifest": "docs/data/seed466-replay-input-bundle/manifest.json",
    "seed466_training": "docs/data/seed466-replay-input-bundle/training.npz",
    "seed466_evaluation": "docs/data/seed466-replay-input-bundle/evaluation.npz",
    "initializer": "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    "seed469_registration": "docs/data/seed469-optimizer-direction-correction/registration.json",
    "seed469_receipt": "docs/data/seed469-optimizer-direction-correction/receipt.json",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_outputs(
    root: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    torch.set_num_threads(1)
    # Reconstructs seed469 proposals from authenticated minibatches and gradients;
    # no seed470 publisher calculation or published prediction is used.
    vectors, _ = reconstruct_seed469(root)
    with np.load(
        root / "docs/data/seed466-replay-input-bundle/training.npz", allow_pickle=False
    ) as z:
        tr = {k: z[k] for k in z.files}
    with np.load(
        root / "docs/data/seed466-replay-input-bundle/evaluation.npz",
        allow_pickle=False,
    ) as z:
        ev = {k: z[k] for k in z.files}
    take = np.flatnonzero(ev["source_ids"] == 0)
    ids = ev["input_identity"][take].tolist()
    if len(take) != 859 or len(set(ids)) != 851:
        raise ValueError("cohort_population_mismatch")
    rows = ev["compact_rows"][take]
    x = ev["inputs"][take] if "inputs" in ev else tr["inputs"][rows]
    p, mask = ev["policy_targets"][take], tr["legal_masks"][rows]
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(model, root / INPUTS["initializer"])
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
    theta = np.concatenate(
        [v.detach().cpu().numpy().reshape(-1) for v in model.parameters()]
    ).astype(np.float32)
    arrays: dict[str, np.ndarray] = {"cohort_targets": p, "cohort_legal_masks": mask}
    arrays.update({f"seed469_{key}": value for key, value in vectors.items()})
    arrays["cohort_identities"] = np.asarray(ids, dtype="U256")
    curves = []

    # Frozen contiguous batches of 128, ascending stored occurrence order.
    def evaluate() -> tuple[np.ndarray, np.ndarray]:
        probs, ces = [], []
        for start in range(0, len(x), 128):
            xx = torch.from_numpy(x[start : start + 128].copy())
            tt = torch.from_numpy(p[start : start + 128].copy())
            mm = torch.from_numpy(mask[start : start + 128].copy())
            logits, _ = model(xx)
            logp = F.log_softmax(logits.masked_fill(mm == 0, -1e9), dim=1)
            probs.append(logp.exp().detach().cpu().numpy())
            ces.append((-(tt * logp).sum(1)).detach().cpu().numpy())
        return np.concatenate(probs), np.concatenate(ces)

    base_pred, base_ce = evaluate()
    for weighting in ("exposure_weighted", "equal_input"):
        counts = {value: ids.count(value) for value in set(ids)}
        weights = np.asarray(
            [
                1 / 859
                if weighting == "exposure_weighted"
                else 1 / (851 * counts[value])
                for value in ids
            ],
            dtype=np.float64,
        )
        arrays[f"cohort_weights_{weighting}"] = weights

        arrays[f"baseline_{weighting}_predictions"] = base_pred
        arrays[f"baseline_{weighting}_losses"] = base_ce
        for batch in range(16):
            for objective in ("P", "J"):
                for direction in ("adam", "negative_gradient"):
                    stem = f"batch{batch:02d}_{objective}_{direction}"
                    proposal = vectors[f"{stem}_proposal"].astype(np.float32)
                    eg = vectors[f"unseen_gradient_{weighting}"].astype(np.float64)
                    one_trial = np.add(theta, proposal, dtype=np.float32).astype(
                        np.float64
                    ) - theta.astype(np.float64)
                    first_order = float(np.dot(eg, one_trial))
                    for si, scale in enumerate(SCALES):
                        product = np.multiply(
                            proposal, np.float32(scale), dtype=np.float32
                        )
                        trial = np.add(theta, product, dtype=np.float32).astype(
                            np.float32
                        )
                        with torch.no_grad():
                            offset = 0
                            for param in model.parameters():
                                n = param.numel()
                                param.copy_(
                                    torch.from_numpy(
                                        trial[offset : offset + n].reshape(param.shape)
                                    )
                                )
                                offset += n
                        delta = trial.astype(np.float64) - theta.astype(np.float64)
                        arrays[f"{stem}_scale{si}_trial_parameters"] = trial.copy()
                        prediction, ce = evaluate()
                        arrays[f"{stem}_scale{si}_predictions"] = prediction
                        arrays[f"{stem}_scale{si}_losses"] = ce
                        change = float(
                            np.dot(
                                ce.astype(np.float64) - base_ce.astype(np.float64),
                                weights,
                            )
                        )
                        first = float(np.dot(eg, delta))
                        curves.append(
                            {
                                "batch": batch,
                                "objective": objective,
                                "direction": direction,
                                "weighting": weighting,
                                "scale_index": si,
                                "scale": scale,
                                "displacement_norm": float(np.linalg.norm(delta)),
                                "ce_change": change,
                                "first_order_prediction": first,
                                "remainder": change - first,
                                "predicted_descent": first < 0,
                                "predicted_descent_at_1": bool(first_order < 0),
                            }
                        )
                    # Restore the same baseline parameters before the next raw proposal.
                    with torch.no_grad():
                        offset = 0
                        for param in model.parameters():
                            n = param.numel()
                            param.copy_(
                                torch.from_numpy(
                                    theta[offset : offset + n].reshape(param.shape)
                                )
                            )
                            offset += n
    if any(not torch.equal(model.state_dict()[k], v) for k, v in frozen.items()):
        raise ValueError("initializer_or_buffer_mutation")
    lookup = {
        (
            item["batch"],
            item["objective"],
            item["direction"],
            item["weighting"],
            item["scale_index"],
        ): item
        for item in curves
    }
    for item in curves:
        other = (
            "equal_input"
            if item["weighting"] == "exposure_weighted"
            else "exposure_weighted"
        )
        item["other_weighting_ce_change"] = lookup[
            (
                item["batch"],
                item["objective"],
                item["direction"],
                other,
                item["scale_index"],
            )
        ]["ce_change"]
    return arrays, curves, {"ids": ids, "indices": take.tolist()}


def expected_classifications(curves: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for obj in ("P", "J"):
        for direction in ("adam", "negative_gradient"):
            for weighting in ("exposure_weighted", "equal_input"):
                selected = [
                    r
                    for r in curves
                    if (r["objective"], r["direction"], r["weighting"])
                    == (obj, direction, weighting)
                ]
                overshoot = sum(
                    r["predicted_descent_at_1"]
                    and r["scale_index"] == 0
                    and r["ce_change"] > 1e-6
                    and any(
                        q["batch"] == r["batch"]
                        and q["objective"] == obj
                        and q["direction"] == direction
                        and q["weighting"] == weighting
                        and q["scale_index"] >= 3
                        and q["ce_change"] < -1e-6
                        for q in selected
                    )
                    for r in selected
                )
                improvement = sum(
                    r["scale_index"] == 0 and r["ce_change"] < -1e-6 for r in selected
                )
                label = (
                    "finite_step_overshoot_consistent"
                    if overshoot >= 12
                    else "finite_step_improvement_consistent"
                    if improvement >= 12
                    else "mixed_or_inconclusive_finite_step"
                )
                result[f"{obj}/{direction}/{weighting}"] = label
    return result


def verify(root: Path) -> None:
    out = root / OUT
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in out.iterdir()
        if p.is_file()
    }
    try:
        verify_bundle(root / "docs/data/seed466-replay-input-bundle", root)
        read_authenticated_bundle(
            root / "docs/data/seed466-replay-input-bundle", MANIFEST_SHA256
        )
        if digest(root / INPUTS["initializer"]) != INIT_SHA:
            raise ValueError("initializer_identity_mismatch")
        reg = json.loads((out / "registration.json").read_text())
        for key, relative in INPUTS.items():
            if reg["input_sha256"].get(key) != digest(root / relative):
                raise ValueError(f"input_binding_mismatch:{key}")
        for rel, expected in reg["source_dependency_closure_sha256"].items():
            if digest(root / rel) != expected:
                raise ValueError(f"source_binding_mismatch:{rel}")
        if reg.get("scales") != list(SCALES):
            raise ValueError("registered_scale_order_mismatch")
        receipt = json.loads((out / "receipt.json").read_text())
        for name in (
            "registration.json",
            "diagnostic.json",
            "evidence.npz",
            "trial-states-00-07.npz",
            "trial-states-08-15.npz",
            "results.md",
        ):
            if receipt.get(name) != digest(out / name):
                raise ValueError(f"receipt_mismatch:{name}")
        arrays, curves, cohort = independent_outputs(root)
        with np.load(out / "evidence.npz", allow_pickle=False) as archive:
            actual = {k: archive[k] for k in archive.files}
        for shard_name in ("trial-states-00-07.npz", "trial-states-08-15.npz"):
            with np.load(out / shard_name, allow_pickle=False) as archive:
                for key in archive.files:
                    if key in actual:
                        raise ValueError(f"duplicate_trial_state:{key}")
                    actual[key] = archive[key]
        if set(actual) != set(arrays):
            raise ValueError("evidence_key_set_mismatch")
        for key, expected in arrays.items():
            found = actual[key]
            finite = (
                np.isfinite(found).all()
                if np.issubdtype(found.dtype, np.number)
                else True
            )
            exact = (
                key in {"cohort_identities", "cohort_targets", "cohort_legal_masks"}
                or "trial_parameters" in key
            )
            equal = (
                np.array_equal(found, expected)
                if exact
                else np.allclose(found, expected, rtol=2e-6, atol=2e-8)
            )
            if (
                found.shape != expected.shape
                or found.dtype != expected.dtype
                or not finite
                or not equal
            ):
                raise ValueError(f"evidence_reconstruction_mismatch:{key}")
        report = json.loads((out / "diagnostic.json").read_text())
        if report.get("curve_records") != curves:
            # JSON float serialization round trips exactly for these Python floats.
            raise ValueError("curve_reconstruction_mismatch")
        if report.get("classifications") != expected_classifications(curves):
            raise ValueError("classification_mismatch")
        expected_counts = {}
        for key in expected_classifications(curves):
            selected = [
                row
                for row in curves
                if f"{row['objective']}/{row['direction']}/{row['weighting']}" == key
            ]
            expected_counts[key] = {
                "predicted_descent_observed_ascent_at_scale_1": sum(
                    row["predicted_descent_at_1"]
                    and row["scale_index"] == 0
                    and row["ce_change"] > 1e-6
                    for row in selected
                ),
                "predicted_descent_observed_ascent_by_scale": {
                    str(scale): sum(
                        row["predicted_descent"] and row["ce_change"] > 1e-6
                        for row in selected
                        if row["scale_index"] == index
                    )
                    for index, scale in enumerate(SCALES)
                },
                "scale_1_descent": sum(
                    row["scale_index"] == 0 and row["ce_change"] < -1e-6
                    for row in selected
                ),
            }
        if report.get("counts") != expected_counts:
            raise ValueError("descriptive_counts_mismatch")
        expected_id_hash = hashlib.sha256("\n".join(cohort["ids"]).encode()).hexdigest()
        if report["cohort"]["selection_indices"] != cohort["indices"]:
            raise ValueError("cohort_order_mismatch")
        if report["cohort"].get("identity_sha256") != expected_id_hash:
            raise ValueError("published_cohort_identity_mismatch")
        if (
            report["cohort"].get("occurrences") != 859
            or report["cohort"].get("identities") != 851
        ):
            raise ValueError("cohort_count_mismatch")
        if (
            reg["seed469_fresh_cohort"].get("ordered_identity_sha256")
            != expected_id_hash
        ):
            raise ValueError("cohort_identity_binding_mismatch")
        if reg["seed469_fresh_cohort"].get("selection_indices") != cohort["indices"]:
            raise ValueError("registered_cohort_order_mismatch")
        if (
            len(curves) != 1024
            or len(
                {
                    (
                        r["batch"],
                        r["objective"],
                        r["direction"],
                        r["weighting"],
                        r["scale_index"],
                    )
                    for r in curves
                }
            )
            != 1024
        ):
            raise ValueError("trial_record_identity_mismatch")
        if report.get("classifications") != expected_classifications(curves):
            raise ValueError("classification_mismatch")
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
    args = parser.parse_args()
    verify(args.root.resolve())
    print("seed470 independent verification passed")


if __name__ == "__main__":
    main()
