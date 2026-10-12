"""Read-only independent verifier for the append-only seed462 correction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import verify_seed462_recovery as recovery

ATTEMPT = Path(recovery.ATTEMPT)
RECOVERY = Path(recovery.RECOVERY)
CORRECTION = Path("docs/data/seed462-historical-policy-removal-correction")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def objective_accounting(result: dict[str, Any]) -> dict[str, Any]:
    """Calculate both objectives only on expanded training occurrences."""
    arrays = result["arrays"]
    x, policies, targets, replay, coefficient, train_positions = arrays[:6]
    fresh = result["fresh"]
    occurrences = replay[train_positions]
    counts = {
        "training_occurrences": int(len(occurrences)),
        "training_fresh_occurrences": int(fresh[occurrences].sum()),
        "training_nonfresh_occurrences": int((~fresh[occurrences]).sum()),
    }
    out: dict[str, Any] = {"occurrence_counts": counts, "arms": {}}
    for name in ("initializer", "C", "F"):
        model = result["models"].get(name)
        if model is None:
            model = recovery.train.PolicyValueNet((96, 3), "residual_v3", 27)
            recovery.train.load_checkpoint_into_model(
                model, result["root"] / recovery.INIT
            )
        model.eval()
        ce_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(occurrences), 2048):
                ids = occurrences[start : start + 2048]
                logits, prediction = model(torch.from_numpy(x[ids]))
                legal = torch.from_numpy(
                    recovery.train.legal_mask_matrix_for_encoded_states(x[ids])
                )
                ce_parts.append(
                    recovery.train.compute_policy_cross_entropy(
                        logits.masked_fill(legal <= 0, -1e9),
                        torch.from_numpy(policies[ids]),
                    )
                    .cpu()
                    .numpy()
                )
                value_parts.append(prediction.reshape(-1).cpu().numpy())
        ce = np.concatenate(ce_parts)
        predicted = np.concatenate(value_parts)
        actual = targets[occurrences].reshape(-1)
        error = np.abs(predicted - actual)
        huber = np.where(error < 1, 0.5 * error**2, error - 0.5)
        weights = coefficient[occurrences].astype(np.float64)
        fresh_mask = fresh[occurrences]
        masses = {
            "original_mixture": float(weights.sum(dtype=np.float64)),
            "fresh_policy": float(weights[fresh_mask].sum(dtype=np.float64)),
        }
        if masses["original_mixture"] <= 0 or masses["fresh_policy"] <= 0:
            raise ValueError("empty_training_policy_mass")
        out["arms"][name] = {
            "policy_masses": masses,
            "original_mixture": float(
                np.average(ce, weights=weights) + 0.3 * huber.mean()
            ),
            "fresh_policy": float(
                np.average(ce, weights=weights * fresh_mask) + 0.3 * huber.mean()
            ),
        }
    return out


def endpoint_reconciliation(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Rebuild occurrence-level validation evidence with original inference order."""
    arrays = result["arrays"]
    x, _policies, _values = arrays[:3]
    membership_path = (
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz"
    )
    with gzip.open(membership_path, "rt", encoding="utf-8") as stream:
        membership = [json.loads(line) for line in stream]
    pooled = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    cohorts = {
        "pooled": pooled,
        "fresh": [row for row in pooled if row["source"] == "fresh"],
    }
    targets = {
        name: recovery._cohort_targets(root, rows) for name, rows in cohorts.items()
    }
    expected_metrics: dict[str, Any] = {
        "schema": "seed462-retrospective-endpoints-v1",
        "timing": "post_execution_reconstruction",
        "cohorts_reused": True,
        "cohort_definition": "membership subset unseen and active_stones > 32; fresh subgroup source=fresh",
        "arms": {},
        "effects_F_minus_C": {},
        "cohorts": {
            name: [
                {
                    key: row[key]
                    for key in ("input_identity", "compact_row", "source", "raw_line")
                }
                for row in rows
            ]
            for name, rows in cohorts.items()
        },
    }
    for arm in ("initializer", "C", "F"):
        model = result["models"].get(arm)
        if model is None:
            model = recovery.train.PolicyValueNet((96, 3), "residual_v3", 27)
            recovery.train.load_checkpoint_into_model(model, root / recovery.INIT)
        model.eval()
        # Match the historical publisher exactly: one full-array forward pass in
        # compact-row order, followed by its float32 log-softmax and row selection.
        with torch.inference_mode():
            logits, prediction = model(torch.from_numpy(x))
            legal = torch.from_numpy(
                recovery.train.legal_mask_matrix_for_encoded_states(x)
            )
            logp = (
                torch.log_softmax(logits.masked_fill(legal <= 0, -torch.inf), dim=1)
                .cpu()
                .numpy()
            )
            pred_value = prediction.reshape(-1).cpu().numpy()
        record: dict[str, Any] = {}
        archive_keys = {
            f"{cohort}_{suffix}"
            for cohort in ("pooled", "fresh")
            for suffix in ("log_policy", "target_policy", "value", "target_value")
        }
        for cohort_name, rows in cohorts.items():
            policy_target, value_target = targets[cohort_name]
            ids = np.asarray([row["compact_row"] for row in rows], dtype=np.int64)
            for row in rows:
                raw_identity = (
                    x[int(row["compact_row"])].astype("<f4", copy=False).tobytes().hex()
                )
                if row["input_identity"] != raw_identity:
                    raise ValueError(f"cohort_input_identity_mismatch:{cohort_name}")
            ce = -(policy_target * np.where(np.isfinite(logp[ids]), logp[ids], 0)).sum(
                axis=1
            )
            mse = np.square(pred_value[ids] - value_target)
            groups: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                groups.setdefault(row["input_identity"], []).append(index)
            record[cohort_name] = {
                "occurrence_count": len(ids),
                "unique_identities": len(groups),
                "exposure_weighted": {
                    "policy_ce": float(np.mean(ce)),
                    "value_mse": float(np.mean(mse)),
                },
                "equal_input": {
                    "policy_ce": float(
                        np.mean([np.mean(ce[ix]) for ix in groups.values()])
                    ),
                    "value_mse": float(
                        np.mean([np.mean(mse[ix]) for ix in groups.values()])
                    ),
                },
            }
            fields = {
                f"{cohort_name}_log_policy": logp[ids],
                f"{cohort_name}_target_policy": policy_target,
                f"{cohort_name}_value": pred_value[ids],
                f"{cohort_name}_target_value": value_target,
            }
            with np.load(
                root / recovery.RECOVERY / f"{arm}-occurrence-predictions.npz",
                allow_pickle=False,
            ) as archived:
                if set(archived.files) != archive_keys:
                    raise ValueError(f"prediction_archive_keys_invalid:{arm}")
                for key, expected in fields.items():
                    if key not in archived.files or not np.array_equal(
                        archived[key], expected
                    ):
                        raise ValueError(f"prediction_semantic_mismatch:{arm}:{key}")
        # Reproduce the old all-replay estimand separately to explain its numbers.
        ce_parts: list[np.ndarray] = []
        replay = arrays[3]
        for start in range(0, len(x), 2048):
            end = min(start + 2048, len(x))
            legal_batch = torch.from_numpy(
                recovery.train.legal_mask_matrix_for_encoded_states(x[start:end])
            )
            with torch.inference_mode():
                logits_batch, _ = model(torch.from_numpy(x[start:end]))
                ce_parts.append(
                    recovery.train.compute_policy_cross_entropy(
                        logits_batch.masked_fill(legal_batch <= 0, -1e9),
                        torch.from_numpy(arrays[1][start:end]),
                    )
                    .cpu()
                    .numpy()
                )
        row_ce = np.concatenate(ce_parts)
        err = np.abs(pred_value[replay] - arrays[2][replay].reshape(-1))
        huber = np.where(err < 1, 0.5 * err**2, err - 0.5)
        weights = arrays[4][replay]
        source_fresh = result["fresh"][replay]
        record["full_training_objectives"] = {
            "original_mixture": float(
                np.average(row_ce[replay], weights=weights) + 0.3 * huber.mean()
            ),
            "fresh_policy": float(
                np.average(row_ce[replay], weights=weights * source_fresh)
                + 0.3 * huber.mean()
            ),
        }
        expected_metrics["arms"][arm] = record
    for cohort_name in cohorts:
        expected_metrics["effects_F_minus_C"][cohort_name] = {}
        for weighting in ("exposure_weighted", "equal_input"):
            expected_metrics["effects_F_minus_C"][cohort_name][weighting] = {
                metric: expected_metrics["arms"]["F"][cohort_name][weighting][metric]
                - expected_metrics["arms"]["C"][cohort_name][weighting][metric]
                for metric in ("policy_ce", "value_mse")
            }
    # Gates and limitation are part of the original descriptive evidence.
    published = json.loads(
        (root / recovery.RECOVERY / "endpoint-metrics.json").read_text()
    )
    # Gate arithmetic is validated independently against reconstructed metric values.
    gates: dict[str, bool] = {}
    for weight in ("exposure_weighted", "equal_input"):
        gates[f"fresh_vs_initializer_{weight}"] = (
            expected_metrics["arms"]["F"]["fresh"][weight]["policy_ce"]
            <= expected_metrics["arms"]["initializer"]["fresh"][weight]["policy_ce"]
            - 0.005
        )
        gates[f"fresh_vs_C_{weight}"] = (
            expected_metrics["arms"]["F"]["fresh"][weight]["policy_ce"]
            <= expected_metrics["arms"]["C"]["fresh"][weight]["policy_ce"] - 0.005
        )
        gates[f"pooled_ce_{weight}"] = (
            expected_metrics["arms"]["F"]["pooled"][weight]["policy_ce"]
            <= expected_metrics["arms"]["initializer"]["pooled"][weight]["policy_ce"]
            + 0.002
        )
        for cohort in ("pooled", "fresh"):
            gates[f"{cohort}_mse_{weight}"] = (
                expected_metrics["arms"]["F"][cohort][weight]["value_mse"]
                <= expected_metrics["arms"]["initializer"][cohort][weight]["value_mse"]
                + 0.002
            )
    gates["fresh_training_objective_decreases"] = (
        expected_metrics["arms"]["F"]["full_training_objectives"]["fresh_policy"]
        < expected_metrics["arms"]["initializer"]["full_training_objectives"][
            "fresh_policy"
        ]
    )
    expected_metrics["gates"] = gates
    expected_metrics["prospective_protocol_limitation"] = published[
        "prospective_protocol_limitation"
    ]
    recovery.validate_endpoint_semantics(expected_metrics)
    for key in ("arms", "effects_F_minus_C", "cohorts"):
        if expected_metrics[key] != published.get(key):
            raise ValueError(f"endpoint_metrics_semantic_mismatch:{key}")
    return {
        "status": "exact",
        "cohort_occurrences": {name: len(rows) for name, rows in cohorts.items()},
        "metrics": expected_metrics,
    }


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    attempt, old, correction = root / ATTEMPT, root / RECOVERY, root / CORRECTION
    if not attempt.is_dir() or not old.is_dir():
        raise ValueError("original_publication_missing")
    preservation_before = {
        str(path.relative_to(root)): {
            "sha256": sha(path),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for directory in (attempt, old)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    registration_path = attempt / "registration.json"
    receipt_path = old / "recovery-receipt.json"
    registration = json.loads(registration_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if sha(registration_path) != receipt.get("registration_sha256"):
        raise ValueError("registration_binding_mismatch")
    if (
        receipt.get("classification")
        != "no_experiment_decision_protocol_estimands_incomplete"
    ):
        raise ValueError("formal_decision_limitation_binding_mismatch")
    for relative, expected in registration.get("source_sha256", {}).items():
        if sha(root / relative) != expected:
            raise ValueError(f"registered_source_mismatch:{relative}")
    for relative, expected in registration.get("input_sha256", {}).items():
        if sha(root / relative) != expected:
            raise ValueError(f"registered_input_mismatch:{relative}")
    for relative, expected in receipt.get("original_attempt_sha256", {}).items():
        if sha(attempt / relative) != expected:
            raise ValueError(f"original_attempt_binding_mismatch:{relative}")
    for relative, expected in receipt.get("recovery_sources_sha256", {}).items():
        if sha(root / relative) != expected:
            raise ValueError(f"recovery_source_binding_mismatch:{relative}")
    for relative, expected in receipt.get("recovery_files_sha256", {}).items():
        if sha(old / relative) != expected:
            raise ValueError(f"recovery_artifact_binding_mismatch:{relative}")

    # Reconstruct in memory; this path deliberately never calls verify_and_publish
    # or endpoint_evidence, both of which historically wrote to the publication.
    trajectory = recovery.reconstruct(root)
    expected_trajectory = {
        key: value
        for key, value in trajectory.items()
        if key not in {"models", "arrays", "fresh", "registration"}
    }
    published_trajectory = json.loads(
        (old / "trajectory-reconstruction.json").read_text()
    )
    if expected_trajectory != published_trajectory:
        raise ValueError("trajectory_artifact_semantic_mismatch")
    trajectory["root"] = root
    accounting = objective_accounting(trajectory)
    endpoint = endpoint_reconciliation(root, trajectory)
    preservation_after = {
        relative: {
            "sha256": sha(root / relative),
            "mtime_ns": (root / relative).stat().st_mtime_ns,
        }
        for relative in preservation_before
    }
    if preservation_before != preservation_after:
        raise ValueError("original_publication_preservation_failed")
    result = {
        "trajectory": "valid",
        "control": "exact",
        "endpoint": endpoint,
        "accounting": accounting,
        "original_bindings": {
            "registration_sha256": sha(registration_path),
            "recovery_receipt_sha256": sha(receipt_path),
            "attempt_inventory": receipt["original_attempt_sha256"],
            "recovery_inventory": receipt["recovery_files_sha256"],
        },
        "preservation": preservation_after,
        "classification": "no_experiment_decision_protocol_estimands_incomplete",
    }
    if correction.exists():
        receipt_file = correction / "correction-receipt.json"
        if not receipt_file.is_file():
            raise ValueError("correction_receipt_missing")
        correction_receipt = json.loads(receipt_file.read_text())
        if correction_receipt.get("classification") != result["classification"]:
            raise ValueError("correction_classification_mismatch")
        for relative, expected in correction_receipt.get(
            "correction_files_sha256", {}
        ).items():
            if sha(correction / relative) != expected:
                raise ValueError(f"correction_artifact_hash_mismatch:{relative}")
        for relative, expected in correction_receipt.get(
            "correction_sources_sha256", {}
        ).items():
            if sha(root / relative) != expected:
                raise ValueError(f"correction_source_hash_mismatch:{relative}")
        report = json.loads((correction / "correction-report.json").read_text())
        if report.get("original_bindings") != result["original_bindings"]:
            raise ValueError("correction_original_binding_mismatch")
        if report.get("accounting") != accounting or report.get("endpoint") != endpoint:
            raise ValueError("correction_semantic_mismatch")
        result["correction_receipt"] = "valid"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
