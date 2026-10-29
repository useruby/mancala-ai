"""Append-only verifier supplement for v2 boundary and status cross-bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train, verify_seed542_fixed_update_cd_v2 as base

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
BOUNDARIES = (27, 54, 81, 108)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_c_boundary_states(root: Path) -> dict[int, dict[str, np.ndarray]]:
    """Independently replay published C updates and retain four boundary arrays."""
    protocol = json.loads(
        (root / "docs/data/seed542-selfplay-4096/training-protocol.json").read_text()
    )
    updates = base.published_schedule(protocol)
    x, p, v, policy_weights, identities = base.load_arm_data("C", root)
    positions = {row_id: index for index, row_id in enumerate(identities)}
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        model, root / "docs/data/seed542-selfplay-4096/initializer.npz"
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    states: dict[int, dict[str, np.ndarray]] = {}
    for step in updates:
        indexes = np.asarray(
            [positions[row_id] for row_id in step["C"]], dtype=np.int64
        )
        bx = torch.from_numpy(x[indexes])
        bp = torch.from_numpy(p[indexes])
        bv = torch.from_numpy(v[indexes])
        bw = torch.from_numpy(policy_weights[indexes])
        legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[indexes]))
        model.train()
        logits, predicted_value = model(bx)
        policy = train.weighted_policy_loss(
            train.compute_policy_cross_entropy(
                logits.masked_fill(legal <= 0.0, -1e9), bp
            ),
            bw,
        )
        value = train.compute_value_loss_vector(
            predicted_value, bv, value_loss="huber", huber_delta=1.0
        ).mean()
        objective = policy + 0.5 * value
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step["update"] in BOUNDARIES:
            states[step["update"]] = {
                name: array.copy()
                for name, array in train.checkpoint_from_model(model).items()
            }
    return states


def verify_production_boundary_checkpoints(
    output: Path, reconstructed_c: dict[int, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """Require each production boundary and C archive to equal replayed C arrays."""
    if set(reconstructed_c) != set(BOUNDARIES):
        raise ValueError("supplement_reconstructed_C_boundary_set_mismatch")
    checked = []
    for update in BOUNDARIES:
        control_path = output / f"C-update-{update:03d}.npz"
        production_path = (
            output / "production-reference" / f"seed504_epoch{update // 27}.npz"
        )
        for label, path in (("C", control_path), ("production", production_path)):
            with np.load(path) as archived:
                arrays = {name: archived[name] for name in archived.files}
            if not base.compare_checkpoint_arrays(reconstructed_c[update], arrays):
                raise ValueError(
                    f"supplement_{label}_boundary_checkpoint_mismatch:{update}"
                )
        checked.append(
            {
                "update": update,
                "C_checkpoint_sha256": sha(control_path),
                "production_checkpoint_sha256": sha(production_path),
                "both_equal_reconstructed_C": True,
            }
        )
    return {
        "status": "production_and_C_boundary_arrays_reconciled",
        "boundaries": checked,
    }


def verify_status_effects(
    output: Path, independently_recomputed_effects: dict[str, Any]
) -> None:
    status = json.loads((output / "run-status.json").read_text())
    base.assert_json_equal(
        independently_recomputed_effects,
        status.get("development_effects"),
        "supplement_run_status_development_effects_mismatch",
    )


def historical_drift_report(root: Path, output: Path) -> dict[str, Any]:
    """Rebuild the descriptive historical checkpoint report for mutation tests."""
    report = {}
    for epoch in range(1, 5):
        production_path = output / "production-reference" / f"seed504_epoch{epoch}.npz"
        old_path = root / f"docs/data/seed504/seed504_epoch{epoch}.npz"
        with np.load(production_path) as production, np.load(old_path) as historical:
            details = {}
            mismatch_count = 0
            for name in sorted(set(production.files) | set(historical.files)):
                if name not in production or name not in historical:
                    details[name] = {"parameter_presence_mismatch": True}
                    continue
                actual, expected = production[name], historical[name]
                differing = int(np.count_nonzero(actual != expected))
                mismatch_count += differing
                details[name] = {
                    "shape_equal": actual.shape == expected.shape,
                    "dtype_equal": actual.dtype == expected.dtype,
                    "bitwise_equal": differing == 0
                    and actual.shape == expected.shape
                    and actual.dtype == expected.dtype,
                    "differing_elements": differing,
                    "maximum_absolute_difference": float(
                        np.max(
                            np.abs(
                                actual.astype(np.float64) - expected.astype(np.float64)
                            ),
                            initial=0.0,
                        )
                    ),
                }
        report[f"E{epoch}"] = {
            "historical_checkpoint_sha256": sha(old_path),
            "current_runtime_production_checkpoint_sha256": sha(production_path),
            "different_elements": mismatch_count,
            "parameter_names_equal": set(production.files) == set(historical.files),
            "bitwise_equal": mismatch_count == 0
            and set(production.files) == set(historical.files)
            and all(row.get("bitwise_equal", False) for row in details.values()),
            "comparison_is_descriptive_only": True,
            "parameters": details,
        }
    return report


def verify_supplement(root: Path, output: Path, tablebase: Path) -> dict[str, Any]:
    """Verify the two additive cross-publication contracts without writing files."""
    registration = json.loads((output / "registration.json").read_text())
    observed = {
        *(root / relative for relative in registration["input_sha256"]),
        tablebase,
        *(path for path in output.rglob("*") if path.is_file()),
    }
    before = _snapshot(observed)
    _registration, _schedule = base.authenticate(root, output, tablebase)
    reconstructed = reconstruct_c_boundary_states(root)
    boundary_report = verify_production_boundary_checkpoints(output, reconstructed)
    base.verify_historical_drift(root, output)
    expected_metrics, expected_predictions = base.independently_recompute_development(
        root, output
    )
    published_metrics = json.loads((output / "development-metrics.json").read_text())
    published_predictions = [
        json.loads(line)
        for line in (output / "development-predictions.jsonl").read_text().splitlines()
    ]
    base.assert_json_equal(
        expected_metrics,
        published_metrics,
        "supplement_development_metrics_mismatch",
    )
    base.assert_json_equal(
        expected_predictions,
        published_predictions,
        "supplement_development_predictions_mismatch",
    )
    verify_status_effects(output, expected_metrics["effects"])
    if before != _snapshot(observed):
        raise ValueError("supplement_verifier_modified_evidence")
    return {
        "status": "supplement_verified",
        "boundary_checkpoints": boundary_report["boundaries"],
        "status_effects_match_independent_development_recomputation": True,
        "historical_drift_remains_descriptive": True,
        "read_only": True,
    }


def _snapshot(paths: set[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
        for path in sorted(paths, key=str)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.root.resolve()
    output = args.out.resolve()
    tablebase = (
        args.tablebase if args.tablebase.is_absolute() else root / args.tablebase
    )
    print(
        json.dumps(verify_supplement(root, output, tablebase), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
