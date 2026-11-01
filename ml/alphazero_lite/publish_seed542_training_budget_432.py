"""Publish fixed-checkpoint development evidence for seed542 update budget 432."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import publish_seed542_fixed_update_cd_v2 as base


def publish(output: Path) -> dict[str, Any]:
    names, x, policy, value = base.retained_development()
    endpoints = {
        "initializer": base.CORPUS / "initializer.npz",
        "update-108": output / "D-update-108.npz",
        "update-216": output / "D-update-216.npz",
        "update-432": output / "D-update-432.npz",
    }
    metrics: dict[str, Any] = {}
    predictions = []
    for endpoint, checkpoint in endpoints.items():
        result, rows = base.evaluate(checkpoint, names, x, policy, value)
        metrics[endpoint] = {
            "checkpoint_sha256": base.sha(checkpoint),
            "cohort_row_count": len(names),
            "cohort_identity_count": len({state.tobytes() for state in x}),
            "metrics": result,
        }
        for row in rows:
            row["endpoint"] = endpoint
            predictions.append(row)

    report = {
        "status": "descriptive development evaluation; update 432 is sole candidate",
        "sole_candidate_checkpoint": 432,
        "development_membership_sha256": base.sha(
            base.CORPUS / "development-membership.json"
        ),
        "checkpoint_identity": {
            endpoint: {
                "sha256": record["checkpoint_sha256"],
                "rows": record["cohort_row_count"],
                "input_identities": record["cohort_identity_count"],
            }
            for endpoint, record in metrics.items()
        },
        "metrics": {
            endpoint: record["metrics"] for endpoint, record in metrics.items()
        },
        "interpretation": "Intermediate checkpoints are diagnostic. Update 432 is the sole candidate. Development loss alone does not authorize advancement; strength requires a separate arena screen.",
        "weightings": {
            "exposure_weighted": "mean over included development exposures",
            "equal_production_float32_input_weighted": "mean loss within each unique production float32 input, then mean equally over unique inputs",
        },
    }
    with (output / "development-predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "development-metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
