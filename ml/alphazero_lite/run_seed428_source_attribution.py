"""Analyze source contributions using only published seed427 predictions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed428_source_attribution import decompose


def analyze(root: Path) -> dict[str, Any]:
    data = root / "docs/data/seed426-canonical-overlap"
    predictions: dict[str, list[dict[str, Any]]] = {"initializer": [], "e4": []}
    with gzip.open(
        data / "seed427-prediction-evidence.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            item = json.loads(line)
            predictions[item.pop("model")].append(item)
    if len(predictions["initializer"]) != len(predictions["e4"]):
        raise ValueError("paired_prediction_count_mismatch")
    paired = []
    for before, after in zip(
        predictions["initializer"], predictions["e4"], strict=True
    ):
        if before["membership"] != after["membership"]:
            raise ValueError("paired_membership_mismatch")
        paired.append(
            {
                "membership": before["membership"],
                "losses": before["losses"],
                "initializer_policy_loss": before["losses"]["policy_loss"],
                "e4_policy_loss": after["losses"]["policy_loss"],
            }
        )
    cohorts = decompose(paired)
    primary = cohorts["unseen/>32"]["source_contributions"]
    tolerance = 1e-12
    reconciled = {
        "exposure_weighted": sum(
            row["exposure_contribution"] for row in primary.values()
        ),
        "equal_canonical_identity": sum(
            row["equal_identity_contribution"] for row in primary.values()
        ),
    }
    expected = {
        "exposure_weighted": 0.05520672117819547,
        "equal_canonical_identity": 0.03754168053554596,
    }
    if any(abs(reconciled[key] - expected[key]) > tolerance for key in expected):
        raise ValueError("seed427_global_delta_reconciliation_failed")
    return {
        "schema": "seed428-source-attribution-v1",
        "analysis_plan": "docs/data/seed426-canonical-overlap/seed428-analysis-plan.md",
        "reconciliation_tolerance": tolerance,
        "primary_reconciliation": {
            "reconciled": reconciled,
            "expected_seed427": expected,
        },
        "cohorts": cohorts,
    }


def main(root: Path) -> None:
    result = analyze(root)
    output = (
        root / "docs/data/seed426-canonical-overlap/seed428-source-contributions.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    main(parser.parse_args().root)
