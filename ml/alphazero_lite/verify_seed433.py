"""Portable verifier for the retrospective seed432 census correction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed433_census_correction import DATA, ROOT, calculate, sha
from ml.alphazero_lite.seed429_policy_normalization import normalize_policy_coefficients


def verify(directory: Path = DATA) -> dict[str, Any]:
    directory = directory.resolve()
    receipt = json.loads((directory / "correction-receipt.json").read_text())
    original = ROOT / "docs/data/seed432-policy-target-compatibility"
    if sha(original / "manifest.json") != receipt["seed432_manifest_sha256"]:
        raise ValueError("seed432_manifest_binding_mismatch")
    if sha(original / "results.json") != receipt["seed432_results_sha256"]:
        raise ValueError("seed432_results_binding_mismatch")
    if sha(directory / "corrected-results.json") != receipt["corrected_results_sha256"]:
        raise ValueError("corrected_results_binding_mismatch")
    for filename, key in (
        ("seed433_census_correction.py", "calculation_source_sha256"),
        ("verify_seed433.py", "verifier_source_sha256"),
    ):
        if sha(ROOT / "ml/alphazero_lite" / filename) != receipt[key]:
            raise ValueError(f"correction_source_binding_mismatch:{filename}")
    manifest = json.loads((original / "manifest.json").read_text())
    if sha(original / "row-accounting.jsonl.gz") != manifest["row_accounting_sha256"]:
        raise ValueError("seed432_row_evidence_hash_mismatch")
    if (
        sha(original / "group-accounting.jsonl.gz")
        != manifest["group_accounting_sha256"]
    ):
        raise ValueError("seed432_group_evidence_hash_mismatch")
    with gzip.open(
        original / "row-accounting.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        rows = [json.loads(line) for line in stream]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        if (
            hashlib.sha256(bytes.fromhex(row["input_hex"])).hexdigest()
            != row["input_sha256"]
        ):
            raise ValueError(f"input_identity_mismatch:{index}")
        if (
            row["exposure_weight"]
            != row["replay_multiplicity"] * row["policy_coefficient"]
        ):
            raise ValueError(f"exposure_accounting_mismatch:{index}")
        groups.setdefault((row["partition"], row["input_sha256"]), []).append(row)
    with gzip.open(
        original / "group-accounting.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        published = [json.loads(line) for line in stream]
    reconstructed = [
        {
            "partition": partition,
            "input_sha256": identity,
            "canonical_states": sorted({row["canonical_state"] for row in members}),
            "compact_rows": [row["compact_row"] for row in members],
            "source_lines": [[row["source"], row["line"]] for row in members],
            "replay_multiplicities": [row["replay_multiplicity"] for row in members],
        }
        for (partition, identity), members in sorted(groups.items())
    ]
    if published != reconstructed:
        raise ValueError("seed432_group_reconstruction_mismatch")
    treatment, _ = normalize_policy_coefficients(
        np.asarray([row["policy_coefficient"] for row in rows]),
        [row["canonical_state"] for row in rows],
        np.asarray([row["active_stones"] for row in rows]),
        [row["source"] for row in rows],
        np.asarray([row["partition"] == "train" for row in rows]),
    )
    if any(
        float(row["treatment_coefficient"]) != float(value)
        for row, value in zip(rows, treatment, strict=True)
    ):
        raise ValueError("treatment_coefficient_reconstruction_mismatch")
    result = json.loads((directory / "corrected-results.json").read_text())
    rebuilt = calculate(original, None)
    if result != rebuilt:
        raise ValueError("corrected_results_mismatch")
    return {
        "status": "valid",
        "classification": result["classification"],
        "primary": result["populations"]["train/gt32"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DATA)
    print(json.dumps(verify(parser.parse_args().directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
