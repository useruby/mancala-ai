"""Portable, read-only verifier for seed432 row evidence and published summary."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed432_policy_target_census import ROOT, summarize


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    rows_path = directory / "row-accounting.jsonl.gz"
    if sha(rows_path) != manifest["row_accounting_sha256"]:
        raise ValueError("accounting_hash_mismatch")
    groups_path = directory / "group-accounting.jsonl.gz"
    if sha(groups_path) != manifest["group_accounting_sha256"]:
        raise ValueError("group_accounting_hash_mismatch")
    for relative, digest in manifest["execution_sources"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"execution_source_hash_mismatch:{relative}")
    rows = []
    with gzip.open(rows_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 87625 or [r["compact_row"] for r in rows] != list(range(len(rows))):
        raise ValueError("compact_row_accounting_mismatch")
    for row in rows:
        raw = bytes.fromhex(row["input_hex"])
        if hashlib.sha256(raw).hexdigest() != row["input_sha256"]:
            raise ValueError("input_identity_mismatch")
        if abs(sum(row["target"]) - row["target_sum"]) > 1e-7:
            raise ValueError("target_sum_accounting_mismatch")
        if (
            row["exposure_weight"]
            != row["replay_multiplicity"] * row["policy_coefficient"]
        ):
            raise ValueError("multiplicity_accounting_mismatch")
    with gzip.open(groups_path, "rt", encoding="utf-8") as stream:
        groups = [json.loads(line) for line in stream]
    if sum(len(group["compact_rows"]) for group in groups) != len(rows):
        raise ValueError("group_row_accounting_mismatch")
    grouped_rows = [index for group in groups for index in group["compact_rows"]]
    if sorted(grouped_rows) != list(range(len(rows))):
        raise ValueError("group_membership_partition_mismatch")
    if any(
        len(group["compact_rows"]) != len(group["source_lines"])
        or len(group["compact_rows"]) != len(group["replay_multiplicities"])
        for group in groups
    ):
        raise ValueError("group_member_accounting_mismatch")
    populations: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        bucket = (
            "le16"
            if row["active_stones"] <= 16
            else "17-32"
            if row["active_stones"] <= 32
            else "gt32"
        )
        populations.setdefault(f"{row['partition']}/{bucket}", []).append(row)
    expected = {
        name: {kind: summarize(group, kind) for kind in ("exposure", "equal_input")}
        | {
            f"seed429_treatment_{kind}": summarize(group, kind, treatment=True)
            for kind in ("exposure", "equal_input")
        }
        for name, group in populations.items()
    }
    result = json.loads((directory / "results.json").read_text())
    if result["populations"] != expected:
        raise ValueError("published_results_mismatch")
    if (
        result["expanded_positions"],
        result["training_positions"],
        result["validation_positions"],
    ) != (149448, 134502, 14946):
        raise ValueError("partition_accounting_mismatch")
    if (
        sum(r["replay_multiplicity"] for r in rows if r["partition"] == "train")
        != 134502
    ):
        raise ValueError("training_copy_accounting_mismatch")
    if (
        sum(r["replay_multiplicity"] for r in rows if r["partition"] == "validation")
        != 14946
    ):
        raise ValueError("validation_copy_accounting_mismatch")
    primary = expected["train/gt32"]
    classification = (
        "material_policy_target_disagreement"
        if all(
            primary[w]["mean_js_nats"] >= 0.05
            and primary[w]["weighted_pairwise_top_action_disagreement"] >= 0.10
            for w in primary
        )
        else "target_disagreement_not_material_under_registered_thresholds"
    )
    if result["classification"] != classification:
        raise ValueError("classification_mismatch")
    return {
        "status": "valid",
        "compact_rows": len(rows),
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=ROOT / "docs/data/seed432-policy-target-compatibility",
    )
    print(json.dumps(verify(parser.parse_args().directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
