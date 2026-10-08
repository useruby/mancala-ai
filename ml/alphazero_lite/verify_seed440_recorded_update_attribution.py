"""Portable verifier for the seed440 recorded-update attribution publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed440_recorded_update_attribution import classify, run


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / "docs/data/seed440-recorded-update-attribution"
    protocol = json.loads((out / "protocol.json").read_text())
    bindings = json.loads((out / "evidence-bindings.json").read_text())
    for item in bindings["seed435"].values():
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError(f"historical_evidence_changed:{item['path']}")
    gradient = bindings["seed439_U_gradient_archive"]
    if sha(root / gradient["path"]) != gradient["sha256"]:
        raise ValueError("seed439_gradient_archive_changed")
    membership = bindings["authoritative_membership"]
    if sha(root / membership["path"]) != membership["sha256"]:
        raise ValueError("membership_changed")
    for relative, expected in protocol["source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"frozen_source_changed:{relative}")
    published = json.loads((out / "step-ledger.json").read_text())
    for arm in ("A", "B"):
        steps = published["arms"][arm]["step_ledger"]
        if len(steps) != 16 or [row["step"] for row in steps] != list(range(1, 17)):
            raise ValueError(f"missing_or_reordered_steps:{arm}")
        totals = published["arms"][arm]["totals"]
        for weighting in ("exposure_weighted", "equal_input"):
            row = totals[weighting]
            for field in (
                "telescoping_error",
                "identity_error",
                "group_reconciliation_error",
            ):
                if abs(row[field]) > protocol["arithmetic"]["tolerance"]["absolute"]:
                    raise ValueError(f"reconciliation_failed:{arm}:{weighting}:{field}")
    recomputed = run(root)
    if recomputed != published:
        raise ValueError("published_attribution_does_not_recompute")
    if (
        classify(
            {
                w: {
                    "D": published["arms"]["A"]["totals"][w]["D"],
                    "S": published["arms"]["A"]["totals"][w]["s"],
                    "R": published["arms"]["A"]["totals"][w]["r"],
                }
                for w in ("exposure_weighted", "equal_input")
            }
        )
        != published["classification"]
    ):
        raise ValueError("classification_mismatch")
    for arm in ("A", "B"):
        hashes = np.asarray(published["arms"][arm]["path_parameter_sha256"])
        with np.load(
            out / "reconstructed-path-hashes.npz", allow_pickle=False
        ) as archive:
            if not np.array_equal(hashes, archive[f"{arm}_sha256"]):
                raise ValueError(f"reconstructed_path_hash_mismatch:{arm}")
    return {
        "status": "valid",
        "classification": published["classification"],
        "reconstructed_paths": "archive-defined",
        "updates_per_arm": 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
