"""Supplemental, read-only checks for the seed426 publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from ml.alphazero_lite.verify_seed426_overlap_audit import verify as verify_seed426  # noqa: E402


def verify(root: Path) -> dict[str, object]:
    """Run original verification, then independently reconcile three identities."""
    verify_seed426(root)
    data = root / "docs/data/seed426-canonical-overlap"
    manifest = json.loads((data / "manifest.json").read_text())
    results = json.loads((data / "results.json").read_text())
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    split_spec = registration["training"]["source_row_split"]
    split_bytes = (root / split_spec["path"]).read_bytes()
    split = json.loads(gzip.decompress(split_bytes))
    actual = {
        "train": len(split["train_positions"]),
        "validation": len(split["validation_positions"]),
    }
    registered = {
        "train": sum(
            item["train_weighted_positions"]
            for item in results["source_accounting"].values()
        ),
        "validation": sum(
            item["validation_weighted_positions"]
            for item in results["source_accounting"].values()
        ),
    }
    if results.get("split_positions") != actual or actual != registered:
        raise ValueError("supplemental_split_positions_mismatch")

    expected_inputs = {item["name"]: item["sha256"] for item in registration["replays"]}
    for name, expected_hash in expected_inputs.items():
        raw = gzip.decompress((data / "sources" / f"{name}.jsonl.gz").read_bytes())
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"supplemental_replay_hash_mismatch:{name}")
    if (
        results.get("input_hashes") != expected_inputs
        or {item["name"]: item["sha256"] for item in manifest["sources"]}
        != expected_inputs
    ):
        raise ValueError("supplemental_input_hashes_mismatch")

    registered_loader = registration["source_hashes"]["ml/alphazero_lite/train.py"]
    manifest_loader = manifest["registrations"]["frozen_train"]
    snapshot_hash = hashlib.sha256(
        (data / "execution-source-snapshots/train.py").read_bytes()
    ).hexdigest()
    if not (
        results.get("loader_parity", {}).get("frozen_train_sha256")
        == registered_loader
        == manifest_loader
        == snapshot_hash
    ):
        raise ValueError("supplemental_frozen_train_identity_mismatch")
    return {
        "status": "verified",
        "split_positions": actual,
        "input_hashes": expected_inputs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True))
