"""Publish the already-reproduced seed504 C checkpoint arrays for verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed504-policy-only-ablation/control-reproduction-checkpoints"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def capture(source: Path, out: Path = OUT) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "status": "captured_exact_control_reproduction",
        "comparison": "all serialized parameter arrays compared bitwise against authenticated seed504 C checkpoints",
        "epochs": {},
    }
    for epoch in range(1, 5):
        name = f"seed504_epoch{epoch}.npz"
        historical = ROOT / "docs/data/seed504" / name
        reproduced = source / name
        actual = parameter_arrays(reproduced)
        expected = parameter_arrays(historical)
        if set(actual) != set(expected) or any(
            not np.array_equal(actual[key], expected[key]) for key in expected
        ):
            raise ValueError(f"control_parameter_array_mismatch:E{epoch}")
        destination = out / name
        shutil.copyfile(reproduced, destination)
        if sha(destination) != sha(historical):
            raise ValueError(f"control_archive_hash_mismatch:E{epoch}")
        evidence["epochs"][f"E{epoch}"] = {
            "path": str(destination.relative_to(ROOT)),
            "parameter_arrays": len(actual),
            "bitwise_equal": True,
            "historical_sha256": sha(historical),
            "reproduced_sha256": sha(destination),
        }
    result_path = out.parent / "control-reproduction-arrays.json"
    result_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=ROOT / ".tmp/seed504-control-repro"
    )
    args = parser.parse_args()
    print(json.dumps(capture(args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
