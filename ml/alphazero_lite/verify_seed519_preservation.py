"""Run seed519 in-place and prove historical bytes and mtimes stayed fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed519


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory() -> dict[str, dict[str, Any]]:
    root = verify_seed519.ROOT
    freeze = json.loads(
        (verify_seed519.PUBLICATION / "amended-timing-freeze.json").read_text()
    )
    registration = json.loads(verify_seed519.REGISTRATION.read_text())
    paths = {path for path in (verify_seed519.PUBLICATION).rglob("*") if path.is_file()}
    paths.add(verify_seed519.REGISTRATION)
    for component in ("initializer", "E4"):
        artifact = root / registration["components"][component]["artifact"]
        paths.update(
            artifact / name for name in registration["components"][component]["files"]
        )
    for runtime in freeze["runtime_identities"].values():
        artifact = Path(runtime["path"])
        if artifact.is_absolute():
            parts = artifact.parts
            marker = parts.index("model-artifact")
            artifact = root.joinpath(*parts[marker:])
        else:
            artifact = root / artifact
        paths.add(artifact)
    return {
        str(path.relative_to(root) if path.is_relative_to(root) else path): {
            "sha256": sha(path),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths)
    }


def verify(output: Path | None = None) -> dict[str, Any]:
    before = inventory()
    verification = verify_seed519.verify()
    after = inventory()
    if before != after:
        raise ValueError("seed519_historical_file_preservation")
    report = {
        "schema": "seed519-physical-copy-preservation-v1",
        "status": "verified",
        "verification": verification,
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "dependency_paths": {
            "verifier": str(Path(verify_seed519.__file__).resolve()),
            "seed515_runtime_helper": str(
                Path(verify_seed519.verify_seed515.__file__).resolve()
            ),
            "seed518_semantic_verifier": str(
                Path(verify_seed519.seed518.__file__).resolve()
            ),
        },
        "historical_inventory_before": before,
        "historical_inventory_after": after,
        "unchanged_sha256_and_mtime_ns": True,
    }
    if output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
