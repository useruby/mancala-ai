#!/usr/bin/env python3
"""Generate and record the canonical KVTB1 tier-21 compatibility artifact."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.run_kalah_v1_native_tablebase_preflight import generate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    disk = os.statvfs(args.tablebase.parent)
    preflight = {
        "available_disk_bytes": disk.f_bavail * disk.f_frsize,
        "file_descriptor_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "available_memory_bytes": int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024,
    }
    result = generate(
        args.generator,
        21,
        args.tablebase,
        memory_limit_kib=8 * 1024 * 1024,
        timeout_seconds=30 * 60,
    )
    header = args.tablebase.read_bytes()[:432]
    if (
        len(header) != 432
        or header[:5] != b"KVTB1"
        or header[18] != 21
        or header[23] != 21
    ):
        raise ValueError("generated artifact is not a canonical tier-21 KVTB1 file")
    report = {
        "schema": "kvtb21_generation_v1",
        "command": f"{args.generator} generate 21 {args.tablebase}",
        "resource_preflight": preflight,
        "generation": result,
        "header": {
            "magic": header[:5].decode(),
            "tier": header[18],
            "declared_max_tier": header[23],
            "bytes": len(header),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
