#!/usr/bin/env python3
"""Build the deterministic exact-aware forensic v2 artifact from frozen labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_exact_references import build_artifact, validate_v2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-results", type=Path)
    args = parser.parse_args()
    native = (
        []
        if args.native_results is None
        else json.loads(args.native_results.read_text())["rows"]
    )
    artifact = build_artifact(args.suite, native)
    errors = validate_v2(artifact, args.suite)
    if errors:
        raise SystemExit("; ".join(errors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
