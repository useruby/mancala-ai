"""Measure model resident memory in a fresh process without running searches."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("seed514_memory_rss_unavailable")


def snapshot() -> dict[str, Any]:
    baseline = arena.ArtifactEvaluator(ARTIFACTS / "initializer")
    baseline_rss = rss_bytes()
    policy = arena.ArtifactEvaluator(ARTIFACTS / "E4")
    composed = arena.ComposedArtifactEvaluator(
        baseline,
        policy,
        policy_source="candidate",
        value_source="current",
    )
    p_rss = rss_bytes()
    return {
        "schema": "seed514-model-memory-snapshot-v1",
        "interpretation": "fresh-process model-resident-memory snapshot; no search, timing, allocation tracing, or profiler",
        "configurations": {
            "baseline": {"resident_bytes": baseline_rss},
            "P": {
                "resident_bytes": p_rss,
                "incremental_over_baseline_bytes": p_rss - baseline_rss,
                "policy_source": "seed504 E4, own trunk",
                "value_source": "frozen initializer, own trunk",
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "command": f"{sys.executable} -m ml.alphazero_lite.run_seed514_memory_snapshot",
        },
        "artifact_sha256": {
            name: sha256(ARTIFACTS / name / "model.npz")
            for name in ("initializer", "E4")
        },
        "implementation_sha256": sha256(Path(__file__)),
        "retained_evaluators": [
            type(baseline).__name__,
            type(policy).__name__,
            type(composed).__name__,
        ],
        "limitations": "Does not measure transient search allocations or exact-root native tablebase working memory; production-hardware memory qualification remains required.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "docs/data/seed514-operational-benchmark-v2/memory-snapshot.json",
    )
    args = parser.parse_args()
    result = snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
