"""Independently verify seed514 raw measurements and reported summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    """Linear interpolation percentile, independently implemented from numpy."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def verify(directory: Path) -> dict[str, Any]:
    freeze_path = directory / "protocol-freeze.json"
    cohort_path = directory / "cohort.json"
    raw_path = directory / "raw-measurements.jsonl"
    evidence_path = directory / "evidence.json"
    freeze = json.loads(freeze_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    cohort = json.loads(cohort_path.read_text())
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    if sha256(cohort_path) != freeze["cohort_sha256"]:
        raise ValueError("seed514_verify_cohort_hash")
    if evidence["protocol_freeze_sha256"] != sha256(freeze_path):
        raise ValueError("seed514_verify_protocol_hash")
    if len(cohort) != 64 or len(raw) != 640:
        raise ValueError("seed514_verify_record_counts")
    state_ids = {row["state_sha256"] for row in cohort}
    if len(state_ids) != 64 or {row["state_sha256"] for row in raw} != state_ids:
        raise ValueError("seed514_verify_state_coverage")
    root = directory.resolve().parents[2]
    expected_seeds = {
        row["state_sha256"]: 514000 + index for index, row in enumerate(cohort)
    }
    expected_pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in raw:
        if row["seed"] != expected_seeds[row["state_sha256"]]:
            raise ValueError("seed514_verify_seed")
        if row["repetition"] not in range(5) or row["config"] not in {"baseline", "P"}:
            raise ValueError("seed514_verify_measurement_key")
        expected_pairs.setdefault((row["state_sha256"], int(row["repetition"])), {})[
            row["config"]
        ] = row
    if len(expected_pairs) != 320 or any(
        set(pair) != {"baseline", "P"} for pair in expected_pairs.values()
    ):
        raise ValueError("seed514_verify_pair_coverage")
    protocol = evidence["protocol"]
    if (
        protocol["simulations"] != 1200
        or protocol["c_puct"] != 1.25
        or protocol["workers"] != 1
    ):
        raise ValueError("seed514_verify_search_protocol")
    for relative, expected in freeze["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"seed514_verify_source:{relative}")
    registration = (
        root / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    if sha256(registration) != freeze["seed512_archive"]["registration_sha256"]:
        raise ValueError("seed514_verify_seed512_registration")
    for relative, expected in freeze["seed512_archive"]["game_batches"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"seed514_verify_seed512_batch:{relative}")

    rebuilt_configs = {}
    for config_name in ("baseline", "P"):
        measurements = [
            float(row["seconds"]) for row in raw if row["config"] == config_name
        ]
        rebuilt_configs[config_name] = {
            "searches": len(measurements),
            "p50_seconds": percentile(measurements, 50),
            "p95_seconds": percentile(measurements, 95),
            "throughput_searches_per_second": len(measurements) / sum(measurements),
        }
    ratios = []
    by_state: dict[str, dict[str, list[float]]] = {}
    for (identity, repetition), pair in sorted(expected_pairs.items()):
        baseline_seconds = float(pair["baseline"]["seconds"])
        p_seconds = float(pair["P"]["seconds"])
        ratios.append(p_seconds / baseline_seconds)
        by_state.setdefault(identity, {"baseline": [], "P": []})["baseline"].append(
            baseline_seconds
        )
        by_state[identity]["P"].append(p_seconds)
    rebuilt_state_ratios = {
        identity: statistics.median(times["P"]) / statistics.median(times["baseline"])
        for identity, times in by_state.items()
    }
    summary = evidence["summary"]
    for config_name, rebuilt in rebuilt_configs.items():
        reported = summary["configs"][config_name]
        if reported["searches"] != rebuilt["searches"]:
            raise ValueError(f"seed514_verify_search_count:{config_name}")
        for key in ("p50_seconds", "p95_seconds", "throughput_searches_per_second"):
            if abs(float(reported[key]) - rebuilt[key]) > 1e-12:
                raise ValueError(f"seed514_verify_summary:{config_name}:{key}")
    for key, expected in (
        ("paired_ratio_p50", percentile(ratios, 50)),
        ("paired_ratio_p95", percentile(ratios, 95)),
    ):
        if abs(float(summary[key]) - expected) > 1e-12:
            raise ValueError(f"seed514_verify_ratio_summary:{key}")
    reported_state_ratios = {
        row["state_sha256"]: float(row["P_over_baseline"])
        for row in summary["per_state_latency_ratios"]
    }
    if reported_state_ratios.keys() != rebuilt_state_ratios.keys() or any(
        abs(reported_state_ratios[key] - rebuilt_state_ratios[key]) > 1e-12
        for key in rebuilt_state_ratios
    ):
        raise ValueError("seed514_verify_per_state_ratios")
    path_counts = {
        name: {
            "exact_root": sum(
                bool(row["exact_root"]) for row in raw if row["config"] == name
            ),
            "puct": sum(not row["exact_root"] for row in raw if row["config"] == name),
        }
        for name in ("baseline", "P")
    }
    if path_counts != summary["search_path_counts"]:
        raise ValueError("seed514_verify_search_path_counts")
    return {
        "status": "independently_recomputed",
        "cohort_states": len(cohort),
        "timed_searches": len(raw),
        "paired_searches": len(expected_pairs),
        "rebuilt_configs": rebuilt_configs,
        "paired_ratio_p50": percentile(ratios, 50),
        "paired_ratio_p95": percentile(ratios, 95),
        "per_state_ratios": len(rebuilt_state_ratios),
        "search_path_counts": path_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
