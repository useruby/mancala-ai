"""Independent, read-only verification of seed518 evidence and arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare dictionaries except for the two declared exact telemetry leaves."""
    a, b = json.loads(json.dumps(left)), json.loads(json.dumps(right))
    a.pop("root_latency_ms", None)
    b.pop("root_latency_ms", None)
    for item in (a, b):
        if isinstance(item.get("exact_root_decision"), dict):
            item["exact_root_decision"].pop("root_latency_ms", None)
    return a == b


def summaries_equal(actual: Any, expected: Any, path: str = "summary") -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            summaries_equal(actual[key], expected[key], f"{path}.{key}")
            for key in actual
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return (
            math.isfinite(float(actual))
            and abs(float(actual) - float(expected)) <= 1e-12
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            summaries_equal(a, b, f"{path}[{i}]")
            for i, (a, b) in enumerate(zip(actual, expected))
        )
    return actual == expected


def independent_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute required summaries without importing publisher code."""

    def pct(vals: list[float], p: float) -> float:
        ordered = sorted(vals)
        x = (len(ordered) - 1) * p / 100
        low = int(x)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] * (high - x) + ordered[high] * (x - low)

    names = ("reference_P", "selective_P")
    out: dict[str, Any] = {"configs": {}, "buckets": {}, "per_state": []}
    for name in names:
        vals = [float(row["seconds"]) for row in rows if row["config"] == name]
        out["configs"][name] = {
            "count": len(vals),
            "p50_seconds": pct(vals, 50),
            "p95_seconds": pct(vals, 95),
            "throughput_searches_per_second": len(vals) / sum(vals),
        }
    states = sorted({row["state_sha256"] for row in rows})
    for sid in states:
        vals = {
            name: [
                float(r["seconds"])
                for r in rows
                if r["state_sha256"] == sid and r["config"] == name
            ]
            for name in names
        }
        out["per_state"].append(
            {
                "state_sha256": sid,
                "reference_median_seconds": statistics.median(vals[names[0]]),
                "selective_median_seconds": statistics.median(vals[names[1]]),
                "selective_over_reference": statistics.median(vals[names[1]])
                / statistics.median(vals[names[0]]),
            }
        )
    for bucket in ("gt32", "17to32", "le16"):
        out["buckets"][bucket] = {}
        for path in ("puct", "exact_root", "all"):
            subset = [
                r
                for r in rows
                if r["bucket"] == bucket
                and (path == "all" or r["dispatch_path"] == path)
            ]
            pairs: dict[tuple[str, int], dict[str, float]] = {}
            for row in subset:
                pairs.setdefault((row["state_sha256"], row["repetition"]), {})[
                    row["config"]
                ] = float(row["seconds"])
            ratios = [
                pair[names[1]] / pair[names[0]]
                for pair in pairs.values()
                if set(pair) == set(names)
            ]
            stats = {}
            for name in names:
                vals = [float(r["seconds"]) for r in subset if r["config"] == name]
                stats[name] = {
                    "count": len(vals),
                    "p50_seconds": pct(vals, 50) if vals else None,
                    "p95_seconds": pct(vals, 95) if vals else None,
                    "throughput_searches_per_second": len(vals) / sum(vals)
                    if vals
                    else None,
                }
            out["buckets"][bucket][path] = {
                "count": len(subset),
                "paired_count": len(ratios),
                "configs": stats,
                "paired_ratio_p50": pct(ratios, 50) if ratios else None,
                "paired_ratio_p95": pct(ratios, 95) if ratios else None,
            }
    ratios = {
        b: out["buckets"][b]["puct"]["paired_ratio_p50"] for b in ("gt32", "17to32")
    }
    out["decision"] = {
        "parity_required": True,
        "minimum_improvement": 0.05,
        "puct_bucket_median_ratios": ratios,
        "advance": all(v is not None and v <= 0.95 for v in ratios.values()),
    }
    return out


def verify(directory: Path) -> dict[str, Any]:
    freeze_path = directory / "amended-timing-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    root = Path(__file__).resolve().parents[2]
    cohort_path = root / freeze["cohort_source"]
    cohort = json.loads(cohort_path.read_text())
    parity_path = directory / "parity-outputs.json"
    parity = json.loads(parity_path.read_text())
    amendment_path = directory / "comparison-contract-amendment.json"
    amendment = json.loads(amendment_path.read_text())
    if (
        amendment["added_exclusion"] != "root_latency_ms"
        or amendment["observed_leaf_differences"]["other_undeclared_paths"] != 0
    ):
        raise ValueError("seed518_amendment_scope")
    evidence = json.loads((directory / "evidence.json").read_text())
    if len(cohort) != 64 or sha(cohort_path) != freeze["cohort_sha256"]:
        raise ValueError("seed518_cohort_identity")
    ids = [row["state_sha256"] for row in cohort]
    if len(parity) != 64 or [row["state_sha256"] for row in parity] != ids:
        raise ValueError("seed518_parity_coverage")
    for index, row in enumerate(parity):
        if row["seed"] != 518000 + index:
            raise ValueError(f"seed518_parity_seed:{index}")
        if (
            row["evaluator_outputs"]["reference_P"]
            != row["evaluator_outputs"]["selective_P"]
        ):
            raise ValueError(f"seed518_evaluator_output_mismatch:{row['state_sha256']}")
        if not search_equal(
            row["search_outputs"]["reference_P"], row["search_outputs"]["selective_P"]
        ):
            raise ValueError(f"seed518_search_output_mismatch:{row['state_sha256']}")
        # This field preserves the original pre-amendment failure; compare the
        # archived outputs above under the bound amended contract instead.
    if evidence["protocol_freeze_sha256"] != sha(freeze_path) or evidence[
        "parity_sha256"
    ] != sha(parity_path):
        raise ValueError("seed518_evidence_hash")
    if evidence.get("post_execution_reanalysis_sha256") != sha(
        directory / "post-execution-reanalysis.json"
    ):
        raise ValueError("seed518_reanalysis_hash")
    if sha(amendment_path) != freeze["amendment_sha256"]:
        raise ValueError("seed518_amendment_hash")
    raw_path = directory / "raw-measurements.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    names = ("reference_P", "selective_P")
    expected = {(sid, rep, name) for sid in ids for rep in range(5) for name in names}
    observed = [(r["state_sha256"], r["repetition"], r["config"]) for r in raw]
    if len(raw) != 640 or len(set(observed)) != 640 or set(observed) != expected:
        raise ValueError("seed518_measurement_coverage_duplicate")
    order_by_state = {sid: index for index, sid in enumerate(ids)}
    lookup = {}
    for row in raw:
        i = order_by_state[row["state_sha256"]]
        rep = row["repetition"]
        if row["seed"] != 518000 + i:
            raise ValueError(f"seed518_measurement_seed:{row['state_sha256']}")
        expected_pos = 0 if ((i + rep) % 2 == 0) == (row["config"] == names[0]) else 1
        if row["order_position"] != expected_pos:
            raise ValueError(f"seed518_alternating_order:{row['state_sha256']}:{rep}")
        if not math.isfinite(float(row["seconds"])) or float(row["seconds"]) <= 0:
            raise ValueError(f"seed518_invalid_timing:{row['state_sha256']}:{rep}")
        if row["dispatch_path"] not in {"puct", "exact_root"}:
            raise ValueError("seed518_dispatch_path")
        lookup[(row["state_sha256"], rep, row["config"])] = row
    expected_buckets = {row["state_sha256"]: row["bucket"] for row in cohort}
    for (sid, rep, _name), row in lookup.items():
        if row["bucket"] != expected_buckets[sid]:
            raise ValueError("seed518_bucket_routing")
        pair = [lookup[(sid, rep, name)] for name in names]
        if pair[0]["dispatch_path"] != pair[1]["dispatch_path"]:
            raise ValueError("seed518_search_path_routing")
        parity_row = parity[ids.index(sid)]
        result_obj = parity_row["search_outputs"]["reference_P"]
        inferred = "exact_root" if "exact_root_decision" in result_obj else "puct"
        if row["dispatch_path"] != inferred:
            raise ValueError("seed518_dispatch_routing")
        if row["dispatch_path"] == "puct" and row["puct_simulations_executed"] != 1200:
            raise ValueError("seed518_puct_simulation_accounting")
    rebuilt = independent_summary(raw)
    if not summaries_equal(rebuilt, evidence["summary"]):
        raise ValueError("seed518_summary_arithmetic")
    if evidence["timed_searches"] != len(raw):
        raise ValueError("seed518_timed_search_count")
    return {
        "status": "verified",
        "cohort_states": 64,
        "parity_states": 64,
        "timed_searches": 640,
        "decision": rebuilt["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.directory), indent=2, sort_keys=True) + "\n"
    if args.report:
        with args.report.open("x") as stream:
            stream.write(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
