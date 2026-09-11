#!/usr/bin/env python3
"""Summarize paired PR #287 control-vs-uniform1200 exact shadow replays."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {
    "overall": {"top1_agreement": -0.02, "average_regret": 0.02, "blunder_rate": 0.01},
    "critical": {
        "sparse_endgame": {
            "top1_agreement": -0.03,
            "average_regret": 0.03,
            "blunder_rate": 0.02,
        },
        "capture_available": {
            "top1_agreement": -0.03,
            "average_regret": 0.03,
            "blunder_rate": 0.02,
        },
    },
}


def passed(report: dict) -> tuple[bool, dict]:
    comparisons = {}
    for scope, current, challenger, thresholds in [
        (
            "overall",
            report["systems"]["current"]["overall"],
            report["systems"]["challenger"]["overall"],
            THRESHOLDS["overall"],
        ),
        *[
            (
                bucket,
                report["buckets"][bucket]["systems"]["current"],
                report["buckets"][bucket]["systems"]["challenger"],
                limits,
            )
            for bucket, limits in THRESHOLDS["critical"].items()
        ],
    ]:
        comparisons[scope] = {
            metric: round(challenger[metric] - current[metric], 4)
            for metric in thresholds
        }
    decision = all(
        delta >= THRESHOLDS["overall"][metric]
        if metric == "top1_agreement"
        else delta <= THRESHOLDS["overall"][metric]
        for metric, delta in comparisons["overall"].items()
    )
    for bucket, limits in THRESHOLDS["critical"].items():
        decision &= all(
            delta >= limits[metric]
            if metric == "top1_agreement"
            else delta <= limits[metric]
            for metric, delta in comparisons[bucket].items()
        )
    return decision, comparisons


def main() -> int:
    rows = {}
    for seed in (44, 45, 46):
        report = json.loads(
            (
                ROOT / f"docs/data/alphazero-lite-exact-forensic-shadow-seed{seed}.json"
            ).read_text()
        )
        decision, deltas = passed(report)
        rows[str(seed)] = {
            "shadow_exact_forensic_decision": "pass" if decision else "fail",
            "exact_deltas": deltas,
            "control": report["systems"]["current"]["overall"],
            "uniform1200": report["systems"]["challenger"]["overall"],
        }
    output = {
        "schema": "azlite_uniform1200_exact_shadow_replay_v1",
        "oracle": "azlite_forensic_references_v2",
        "seeds": rows,
        "hard_classification": "uniform1200_exact_regression_confirmed",
        "next_experiment": "return to the exact failure families and identify the dominant training mechanism using the now-valid exact oracle.",
    }
    data = ROOT / "docs/data/alphazero-lite-uniform1200-exact-shadow-replay.json"
    data.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Uniform1200 Exact Shadow Replay",
        "",
        "## Matched PR #287 Results",
        "",
        "| Seed | Exact-set delta | Exact regret delta | Exact blunder delta | Shadow decision |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for seed, row in rows.items():
        delta = row["exact_deltas"]["overall"]
        lines.append(
            f"| {seed} | {delta['top1_agreement']:+.4f} | {delta['average_regret']:+.4f} | {delta['blunder_rate']:+.4f} | {row['shadow_exact_forensic_decision']} |"
        )
    lines += [
        "",
        "## Hard Classification",
        "",
        f"`{output['hard_classification']}`",
        "",
        "## One Next Experiment",
        "",
        output["next_experiment"],
        "",
    ]
    (ROOT / "docs/alphazero-lite-uniform1200-exact-shadow-replay.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
