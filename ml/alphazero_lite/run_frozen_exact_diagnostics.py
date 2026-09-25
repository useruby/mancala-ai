#!/usr/bin/env python3
"""Evaluate named artifacts on a fixed exact-reference corpus."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_seed48_target_quality_audit import (
    paired_bootstrap,
    quality,
    search,
)


def evaluate(
    artifact: Path, rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact)
    values: dict[str, list[dict[str, Any]]] = {"raw": [], "mcts_384": [], "value": []}
    for index, row in enumerate(rows):
        exact = {
            **row["exact"],
            "exact_regret_by_move": {
                int(move): value
                for move, value in row["exact"]["exact_regret_by_move"].items()
            },
            "exact_optimal_moves": [
                int(move) for move in row["exact"]["exact_optimal_moves"]
            ],
        }
        game = KalahGame.from_state(row["canonical_state"])
        policy, value = evaluator.evaluate(game)
        values["raw"].append(quality(policy.tolist(), exact, row["legal_moves"]))
        searched = search(evaluator, row["canonical_state"], 384, seed + index)
        metric = quality(searched["policy"], exact, row["legal_moves"])
        metric["selected_regret"] = metric["top_move_regret"]
        values["mcts_384"].append(metric)
        target = (exact["best_exact_score"] > 0) - (exact["best_exact_score"] < 0)
        values["value"].append({"prediction": float(value), "target": target})
    return {
        "rows": values,
        "summary": {
            "raw": {
                key: statistics.fmean(float(row[key]) for row in values["raw"])
                for key in ("top_move_optimal", "optimal_mass", "expected_regret")
            },
            "mcts_384": {
                key: statistics.fmean(float(row[key]) for row in values["mcts_384"])
                for key in (
                    "top_move_optimal",
                    "optimal_mass",
                    "expected_regret",
                    "selected_regret",
                )
            },
            "value": {
                "mae": statistics.fmean(
                    abs(row["prediction"] - row["target"]) for row in values["value"]
                ),
                "sign_accuracy": statistics.fmean(
                    (row["prediction"] > 0) == (row["target"] > 0)
                    for row in values["value"]
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--artifact", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--min-active-stones", type=int, default=0)
    parser.add_argument("--seed", type=int, default=370)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))["states"]
    rows = [
        row for row in corpus if int(row["active_stones"]) >= args.min_active_stones
    ]
    artifacts = dict(item.split("=", 1) for item in args.artifact)
    results = {
        name: evaluate(Path(path), rows, seed=args.seed)
        for name, path in artifacts.items()
    }
    baseline, treatment = results[args.baseline], results[args.treatment]
    pairs = {
        "raw_optimal_mass": ("raw", "optimal_mass"),
        "raw_expected_regret": ("raw", "expected_regret"),
        "puct384_optimal_mass": ("mcts_384", "optimal_mass"),
        "puct384_expected_regret": ("mcts_384", "expected_regret"),
        "value_mae": ("value", None),
    }
    bootstrap = {}
    for name, (method, metric) in pairs.items():
        if metric is None:
            left = [
                abs(row["prediction"] - row["target"])
                for row in treatment["rows"][method]
            ]
            right = [
                abs(row["prediction"] - row["target"])
                for row in baseline["rows"][method]
            ]
        else:
            left = [row[metric] for row in treatment["rows"][method]]
            right = [row[metric] for row in baseline["rows"][method]]
        bootstrap[name] = paired_bootstrap(left, right, seed=args.seed)
    report = {
        "schema": "azlite_frozen_exact_diagnostics_v1",
        "corpus": str(args.corpus),
        "positions": len(rows),
        "minimum_active_stones": args.min_active_stones,
        "puct_simulations": 384,
        "seed": args.seed,
        "results": {name: value["summary"] for name, value in results.items()},
        "paired_bootstrap_treatment_minus_baseline": bootstrap,
    }
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
