#!/usr/bin/env python3
"""Search ablation runner skeleton."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, build_eval_search_options, evaluate_artifact_position
from ml.alphazero_lite.forensic_suite import load_suite, summarize_bucket
from ml.alphazero_lite.run_forensic_suite import build_row, run_reference


DEFAULT_ARTIFACT_PATH = "model-artifact/current"
DEFAULT_SUITE = "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
DEFAULT_BUDGETS = "128,384,1200"
DEFAULT_SCHEMA = "search_ablation_report_v1"
DEFAULT_MODES = ["classic_only", "policy_only", "value_only", "full"]


def build_stub_rows_by_budget_and_mode(budgets: list[int], modes: list[str]) -> dict[int, dict[str, list[dict]]]:
    return {
        budget: {
            mode: [
                {
                    "bucket": "opening_plies_1_8",
                    "agrees_top1": mode in ("policy_only", "full"),
                    "regret": 0.0 if mode == "full" else 0.1,
                    "value_error": 0.0 if mode in ("value_only", "full") else 0.1,
                }
            ]
            for mode in modes
        }
        for budget in budgets
    }


def write_stub_report(args: argparse.Namespace) -> None:
    budgets = parse_budgets(args.budgets)
    modes = list(DEFAULT_MODES)
    rows_by_budget_and_mode = build_stub_rows_by_budget_and_mode(budgets, modes)
    matrix = build_position_matrix(rows_by_budget_and_mode)

    report = {
        "schema": args.schema,
        "artifact_path": args.artifact_path,
        "suite_path": args.suite,
        "budgets": budgets,
        "modes": modes,
        "reference": {
            "kind": "stub",
            "seed": args.seed,
        },
        "overall": matrix["overall"],
        "buckets": matrix["buckets"],
        "attribution_summary": build_attribution_summary(
            overall=matrix["overall"],
            buckets=matrix["buckets"],
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_real_rows_by_budget_and_mode(args: argparse.Namespace) -> dict[int, dict[str, list[dict]]]:
    suite = load_suite(args.suite)
    search_options = build_eval_search_options()
    evaluator = ArtifactEvaluator(Path(args.artifact_path))
    rows_by_budget_and_mode: dict[int, dict[str, list[dict]]] = {}

    for budget in parse_budgets(args.budgets):
        references = [
            run_reference(position.state, budget, budget, args.seed + index, index)
            for index, position in enumerate(suite)
        ]
        rows_by_mode: dict[str, list[dict]] = {}
        for mode in DEFAULT_MODES:
            rows: list[dict] = []
            for index, position in enumerate(suite):
                system = evaluate_artifact_position(
                    artifact_path=args.artifact_path,
                    evaluator=evaluator,
                    state=position.state,
                    simulations=budget,
                    seed=args.seed + 10_000 + index,
                    c_puct=1.25,
                    search_options=search_options,
                    ablation_mode=mode,
                )
                rows.append(
                    build_row(
                        position_id=position.id,
                        bucket=position.bucket,
                        reference=references[index],
                        system=system,
                    )
                )
            rows_by_mode[mode] = rows
        rows_by_budget_and_mode[budget] = rows_by_mode

    return rows_by_budget_and_mode


def write_real_report(args: argparse.Namespace) -> None:
    budgets = parse_budgets(args.budgets)
    rows_by_budget_and_mode = build_real_rows_by_budget_and_mode(args)
    matrix = build_position_matrix(rows_by_budget_and_mode)
    report = {
        "schema": args.schema,
        "artifact_path": args.artifact_path,
        "suite_path": args.suite,
        "budgets": budgets,
        "modes": list(DEFAULT_MODES),
        "reference": {
            "kind": "classic_mcts_policy_reference",
            "mode_note": "classic_only uses classic MCTS for both move selection and reported value",
        },
        "overall": matrix["overall"],
        "buckets": matrix["buckets"],
        "attribution_summary": build_attribution_summary(
            overall=matrix["overall"],
            buckets=matrix["buckets"],
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_budgets(raw: str) -> list[int]:
    budgets: list[int] = []

    for piece in raw.split(","):
        budget = int(piece.strip())
        if budget <= 0:
            raise ValueError("budgets must be positive integers")
        budgets.append(budget)

    if len(set(budgets)) != len(budgets):
        raise ValueError("duplicate budgets")

    return budgets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(schema=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def build_position_matrix(rows_by_budget_and_mode: dict[int, dict[str, list[dict]]]) -> dict[str, dict]:
    budgets = sorted(rows_by_budget_and_mode)
    modes = sorted(
        {
            mode
            for rows_by_mode in rows_by_budget_and_mode.values()
            for mode in rows_by_mode
        }
    )
    bucket_names = sorted(
        {
            str(row["bucket"])
            for rows_by_mode in rows_by_budget_and_mode.values()
            for rows in rows_by_mode.values()
            for row in rows
        }
    )

    return {
        "overall": {
            budget: {
                mode: summarize_bucket(rows_by_budget_and_mode[budget].get(mode, []))
                for mode in modes
            }
            for budget in budgets
        },
        "buckets": {
            bucket_name: {
                budget: {
                    mode: summarize_bucket(
                        [row for row in rows_by_budget_and_mode[budget].get(mode, []) if row["bucket"] == bucket_name]
                    )
                    for mode in modes
                }
                for budget in budgets
            }
            for bucket_name in bucket_names
        },
    }


def build_attribution_summary(*, overall: dict[int, dict[str, dict]], buckets: dict[str, dict[int, dict[str, dict]]]) -> dict[str, dict]:
    def extract_score(summary: dict) -> float:
        if "score" in summary:
            return float(summary["score"])
        return float(summary["top1_agreement"])

    def summarize_scores(scores_by_mode: dict[str, dict]) -> dict[str, float | str]:
        classic_score = extract_score(scores_by_mode["classic_only"])
        policy_score = extract_score(scores_by_mode["policy_only"])
        value_score = extract_score(scores_by_mode["value_only"])
        full_score = extract_score(scores_by_mode["full"])

        policy_delta = round(policy_score - classic_score, 4)
        value_delta = round(value_score - classic_score, 4)
        larger_contributor = "policy" if policy_delta >= value_delta else "value"
        if policy_delta <= 0 and value_delta <= 0:
            larger_contributor = "neither"

        return {
            "full_minus_classic_only": round(full_score - classic_score, 4),
            "full_minus_policy_only": round(full_score - policy_score, 4),
            "full_minus_value_only": round(full_score - value_score, 4),
            "policy_only_minus_classic_only": policy_delta,
            "value_only_minus_classic_only": value_delta,
            "larger_contributor": larger_contributor,
        }

    return {
        "overall": {
            budget: summarize_scores(scores_by_mode)
            for budget, scores_by_mode in sorted(overall.items())
        },
        "buckets": {
            bucket_name: {
                budget: summarize_scores(scores_by_mode)
                for budget, scores_by_mode in sorted(scores_by_budget.items())
            }
            for bucket_name, scores_by_budget in sorted(buckets.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if os.environ.get("AZLITE_SEARCH_ABLATION_STUB") == "1":
        write_stub_report(args)
        return 0

    write_real_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
