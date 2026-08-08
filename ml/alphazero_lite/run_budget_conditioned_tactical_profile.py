#!/usr/bin/env python3
"""Evaluate the one budget-conditioned tactical runtime profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.evaluation_seed_contract import SEED_CONTRACT_VERSION
from ml.alphazero_lite.run_canonical_runtime_profile_revalidation import (
    BUDGETS,
    PROFILE_C,
    PROFILE_D,
    PROFILE_E,
    aggregate_seed_opening_deltas,
    direct_match,
    profile_definition,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRASTS = {
    "E_minus_D": (PROFILE_E, PROFILE_D),
    "E_minus_C": (PROFILE_E, PROFILE_C),
    "D_minus_D": (PROFILE_D, PROFILE_D),
    "E_minus_E": (PROFILE_E, PROFILE_E),
}


def evaluate(args: argparse.Namespace, label: str, suite: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for contrast, (first, second) in CONTRASTS.items():
        result[contrast] = {}
        for budget in BUDGETS:
            blocks = [
                direct_match(
                    args=args,
                    first=first,
                    second=second,
                    budget=budget,
                    suite=suite,
                    seed=seed,
                    out=Path(args.workdir)
                    / label
                    / contrast
                    / budget.replace(":", "_")
                    / f"seed_{seed}",
                )
                for seed in args.base_seeds
            ]
            by_seed = {
                str(block["base_seed"]): block["paired_per_opening_delta"]
                for block in blocks
            }
            aggregate = aggregate_seed_opening_deltas(by_seed, seed=42)
            result[contrast][budget] = {
                **blocks[0],
                "orientation_normalized_ds": aggregate["mean"],
                "opening_cluster_ci95": aggregate["opening_cluster_ci95"],
                "hierarchical_ci95": aggregate["hierarchical_ci95"],
                "opening_deltas_by_seed": by_seed,
                "seed_block_sensitivity": {
                    str(block["base_seed"]): block["orientation_normalized_ds"]
                    for block in blocks
                },
            }
    return result


def expected_nulls(rows: dict[str, Any]) -> list[str]:
    errors = []
    contexts = {
        "E_minus_D": ("384:256", "768:768"),
        "E_minus_C": ("768:256", "1200:1200", "1200:256", "256:768"),
        "D_minus_D": BUDGETS,
        "E_minus_E": BUDGETS,
    }
    for contrast, budgets in contexts.items():
        for budget in budgets:
            if rows[contrast][budget]["orientation_normalized_ds"] != 0.0:
                errors.append(f"{contrast} {budget} expected exact null")
    return errors


def fixed_pass(rows: dict[str, Any]) -> bool:
    if expected_nulls(rows):
        return False
    d = rows["E_minus_D"]
    c = rows["E_minus_C"]
    return (
        d["768:256"]["orientation_normalized_ds"] >= 0.02
        and d["768:256"]["opening_cluster_ci95"]["lower_95"] > 0
        and d["1200:1200"]["orientation_normalized_ds"] >= 0.01
        and d["1200:1200"]["opening_cluster_ci95"]["lower_95"] >= 0
        and d["1200:256"]["orientation_normalized_ds"] >= 0.03
        and d["1200:256"]["opening_cluster_ci95"]["lower_95"] > 0
        and d["256:768"]["orientation_normalized_ds"] >= 0.02
        and d["256:768"]["opening_cluster_ci95"]["lower_95"] > 0
        and c["384:256"]["orientation_normalized_ds"] >= 0
        and c["768:768"]["orientation_normalized_ds"] >= 0.02
        and c["768:768"]["opening_cluster_ci95"]["lower_95"] > 0
    )


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Budget-Conditioned Tactical Profile",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "| Suite | Contrast | Budget | DS | 95% CI | +/0/- |",
        "|---|---|---|---:|---|---:|",
    ]
    for suite in ("medium", "fixed_large"):
        for contrast, budgets in summary.get(suite, {}).items():
            for budget, row in budgets.items():
                ci = row["opening_cluster_ci95"]
                lines.append(
                    f"| {suite} | {contrast} | {budget} | {row['orientation_normalized_ds']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {row['positive_openings']}/{row['zero_openings']}/{row['negative_openings']} |"
                )
    lines += [
        "",
        "Profile E uses 0.00 only at 384:256 and 768:768; all other listed budgets use 0.10. Profile D is retained as the runtime default.",
        "",
    ]
    return "\n".join(lines)


def compact(value: Any) -> None:
    """Keep detailed opening ledgers in the workdir, not committed summaries."""
    if isinstance(value, dict):
        value.pop("paired_per_opening_delta", None)
        value.pop("opening_deltas_by_seed", None)
        value.pop("games", None)
        value.pop("per_opening_score", None)
        if isinstance(value.get("p0_p1"), dict):
            value["p0_p1"] = {
                name: {
                    "p0_score": metrics.get("p0_score"),
                    "p1_score": metrics.get("p1_score"),
                }
                for name, metrics in value["p0_p1"].items()
                if isinstance(metrics, dict)
            }
        for child in value.values():
            compact(child)
    elif isinstance(value, list):
        for child in value:
            compact(child)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--seed-contract", default=SEED_CONTRACT_VERSION)
    parser.add_argument("--base-seeds", default="42,43,44,45")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.base_seeds = [int(value) for value in args.base_seeds.split(",")]
    if args.seed_contract != SEED_CONTRACT_VERSION:
        raise ValueError("only azlite_eval_seed_v2 is supported")
    if (
        sha256_file(Path(args.current) / "weights.json")
        != args.expected_current_weights_sha256
    ):
        raise ValueError("current weights SHA256 does not match expected artifact")
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    medium = evaluate(args, "medium", Path(args.medium_suite))
    medium_errors = expected_nulls(medium)
    if medium_errors:
        classification = "runtime_profile_resolver_or_seed_contract_failed"
        fixed: dict[str, Any] = {}
    else:
        fixed = evaluate(args, "fixed_large", Path(args.fixed_large_suite))
        classification = (
            "budget_conditioned_runtime_profile_candidate"
            if fixed_pass(fixed)
            else "budget_conditioned_tactical_profile_rejected"
        )
    summary = {
        "schema": "azlite_budget_conditioned_tactical_profile_v1",
        "artifact_weights_sha256": args.expected_current_weights_sha256,
        "seed_contract": args.seed_contract,
        "base_seeds": args.base_seeds,
        "profiles": {
            name: profile_definition(name) for name in (PROFILE_C, PROFILE_D, PROFILE_E)
        },
        "medium": medium,
        "fixed_large": fixed,
        "expected_null_errors": medium_errors
        + (expected_nulls(fixed) if fixed else []),
        "classification": classification,
    }
    workdir = Path(args.workdir)
    (workdir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    committed = json.loads(json.dumps(summary))
    compact(committed)
    (
        REPO_ROOT
        / "docs/data/alphazero-lite-budget-conditioned-tactical-profile-summary.json"
    ).write_text(json.dumps(committed, indent=2), encoding="utf-8")
    (
        REPO_ROOT / "docs/alphazero-lite-budget-conditioned-tactical-profile-results.md"
    ).write_text(markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
