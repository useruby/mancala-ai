#!/usr/bin/env python3
"""Treatment-invariant, direct 2x2 runtime-profile revalidation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    stable_hash,
)
from ml.alphazero_lite.runtime_profiles import (  # noqa: E402
    resolve_runtime_profile,
    runtime_profile_definition,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import compute_seat_metrics  # noqa: E402

PROFILES = {
    "legacy_tactical_no_schedule": {"tactical_root_bias": 0.10, "schedule": {}},
    "no_tactical_no_schedule": {"tactical_root_bias": 0.00, "schedule": {}},
    "legacy_tactical_with_schedule": {
        "tactical_root_bias": 0.10,
        "schedule": {"768:768": 0.90},
    },
    "current_promoted_profile": {
        "tactical_root_bias": 0.00,
        "schedule": {"768:768": 0.90},
    },
}
PROFILE_A = "legacy_tactical_no_schedule"
PROFILE_B = "no_tactical_no_schedule"
PROFILE_C = "legacy_tactical_with_schedule"
PROFILE_D = "current_promoted_profile"
PROFILE_E = "budget_conditioned_tactical_profile"
BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
DIRECT_CONTRASTS = {
    "B_minus_A": (PROFILE_B, PROFILE_A),
    "D_minus_C": (PROFILE_D, PROFILE_C),
    "C_minus_A": (PROFILE_C, PROFILE_A),
    "D_minus_B": (PROFILE_D, PROFILE_B),
}


def _runtime_profile(name: str) -> dict[str, Any]:
    profile = PROFILES[name]
    return runtime_profile_definition(
        name=name,
        default_tactical_root_bias=profile["tactical_root_bias"],
        tactical_root_bias_overrides=profile.get("tactical_overrides", {}),
        default_c_puct=1.25,
        c_puct_overrides=profile["schedule"],
    )


PROFILES[PROFILE_E] = {
    "tactical_root_bias": 0.10,
    "tactical_overrides": {"384:256": 0.00, "768:768": 0.00},
    "schedule": {"768:768": 0.90},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_definition(name: str) -> dict[str, Any]:
    # Preserve historical field names in reports while using one normalized resolver.
    profile = _runtime_profile(name)
    return {
        **profile,
        "c_puct_schedule": profile["c_puct_overrides"],
        "tactical_root_bias": profile["default_tactical_root_bias"],
    }


def bootstrap_ci(
    values: list[float], *, seed: int, samples: int = 10_000
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0, "samples": samples}
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(samples, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower_95": float(np.percentile(means, 2.5)),
        "upper_95": float(np.percentile(means, 97.5)),
        "samples": samples,
    }


def aggregate_seed_opening_deltas(
    opening_deltas_by_seed: dict[str, list[float]], *, seed: int, samples: int = 10_000
) -> dict[str, Any]:
    """Aggregate seed blocks without treating repeated openings as replicates."""
    blocks = list(opening_deltas_by_seed.values())
    if not blocks:
        return {
            "mean": 0.0,
            "opening_cluster_ci95": bootstrap_ci([], seed=seed, samples=samples),
            "hierarchical_ci95": bootstrap_ci([], seed=seed, samples=samples),
        }
    if len({tuple(block) for block in blocks}) == 1:
        unique_openings = blocks[0]
    else:
        unique_openings = [
            statistics.fmean(values) for values in zip(*blocks, strict=True)
        ]
    rng = np.random.default_rng(seed)
    hierarchical = np.asarray(
        [
            rng.choice(
                blocks[rng.integers(len(blocks))], size=len(blocks[0]), replace=True
            ).mean()
            for _ in range(samples)
        ]
    )
    return {
        "mean": statistics.fmean(statistics.fmean(block) for block in blocks),
        "opening_cluster_ci95": bootstrap_ci(
            unique_openings, seed=seed, samples=samples
        ),
        "hierarchical_ci95": {
            "mean": statistics.fmean(statistics.fmean(block) for block in blocks),
            "lower_95": float(np.percentile(hierarchical, 2.5)),
            "upper_95": float(np.percentile(hierarchical, 97.5)),
            "samples": samples,
        },
    }


def opening_ds(entries: list[dict], games_per_opening: int = 2) -> dict[int, float]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["game_index"]) // games_per_opening].append(entry)
    return {
        index: float(compute_seat_metrics(rows)["ds"])
        for index, rows in grouped.items()
    }


def paired_delta(left: list[dict], right: list[dict], *, seed: int) -> dict[str, Any]:
    left_by_opening, right_by_opening = opening_ds(left), opening_ds(right)
    indices = sorted(set(left_by_opening) & set(right_by_opening))
    deltas = [left_by_opening[i] - right_by_opening[i] for i in indices]
    return {
        "paired_per_opening_delta": deltas,
        "opening_cluster_ci95": bootstrap_ci(deltas, seed=seed),
        "positive_openings": sum(x > 0 for x in deltas),
        "zero_openings": sum(x == 0 for x in deltas),
        "negative_openings": sum(x < 0 for x in deltas),
    }


def profile_cpuct(profile: str, budget: str) -> float:
    return float(resolve_runtime_profile(_runtime_profile(profile), budget)["c_puct"])


def profile_tactical_root_bias(profile: str, budget: str) -> float:
    return float(
        resolve_runtime_profile(_runtime_profile(profile), budget)["tactical_root_bias"]
    )


def _load_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _score_by_opening(
    entries: list[dict], *, games_per_opening: int = 2
) -> dict[int, float]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["game_index"]) // games_per_opening].append(entry)
    return {
        key: statistics.fmean(
            1.0
            if row["winner"] == "challenger"
            else 0.5
            if row["winner"] == "draw"
            else 0.0
            for row in rows
        )
        for key, rows in grouped.items()
    }


def _run_orientation(
    *,
    args: argparse.Namespace,
    first: str,
    second: str,
    budget: str,
    suite: Path,
    seed: int,
    out: Path,
) -> dict[str, Any]:
    challenger_sims, current_sims = (int(value) for value in budget.split(":"))
    out.mkdir(parents=True, exist_ok=True)
    all_games, reports = [], []
    suite_size = len(_load_jsonl(suite))
    for seat in (0, 1):
        seat_dir = out / f"starts_{seat}"
        report_path, games_path = seat_dir / "arena.json", seat_dir / "games.jsonl"
        cached_games_path = (
            games_path if games_path.is_file() else games_path.with_suffix(".jsonl.gz")
        )
        if not (report_path.is_file() and cached_games_path.is_file()):
            seat_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(REPO_ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                args.current,
                "--current",
                args.current,
                "--challenger-simulations",
                str(challenger_sims),
                "--current-simulations",
                str(current_sims),
                "--games",
                str(suite_size * 2),
                "--base-seed",
                str(seed),
                "--seed-contract",
                args.seed_contract,
                "--workers",
                str(args.workers),
                "--min-score",
                "0",
                "--out",
                str(report_path),
                "--game-jsonl",
                str(games_path),
                "--challenger-starts",
                str(seat),
                "--games-per-opening",
                "2",
                "--opening-prefixes-jsonl",
                str(suite),
                "--root-policy-mode",
                "deterministic",
                "--challenger-c-puct",
                str(profile_cpuct(first, budget)),
                "--current-c-puct",
                str(profile_cpuct(second, budget)),
                "--challenger-search-options-json",
                json.dumps(
                    {"tactical_root_bias": profile_tactical_root_bias(first, budget)}
                ),
                "--current-search-options-json",
                json.dumps(
                    {"tactical_root_bias": profile_tactical_root_bias(second, budget)}
                ),
                "--challenger-runtime-profile-hash",
                profile_definition(first)["hash"],
                "--current-runtime-profile-hash",
                profile_definition(second)["hash"],
                "--seed-ledger-output",
                str(seat_dir / "seed_identity_ledger.jsonl.gz"),
                "--search-configuration-ledger-output",
                str(seat_dir / "search_configuration_ledger.jsonl.gz"),
                "--search-outcome-ledger-output",
                str(seat_dir / "search_outcome_ledger.jsonl.gz"),
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            cached_games_path = games_path
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        all_games.extend(_load_jsonl(cached_games_path))
    notes = [report["notes"] for report in reports]
    return {
        "games": all_games,
        "metrics": compute_seat_metrics(all_games),
        "per_opening_score": _score_by_opening(all_games),
        "provenance": {
            key: stable_hash([note[key] for note in notes])
            for key in (
                "seed_identity_ledger_sha256",
                "search_configuration_ledger_sha256",
                "search_outcome_ledger_sha256",
            )
        },
        "latency": {
            "mean_ms": statistics.fmean(note["move_time_mean_ms"] for note in notes),
            "p95_ms": statistics.fmean(note["move_time_p95_ms"] for note in notes),
        },
    }


def direct_match(
    *,
    args: argparse.Namespace,
    first: str,
    second: str,
    budget: str,
    suite: Path,
    seed: int,
    out: Path,
) -> dict[str, Any]:
    forward = _run_orientation(
        args=args,
        first=first,
        second=second,
        budget=budget,
        suite=suite,
        seed=seed,
        out=out / "first_challenger",
    )
    reverse = _run_orientation(
        args=args,
        first=second,
        second=first,
        budget=budget,
        suite=suite,
        seed=seed,
        out=out / "second_challenger",
    )
    keys = sorted(set(forward["per_opening_score"]) & set(reverse["per_opening_score"]))
    # The reverse challenger score is second-minus-first, so negate before pooling.
    deltas = [
        (forward["per_opening_score"][key] - reverse["per_opening_score"][key]) / 2.0
        for key in keys
    ]
    orientation = {
        "first_challenger": bootstrap_ci(
            [forward["per_opening_score"][key] - 0.5 for key in keys], seed=seed
        ),
        "second_challenger_normalized": bootstrap_ci(
            [0.5 - reverse["per_opening_score"][key] for key in keys], seed=seed
        ),
    }
    return {
        "first_profile": first,
        "second_profile": second,
        "budget": budget,
        "base_seed": seed,
        "orientation_normalized_ds": statistics.fmean(deltas) if deltas else 0.0,
        "paired_per_opening_delta": deltas,
        "opening_cluster_ci95": bootstrap_ci(deltas, seed=seed),
        "positive_openings": sum(x > 0 for x in deltas),
        "zero_openings": sum(x == 0 for x in deltas),
        "negative_openings": sum(x < 0 for x in deltas),
        "orientation_decomposition": orientation,
        "p0_p1": {
            "first_challenger": forward["metrics"],
            "second_challenger": reverse["metrics"],
        },
        "trajectory_agreement": {
            "first_challenger_unique": forward["metrics"]["unique_trajectories"],
            "second_challenger_unique": reverse["metrics"]["unique_trajectories"],
        },
        "latency": {
            "mean_ms": statistics.fmean(
                [forward["latency"]["mean_ms"], reverse["latency"]["mean_ms"]]
            ),
            "p95_ms": max(forward["latency"]["p95_ms"], reverse["latency"]["p95_ms"]),
        },
        "provenance": {
            key: stable_hash([forward["provenance"][key], reverse["provenance"][key]])
            for key in forward["provenance"]
        },
    }


def evaluate_suite(
    *,
    args: argparse.Namespace,
    label: str,
    suite: Path,
    contrasts: dict[str, tuple[str, str]],
    budgets: tuple[str, ...] = BUDGETS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for contrast, (first, second) in contrasts.items():
        result[contrast] = {}
        for budget in budgets:
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
            opening_deltas_by_seed = {
                str(block["base_seed"]): block["paired_per_opening_delta"]
                for block in blocks
            }
            # Base seeds 42--45 deliberately share the same opening block.  Bootstrap
            # unique opening indexes once, never seed x opening pseudo-replicates.
            aggregate = aggregate_seed_opening_deltas(opening_deltas_by_seed, seed=42)
            result[contrast][budget] = {
                **blocks[0],
                "orientation_normalized_ds": statistics.fmean(
                    block["orientation_normalized_ds"] for block in blocks
                ),
                "opening_cluster_ci95": aggregate["opening_cluster_ci95"],
                "hierarchical_ci95": aggregate["hierarchical_ci95"],
                "opening_deltas_by_seed": opening_deltas_by_seed,
                "seed_sensitivity": {
                    "min": min(block["orientation_normalized_ds"] for block in blocks),
                    "max": max(block["orientation_normalized_ds"] for block in blocks),
                    "sign_consistent": len(
                        {
                            (value > 0) - (value < 0)
                            for value in (
                                block["orientation_normalized_ds"] for block in blocks
                            )
                        }
                    )
                    == 1,
                },
                "seed_block_sensitivity": {
                    str(block["base_seed"]): block["orientation_normalized_ds"]
                    for block in blocks
                },
                "provenance_by_seed": {
                    str(block["base_seed"]): block["provenance"] for block in blocks
                },
            }
    return result


def factorial_effects(fixed: dict[str, Any]) -> dict[str, Any]:
    rows = fixed
    effects = {name: rows[name]["768:768"] for name in DIRECT_CONTRASTS}
    scheduled_no_tactical = effects["D_minus_B"]["paired_per_opening_delta"]
    scheduled_tactical = effects["C_minus_A"]["paired_per_opening_delta"]
    interaction = [a - b for a, b in zip(scheduled_no_tactical, scheduled_tactical)]
    effects["interaction_(D_minus_B)_minus_(C_minus_A)"] = {
        "point_estimate": statistics.fmean(interaction) if interaction else 0.0,
        "paired_per_opening_delta": interaction,
        "opening_cluster_ci95": bootstrap_ci(interaction, seed=42),
        "positive_openings": sum(x > 0 for x in interaction),
        "negative_openings": sum(x < 0 for x in interaction),
    }
    return effects


def classify(summary: dict[str, Any]) -> str:
    fixed = summary["fixed_large"]
    schedule = fixed["D_minus_B"]["768:768"]["opening_cluster_ci95"]
    tactical = fixed["D_minus_C"]["768:768"]["opening_cluster_ci95"]
    nonactive_failure = any(
        abs(fixed[name][budget]["orientation_normalized_ds"]) > 1e-12
        for name in ("C_minus_A", "D_minus_B")
        for budget in BUDGETS
        if budget != "768:768"
    )
    if (
        nonactive_failure
        or summary["nulls"]["D_minus_D"]["768:768"]["orientation_normalized_ds"] != 0
        or summary["nulls"]["A_minus_A"]["768:768"]["orientation_normalized_ds"] != 0
    ):
        return "evaluation_seed_v2_failed"
    if summary.get("gate", {}).get("status") == "invalid_original_gate":
        return "global_no_tactical_bias_not_robust"
    if schedule["mean"] <= 0:
        return "cpuct_schedule_promotion_invalidated"
    if tactical["upper_95"] < 0:
        return "no_tactical_bias_promotion_invalidated"
    gate = summary.get("gate", {})
    gate_result = gate.get("result", {}) if isinstance(gate, dict) else {}
    if (
        gate.get("status") == "executed"
        and gate_result.get("classification") == "regression_masked_by_seat"
    ):
        return "current_runtime_profile_statistically_inconclusive"
    if schedule["lower_95"] > 0:
        return "current_runtime_profile_revalidated"
    return "current_runtime_profile_statistically_inconclusive"


def _strip_details(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("paired_per_opening_delta", None)
        value.pop("opening_deltas_by_seed", None)
        value.pop("games", None)
        value.pop("per_opening_score", None)
        p0_p1 = value.get("p0_p1")
        if isinstance(p0_p1, dict):
            value["p0_p1"] = {
                orientation: {
                    "p0_score": metrics.get("p0_score"),
                    "p1_score": metrics.get("p1_score"),
                }
                for orientation, metrics in p0_p1.items()
                if isinstance(metrics, dict)
            }
        value.pop("provenance_by_seed", None)
        value.pop("orientation_decomposition", None)
        for child in value.values():
            _strip_details(child)
    elif isinstance(value, list):
        for child in value:
            _strip_details(child)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Treatment-Invariant Runtime Profile Revalidation",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "## Direct Contrasts",
        "",
        "| Suite | Contrast | Budget | DS | 95% CI | +/0/- |",
        "|---|---|---:|---:|---|---:|",
    ]
    for suite in ("medium", "fixed_large"):
        for name, budgets in summary[suite].items():
            for budget, row in budgets.items():
                ci = row["opening_cluster_ci95"]
                lines.append(
                    f"| {suite} | {name} | {budget} | {row['orientation_normalized_ds']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {row['positive_openings']}/{row['zero_openings']}/{row['negative_openings']} |"
                )
    lines += [
        "",
        "## Factorial Effects",
        "",
        "| Effect | Estimate | 95% CI |",
        "|---|---:|---|",
    ]
    for name, row in summary["factorial_effects"].items():
        ci = row["opening_cluster_ci95"]
        estimate = row.get("point_estimate", row.get("orientation_normalized_ds", 0.0))
        lines.append(
            f"| {name} | {estimate:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
        )
    lines += [
        "",
        "## Held-Out Primary Decisions",
        "",
        "| Suite | Contrast | DS | 95% CI |",
        "|---|---|---:|---|",
    ]
    for suite, contrasts in summary["heldout"].items():
        for name, budgets in contrasts.items():
            row = budgets["768:768"]
            ci = row["opening_cluster_ci95"]
            lines.append(
                f"| {suite} | {name} | {row['orientation_normalized_ds']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
            )
    lines += [
        "",
        "## Null Comparisons",
        "",
        "Duplicate A-v-A and D-v-D are required to be exactly zero per opening and in aggregate. Detailed ledgers remain in the work directory.",
        "",
        "## Gate Correction",
        "",
        "The original gate is invalid candidate evidence because it supplied one shared runtime profile to both equal-weight sides. Confidence intervals bootstrap unique opening indexes once; repeated base-seed opening blocks are reported as sensitivity, not independent observations.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--seed-contract", default=SEED_CONTRACT_VERSION)
    parser.add_argument("--base-seeds", default="42,43,44,45")
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_seeds = [int(seed) for seed in args.base_seeds.split(",")]
    if args.seed_contract != SEED_CONTRACT_VERSION:
        raise ValueError("new revalidation requires azlite_eval_seed_v2")
    if (
        sha256_file(Path(args.current) / "weights.json")
        != args.expected_current_weights_sha256
    ):
        raise ValueError("current weights SHA256 does not match expected artifact")
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    medium = evaluate_suite(
        args=args,
        label="medium",
        suite=Path(args.medium_suite),
        contrasts=DIRECT_CONTRASTS,
    )
    fixed = evaluate_suite(
        args=args,
        label="fixed_large",
        suite=Path(args.fixed_large_suite),
        contrasts=DIRECT_CONTRASTS,
    )
    nulls = evaluate_suite(
        args=args,
        label="nulls",
        suite=Path(args.fixed_large_suite),
        contrasts={
            "D_minus_D": (PROFILE_D, PROFILE_D),
            "A_minus_A": (PROFILE_A, PROFILE_A),
        },
    )
    heldout = {
        Path(text).stem: evaluate_suite(
            args=args,
            label=f"heldout/{Path(text).stem}",
            suite=Path(text),
            contrasts={
                "D_minus_B": (PROFILE_D, PROFILE_B),
                "D_minus_C": (PROFILE_D, PROFILE_C),
                "D_minus_A": (PROFILE_D, PROFILE_A),
            },
            budgets=("768:768",),
        )
        for text in args.heldout_suites.split(",")
    }
    primary_rows = [
        suite[contrast]["768:768"]["opening_cluster_ci95"]
        for suite in heldout.values()
        for contrast in ("D_minus_B", "D_minus_C", "D_minus_A")
    ]
    gate_eligible = bool(primary_rows) and all(
        row["lower_95"] > 0 for row in primary_rows
    )
    gate: dict[str, Any] = {
        "status": "invalid_original_gate",
        "reason": "PR #172 supplied one shared profile to both same-weight sides; it is not candidate evidence.",
        "would_have_been_eligible": gate_eligible,
    }
    summary = {
        "schema": "azlite_treatment_invariant_runtime_revalidation_v2",
        "artifact_weights_sha256": args.expected_current_weights_sha256,
        "seed_contract": args.seed_contract,
        "base_seeds": args.base_seeds,
        "profiles": {name: profile_definition(name) for name in PROFILES},
        "medium": medium,
        "fixed_large": fixed,
        "nulls": nulls,
        "heldout": heldout,
        "factorial_effects": factorial_effects(fixed),
        "gate": gate,
    }
    summary["classification"] = classify(summary)
    detailed = Path(args.workdir) / "summary_metrics.json"
    detailed.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    committed = json.loads(json.dumps(summary))
    _strip_details(committed)
    (
        REPO_ROOT
        / "docs/data/alphazero-lite-treatment-invariant-runtime-revalidation-summary.json"
    ).write_text(json.dumps(committed, indent=2), encoding="utf-8")
    (
        REPO_ROOT
        / "docs/alphazero-lite-treatment-invariant-runtime-revalidation-results.md"
    ).write_text(markdown(summary), encoding="utf-8")
    print(f"wrote {detailed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
