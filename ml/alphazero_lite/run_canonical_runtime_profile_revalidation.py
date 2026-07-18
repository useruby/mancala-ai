#!/usr/bin/env python3
"""Canonical 2x2 runtime-profile revalidation without training or promotion."""

from __future__ import annotations

import argparse
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

from ml.alphazero_lite.cpuct_schedule import resolve_budget_cpuct  # noqa: E402
from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    stable_hash,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_seat_metrics,
)

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
BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_definition(name: str) -> dict[str, Any]:
    profile = PROFILES[name]
    return {
        "name": name,
        "default_c_puct": 1.25,
        "c_puct_schedule": profile["schedule"],
        "tactical_root_bias": profile["tactical_root_bias"],
        "root_policy_mode": "deterministic",
        "normalize_values": False,
        "value_transform": None,
        "root_prior_transform": None,
        "hash": stable_hash(profile),
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


def opening_ds(entries: list[dict], games_per_opening: int = 2) -> dict[int, float]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["game_index"]) // games_per_opening].append(entry)
    return {
        index: float(compute_seat_metrics(rows)["ds"])
        for index, rows in grouped.items()
    }


def paired_delta(left: list[dict], right: list[dict], *, seed: int) -> dict[str, Any]:
    left_by_opening = opening_ds(left)
    right_by_opening = opening_ds(right)
    indices = sorted(set(left_by_opening) & set(right_by_opening))
    deltas = [left_by_opening[i] - right_by_opening[i] for i in indices]
    return {
        "paired_per_opening_delta": deltas,
        "opening_cluster_ci95": bootstrap_ci(deltas, seed=seed),
        "positive_openings": sum(delta > 0 for delta in deltas),
        "zero_openings": sum(delta == 0 for delta in deltas),
        "negative_openings": sum(delta < 0 for delta in deltas),
    }


def factorial_contrasts(
    results: dict[str, dict[str, dict]], *, seed: int
) -> dict[str, dict]:
    contrasts: dict[str, dict] = {}
    definitions = {
        "no_tactical_without_schedule_B_minus_A": (PROFILE_B, PROFILE_A),
        "no_tactical_with_schedule_D_minus_C": (PROFILE_D, PROFILE_C),
        "schedule_with_tactical_C_minus_A": (PROFILE_C, PROFILE_A),
        "schedule_without_tactical_D_minus_B": (PROFILE_D, PROFILE_B),
    }
    for budget in BUDGETS:
        rows = {}
        for label, (left, right) in definitions.items():
            rows[label] = paired_delta(
                results[left][budget]["games"],
                results[right][budget]["games"],
                seed=seed,
            )
        interaction = [
            a - b
            for a, b in zip(
                rows["schedule_without_tactical_D_minus_B"]["paired_per_opening_delta"],
                rows["schedule_with_tactical_C_minus_A"]["paired_per_opening_delta"],
            )
        ]
        rows["interaction_(D_minus_B)_minus_(C_minus_A)"] = {
            "paired_per_opening_delta": interaction,
            "opening_cluster_ci95": bootstrap_ci(interaction, seed=seed),
        }
        contrasts[budget] = rows
    return contrasts


def classify(summary: dict[str, Any]) -> str:
    fixed = summary["fixed_large"]["profiles"]
    heldout = summary["heldout"]
    d_768 = fixed[PROFILE_D]["768:768"]["profile_minus_B"]["opening_cluster_ci95"]
    d_primary = fixed[PROFILE_D]["384:256"]
    interaction = summary["fixed_large"]["factorial_contrasts"]["768:768"][
        "interaction_(D_minus_B)_minus_(C_minus_A)"
    ]["opening_cluster_ci95"]
    if interaction["lower_95"] > 0 or interaction["upper_95"] < 0:
        return "runtime_profile_interaction_detected"
    if (
        d_768["mean"] <= 0
        or min(
            fixed[PROFILE_D][budget]["profile_minus_B"]["opening_cluster_ci95"]["mean"]
            for budget in ("1200:1200", "1200:256")
        )
        < -0.03
    ):
        return "cpuct_schedule_promotion_invalidated"
    if any(
        paired_delta(
            fixed[left][budget]["games"], fixed[right][budget]["games"], seed=42
        )["opening_cluster_ci95"]["lower_95"]
        > 0
        for left, right in ((PROFILE_A, PROFILE_B), (PROFILE_C, PROFILE_D))
        for budget in BUDGETS
    ):
        return "no_tactical_bias_promotion_invalidated"
    if heldout.get("worse_suite_collapse"):
        return "current_runtime_profile_statistically_inconclusive"
    primary_ci = d_primary["profile_minus_D"]["opening_cluster_ci95"]
    if primary_ci["lower_95"] >= 0 and d_768["lower_95"] > 0:
        return "current_runtime_profile_revalidated"
    return "current_runtime_profile_statistically_inconclusive"


def run_arena(
    *, args: argparse.Namespace, profile: str, budget: str, suite: Path, out: Path
) -> dict:
    challenger_sims, current_sims = (int(value) for value in budget.split(":"))
    definition = profile_definition(profile)
    c_puct = resolve_budget_cpuct(
        schedule=definition["c_puct_schedule"],
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=1.25,
    )
    suite_size = sum(
        1 for line in suite.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    games = suite_size * 2
    all_games: list[dict] = []
    reports = []
    for seat in (0, 1):
        seat_dir = out / f"starts_{seat}"
        seat_dir.mkdir(parents=True, exist_ok=True)
        report_path, games_path = seat_dir / "arena.json", seat_dir / "games.jsonl"
        if report_path.is_file() and games_path.is_file():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            all_games.extend(
                json.loads(line)
                for line in games_path.read_text(encoding="utf-8").splitlines()
                if line
            )
            continue
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
            str(games),
            "--base-seed",
            str(args.base_seed),
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
            "--c-puct",
            "1.25",
            "--challenger-c-puct",
            str(c_puct),
            "--challenger-search-options-json",
            json.dumps({"tactical_root_bias": definition["tactical_root_bias"]}),
            "--seed-ledger-output",
            str(seat_dir / "seed_identity_ledger.jsonl"),
            "--search-outcome-ledger-output",
            str(seat_dir / "search_outcome_ledger.jsonl"),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        all_games.extend(
            json.loads(line)
            for line in games_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    metrics = compute_seat_metrics(all_games)
    notes = reports[0]["notes"]
    return {
        **metrics,
        "games": all_games,
        "effective_c_puct": c_puct,
        "seed_identity_ledger_sha256": stable_hash(
            [r["notes"]["seed_identity_ledger_sha256"] for r in reports]
        ),
        "search_outcome_ledger_sha256": stable_hash(
            [r["notes"]["search_outcome_ledger_sha256"] for r in reports]
        ),
        "effective_runtime_profile_hash": definition["hash"],
        "move_time_mean_ms": statistics.fmean(
            r["notes"]["move_time_mean_ms"] for r in reports
        ),
        "move_time_p95_ms": statistics.fmean(
            r["notes"]["move_time_p95_ms"] for r in reports
        ),
        "arena_search_profile_hash": notes.get("search_profile_hash"),
    }


def add_comparisons(profiles: dict[str, dict], *, seed: int) -> None:
    for profile, budgets in profiles.items():
        for budget, row in budgets.items():
            for reference, suffix in (
                (PROFILE_A, "A"),
                (PROFILE_B, "B"),
                (PROFILE_D, "D"),
            ):
                row[f"profile_minus_{suffix}"] = paired_delta(
                    row["games"], profiles[reference][budget]["games"], seed=seed
                )


def heldout_summary(suites: dict[str, dict[str, dict]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"by_budget": {}}
    profiles = sorted({profile for suite in suites.values() for profile in suite})
    for budget in BUDGETS:
        rows: dict[str, Any] = {}
        for profile in profiles:
            values = [
                suite[profile][budget]["ds"]
                for suite in suites.values()
                if profile in suite
            ]
            rows[profile] = {
                "mean_ds": statistics.fmean(values),
                "worst_suite_ds": min(values),
                "suite_count": len(values),
            }
        if PROFILE_D in rows:
            for profile in profiles:
                if profile == PROFILE_D:
                    continue
                deltas = [
                    suite[profile][budget]["ds"] - suite[PROFILE_D][budget]["ds"]
                    for suite in suites.values()
                    if profile in suite
                ]
                rows[profile]["profile_minus_current"] = {
                    "suite_mean_delta": statistics.fmean(deltas),
                    # Suite-level rows are already opening-cluster paired internally.
                    "suite_cluster_ci95": bootstrap_ci(deltas, seed=42),
                }
        aggregate["by_budget"][budget] = rows
    aggregate["worse_suite_collapse"] = any(
        row[PROFILE_D]["worst_suite_ds"] < -0.03
        for row in aggregate["by_budget"].values()
        if PROFILE_D in row
    )
    return aggregate


def remove_detailed_opening_rows(value: Any) -> None:
    """Keep committed summaries aggregate-only; detailed rows stay in the workdir."""
    if isinstance(value, dict):
        value.pop("games", None)
        value.pop("paired_per_opening_delta", None)
        for child in value.values():
            remove_detailed_opening_rows(child)
    elif isinstance(value, list):
        for child in value:
            remove_detailed_opening_rows(child)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--seed-contract", default=SEED_CONTRACT_VERSION)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed_contract != SEED_CONTRACT_VERSION:
        raise ValueError(f"only {SEED_CONTRACT_VERSION} is supported")
    selected = [name for name in args.profiles.split(",") if name]
    if selected != list(PROFILES):
        raise ValueError("the revalidation requires the exact four canonical profiles")
    weights_sha = sha256_file(Path(args.current) / "weights.json")
    if weights_sha != args.expected_current_weights_sha256:
        raise ValueError("current weights SHA256 does not match the expected artifact")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def evaluate_suite(label: str, suite: Path, profiles: list[str]) -> dict[str, dict]:
        result = {profile: {} for profile in profiles}
        for profile in profiles:
            for budget in BUDGETS:
                result[profile][budget] = run_arena(
                    args=args,
                    profile=profile,
                    budget=budget,
                    suite=suite,
                    out=workdir / label / profile / budget.replace(":", "_"),
                )
        add_comparisons(result, seed=args.base_seed)
        return result

    medium = evaluate_suite("medium", Path(args.medium_suite), selected)
    fixed = evaluate_suite("fixed_large", Path(args.fixed_large_suite), selected)
    fixed_contrasts = factorial_contrasts(fixed, seed=args.base_seed)
    heldout_profiles = [PROFILE_D, PROFILE_B, PROFILE_A]
    interaction = fixed_contrasts["768:768"][
        "interaction_(D_minus_B)_minus_(C_minus_A)"
    ]["opening_cluster_ci95"]
    if interaction["lower_95"] > 0 or interaction["upper_95"] < 0:
        heldout_profiles.append(PROFILE_C)
    heldout = {}
    for suite_text in args.heldout_suites.split(","):
        suite = Path(suite_text)
        heldout[suite.stem] = evaluate_suite(
            f"heldout/{suite.stem}", suite, heldout_profiles
        )
    summary = {
        "schema": "azlite_canonical_runtime_profile_revalidation_v1",
        "artifact_weights_sha256": weights_sha,
        "seed_contract": args.seed_contract,
        "base_seed": args.base_seed,
        "profiles": {name: profile_definition(name) for name in selected},
        "medium": {"profiles": medium},
        "fixed_large": {"profiles": fixed, "factorial_contrasts": fixed_contrasts},
        "heldout": {"suites": heldout, **heldout_summary(heldout)},
        "gate": {
            "status": "not_run",
            "reason": "A gate is only eligible for a held-out primary effect of at least 0.05 with a non-zero lower CI and robustness limits passing.",
        },
    }
    summary["classification"] = classify(summary)
    remove_detailed_opening_rows(summary)
    output = workdir / "summary_metrics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    docs_data = (
        REPO_ROOT
        / "docs/data/alphazero-lite-canonical-runtime-profile-revalidation-summary.json"
    )
    docs_data.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = (
        REPO_ROOT
        / "docs/alphazero-lite-canonical-runtime-profile-revalidation-results.md"
    )
    report.write_text(
        "# Canonical Runtime Profile Revalidation\n\n"
        f"Classification: `{summary['classification']}`.\n\n"
        f"Artifact SHA256: `{weights_sha}`. Seed contract: `{args.seed_contract}`; base seed: `{args.base_seed}`.\n\n"
        "Seed identity ledgers contain only canonical seed inputs and derived identities; outcome ledgers contain moves, visits, cache execution, trajectories, and profile hashes. Detailed ledgers remain in the workdir.\n\n"
        "## Profiles\n\n```json\n"
        + json.dumps(summary["profiles"], indent=2)
        + "\n```\n\n"
        "## Results\n\nAggregate medium, fixed-large, held-out, P0/P1, latency, paired opening-cluster CI, and factorial contrast tables are in `docs/data/alphazero-lite-canonical-runtime-profile-revalidation-summary.json`.\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
