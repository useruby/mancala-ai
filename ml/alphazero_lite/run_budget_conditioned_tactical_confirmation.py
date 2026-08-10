#!/usr/bin/env python3
"""Preregistered independent confirmation of tactical profile E against D."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from ml.alphazero_lite.build_opening_suite import (  # noqa: E402
    canonical_key,
    deduplicate_openings,
    enumerate_legal_prefixes,
    select_diverse,
    stratify_openings,
    suite_sha256,
    write_suite_jsonl,
)
from ml.alphazero_lite.evaluation_seed_contract import SEED_CONTRACT_VERSION  # noqa: E402
from ml.alphazero_lite.run_canonical_runtime_profile_revalidation import (  # noqa: E402
    BUDGETS,
    PROFILE_D,
    PROFILE_E,
    direct_match,
    profile_definition,
    sha256_file,
)
from ml.alphazero_lite.runtime_profiles import resolve_runtime_profile  # noqa: E402

WORKDIR = Path("/tmp/azlite_budget_tactical_confirmation")
SEEDS = (49, 50, 51, 52, 53, 54)
SUITE_SIZE = 384
CHANGED_BUDGETS = ("768:256", "1200:1200", "1200:256", "256:768")
NULL_BUDGETS = ("384:256", "768:768")
CONTRASTS = {
    "E_minus_D": (PROFILE_E, PROFILE_D),
    "D_minus_D": (PROFILE_D, PROFILE_D),
    "E_minus_E": (PROFILE_E, PROFILE_E),
}
THRESHOLDS = {
    "768:256": {"mean": 0.02, "lower": 0.0, "lower_strict": True, "worst": 0.0},
    "1200:1200": {"mean": 0.01, "lower": 0.0, "lower_strict": False, "worst": -0.02},
    "1200:256": {"mean": 0.03, "lower": 0.0, "lower_strict": True, "worst": 0.0},
    "256:768": {"mean": 0.02, "lower": 0.0, "lower_strict": True, "worst": 0.0},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def prefix_key(entry: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(move) for move in entry["prefix_moves"])


def suite_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "openings": len(entries),
        "duplicate_states": len(entries)
        - len({canonical_key(row["state"]) for row in entries}),
        "duplicate_prefixes": len(entries) - len({prefix_key(row) for row in entries}),
        "phase_distribution": dict(
            sorted(Counter(row["phase_bucket"] for row in entries).items())
        ),
        "prefix_length_distribution": dict(
            sorted(Counter(len(row["prefix_moves"]) for row in entries).items())
        ),
        "player_to_move_distribution": dict(
            sorted(Counter(row["side_to_move"] for row in entries).items())
        ),
    }


def prior_suite_paths() -> list[Path]:
    paths = [
        Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
        Path("/tmp/azlite_opening_suite/large_eval.jsonl"),
        *sorted(
            path
            for path in Path("/tmp").glob("azlite*/suites/heldout_seed4*_large.jsonl")
            if WORKDIR not in path.parents
        ),
    ]
    unique = list(dict.fromkeys(paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prior evaluation suites: {missing}")
    return unique


def build_suites(
    suite_dir: Path, previous: list[Path]
) -> tuple[list[Path], dict[str, Any]]:
    prior_rows = [row for path in previous for row in load_jsonl(path)]
    prior_states = {canonical_key(row["state"]) for row in prior_rows}
    prior_prefixes = {prefix_key(row) for row in prior_rows}
    all_prefixes = [
        entry for ply in (2, 4, 6) for entry in enumerate_legal_prefixes(ply)
    ]
    unique, _duplicates, _count = deduplicate_openings(all_prefixes)
    eligible = [
        entry
        for entry in stratify_openings(unique)
        if canonical_key(entry["state"]) not in prior_states
        and prefix_key(entry) not in prior_prefixes
    ]
    paths, selected_states, selected_prefixes = [], set(), set()
    audit: dict[str, Any] = {
        "schema": "azlite_budget_tactical_suite_independence_v1",
        "generator": "build_opening_suite.py: enumerate plies 2,4,6; state deduplication; stratified select_diverse",
        "exclusion_layer": "exclude prior exact states and prefixes, then exclude already-selected exact states and prefixes",
        "prior_suites": {str(path): suite_sha256(str(path)) for path in previous},
        "new_suites": {},
    }
    for seed in SEEDS:
        path = suite_dir / f"heldout_seed{seed}_large.jsonl"
        if path.is_file():
            selected = load_jsonl(path)
        else:
            candidates = [
                entry
                for entry in eligible
                if canonical_key(entry["state"]) not in selected_states
                and prefix_key(entry) not in selected_prefixes
            ]
            selected = select_diverse(candidates, SUITE_SIZE, seed)
            if len(selected) != SUITE_SIZE:
                raise ValueError("insufficient independent openings after exclusions")
            write_suite_jsonl(selected, str(path))
        states = {canonical_key(row["state"]) for row in selected}
        prefixes = {prefix_key(row) for row in selected}
        audit["new_suites"][path.name] = {
            "sha256": suite_sha256(str(path)),
            **suite_summary(selected),
            "exact_opening_state_overlap_with_prior": len(states & prior_states),
            "exact_prefix_overlap_with_prior": len(prefixes & prior_prefixes),
        }
        if any(
            audit["new_suites"][path.name][key]
            for key in (
                "duplicate_states",
                "duplicate_prefixes",
                "exact_opening_state_overlap_with_prior",
                "exact_prefix_overlap_with_prior",
            )
        ):
            raise ValueError(f"suite independence failure for {path.name}")
        paths.append(path)
        selected_states.update(states)
        selected_prefixes.update(prefixes)
    audit["passed"] = True
    return paths, audit


def contract(suites: list[Path], artifact_sha: str) -> dict[str, Any]:
    profiles = {name: profile_definition(name) for name in (PROFILE_D, PROFILE_E)}
    return {
        "schema": "azlite_budget_tactical_confirmation_contract_v1",
        "artifact": "model-artifact/current",
        "artifact_weights_sha256": artifact_sha,
        "seed_contract": SEED_CONTRACT_VERSION,
        "base_search_seed": 42,
        "profiles": profiles,
        "contrasts": {name: list(value) for name, value in CONTRASTS.items()},
        "budgets": list(BUDGETS),
        "suites": {path.name: suite_sha256(str(path)) for path in suites},
        "games_per_opening": 2,
        "orientations": ["first_challenger", "second_challenger"],
        "seats": [0, 1],
        "bootstrap": {
            "method": "hierarchical_suite_then_opening",
            "samples": 10000,
            "seed": 42,
        },
        "thresholds": THRESHOLDS,
        "null_budgets": list(NULL_BUDGETS),
        "seat_thresholds": {
            "aggregate_minimum": -0.02,
            "suite_collapse_minimum": -0.05,
        },
    }


def hierarchical(suites: list[list[float]], samples: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    arrays = [np.asarray(suite, dtype=np.float64) for suite in suites]
    values = np.asarray(
        [
            np.mean(
                [
                    np.mean(
                        arrays[index][
                            rng.integers(len(arrays[index]), size=len(arrays[index]))
                        ]
                    )
                    for index in rng.integers(len(arrays), size=len(arrays))
                ]
            )
            for _ in range(samples)
        ]
    )
    mean = statistics.fmean(statistics.fmean(row) for row in suites)
    return {
        "mean": mean,
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
        "samples": samples,
    }


def leave_one_out(suites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: hierarchical(
            [
                other["paired_per_opening_delta"]
                for other_name, other in suites.items()
                if other_name != name
            ]
        )
        for name in suites
    }


def evaluate(args: argparse.Namespace, suites: list[Path]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for suite in suites:
        label = suite.stem
        rows: dict[str, Any] = {}
        for contrast, (first, second) in CONTRASTS.items():
            rows[contrast] = {}
            for budget in BUDGETS:
                rows[contrast][budget] = direct_match(
                    args=args,
                    first=first,
                    second=second,
                    budget=budget,
                    suite=suite,
                    seed=42,
                    out=Path(args.workdir)
                    / "arena"
                    / label
                    / contrast
                    / budget.replace(":", "_"),
                )
        results[label] = rows
    return results


def exact_null_errors(results: dict[str, Any]) -> list[str]:
    errors = []
    for suite, contrasts in results.items():
        for contrast, budgets in contrasts.items():
            required = (
                BUDGETS if contrast in ("D_minus_D", "E_minus_E") else NULL_BUDGETS
            )
            for budget in required:
                row = budgets[budget]
                if row["orientation_normalized_ds"] != 0.0 or any(
                    row["paired_per_opening_delta"]
                ):
                    errors.append(f"{suite} {contrast} {budget}")
    return errors


def aggregate(results: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for budget in CHANGED_BUDGETS:
        suites = {name: rows["E_minus_D"][budget] for name, rows in results.items()}
        suite_ds = [row["orientation_normalized_ds"] for row in suites.values()]
        output[budget] = {
            "per_suite": suites,
            "mean_ds": statistics.fmean(suite_ds),
            "median_ds": statistics.median(suite_ds),
            "worst_suite_ds": min(suite_ds),
            "suite_standard_deviation": statistics.stdev(suite_ds),
            "hierarchical_ci95": hierarchical(
                [row["paired_per_opening_delta"] for row in suites.values()]
            ),
            "leave_one_suite_out": leave_one_out(suites),
        }
    return output


def seat_failures(aggregate_rows: dict[str, Any]) -> list[str]:
    failures = []
    for budget, row in aggregate_rows.items():
        for orientation in ("first_challenger", "second_challenger_normalized"):
            values = [
                suite["orientation_decomposition"][orientation]["mean"]
                for suite in row["per_suite"].values()
            ]
            if statistics.fmean(values) < -0.02:
                failures.append(f"{budget} {orientation} aggregate")
            if min(values) < -0.05:
                failures.append(f"{budget} {orientation} suite")
    return failures


def strength_failures(aggregate_rows: dict[str, Any]) -> list[str]:
    failures = []
    for budget, row in aggregate_rows.items():
        threshold = THRESHOLDS[budget]
        ci = row["hierarchical_ci95"]
        if row["mean_ds"] < threshold["mean"]:
            failures.append(f"{budget} mean")
        if (
            ci["lower_95"] <= threshold["lower"]
            if threshold["lower_strict"]
            else ci["lower_95"] < threshold["lower"]
        ):
            failures.append(f"{budget} lower_ci")
        if row["worst_suite_ds"] < threshold["worst"]:
            failures.append(f"{budget} worst_suite")
    return failures


def compact(value: Any) -> None:
    if isinstance(value, dict):
        value.pop("paired_per_opening_delta", None)
        value.pop("games", None)
        value.pop("per_opening_score", None)
        for child in value.values():
            compact(child)
    elif isinstance(value, list):
        for child in value:
            compact(child)


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Budget-Conditioned Tactical Confirmation",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        f"Decision contract SHA256: `{summary['decision_contract_sha256']}`.",
        "",
        "| Budget | Mean DS | Median | Worst suite | Hierarchical 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for budget, row in summary["changed_budgets"].items():
        ci = row["hierarchical_ci95"]
        lines.append(
            f"| {budget} | {row['mean_ds']:+.4f} | {row['median_ds']:+.4f} | {row['worst_suite_ds']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            f"Suite independence audit: `{'passed' if summary['suite_independence_audit']['passed'] else 'failed'}`.",
            f"Expected-null errors: `{len(summary['null_errors'])}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    """Run the repaired gate only after the frozen held-out decision passes."""
    output = workdir / "runtime_profile_gate.json"
    candidate = profile_definition(PROFILE_E)
    current = profile_definition(PROFILE_D)
    command = [
        sys.executable,
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        args.current,
        "--current-path",
        args.current,
        "--out",
        str(output),
        "--workdir",
        str(workdir / "gate_work"),
        "--seed-contract",
        SEED_CONTRACT_VERSION,
        "--base-seed",
        "42",
        "--budget-pairs",
        ",".join(BUDGETS),
        "--candidate-runtime-profile-json",
        json.dumps(candidate, sort_keys=True),
        "--current-runtime-profile-json",
        json.dumps(current, sort_keys=True),
        "--workers",
        str(args.workers),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    validation = result.get("runtime_gate_validation", {})
    return {
        "status": "executed",
        "result": result,
        "passed": bool(validation.get("valid"))
        and result.get("classification") != "runtime_profile_gate_invalid",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=str(WORKDIR))
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default="8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a",
    )
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.seed_contract = SEED_CONTRACT_VERSION
    if (
        sha256_file(Path(args.current) / "weights.json")
        != args.expected_current_weights_sha256
    ):
        raise ValueError("current weights SHA256 does not match expected artifact")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    suites, audit = build_suites(workdir / "suites", prior_suite_paths())
    (workdir / "suite_independence_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    decision = contract(suites, args.expected_current_weights_sha256)
    contract_path = workdir / "decision_contract.json"
    contract_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    decision_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    results = evaluate(args, suites)
    null_errors = exact_null_errors(results)
    changed = aggregate(results)
    failures = strength_failures(changed)
    seats = seat_failures(changed)
    if null_errors:
        classification = "runtime_profile_resolver_or_seed_contract_failed"
    elif failures or seats:
        other = [failure for failure in failures if not failure.startswith("1200:1200")]
        classification = (
            "tactical_profile_1200_equal_inconclusive"
            if failures and not other and not seats
            else "budget_conditioned_tactical_profile_rejected_confirmed"
        )
    else:
        classification = "budget_conditioned_runtime_profile_candidate"
    gate = (
        run_gate(args, workdir)
        if classification == "budget_conditioned_runtime_profile_candidate"
        else {"status": "not_run", "reason": "phase_f_did_not_pass"}
    )
    if (
        classification == "budget_conditioned_runtime_profile_candidate"
        and not gate["passed"]
    ):
        classification = "runtime_profile_gate_invalid"
    summary = {
        "schema": "azlite_budget_tactical_confirmation_v1",
        "artifact_weights_sha256": args.expected_current_weights_sha256,
        "seed_contract": SEED_CONTRACT_VERSION,
        "base_search_seed": 42,
        "decision_contract_sha256": decision_hash,
        "suite_independence_audit": audit,
        "profiles": {name: profile_definition(name) for name in (PROFILE_D, PROFILE_E)},
        "resolved_treatment_hashes": {
            budget: {
                name: resolve_runtime_profile(profile_definition(name), budget)[
                    "runtime_treatment_hash"
                ]
                for name in (PROFILE_D, PROFILE_E)
            }
            for budget in BUDGETS
        },
        "null_results": {
            name: {
                budget: {suite: rows[name][budget] for suite, rows in results.items()}
                for budget in (BUDGETS if name != "E_minus_D" else NULL_BUDGETS)
            }
            for name in ("D_minus_D", "E_minus_E", "E_minus_D")
        },
        "changed_budgets": changed,
        "null_errors": null_errors,
        "strength_failures": failures,
        "seat_failures": seats,
        "gate": gate,
        "classification": classification,
    }
    detailed = workdir / "summary_metrics.json"
    detailed.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    committed = json.loads(json.dumps(summary))
    compact(committed)
    (
        REPO_ROOT
        / "docs/data/alphazero-lite-budget-conditioned-tactical-confirmation-summary.json"
    ).write_text(json.dumps(committed, indent=2), encoding="utf-8")
    (
        REPO_ROOT
        / "docs/alphazero-lite-budget-conditioned-tactical-confirmation-results.md"
    ).write_text(markdown(summary), encoding="utf-8")
    print(f"classification={classification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
