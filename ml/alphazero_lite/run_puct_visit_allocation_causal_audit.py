#!/usr/bin/env python3
# ruff: noqa: E402
"""Causal audit of PUCT child-Q versus final visit allocation.

This diagnostic reuses PR #180's frozen search and continuation records and
only evaluates forced Q-best moves that were absent from its bounded move set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_seed_contract import stable_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_denoised_puct_convergence_audit import (
    CONTINUATION_BUDGETS,
    EXPECTED_CURRENT_SHA256,
    run_forced_tasks,
    sha256_file,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_search_q_value_attribution_audit import (
    BOOTSTRAP_SAMPLES,
    _state_metric,
    calibration_summary,
    read_jsonl,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator

DEFAULT_PR180_WORKDIR = Path("/tmp/azlite_search_q_value_attribution")
DEFAULT_WORKDIR = Path("/tmp/azlite_puct_visit_allocation_causal")
SUMMARY_PATH = (
    REPO_ROOT / "docs/data/alphazero-lite-search-q-value-attribution-summary.json"
)
TEACHERS = (384, 768, 1200)
SCHEMA = "azlite_puct_visit_allocation_causal_audit_v1"


def _bootstrap(values: list[float], *, seed: int) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(BOOTSTRAP_SAMPLES, len(data)), replace=True).mean(
        axis=1
    )
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "n": len(data),
    }


def _best_move(teacher: dict[str, Any], *, field: str) -> int:
    if field == "q_value":
        return int(
            max(
                teacher["child_stats"],
                key=lambda row: (float(row["q_value"]), -int(row["move"])),
            )["move"]
        )
    return int(
        max(
            teacher["child_stats"],
            key=lambda row: (
                int(row["visits"]),
                float(row["q_value"]),
                -int(row["move"]),
            ),
        )["move"]
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"PR #180 reconstruction failed: {message}")


def load_pr180(
    workdir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    paths = {
        "manifest": workdir / "q_calibration_probe_manifest.json",
        "teachers": workdir / "teacher_search_records.jsonl",
        "moves": workdir / "audited_move_records.jsonl",
        "forced": workdir / "forced_continuation_records.jsonl",
    }
    _require(
        all(path.is_file() for path in paths.values()),
        "missing PR #180 workdir artifact",
    )
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    teachers, moves, forced = (
        read_jsonl(paths[name]) for name in ("teachers", "moves", "forced")
    )
    _require(
        manifest == summary["probe_manifest"],
        "probe manifest differs from committed summary",
    )
    _require(
        manifest["state_count"] == 768
        and len({row["state_hash"] for row in teachers}) == 768,
        "probe is not 768 unique states",
    )
    _require(
        summary["current_weights_sha256"] == EXPECTED_CURRENT_SHA256,
        "committed weights hash differs",
    )
    # PR #180 did not commit these record hashes; preserve observed values and
    # prove reuse by exact aggregate reconstruction below.
    provenance = {
        name: sha256_file(path) for name, path in paths.items() if name != "manifest"
    }
    rebuilt = calibration_summary(forced, seed=42)
    for signal in ("q_D768", "visit_share_D768", "q_D1200", "visit_share_D1200"):
        committed = summary["calibration"]["margin"]["1200"][signal]["mean"]
        observed = rebuilt["margin"]["1200"][signal]["mean"]
        _require(np.isclose(committed, observed, atol=1e-12), f"{signal} concordance")
        committed_regret = summary["calibration"]["selected_move_regret"]["1200"][
            signal
        ]["mean"]
        observed_regret = rebuilt["selected_move_regret"]["1200"][signal]["mean"]
        _require(
            np.isclose(committed_regret, observed_regret, atol=1e-12),
            f"{signal} regret",
        )
    return manifest, teachers, moves, forced, provenance


def _forced_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (row["state_hash"], int(row["continuation_budget"]), int(row["move"])): row
        for row in rows
    }


def missing_q_tasks(
    teachers: list[dict[str, Any]], forced: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    existing = _forced_index(forced)
    tasks = []
    for record in teachers:
        missing = {
            _best_move(record["teachers"][f"D{budget}"], field="q_value")
            for budget in TEACHERS
        }
        for continuation_budget in CONTINUATION_BUDGETS:
            absent = sorted(
                move
                for move in missing
                if (record["state_hash"], continuation_budget, move) not in existing
            )
            if absent:
                tasks.append(
                    {
                        "state": record["state"],
                        "state_hash": record["state_hash"],
                        "continuation_budget": continuation_budget,
                        "experiment_seed": seed,
                        "moves": absent,
                        "comparison": "missing_q_best",
                    }
                )
    return tasks


def run_missing(
    tasks: list[dict[str, Any]],
    *,
    checkpoint: Path,
    workers: int,
    teachers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    metadata = {row["state_hash"]: row for row in teachers}
    results = run_forced_tasks(tasks=tasks, checkpoint=checkpoint, workers=workers)
    rows = []
    for record in results:
        source = metadata[record["state_hash"]]
        for raw_move, result in record["interventions"].items():
            move = int(raw_move)
            row = {
                key: source[key]
                for key in (
                    "state_hash",
                    "state",
                    "player",
                    "phase",
                    "source_domain",
                    "legal_move_count",
                    "current_policy_entropy",
                    "policy_entropy_quartile",
                )
            }
            row.update(
                {
                    "move": move,
                    "continuation_budget": record["continuation_budget"],
                    **result,
                    "normalized_store_margin_root": float(result["store_margin_root"])
                    / 48.0,
                }
            )
            for budget in TEACHERS:
                child = next(
                    item
                    for item in source["teachers"][f"D{budget}"]["child_stats"]
                    if int(item["move"]) == move
                )
                row[f"q_D{budget}"] = float(child["q_value"])
                row[f"visit_count_D{budget}"] = int(child["visits"])
                row[f"visit_share_D{budget}"] = float(child["visits"]) / budget
            rows.append(row)
    return rows


def _quality_row(
    index: dict[tuple[str, int, int], dict[str, Any]],
    state_hash: str,
    continuation: int,
    move: int,
) -> dict[str, Any]:
    try:
        return index[(state_hash, continuation, move)]
    except KeyError as exc:
        raise RuntimeError(
            f"missing completed forced move: {state_hash} D{continuation} move {move}"
        ) from exc


def disagreement_rows(
    teachers: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    *,
    evaluator: CheckpointEvaluator,
) -> list[dict[str, Any]]:
    index = _forced_index(completed)
    output = []
    for record in teachers:
        game = KalahGame.from_state(record["state"])
        policy, _ = evaluator.evaluate(game)
        for budget in TEACHERS:
            teacher = record["teachers"][f"D{budget}"]
            q_move, visit_move = (
                _best_move(teacher, field="q_value"),
                _best_move(teacher, field="visits"),
            )
            prior_move = int(
                max(
                    record["legal_moves"], key=lambda move: (float(policy[move]), -move)
                )
            )
            if q_move == visit_move:
                continue
            q_child = next(
                row for row in teacher["child_stats"] if int(row["move"]) == q_move
            )
            visit_child = next(
                row for row in teacher["child_stats"] if int(row["move"]) == visit_move
            )
            for continuation in CONTINUATION_BUDGETS:
                q_result = _quality_row(
                    index, record["state_hash"], continuation, q_move
                )
                visit_result = _quality_row(
                    index, record["state_hash"], continuation, visit_move
                )
                parent_visits = budget
                q_prior, visit_prior = float(policy[q_move]), float(policy[visit_move])
                q_visits, visit_visits = (
                    int(q_child["visits"]),
                    int(visit_child["visits"]),
                )
                q_u = 1.25 * q_prior * np.sqrt(parent_visits) / (1 + q_visits)
                visit_u = (
                    1.25 * visit_prior * np.sqrt(parent_visits) / (1 + visit_visits)
                )
                output.append(
                    {
                        "teacher_budget": budget,
                        "continuation_budget": continuation,
                        "state_hash": record["state_hash"],
                        "q_best_move": q_move,
                        "visit_best_move": visit_move,
                        "prior_best_move": prior_move,
                        "prior_q_best": q_prior,
                        "prior_visit_best": visit_prior,
                        "q_q_best": float(q_child["q_value"]),
                        "q_visit_best": float(visit_child["q_value"]),
                        "visits_q_best": q_visits,
                        "visits_visit_best": visit_visits,
                        "visit_share_q_best": q_visits / budget,
                        "visit_share_visit_best": visit_visits / budget,
                        "prior_advantage_for_visit": visit_prior - q_prior,
                        "q_advantage_for_q": float(
                            q_child["q_value"] - visit_child["q_value"]
                        ),
                        "visit_advantage": (visit_visits - q_visits) / budget,
                        "q_u": float(q_u),
                        "visit_u": float(visit_u),
                        "q_plus_u": float(q_child["q_value"] + q_u),
                        "visit_plus_u": float(visit_child["q_value"] + visit_u),
                        "outcome_delta": float(
                            q_result["outcome_root"] - visit_result["outcome_root"]
                        ),
                        "store_margin_delta": float(
                            q_result["store_margin_root"]
                            - visit_result["store_margin_root"]
                        ),
                        "normalized_store_margin_delta": float(
                            q_result["normalized_store_margin_root"]
                            - visit_result["normalized_store_margin_root"]
                        ),
                        **{
                            key: record[key]
                            for key in (
                                "player",
                                "phase",
                                "source_domain",
                                "legal_move_count",
                                "policy_entropy_quartile",
                            )
                        },
                    }
                )
    return output


def _causal(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    margins = [row["normalized_store_margin_delta"] for row in rows]
    outcomes = [row["outcome_delta"] for row in rows]
    return {
        "n": len(rows),
        "outcome_delta": _bootstrap(outcomes, seed=seed),
        "store_margin_delta": _bootstrap(
            [row["store_margin_delta"] for row in rows], seed=seed
        ),
        "normalized_store_margin_delta": _bootstrap(margins, seed=seed),
        "q_better_fraction": float(np.mean(np.asarray(margins) > 0)) if rows else 0.0,
        "visit_better_fraction": float(np.mean(np.asarray(margins) < 0))
        if rows
        else 0.0,
        "tied_fraction": float(np.mean(np.asarray(margins) == 0)) if rows else 0.0,
    }


def paired_differences(completed: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(row["state_hash"], row["continuation_budget"])].append(row)
    result = {}
    for teacher in TEACHERS:
        result[f"D{teacher}"] = {}
        for continuation in CONTINUATION_BUDGETS:
            groups = [
                rows
                for (_state, budget), rows in grouped.items()
                if budget == continuation
            ]
            concordance = [
                _state_metric(rows, f"q_D{teacher}", "normalized_store_margin_root")[
                    "concordance"
                ]
                - _state_metric(
                    rows, f"visit_share_D{teacher}", "normalized_store_margin_root"
                )["concordance"]
                for rows in groups
            ]
            regrets = [
                float(
                    max(row["normalized_store_margin_root"] for row in rows)
                    - _quality_row(
                        {
                            (r["state_hash"], r["continuation_budget"], r["move"]): r
                            for r in rows
                        },
                        rows[0]["state_hash"],
                        continuation,
                        max(
                            rows,
                            key=lambda r: (r[f"visit_share_D{teacher}"], -r["move"]),
                        )["move"],
                    )["normalized_store_margin_root"]
                )
                - float(
                    max(row["normalized_store_margin_root"] for row in rows)
                    - _quality_row(
                        {
                            (r["state_hash"], r["continuation_budget"], r["move"]): r
                            for r in rows
                        },
                        rows[0]["state_hash"],
                        continuation,
                        max(rows, key=lambda r: (r[f"q_D{teacher}"], -r["move"]))[
                            "move"
                        ],
                    )["normalized_store_margin_root"]
                )
                for rows in groups
            ]
            result[f"D{teacher}"][str(continuation)] = {
                "q_minus_visit_concordance": _bootstrap(
                    concordance,
                    seed=stable_seed(seed, teacher, continuation, "concordance"),
                ),
                "visit_minus_q_regret": _bootstrap(
                    regrets, seed=stable_seed(seed, teacher, continuation, "regret")
                ),
            }
    return result


def classify(
    direct: dict[str, Any], paired: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[str], str]:
    d768 = direct["D768"]
    confirmed = (
        len({row["state_hash"] for row in rows if row["teacher_budget"] == 768}) >= 64
        and all(
            d768[str(b)]["normalized_store_margin_delta"]["mean"] > 0
            and d768[str(b)]["normalized_store_margin_delta"]["lower"] >= 0
            for b in CONTINUATION_BUDGETS
        )
        and all(
            paired["D768"][str(b)]["q_minus_visit_concordance"]["lower"] > 0
            and paired["D768"][str(b)]["visit_minus_q_regret"]["lower"] > 0
            for b in CONTINUATION_BUDGETS
        )
    )
    labels = (
        ["visit_allocation_bottleneck_confirmed"]
        if confirmed
        else ["visit_allocation_not_primary"]
    )
    harmful = [
        row
        for row in rows
        if row["teacher_budget"] == 768 and row["normalized_store_margin_delta"] > 0
    ]
    prior_fraction = (
        float(np.mean([row["prior_advantage_for_visit"] > 0 for row in harmful]))
        if harmful
        else 0.0
    )
    favor = _causal(
        [row for row in harmful if row["prior_advantage_for_visit"] > 0], seed=1
    )["normalized_store_margin_delta"]["mean"]
    other = _causal(
        [row for row in harmful if row["prior_advantage_for_visit"] <= 0], seed=2
    )["normalized_store_margin_delta"]["mean"]
    if confirmed and prior_fraction >= 0.70 and favor > other:
        labels.append("policy_prior_driven_visit_misallocation")
    if confirmed:
        labels.append("q_greedy_better_but_not_visit_target_ready")
    action = (
        "Retire the PR #180 visit-allocation label and proceed to the stronger-teacher decision."
        if not confirmed
        else "Run a no-training prior-pressure audit before changing c_puct or policy targets."
    )
    return labels, action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr180-workdir", default=str(DEFAULT_PR180_WORKDIR))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def report(summary: dict[str, Any]) -> str:
    sections = [
        ("PR #180 Reproduction", summary["pr180_reproduction"]),
        ("Q/Visit Disagreement Counts", summary["disagreement_counts"]),
        ("Missing-Q Forced Interventions", summary["missing_q_interventions"]),
        ("Direct Paired Causal Results", summary["direct_causal_results"]),
        ("Paired Concordance And Regret Differences", summary["paired_differences"]),
        ("Prior-Pressure Attribution", summary["prior_pressure"]),
        ("Root PUCT Pressure Diagnostic", summary["root_puct_pressure"]),
        ("Search-Budget Trend", summary["search_budget_trend"]),
        ("Phase, Player, Entropy, And Domain Slices", summary["slices"]),
    ]
    lines = [
        "# PUCT Visit Allocation Causal Audit",
        "",
        f"- Classification: `{', '.join(summary['classifications'])}`",
        f"- Next action: {summary['next_action']}",
    ]
    for title, value in sections:
        lines += [
            "",
            f"## {title}",
            "",
            "```json",
            json.dumps(value, indent=2, sort_keys=True),
            "```",
        ]
    lines += [
        "",
        "## Exact Classification Criteria",
        "",
        "D768 requires >=64 disagreements, positive direct Q-minus-visit margin with lower 95% CI >=0 at both continuations, and positive paired concordance and regret-difference lower CIs.",
        "",
        "Per-state traces remain in the workdir. No model training, replay generation, or runtime tuning was performed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    manifest, teachers, moves, forced, hashes = load_pr180(Path(args.pr180_workdir))
    weights = Path(args.current) / "weights.json"
    _require(sha256_file(weights) == EXPECTED_CURRENT_SHA256, "current weights hash")
    checkpoint = materialize_weights_json_checkpoint(
        weights_path=weights, out_path=workdir / "current.npz"
    )
    tasks = missing_q_tasks(teachers, forced, seed=args.seed)
    missing_path = workdir / "missing_q_move_interventions.jsonl"
    missing = (
        read_jsonl(missing_path)
        if missing_path.is_file()
        else run_missing(
            tasks, checkpoint=checkpoint, workers=args.workers, teachers=teachers
        )
    )
    write_jsonl(missing_path, missing)
    completed = forced + missing
    evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    completed_index = _forced_index(completed)
    for record in teachers:
        policy, _ = evaluator.evaluate(KalahGame.from_state(record["state"]))
        prior_move = int(
            max(record["legal_moves"], key=lambda move: (float(policy[move]), -move))
        )
        for budget in TEACHERS:
            required_moves = {
                prior_move,
                _best_move(record["teachers"][f"D{budget}"], field="q_value"),
                _best_move(record["teachers"][f"D{budget}"], field="visits"),
            }
            for continuation in CONTINUATION_BUDGETS:
                _require(
                    all(
                        (record["state_hash"], continuation, move) in completed_index
                        for move in required_moves
                    ),
                    f"incomplete D{budget} required move quality",
                )
    completed_manifest = {
        "pr180_workdir": str(args.pr180_workdir),
        "pr180_artifact_sha256": hashes,
        "missing_intervention_count": len(missing),
        "completed_record_count": len(completed),
        "state_count": 768,
        "required_moves_present": True,
    }
    write_json(workdir / "completed_move_quality_manifest.json", completed_manifest)
    rows = disagreement_rows(teachers, completed, evaluator=evaluator)
    write_jsonl(workdir / "q_visit_disagreement_records.jsonl", rows)
    direct = {
        f"D{teacher}": {
            str(cont): _causal(
                [
                    row
                    for row in rows
                    if row["teacher_budget"] == teacher
                    and row["continuation_budget"] == cont
                ],
                seed=stable_seed(args.seed, teacher, cont),
            )
            for cont in CONTINUATION_BUDGETS
        }
        for teacher in TEACHERS
    }
    paired = paired_differences(completed, seed=args.seed)
    memberships = {
        row["state_hash"]: {
            candidate["teacher_budget"]
            for candidate in rows
            if candidate["state_hash"] == row["state_hash"]
        }
        for row in rows
    }
    counts = {
        f"D{teacher}": {
            "count": len(
                {row["state_hash"] for row in rows if row["teacher_budget"] == teacher}
            ),
            "fraction": len(
                {row["state_hash"] for row in rows if row["teacher_budget"] == teacher}
            )
            / 768,
        }
        for teacher in TEACHERS
    }
    counts["intersections"] = {
        "|".join(f"D{budget}" for budget in sorted(values)): sum(
            candidate == values for candidate in memberships.values()
        )
        for values in {frozenset(value) for value in memberships.values()}
    }
    slices: dict[str, Any] = {
        f"D{teacher}": {
            str(cont): {
                field: {
                    str(value): _causal(
                        [
                            row
                            for row in rows
                            if row["teacher_budget"] == teacher
                            and row["continuation_budget"] == cont
                            and str(row[field]) == str(value)
                        ],
                        seed=stable_seed(args.seed, teacher, cont, field, value),
                    )
                    for value in sorted(
                        {
                            row[field]
                            for row in rows
                            if row["teacher_budget"] == teacher
                        },
                        key=str,
                    )
                }
                for field in (
                    "phase",
                    "player",
                    "policy_entropy_quartile",
                    "source_domain",
                )
            }
            for cont in CONTINUATION_BUDGETS
        }
        for teacher in TEACHERS
    }
    slices["minimum_interpretable_disagreement_states"] = 32
    prior = {
        f"D{teacher}": {
            str(cont): {
                "visit_higher_prior_fraction": float(
                    np.mean([row["prior_advantage_for_visit"] > 0 for row in subset])
                )
                if (
                    subset := [
                        row
                        for row in rows
                        if row["teacher_budget"] == teacher
                        and row["continuation_budget"] == cont
                    ]
                )
                else 0.0,
                "prior_favors_visit": _causal(
                    [row for row in subset if row["prior_advantage_for_visit"] > 0],
                    seed=stable_seed(args.seed, teacher, cont, "favor"),
                ),
                "prior_does_not_favor_visit": _causal(
                    [row for row in subset if row["prior_advantage_for_visit"] <= 0],
                    seed=stable_seed(args.seed, teacher, cont, "other"),
                ),
            }
            for cont in CONTINUATION_BUDGETS
        }
        for teacher in TEACHERS
    }
    pressure = {
        f"D{teacher}": {
            str(cont): {
                "n": len(subset),
                "mean_q_best_q": float(np.mean([row["q_q_best"] for row in subset]))
                if subset
                else 0.0,
                "mean_visit_best_q": float(
                    np.mean([row["q_visit_best"] for row in subset])
                )
                if subset
                else 0.0,
                "mean_q_best_prior": float(
                    np.mean([row["prior_q_best"] for row in subset])
                )
                if subset
                else 0.0,
                "mean_visit_best_prior": float(
                    np.mean([row["prior_visit_best"] for row in subset])
                )
                if subset
                else 0.0,
                "mean_q_best_u": float(np.mean([row["q_u"] for row in subset]))
                if subset
                else 0.0,
                "mean_visit_best_u": float(np.mean([row["visit_u"] for row in subset]))
                if subset
                else 0.0,
                "mean_q_margin": float(
                    np.mean([row["q_advantage_for_q"] for row in subset])
                )
                if subset
                else 0.0,
                "mean_visit_margin": float(
                    np.mean([row["visit_advantage"] for row in subset])
                )
                if subset
                else 0.0,
                "mean_prior_ratio_visit_over_q": float(
                    np.mean(
                        [
                            (row["prior_visit_best"] + 1e-12)
                            / (row["prior_q_best"] + 1e-12)
                            for row in subset
                        ]
                    )
                )
                if subset
                else 0.0,
            }
            for cont in CONTINUATION_BUDGETS
            if (
                subset := [
                    row
                    for row in rows
                    if row["teacher_budget"] == teacher
                    and row["continuation_budget"] == cont
                ]
            )
            is not None
        }
        for teacher in TEACHERS
    }
    labels, next_action = classify(direct, paired, rows)
    summary = {
        "schema": SCHEMA,
        "current_weights_sha256": sha256_file(weights),
        "probe_manifest": manifest,
        "pr180_reproduction": {
            "artifact_sha256": hashes,
            "recomputed_metrics_verified": True,
        },
        "missing_q_interventions": {
            "requested_task_count": len(tasks),
            "completed_count": len(missing),
        },
        "completed_move_quality_manifest": completed_manifest,
        "disagreement_counts": counts,
        "direct_causal_results": direct,
        "paired_differences": paired,
        "prior_pressure": prior,
        "root_puct_pressure": pressure,
        "search_budget_trend": {
            f"D{teacher}": {
                "disagreement_fraction": counts[f"D{teacher}"]["fraction"],
                "paired_1200": paired[f"D{teacher}"]["1200"],
                "causal_1200": direct[f"D{teacher}"]["1200"],
            }
            for teacher in TEACHERS
        },
        "slices": slices,
        "classification": labels[0],
        "classifications": labels,
        "next_action": next_action,
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-puct-visit-allocation-causal-summary.json",
        summary,
    )
    (
        REPO_ROOT / "docs/alphazero-lite-puct-visit-allocation-causal-results.md"
    ).write_text(report(summary), encoding="utf-8")
    print(json.dumps({"classifications": labels, "workdir": str(workdir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
