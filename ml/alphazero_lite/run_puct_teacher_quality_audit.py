#!/usr/bin/env python3
"""Audit whether frozen PUCT search improves the incumbent policy.

This is evaluation-only: it reads exact holdout labels and the current artifact,
never writes replay/training rows, and never calls promotion code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import exact_teacher_labeling as exact
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    _has_immediate_capture,
    _has_immediate_extra_turn,
    policy_from_visits,
)
from ml.alphazero_lite.kalah_rules import KalahGame

BUDGETS = (48, 96, 192, 384, 768, 1200)
SEED = 284001
EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def audit_seed(canonical_state: str, budget: int) -> int:
    digest = hashlib.sha256(f"{SEED}:{budget}:{canonical_state}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def phase_for(row: dict[str, Any]) -> str:
    phase = str(row.get("phase", ""))
    if phase in ("early", "opening"):
        return "opening"
    if phase in ("mid", "midgame"):
        return "midgame"
    return "late"


def normalize_policy(values: list[float] | np.ndarray, legal: list[int]) -> list[float]:
    result = [0.0] * 6
    total = sum(max(0.0, float(values[move])) for move in legal)
    if total <= 0:
        for move in legal:
            result[move] = 1.0 / len(legal)
    else:
        for move in legal:
            result[move] = max(0.0, float(values[move])) / total
    return result


def top_action(policy: list[float], legal: list[int]) -> int:
    return max(legal, key=lambda move: (float(policy[move]), -move))


def optimal_metrics(
    policy: list[float], optimal: list[int], legal: list[int]
) -> dict[str, float | bool]:
    optimal_set = set(optimal)
    top = top_action(policy, legal)
    mass = sum(float(policy[move]) for move in optimal)
    cross_entropy = -sum(
        np.log(max(float(policy[move]), EPSILON)) / len(optimal) for move in optimal
    )
    return {
        "top": top,
        "correct": top in optimal_set,
        "optimal_mass": mass,
        "cross_entropy_uniform_optimal": float(cross_entropy),
    }


def transition(network_correct: bool, search_correct: bool) -> str:
    if not network_correct and search_correct:
        return "wrong_to_correct"
    if network_correct and search_correct:
        return "correct_to_correct"
    if not network_correct:
        return "wrong_to_wrong"
    return "correct_to_wrong"


def actual_target(
    visits: list[float], legal: list[int], move_index: int
) -> list[float]:
    # Current selected self-play config: T=1.1 through ply 11, T=0.15 after.
    # The frozen exact source phase is the only available production-compatible
    temperature = 1.1 if move_index < 12 else 0.15
    base = policy_from_visits(np.asarray(visits, dtype=np.float32), legal, temperature)
    sharpened = [value * value for value in base]
    return normalize_policy(sharpened, legal)


def tactical_metadata(state: dict[str, Any], legal: list[int]) -> dict[str, bool | str]:
    extra = _has_immediate_extra_turn(state, side=int(state["current_player"]))
    capture = _has_immediate_capture(state, side=int(state["current_player"]))
    return {
        "immediate_extra_turn": extra,
        "immediate_capture": capture,
        "tactical_kind": "both"
        if extra and capture
        else "extra"
        if extra
        else "capture"
        if capture
        else "neither",
        "legal_bucket": "2" if len(legal) == 2 else "3-4" if len(legal) <= 4 else "5-6",
    }


def build_cohort(
    sources: list[dict[str, Any]], labels: list[dict[str, Any]], *, source_lane: str
) -> list[dict[str, Any]]:
    """Deterministically select 1,000 opening and 1,000 midgame holdouts."""
    sources_by_id = {str(row["source_id"]): row for row in sources}
    seen: set[str] = set()
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        source = sources_by_id.get(str(label["source_id"]))
        # Opening shard labels preserve canonical state and move index even when
        # their original source shard is no longer materialized.
        if source is None:
            source = {
                "source_id": label["source_id"],
                "canonical_state": label["canonical_state"],
                "state": json.loads(label["canonical_state"]),
                "phase": "early" if int(label.get("move_index", 0)) <= 8 else "mid",
                "move_index": label.get("move_index", 0),
            }
        phase = phase_for(source)
        canonical = str(label["canonical_state"])
        if phase not in ("opening", "midgame") or canonical in seen:
            continue
        seen.add(canonical)
        state = source["state"]
        game = KalahGame.from_state(state)
        legal = game.possible_moves()
        candidates[phase].append(
            {
                "canonical_state": canonical,
                "source_id": str(label["source_id"]),
                "state": state,
                "phase": phase,
                "move_index": int(source["move_index"]),
                "legal_actions": legal,
                "exact_root_margin": int(label["exact_root_margin"]),
                "exact_root_value": float(label["exact_value_training"]),
                "exact_action_values": label["exact_action_margins"],
                "exact_optimal_actions": [
                    int(v) for v in label["exact_optimal_actions"]
                ],
                "training_ineligible": True,
                "source_lane": source_lane,
                **tactical_metadata(state, legal),
            }
        )
    return [row for phase_rows in candidates.values() for row in phase_rows]


def select_cohort(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohort: list[dict[str, Any]] = []
    for phase in ("opening", "midgame"):
        by_canonical = {
            row["canonical_state"]: row for row in candidates if row["phase"] == phase
        }
        selected = [by_canonical[key] for key in sorted(by_canonical)[:1000]]
        if len(selected) != 1000:
            raise ValueError(
                f"need 1000 exact holdout {phase} states, found {len(selected)}"
            )
        cohort.extend(selected)
    return cohort


def write_manifest(
    cohort: list[dict[str, Any]], out: Path, *, source_sha: str, label_sha: str
) -> str:
    rows_sha = hashlib.sha256(canonical_json(cohort).encode()).hexdigest()
    payload = {
        "schema": "puct_teacher_quality_frozen_cohort_v1",
        "selection": "canonical-state sort; first 1000 each from exact-scale holdout",
        "training_ineligible": True,
        "promotion_suite_disjoint": "not applicable: exact-scale holdout source is separate from promotion suites",
        "source_states_sha256": source_sha,
        "exact_holdout_sha256": label_sha,
        "cohort_sha256": rows_sha,
        "phase_counts": dict(Counter(row["phase"] for row in cohort)),
        "rows": cohort,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows_sha


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "exact_set_agreement": sum(row["correct"] for row in rows) / n,
        "single_optimum_top1": sum(row["correct"] for row in rows if row["single"])
        / max(1, sum(row["single"] for row in rows)),
        "optimal_mass": float(np.mean([row["optimal_mass"] for row in rows])),
        "cross_entropy_uniform_optimal": float(
            np.mean([row["cross_entropy_uniform_optimal"] for row in rows])
        ),
        "multi_optimum_states": sum(not row["single"] for row in rows),
        "value_mae": float(
            np.mean([abs(row["value"] - row["exact_value"]) for row in rows])
        ),
        "value_signed_bias": float(
            np.mean([row["value"] - row["exact_value"] for row in rows])
        ),
    }


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PUCT Teacher-Quality Audit",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "## Frozen Inputs",
        "",
        f"- Incumbent weights SHA-256: `{summary['checkpoint_sha256']}`",
        f"- Cohort SHA-256: `{summary['cohort_sha256']}`",
        f"- Exact provenance: `{summary['exact_provenance']}`",
        "",
        "## Production Search Configuration",
        "",
        "```json",
        json.dumps(summary["production_search_config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Policy Results",
        "",
        "| phase | budget | network exact-set | search exact-set | lift | wrong->correct | correct->wrong |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("opening", "midgame"):
        network = summary["network"][phase]["exact_set_agreement"]
        for budget in BUDGETS:
            stat = summary["search"][str(budget)][phase]
            transitions = stat["transitions"]
            lines.append(
                f"| {phase} | {budget} | {network:.3f} | {stat['raw_visits']['exact_set_agreement']:.3f} | {stat['correction_lift']:.3f} | {transitions['wrong_to_correct']} | {transitions['correct_to_wrong']} |"
            )
    lines += [
        "",
        "## Raw Visits Versus Stored Target",
        "",
        "| phase | budget | raw exact-set | target exact-set | raw optimal mass | target optimal mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("opening", "midgame"):
        for budget in BUDGETS:
            stat = summary["search"][str(budget)][phase]
            raw, target = stat["raw_visits"], stat["actual_training_target"]
            lines.append(
                f"| {phase} | {budget} | {raw['exact_set_agreement']:.3f} | {target['exact_set_agreement']:.3f} | {raw['optimal_mass']:.3f} | {target['optimal_mass']:.3f} |"
            )
    lines += [
        "",
        "## Value Error",
        "",
        "| phase | budget | network MAE | PUCT root-Q MAE | MAE improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("opening", "midgame"):
        for budget in BUDGETS:
            raw = summary["search"][str(budget)][phase]["raw_visits"]
            network = summary["network"][phase]
            lines.append(
                f"| {phase} | {budget} | {network['value_mae']:.3f} | {raw['value_mae']:.3f} | {network['value_mae'] - raw['value_mae']:.3f} |"
            )
    lines += [
        "",
        "## Adjacent-Budget Stability",
        "",
        "| pair | top changes | correctness changes | wrong->correct | correct->wrong | TVD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, stat in summary["budget_stability"].items():
        lines.append(
            f"| {pair} | {stat['top_action_changes']:.3f} | {stat['correctness_changes']:.3f} | {stat['wrong_to_correct']:.3f} | {stat['correct_to_wrong']:.3f} | {stat['visit_policy_tvd']:.3f} |"
        )
    lines += [
        "",
        "## Compute",
        "",
        "| budget | wall seconds | states/sec | simulations/sec | x96 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in BUDGETS:
        stat = summary["compute"][str(budget)]
        lines.append(
            f"| {budget} | {stat['wall_seconds']:.1f} | {stat['states_per_second']:.2f} | {stat['simulations_per_second']:.0f} | {stat['relative_compute_vs_96']:.1f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Recommended self-play simulation budget: `{summary['recommended_self_play_simulation_budget']}`",
        f"- Next experiment: `{summary['next_experiment']}`",
        "",
    ]
    return "\n".join(lines)


def run_search(
    evaluator: CheckpointEvaluator, row: dict[str, Any], budget: int
) -> dict[str, Any]:
    game = KalahGame.from_state(row["state"])
    search = PUCT(
        evaluator,
        budget,
        1.25,
        random.Random(audit_seed(row["canonical_state"], budget)),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
    )
    visits, _root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    raw_policy = normalize_policy(visits, row["legal_actions"])
    target = actual_target(visits.tolist(), row["legal_actions"], row["move_index"])
    root = search.root_summary()
    return {
        "raw_policy": raw_policy,
        "target": target,
        "visits": visits.tolist(),
        "root": root,
    }


def classify_result(
    network: dict[str, Any], search: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    raw = optimal_metrics(
        search["raw_policy"], row["exact_optimal_actions"], row["legal_actions"]
    )
    target = optimal_metrics(
        search["target"], row["exact_optimal_actions"], row["legal_actions"]
    )
    return {
        "raw": raw,
        "target": target,
        "transition": transition(bool(network["correct"]), bool(raw["correct"])),
    }


def stability(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, float]:
    paired = zip(left, right, strict=True)
    rows = list(paired)
    n = len(rows)
    return {
        "top_action_changes": sum(
            a["classification"]["raw"]["top"] != b["classification"]["raw"]["top"]
            for a, b in rows
        )
        / n,
        "correctness_changes": sum(
            a["classification"]["raw"]["correct"]
            != b["classification"]["raw"]["correct"]
            for a, b in rows
        )
        / n,
        "wrong_to_correct": sum(
            not a["classification"]["raw"]["correct"]
            and b["classification"]["raw"]["correct"]
            for a, b in rows
        )
        / n,
        "correct_to_wrong": sum(
            a["classification"]["raw"]["correct"]
            and not b["classification"]["raw"]["correct"]
            for a, b in rows
        )
        / n,
        "visit_policy_tvd": float(
            np.mean(
                [
                    sum(
                        abs(x - y)
                        for x, y in zip(
                            a["search"]["raw_policy"],
                            b["search"]["raw_policy"],
                            strict=True,
                        )
                    )
                    / 2
                    for a, b in rows
                ]
            )
        ),
        "optimal_mass_delta": float(
            np.mean(
                [
                    b["classification"]["raw"]["optimal_mass"]
                    - a["classification"]["raw"]["optimal_mass"]
                    for a, b in rows
                ]
            )
        ),
    }


def tactical_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for key in ("tactical_kind", "legal_bucket", "exact_root_value"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            groups[str(row[key])].append(row)
        result[key] = {
            name: {
                "n": len(group),
                "exact_set_agreement": sum(
                    r["classification"]["raw"]["correct"] for r in group
                )
                / len(group),
                "correction_lift": sum(
                    r["classification"]["raw"]["correct"] - r["network"]["correct"]
                    for r in group
                )
                / len(group),
            }
            for name, group in sorted(groups.items())
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-states",
        type=Path,
        default=Path("/tmp/azlite_exact_scale/source_states.jsonl"),
    )
    parser.add_argument(
        "--exact-holdout",
        type=Path,
        default=Path("/tmp/azlite_exact_scale/holdout.jsonl"),
    )
    parser.add_argument(
        "--opening-source-states",
        type=Path,
        default=Path("/tmp/azlite_exact_opening/source_states.jsonl"),
    )
    parser.add_argument(
        "--opening-exact-holdout",
        type=Path,
        default=Path("/tmp/azlite_exact_opening/opening_labeled_holdout.jsonl"),
    )
    parser.add_argument(
        "--opening-shard-dir", type=Path, default=Path("/tmp/azlite_exact_opening")
    )
    parser.add_argument("--current", type=Path, default=Path("model-artifact/current"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/data/alphazero-lite-puct-teacher-quality-cohort.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/data/alphazero-lite-puct-teacher-quality-audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/alphazero-lite-puct-teacher-quality-audit-results.md"),
    )
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    if args.render_only:
        args.report.write_text(
            report_markdown(json.loads(args.out.read_text(encoding="utf-8"))),
            encoding="utf-8",
        )
        return 0
    sources, labels = (
        exact.read_jsonl(args.source_states),
        exact.read_jsonl(args.exact_holdout),
    )
    opening_sources = exact.read_jsonl(args.opening_source_states)
    opening_labels = exact.read_jsonl(args.opening_exact_holdout)
    for path in sorted(args.opening_shard_dir.glob("holdout_*.jsonl")):
        opening_labels.extend(exact.read_jsonl(path))
    cohort = select_cohort(
        build_cohort(sources, labels, source_lane="exact_scale_holdout")
        + build_cohort(
            opening_sources, opening_labels, source_lane="exact_opening_holdout"
        )
    )
    cohort_sha = write_manifest(
        cohort,
        args.manifest,
        source_sha=hashlib.sha256(
            f"{sha256_file(args.source_states)}:{sha256_file(args.opening_source_states)}".encode()
        ).hexdigest(),
        label_sha=hashlib.sha256(
            f"{sha256_file(args.exact_holdout)}:{sha256_file(args.opening_exact_holdout)}:{canonical_json([sha256_file(path) for path in sorted(args.opening_shard_dir.glob('holdout_*.jsonl'))])}".encode()
        ).hexdigest(),
    )
    if args.build_manifest_only:
        return 0
    metadata = json.loads((args.current / "metadata.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="puct_teacher_audit_") as tmp:
        checkpoint = materialize_weights_json_checkpoint(
            weights_path=args.current / "weights.json",
            out_path=Path(tmp) / "incumbent.npz",
        )
        evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
        network_rows, results, failures = [], defaultdict(list), []
        for row in cohort:
            priors, value = evaluator.evaluate(KalahGame.from_state(row["state"]))
            policy = normalize_policy(priors, row["legal_actions"])
            metric = optimal_metrics(
                policy, row["exact_optimal_actions"], row["legal_actions"]
            )
            network = {
                **metric,
                "value": float(value),
                "exact_value": row["exact_root_value"],
                "single": len(row["exact_optimal_actions"]) == 1,
            }
            network_rows.append({**network, "phase": row["phase"]})
            for budget in BUDGETS:
                started_wall, started_cpu = time.monotonic(), time.process_time()
                search = run_search(evaluator, row, budget)
                classified = classify_result(network, search, row)
                record = {
                    **row,
                    "budget": budget,
                    "network": network,
                    "search": search,
                    "classification": classified,
                    "wall_seconds": time.monotonic() - started_wall,
                    "cpu_seconds": time.process_time() - started_cpu,
                }
                results[budget].append(record)
                if classified["transition"] in (
                    "correct_to_wrong",
                    "wrong_to_wrong",
                ) and (
                    classified["transition"] == "correct_to_wrong" or budget == 1200
                ):
                    failures.append(record)
    network_summary = {
        phase: summarize([r for r in network_rows if r["phase"] == phase])
        for phase in ("opening", "midgame")
    }
    search_summary: dict[str, Any] = {}
    for budget, records in results.items():
        phases = {}
        for phase in ("opening", "midgame"):
            subset = [r for r in records if r["phase"] == phase]
            raw_rows = [
                {
                    **r["classification"]["raw"],
                    "value": r["search"]["root"]["root_q_value"],
                    "exact_value": r["exact_root_value"],
                    "single": len(r["exact_optimal_actions"]) == 1,
                }
                for r in subset
            ]
            target_rows = [
                {
                    **r["classification"]["target"],
                    "value": 0.0,
                    "exact_value": 0.0,
                    "single": len(r["exact_optimal_actions"]) == 1,
                }
                for r in subset
            ]
            counts = Counter(r["classification"]["transition"] for r in subset)
            phases[phase] = {
                "raw_visits": summarize(raw_rows),
                "actual_training_target": summarize(target_rows),
                "transitions": dict(counts),
                "correction_lift": summarize(raw_rows)["exact_set_agreement"]
                - network_summary[phase]["exact_set_agreement"],
                "wall_seconds": sum(r["wall_seconds"] for r in subset),
                "cpu_seconds": sum(r["cpu_seconds"] for r in subset),
            }
        search_summary[str(budget)] = phases
    best = {
        phase: max(
            search_summary[str(b)][phase]["raw_visits"]["exact_set_agreement"]
            for b in BUDGETS
        )
        for phase in ("opening", "midgame")
    }
    recommended = next(
        (
            b
            for b in BUDGETS
            if all(
                search_summary[str(b)][p]["raw_visits"]["exact_set_agreement"]
                >= best[p] - 0.01
                for p in best
            )
        ),
        1200,
    )
    high = search_summary["1200"]
    failure = any(
        high[p]["correction_lift"] <= 0
        or high[p]["transitions"].get("correct_to_wrong", 0)
        >= high[p]["transitions"].get("wrong_to_correct", 0)
        for p in high
    )
    monotonic_high = all(
        search_summary[str(right)][phase]["raw_visits"]["exact_set_agreement"]
        >= search_summary[str(left)][phase]["raw_visits"]["exact_set_agreement"]
        for phase in ("opening", "midgame")
        for left, right in zip(BUDGETS, BUDGETS[1:])
    )
    # A sub-five-point correction lift at the production 96-simulation region
    # is weak relative to the continued multi-point high-budget improvement.
    weak_current = any(
        search_summary["96"][phase]["correction_lift"] < 0.05
        for phase in ("opening", "midgame")
    )
    classification = (
        "mcts_teacher_quality_search_failure"
        if failure
        else "mcts_teacher_quality_budget_limited"
        if weak_current and monotonic_high
        else "mcts_teacher_quality_sufficient"
    )
    next_experiment = (
        "diagnose PUCT prior/Q/value interaction"
        if classification == "mcts_teacher_quality_search_failure"
        else "controlled phase-specific self-play simulation-budget ablation"
        if classification == "mcts_teacher_quality_budget_limited"
        else "replay/state-distribution quality"
    )
    summary = {
        "schema": "puct_teacher_quality_audit_v1",
        "checkpoint_sha256": sha256_file(args.current / "weights.json"),
        "checkpoint_version": metadata["version"],
        "cohort_sha256": cohort_sha,
        "cohort_phase_counts": dict(Counter(r["phase"] for r in cohort)),
        "exact_provenance": "native_hybrid_exact pr281_production_v1; exact holdout lanes",
        "production_search_config": {
            "input_encoding": "kalah_v3",
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "normalize_values": False,
            "value_transform": None,
            "value_trust_schedule": "disabled",
            "root_policy_mode": "visit_count",
            "tactical_root_bias": 0.0,
            "policy_target_mode": "sharpened",
            "policy_target_noise_mode": "denoised",
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.3,
            "tree_reuse": True,
            "normal_simulations": 192,
            "opening_min_simulations": 384,
        },
        "network": network_summary,
        "search": search_summary,
        "budget_stability": {
            f"{left}->{right}": stability(results[left], results[right])
            for left, right in zip(BUDGETS, BUDGETS[1:])
        },
        "tactical_stratification": {
            str(budget): tactical_summary(records)
            for budget, records in results.items()
        },
        "compute": {
            str(budget): {
                "states_processed": len(records),
                "wall_seconds": sum(r["wall_seconds"] for r in records),
                "cpu_seconds": sum(r["cpu_seconds"] for r in records),
                "states_per_second": len(records)
                / max(EPSILON, sum(r["wall_seconds"] for r in records)),
                "simulations_per_second": budget
                * len(records)
                / max(EPSILON, sum(r["wall_seconds"] for r in records)),
                "relative_compute_vs_96": budget / 96,
            }
            for budget, records in results.items()
        },
        "recommended_self_play_simulation_budget": recommended,
        "classification": classification,
        "next_experiment": next_experiment,
        "failure_telemetry_count": len(failures),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out.parent / "alphazero-lite-puct-teacher-quality-failures.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in failures), encoding="utf-8"
    )
    args.report.write_text(report_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
