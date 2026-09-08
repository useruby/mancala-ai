#!/usr/bin/env python3
"""Equal-budget opening/midgame exact-teacher distribution ablation.

This runner intentionally compares different state distributions.  It does
not reuse the exact-vs-MCTS runner's identical-state invariant, which remains
the safeguard for that separate teacher-target experiment.  It never promotes
or writes to the incumbent artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import exact_teacher_labeling as exact  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file  # noqa: E402
from ml.alphazero_lite.run_exact_teacher_training_ablation import (  # noqa: E402
    export_artifact,
    score_holdout,
    train_lane,
    verify_holdout_disjoint,
)

LANE_COUNTS = {
    "opening_100": (2940, 0),
    "midgame_100": (0, 2940),
    "mixed_50_50": (1470, 1470),
}
SEEDS = (42, 43, 44)
BUDGET_PAIRS = "384:256,768:768,1200:1200,1200:256"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-train", type=Path, required=True)
    parser.add_argument("--midgame-train", type=Path, required=True)
    parser.add_argument("--opening-holdout", type=Path, required=True)
    parser.add_argument("--midgame-holdout", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--current", type=Path, default=Path("model-artifact/current"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--opening-source-states", type=Path)
    parser.add_argument("--midgame-source-states", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-arena", action="store_true")
    return parser.parse_args(argv)


def canonical_sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows, key=lambda row: (str(row["canonical_state"]), str(row["source_id"]))
    )


def verify_unique(rows: list[dict[str, Any]], label: str) -> None:
    keys = [str(row.get("canonical_state", "")) for row in rows]
    if not all(keys):
        raise ValueError(f"{label} has a row without canonical_state")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicate canonical states")


def select_lanes(
    opening_rows: list[dict[str, Any]], midgame_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Build canonical-key-stable lanes with mixed rows nested in pure lanes."""
    verify_unique(opening_rows, "opening input")
    verify_unique(midgame_rows, "midgame input")
    opening_keys = {str(row["canonical_state"]) for row in opening_rows}
    if opening_keys & {str(row["canonical_state"]) for row in midgame_rows}:
        raise ValueError("opening and midgame inputs overlap by canonical state")
    opening = canonical_sorted(opening_rows)
    midgame = canonical_sorted(midgame_rows)
    if len(opening) < 2940 or len(midgame) < 2940:
        raise ValueError("both inputs must contain at least 2,940 unique rows")
    pure_opening = opening[:2940]
    pure_midgame = midgame[:2940]
    lanes = {
        "opening_100": pure_opening,
        "midgame_100": pure_midgame,
        "mixed_50_50": canonical_sorted(pure_opening[:1470] + pure_midgame[:1470]),
    }
    for lane, rows in lanes.items():
        opening_n, midgame_n = LANE_COUNTS[lane]
        if len(rows) != opening_n + midgame_n:
            raise AssertionError(f"{lane} row count is not fixed at 2,940")
        verify_unique(rows, lane)
    return lanes


def suite_canonical_keys(path: Path) -> set[str]:
    from ml.alphazero_lite.forensic_suite import canonical_state_key

    return {canonical_state_key(row["state"]) for row in exact.read_jsonl(path)}


def verify_leakage(
    lanes: dict[str, list[dict[str, Any]]],
    holdouts: dict[str, list[dict[str, Any]]],
    suite: Path,
) -> None:
    suite_keys = suite_canonical_keys(suite)
    holdout_keys = {
        name: {str(row["canonical_state"]) for row in rows}
        for name, rows in holdouts.items()
    }
    for lane, rows in lanes.items():
        keys = {str(row["canonical_state"]) for row in rows}
        if keys & suite_keys:
            raise ValueError(f"{lane} overlaps the evaluation suite")
        for name, keys_holdout in holdout_keys.items():
            if keys & keys_holdout:
                raise ValueError(f"{lane} overlaps {name} holdout")


def provenance(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"teacher": teacher, "teacher_version": version}
        for teacher, version in sorted(
            {
                (str(row.get("teacher", "")), str(row.get("teacher_version", "")))
                for row in rows
            }
        )
    ]


def lane_integrity(
    lanes: dict[str, list[dict[str, Any]]],
    opening_source: Path | None,
    midgame_source: Path | None,
    holdouts: dict[str, list[dict[str, Any]]],
    suite: Path,
) -> dict[str, Any]:
    suite_keys = suite_canonical_keys(suite)
    source_paths = {"opening": opening_source, "midgame": midgame_source}
    source_shas = {
        name: sha256_file(path) if path is not None else None
        for name, path in source_paths.items()
    }
    result: dict[str, Any] = {
        "source_cohort_sha256": source_shas,
        "suite_sha256": sha256_file(suite),
        "lanes": {},
    }
    for lane, rows in lanes.items():
        opening_n, midgame_n = LANE_COUNTS[lane]
        keys = [str(row["canonical_state"]) for row in rows]
        result["lanes"][lane] = {
            "row_count": len(rows),
            "canonical_unique_count": len(set(keys)),
            "exact_duplicate_count": len(keys) - len(set(keys)),
            "composition": {"opening": opening_n, "midgame": midgame_n},
            "suite_canonical_overlap": len(set(keys) & suite_keys),
            "holdout_canonical_overlap": {
                name: len(set(keys) & {str(row["canonical_state"]) for row in holdout})
                for name, holdout in holdouts.items()
            },
            "exact_teacher_provenance": provenance(rows),
        }
    return result


def write_lane_files(
    workdir: Path, lanes: dict[str, list[dict[str, Any]]]
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for lane, rows in lanes.items():
        path = workdir / "datasets" / f"{lane}.jsonl"
        exact.write_jsonl(path, rows)
        paths[lane] = path
    return paths


def gate_pass(budgets: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, bool]]:
    """Apply PR #282's frozen DS gate, not the diagnostic paired effect."""
    names = ("standard", "equal_768", "equal_high", "1200_vs_256")
    thresholds = (0.0, 0.0, 0.0, 0.0)
    checks = {
        name: float(budgets.get(name, {}).get("ds", float("nan"))) >= threshold
        and (name != "standard" or float(budgets[name]["ds"]) > 0.0)
        for name, threshold in zip(names, thresholds)
    }
    return all(checks.values()), checks


def benchmark_results(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for seed_report in report["temperature_reports"][0]["seed_reports"]:
        for candidate in seed_report["candidate_reports"]:
            result[candidate["candidate"]] = candidate["budget_results"]
    return result


def pearson(values: list[float], effects: list[float]) -> float | None:
    if len(values) < 2 or len(set(values)) < 2 or len(set(effects)) < 2:
        return None
    return float(np.corrcoef(values, effects)[0, 1])


def distribution_deltas(result: dict[str, Any]) -> dict[str, Any]:
    """Compare each mixed checkpoint with both pure lanes at each fixed seed."""
    output: dict[str, Any] = {}
    for seed in (f"seed{seed}" for seed in SEEDS):
        mixed = result["lane_results"]["mixed_50_50"]["seeds"][seed].get("arena", {})
        output[seed] = {}
        for pure in ("opening_100", "midgame_100"):
            baseline = result["lane_results"][pure]["seeds"][seed].get("arena", {})
            output[seed][f"mixed_minus_{pure}"] = {
                budget: float(mixed[budget]["paired_candidate_effect"])
                - float(baseline[budget]["paired_candidate_effect"])
                for budget in ("standard", "equal_768", "equal_high", "1200_vs_256")
                if budget in mixed and budget in baseline
            }
    return output


def ranking_consistency(result: dict[str, Any]) -> dict[str, Any]:
    """Record whether lane rankings by paired effect agree across train seeds."""
    output: dict[str, Any] = {}
    for budget in ("standard", "equal_768", "equal_high", "1200_vs_256"):
        rankings = {
            seed: [
                lane
                for lane, _ in sorted(
                    (
                        (
                            lane,
                            float(
                                result["lane_results"][lane]["seeds"][seed]["arena"][
                                    budget
                                ]["paired_candidate_effect"]
                            ),
                        )
                        for lane in LANE_COUNTS
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            for seed in (f"seed{seed}" for seed in SEEDS)
        }
        output[budget] = {
            "rankings_by_seed": rankings,
            "consistent_across_seeds": len({tuple(v) for v in rankings.values()}) == 1,
        }
    return output


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Equal-Budget Exact-Teacher Distribution Ablation",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "## Pre-registered decision",
        "",
        "Continue the exact-teacher program only when `mixed_50_50` passes the unchanged gate on at least 2 of 3 seeds. No checkpoint is promoted.",
        "",
        "## Dataset integrity",
        "",
        f"Opening source cohort SHA: `{summary['dataset_integrity']['source_cohort_sha256']['opening']}`. Midgame source cohort SHA: `{summary['dataset_integrity']['source_cohort_sha256']['midgame']}`.",
        "",
        "| lane | rows | opening | midgame | lane SHA |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for lane, data in summary["dataset_integrity"]["lanes"].items():
        composition = data["composition"]
        lines.append(
            f"| {lane} | {data['row_count']} | {composition['opening']} | {composition['midgame']} | {data['lane_sha256']} |"
        )
    lines.extend(
        [
            "",
            "## Gate pass rate",
            "",
            "| lane | passed seeds | pass rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for lane, result in summary["lane_results"].items():
        lines.append(
            f"| {lane} | {result['gate_pass_count']}/3 | {result['gate_pass_rate']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Arena",
            "",
            "Effects are paired candidate effects; brackets are the benchmark's opening bootstrap 95% CI. `gate_ds` is the unchanged PR #282 gate metric; each cell also reports the disadvantaged-seat score. The paired effect is diagnostic only and does not replace the frozen gate.",
            "",
            "| lane / seed | 384:256 | 768:768 | 1200:1200 | 1200:256 | gate |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for lane, result in summary["lane_results"].items():
        for seed, run in result["seeds"].items():
            effects = run.get("arena", {})
            formatted = []
            for name in ("standard", "equal_768", "equal_high", "1200_vs_256"):
                metric = effects.get(name, {})
                value = metric.get("paired_candidate_effect")
                ci = metric.get("opening_bootstrap_ci", {})
                formatted.append(
                    "n/a"
                    if value is None
                    else f"effect={value:+.3f} [{ci.get('lower_95', float('nan')):+.3f}, {ci.get('upper_95', float('nan')):+.3f}]; gate_ds={metric.get('ds', float('nan')):+.3f}; disadvantaged={metric.get('disadvantaged_seat_score', float('nan')):.3f}"
                )
            lines.append(
                f"| {lane} / {seed} | {' | '.join(formatted)} | {'PASS' if run.get('gate_pass') else 'fail'} |"
            )
    lines.extend(
        [
            "",
            "## Holdouts",
            "",
            "| lane / seed | family | rows | top-in-set | single top-1 | policy CE | value MAE | checkpoint SHA |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for lane, result in summary["lane_results"].items():
        for seed, run in result["seeds"].items():
            for family, holdout in run["holdouts"].items():
                single = holdout["single_top1_agreement"]
                single_text = "n/a" if single is None else f"{single:.3%}"
                lines.append(
                    f"| {lane} / {seed} | {family} | {holdout['n']} | {holdout['top_in_exact_set_rate']:.3%} | {single_text} | {holdout['policy_cross_entropy_vs_uniform_optimal']:.3f} | {holdout['value_mae']:.3f} | {run['checkpoint_sha256']} |"
                )
    lines.extend(
        [
            "",
            "## Secondary interpretation",
            "",
            "| seed | comparison | 384:256 | 768:768 | 1200:1200 | 1200:256 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed, comparisons in summary["secondary_interpretation"][
        "mixed_effect_deltas"
    ].items():
        for comparison, deltas in comparisons.items():
            lines.append(
                f"| {seed} | {comparison} | {deltas.get('standard', float('nan')):+.3f} | {deltas.get('equal_768', float('nan')):+.3f} | {deltas.get('equal_high', float('nan')):+.3f} | {deltas.get('1200_vs_256', float('nan')):+.3f} |"
            )
    lines.extend(
        [
            "",
            f"Opening-holdout top-in-set vs standard paired effect Pearson r: {summary['secondary_interpretation']['opening_holdout_vs_standard_gate_pearson']:.3f}. Midgame equivalent: {summary['secondary_interpretation']['midgame_holdout_vs_standard_gate_pearson']:.3f}.",
            "",
            "Paired-effect lane rankings are consistent across seeds by budget: "
            + ", ".join(
                f"{budget}={info['consistent_across_seeds']}"
                for budget, info in summary["secondary_interpretation"][
                    "ranking_consistency"
                ].items()
            )
            + ".",
        ]
    )
    lines.extend(["", "## Interpretation", ""])
    if summary["classification"] == "exact_distribution_mix_gate_met":
        lines.append(
            "Mixed data passed at least two seeds. Recommend the optional 5,880-row scale confirmation; do not promote these checkpoints."
        )
    elif summary["classification"] == "exact_distribution_mix_gate_not_met":
        lines.append(
            "The mixed recipe did not establish a reliable gate result. Stop exact-teacher promotion work and improve the normal AlphaZero/MCTS self-play teacher."
        )
    else:
        lines.append(
            "The required arena evidence is incomplete; the experiment is inconclusive."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    opening_rows = exact.read_jsonl(args.opening_train)
    midgame_rows = exact.read_jsonl(args.midgame_train)
    holdouts = {
        "opening": exact.read_jsonl(args.opening_holdout),
        "midgame": exact.read_jsonl(args.midgame_holdout),
    }
    lanes = select_lanes(opening_rows, midgame_rows)
    verify_leakage(lanes, holdouts, args.suite)
    for rows in lanes.values():
        for holdout in holdouts.values():
            verify_holdout_disjoint(rows, holdout)
    lane_files = write_lane_files(args.workdir, lanes)
    integrity = lane_integrity(
        lanes,
        args.opening_source_states,
        args.midgame_source_states,
        holdouts,
        args.suite,
    )
    integrity["input_label_sha256"] = {
        "opening": sha256_file(args.opening_train),
        "midgame": sha256_file(args.midgame_train),
        "opening_holdout": sha256_file(args.opening_holdout),
        "midgame_holdout": sha256_file(args.midgame_holdout),
    }
    for lane, path in lane_files.items():
        integrity["lanes"][lane]["lane_sha256"] = sha256_file(path)
    result: dict[str, Any] = {
        "schema": "exact_teacher_distribution_ablation_v1",
        "promotion": {"performed": False, "reason": "ablation_only"},
        "pre_registered_success_rule": "mixed_50_50 gate passes >=2 of 3 seeds",
        "dataset_integrity": integrity,
        "training_config": {
            "model_type": "residual_v3",
            "hidden_sizes": [96, 3],
            "epochs": 46,
            "lr": 1e-3,
            "batch_size": 512,
            "value_loss_weight": 0.5,
            "input_encoding": "kalah_v3",
            "lr_scheduler": "cosine",
            "weight_decay": 0.0,
            "trainable_scope": "all",
            "init_checkpoint": None,
            "seeds": list(SEEDS),
        },
        "lane_results": {},
    }
    for lane, rows in lanes.items():
        lane_result: dict[str, Any] = {"seeds": {}}
        for seed in SEEDS:
            lane_dir = args.workdir / "training" / lane / f"seed{seed}"
            checkpoint = lane_dir / "checkpoint.npz"
            artifact = lane_dir / f"artifact_{lane}_seed{seed}"
            if checkpoint.is_file() and (artifact / "weights.json").is_file():
                trained = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "resumed_checkpoint": True,
                }
            else:
                trained = train_lane(
                    rows,
                    model_type="residual_v3",
                    hidden_sizes=(96, 3),
                    input_encoding="kalah_v3",
                    epochs=46,
                    batch_size=512,
                    lr=1e-3,
                    seed=seed,
                    value_loss_weight=0.5,
                    val_split=0.1,
                    device=torch.device(args.device),
                    lane_dir=lane_dir,
                    init_checkpoint=None,
                    lr_scheduler="cosine",
                    weight_decay=0.0,
                    trainable_scope="all",
                )
                artifact = export_artifact(
                    checkpoint,
                    artifact,
                    model_type="residual_v3",
                    input_encoding="kalah_v3",
                    version=f"exact-distribution-{lane}-seed{seed}",
                )
            trained["artifact"] = str(artifact)
            trained["holdouts"] = {
                name: score_holdout(
                    Path(trained["checkpoint"]), rows_holdout, input_encoding="kalah_v3"
                )
                for name, rows_holdout in holdouts.items()
            }
            lane_result["seeds"][f"seed{seed}"] = trained
        result["lane_results"][lane] = lane_result
    if args.skip_arena:
        arena_by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    else:
        candidates = ",".join(
            run["artifact"]
            for lane in result["lane_results"].values()
            for run in lane["seeds"].values()
        )
        benchmark_dir = args.workdir / "benchmark"
        command = [
            sys.executable,
            str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
            "--workdir",
            str(benchmark_dir),
            "--suite",
            str(args.suite),
            "--current",
            str(args.current),
            "--candidates",
            candidates,
            "--budget-pairs",
            BUDGET_PAIRS,
            "--games-per-opening",
            "2",
            "--seed",
            "42",
            "--workers",
            str(args.workers),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        arena_by_candidate = benchmark_results(
            benchmark_dir / "temperature_benchmark_report.json"
        )
    standard_effects: list[float] = []
    opening_rates: list[float] = []
    midgame_rates: list[float] = []
    for lane, lane_result in result["lane_results"].items():
        passes = 0
        for seed, run in lane_result["seeds"].items():
            run["arena"] = arena_by_candidate.get(Path(run["artifact"]).name, {})
            if run["arena"]:
                run["gate_pass"], run["gate_by_budget"] = gate_pass(run["arena"])
                passes += int(run["gate_pass"])
                standard_effects.append(
                    float(run["arena"]["standard"]["paired_candidate_effect"])
                )
                opening_rates.append(
                    float(run["holdouts"]["opening"]["top_in_exact_set_rate"])
                )
                midgame_rates.append(
                    float(run["holdouts"]["midgame"]["top_in_exact_set_rate"])
                )
            else:
                run["gate_pass"] = False
                run["gate_by_budget"] = {}
        lane_result["gate_pass_count"] = passes
        lane_result["gate_pass_rate"] = passes / len(SEEDS)
    mixed_passes = result["lane_results"]["mixed_50_50"]["gate_pass_count"]
    result["classification"] = (
        "exact_distribution_mix_inconclusive"
        if args.skip_arena
        else (
            "exact_distribution_mix_gate_met"
            if mixed_passes >= 2
            else "exact_distribution_mix_gate_not_met"
        )
    )
    result["secondary_interpretation"] = {
        "mixed_effect_deltas": distribution_deltas(result),
        "ranking_consistency": ranking_consistency(result),
        "opening_holdout_vs_standard_gate_pearson": pearson(
            opening_rates, standard_effects
        ),
        "midgame_holdout_vs_standard_gate_pearson": pearson(
            midgame_rates, standard_effects
        ),
    }
    result["elapsed_seconds"] = round(time.monotonic() - started, 1)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
