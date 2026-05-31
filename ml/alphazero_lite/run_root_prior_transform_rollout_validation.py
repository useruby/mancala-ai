#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ml.alphazero_lite.run_capture_002_root_prior_transform_followup import (
    ROW_IDS as LOCAL_GUARD_ROW_IDS,
    C_PUCT,
    SEED as LOCAL_GUARD_SEED,
    SIMULATION_BUDGETS,
    metric_for_move,
    visit_share_by_move,
)
from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    build_row_views,
    probe_artifact_position,
    validated_diagnostic_state,
)
from ml.alphazero_lite.root_prior_transforms import build_root_prior_override
from ml.alphazero_lite.run_capture_002_003_search_prior_control_experiment import (
    SEARCH_CONTROL_OVERRIDES,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    build_probe_row,
    load_json,
    repo_root,
    row_map_from_reference,
    write_json,
)


DEFAULT_OUT_ROOT = "/tmp/azlite_root_prior_transform_rollout_validation"
DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"
DEFAULT_CANDIDATE_PATH = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_TRANSFORM = "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5"
DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_VALIDATION_PATH = "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
DEFAULT_SEEDS = "11,23,37"
SCHEMA = "azlite_root_prior_transform_rollout_validation_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--transform-name", default=DEFAULT_TRANSFORM)
    parser.add_argument("--arena-games", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--fair-budget-random-opening-plies", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    return seeds


def artifact_label(path: Path) -> str:
    if path.name == "current":
        return "current"
    return path.name


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> dict:
    rendered = shlex.join(command)
    if dry_run:
        return {"command": command, "rendered_command": rendered, "executed": False}
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "command": command,
        "rendered_command": rendered,
        "executed": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def arena_matrix_rows(
    *, current_path: Path, candidate_path: Path, transform_name: str
) -> list[dict]:
    candidate_label = artifact_label(candidate_path)
    current_label = artifact_label(current_path)
    return [
        {
            "comparison_id": "A1_candidate_vs_current_baseline",
            "challenger_artifact": candidate_label,
            "current_artifact": current_label,
            "challenger_path": candidate_path,
            "current_path": current_path,
            "challenger_transform": None,
            "current_transform": None,
        },
        {
            "comparison_id": "A2_candidate_transform_vs_current_baseline",
            "challenger_artifact": candidate_label,
            "current_artifact": current_label,
            "challenger_path": candidate_path,
            "current_path": current_path,
            "challenger_transform": transform_name,
            "current_transform": None,
        },
        {
            "comparison_id": "A3_current_transform_vs_current_baseline",
            "challenger_artifact": current_label,
            "current_artifact": current_label,
            "challenger_path": current_path,
            "current_path": current_path,
            "challenger_transform": transform_name,
            "current_transform": None,
        },
        {
            "comparison_id": "A4_candidate_transform_vs_current_transform",
            "challenger_artifact": candidate_label,
            "current_artifact": current_label,
            "challenger_path": candidate_path,
            "current_path": current_path,
            "challenger_transform": transform_name,
            "current_transform": transform_name,
        },
    ]


def arena_budget_rows() -> list[dict]:
    return [
        {
            "budget_id": "equal_low",
            "challenger_simulations": 256,
            "current_simulations": 256,
        },
        {
            "budget_id": "equal_high",
            "challenger_simulations": 640,
            "current_simulations": 640,
        },
        {
            "budget_id": "superhuman_style",
            "challenger_simulations": 640,
            "current_simulations": 256,
        },
    ]


def arena_report_row(
    *, row: dict, budget: dict, seed: int, report: dict, report_path: Path
) -> dict:
    interval = report.get("confidence_interval_95") or {}
    return {
        "comparison_id": row["comparison_id"],
        "challenger_artifact": row["challenger_artifact"],
        "current_artifact": row["current_artifact"],
        "challenger_transform": row["challenger_transform"],
        "current_transform": row["current_transform"],
        "challenger_sims": budget["challenger_simulations"],
        "current_sims": budget["current_simulations"],
        "seed": seed,
        "games": int(report.get("games_played", 0)),
        "score": report.get("score"),
        "wins": int(report.get("wins", 0)),
        "losses": int(report.get("losses", 0)),
        "draws": int(report.get("draws", 0)),
        "ci_low": interval.get("lower"),
        "ci_high": interval.get("upper"),
        "unstable_decision": report.get("unstable_decision"),
        "report_path": str(report_path),
        "root_prior_telemetry": report.get("root_prior_telemetry"),
    }


def hard_state_report_row(
    *,
    artifact: str,
    transform: str | None,
    simulations: int,
    report: dict,
    report_path: Path,
) -> dict:
    telemetry = report.get("root_prior_transform_telemetry") or {}
    return {
        "artifact": artifact,
        "transform": transform,
        "simulations": simulations,
        "positions": report.get("position_count"),
        "average_regret": report.get("average_regret"),
        "blunder_rate": (report.get("overall") or {}).get("blunder_rate"),
        "value_calibration_mae": report.get("value_calibration_mae"),
        "activation_count": telemetry.get("activation_count"),
        "activation_rate": telemetry.get("activation_rate"),
        "average_mass_shift": telemetry.get("average_mass_shift"),
        "report_path": str(report_path),
        "buckets": report.get("buckets"),
        "root_prior_transform_telemetry": telemetry,
    }


def collect_local_guard_results(
    *,
    artifact_path: Path,
    artifact_name: str,
    transform_name: str | None,
    reference_rows: dict[str, dict],
) -> list[dict]:
    arena = __import__("ml.alphazero_lite.arena", fromlist=["ArtifactEvaluator"])
    evaluator = arena.ArtifactEvaluator(artifact_path)
    override = (
        None if transform_name is None else build_root_prior_override(transform_name)
    )
    results = []
    for row_id in LOCAL_GUARD_ROW_IDS:
        reference_row = reference_rows[row_id]
        probe_row = build_probe_row(reference_row)
        state = validated_diagnostic_state(row=probe_row)
        for simulations in SIMULATION_BUDGETS:
            probe_summary = probe_artifact_position(
                artifact_path=str(artifact_path),
                evaluator=evaluator,
                state=state,
                simulations=simulations,
                seed=LOCAL_GUARD_SEED,
                c_puct=C_PUCT,
                search_options=dict(SEARCH_CONTROL_OVERRIDES),
                ablation_mode="full",
                root_prior_override=override,
            )
            row_view = build_row_views(row=probe_row, probe_summary=probe_summary)
            visit_share = visit_share_by_move(row_view.get("search_view") or {})
            selected_move = row_view["search_view"]["searched_selected_move"]
            reference_move = int(probe_row["reference_move"])
            selected_is_reference = selected_move == reference_move
            note_parts: list[str] = []
            passes = selected_is_reference
            if row_id == "capture_available-005":
                passes = True
                note_parts.append("reported_only_not_hard_fail")
            elif row_id in {"capture_available-002", "capture_available-003"}:
                passes = selected_is_reference
            elif row_id in {
                "capture_available-006",
                "capture_available-007",
                "capture_available-008",
            }:
                passes = selected_is_reference
            if not selected_is_reference:
                note_parts.append("selected_non_reference")
            results.append(
                {
                    "artifact": artifact_name,
                    "transform": transform_name,
                    "row_id": row_id,
                    "simulations": simulations,
                    "selected_move": selected_move,
                    "reference_move": reference_move,
                    "selected_is_reference": selected_is_reference,
                    "reference_visit_share": metric_for_move(
                        visit_share, reference_move
                    ),
                    "pass": passes,
                    "notes": ",".join(note_parts) if note_parts else "ok",
                }
            )
    return results


def classify(summary: dict) -> tuple[str, str]:
    local_guard_rows = summary["local_guard_results"]
    arena_rows = summary["arena_results"]
    hard_rows = summary["hard_state_results"]
    guard_fail = any(
        row["row_id"] != "capture_available-005" and not bool(row["pass"])
        for row in local_guard_rows
    )
    candidate_equal_budget_scores = [
        float(row["score"])
        for row in arena_rows
        if row["comparison_id"] == "A2_candidate_transform_vs_current_baseline"
        and row["challenger_sims"] == row["current_sims"]
    ]
    candidate_superhuman_scores = [
        float(row["score"])
        for row in arena_rows
        if row["comparison_id"] == "A2_candidate_transform_vs_current_baseline"
        and row["challenger_sims"] == 640
        and row["current_sims"] == 256
    ]
    current_transform_scores = [
        float(row["score"])
        for row in arena_rows
        if row["comparison_id"] == "A3_current_transform_vs_current_baseline"
    ]
    hard_index = {
        (row["artifact"], row["transform"], row["simulations"]): row
        for row in hard_rows
    }
    hard_worsened = False
    unexpected_high_activation = False
    for simulations in (384, 1200):
        baseline = hard_index.get(("guarded-w2", None, simulations))
        transformed = hard_index.get(
            ("guarded-w2", summary["transform_name"], simulations)
        )
        if baseline and transformed:
            if (
                float(transformed["average_regret"])
                > float(baseline["average_regret"]) + 0.01
            ):
                hard_worsened = True
            if (
                float(transformed["blunder_rate"])
                > float(baseline["blunder_rate"]) + 0.01
            ):
                hard_worsened = True
            activation_rate = transformed.get("activation_rate")
            if activation_rate is not None and float(activation_rate) > 0.1:
                unexpected_high_activation = True
    if hard_worsened or unexpected_high_activation:
        return (
            "unsafe_transform",
            "narrow the activation condition further or abandon search-time transform.",
        )
    if guard_fail and not arena_rows:
        return (
            "local_patch_no_game_strength",
            "abandon this transform for deployment; move to input encoding / target-generation audit.",
        )
    if candidate_superhuman_scores and not candidate_equal_budget_scores:
        return (
            "budget_advantage_not_transform_evidence",
            "do not promote; rerun fair-budget or improve evaluation design.",
        )
    if candidate_superhuman_scores and candidate_equal_budget_scores:
        if (
            max(candidate_superhuman_scores) > 0.55
            and max(candidate_equal_budget_scores) <= 0.55
        ):
            return (
                "budget_advantage_not_transform_evidence",
                "do not promote; rerun fair-budget or improve evaluation design.",
            )
    if current_transform_scores and candidate_equal_budget_scores:
        if (
            abs(max(current_transform_scores) - max(candidate_equal_budget_scores))
            <= 0.05
        ):
            return (
                "search_patch_not_model_improvement",
                "evaluate transform as an optional search mode on current, not as a guarded-w2 model promotion.",
            )
    if (
        candidate_equal_budget_scores
        and max(candidate_equal_budget_scores) > 0.55
        and not guard_fail
        and not hard_worsened
    ):
        return (
            "transform_candidate_validated",
            "run a larger 400+ game promotion-style PUCT arena confirmation with strict CI.",
        )
    max_activation_rate = max(
        (
            float(row["activation_rate"])
            for row in hard_rows
            if row.get("activation_rate") is not None
        ),
        default=0.0,
    )
    if max_activation_rate < 0.01 and any(
        float(row["score"]) > 0.55 for row in arena_rows
    ):
        return (
            "likely_evaluation_artifact",
            "rerun with more games and inspect activation logs before making claims.",
        )
    if not guard_fail:
        return (
            "local_patch_no_game_strength",
            "abandon this transform for deployment; move to input encoding / target-generation audit.",
        )
    return (
        "local_patch_no_game_strength",
        "abandon this transform for deployment; move to input encoding / target-generation audit.",
    )


def build_markdown(summary: dict) -> str:
    lines = [
        "# AlphaZero-lite Root-Prior Transform Rollout Validation Results",
        "",
        "## Context",
        "",
        "This run evaluates whether the narrow root-prior transform is a robust PUCT search-time improvement or only a local diagnostic patch.",
        "",
        f"- transform: `{summary['transform_name']}`",
        f"- current artifact: `{summary['current_path']}`",
        f"- candidate artifact: `{summary['candidate_path']}`",
        f"- output summary: `{summary['summary_path']}`",
        f"- arena supports side-specific transforms: `{str(summary['arena_supports_side_specific_transforms']).lower()}`",
        "",
        "## Matrix Definition",
        "",
        "- Matrix A comparisons: challenger/current baseline, challenger transform vs current baseline, current transform vs current baseline, challenger transform vs current transform.",
        "- Matrix B budgets: equal low 256/256, equal high 640/640, superhuman-style 640/256.",
        f"- fair-budget design improvement: paired random opening prefixes with `{summary['fair_budget_random_opening_plies']}` plies on equal-budget rows.",
        f"- seeds: `{', '.join(str(seed) for seed in summary['seeds'])}`",
        "",
        "## Local Guard Results",
        "",
        "| artifact | transform | row_id | simulations | selected_move | reference_move | selected_is_reference | reference_visit_share | pass | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["local_guard_results"]:
        lines.append(
            "| {artifact} | {transform} | {row_id} | {simulations} | {selected_move} | {reference_move} | {selected_is_reference} | {reference_visit_share} | {pass_value} | {notes} |".format(
                artifact=row["artifact"],
                transform=row["transform"],
                row_id=row["row_id"],
                simulations=row["simulations"],
                selected_move=row["selected_move"],
                reference_move=row["reference_move"],
                selected_is_reference=str(row["selected_is_reference"]).lower(),
                reference_visit_share=row["reference_visit_share"],
                pass_value=str(row["pass"]).lower(),
                notes=row["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Arena Matrix Results",
            "",
            "| comparison_id | challenger_artifact | current_artifact | challenger_transform | current_transform | challenger_sims | current_sims | seed | games | score | wins | losses | draws | ci_low | ci_high | unstable_decision | report_path |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["arena_results"]:
        lines.append(
            "| {comparison_id} | {challenger_artifact} | {current_artifact} | {challenger_transform} | {current_transform} | {challenger_sims} | {current_sims} | {seed} | {games} | {score} | {wins} | {losses} | {draws} | {ci_low} | {ci_high} | {unstable_decision} | {report_path} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## PUCT Hard-State Validation Results",
            "",
            "| artifact | transform | simulations | positions | average_regret | blunder_rate | value_calibration_mae | activation_count | activation_rate | average_mass_shift | report_path |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["hard_state_results"]:
        lines.append(
            "| {artifact} | {transform} | {simulations} | {positions} | {average_regret} | {blunder_rate} | {value_calibration_mae} | {activation_count} | {activation_rate} | {average_mass_shift} | {report_path} |".format(
                **row
            )
        )
    lines.extend(["", "## Transform Activation Telemetry", ""])
    for key, telemetry in summary["activation_telemetry"].items():
        lines.append(f"- `{key}`: `{telemetry}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- classification: `{summary['classification']}`",
            f"- interpretation: {summary['interpretation']}",
            "",
            "## Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out_root = resolve_path(root, args.out_root)
    current_path = resolve_path(root, args.current_path)
    candidate_path = resolve_path(root, args.candidate_path)
    reference_artifact_path = Path(DEFAULT_REFERENCE_ARTIFACT)
    validation_path = resolve_path(root, DEFAULT_VALIDATION_PATH)
    summary_path = out_root / "root_prior_transform_rollout_validation_summary.json"
    report_path = (
        root / "docs/alphazero-lite-root-prior-transform-rollout-validation-results.md"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    reference_rows = row_map_from_reference(load_json(reference_artifact_path))
    arena_rows = arena_matrix_rows(
        current_path=current_path,
        candidate_path=candidate_path,
        transform_name=args.transform_name,
    )
    budget_rows = arena_budget_rows()

    arena_results = []
    arena_plans = []
    for row in arena_rows:
        for budget in budget_rows:
            for seed in seeds:
                report_file = out_root / (
                    f"arena_{row['comparison_id']}_{budget['budget_id']}_seed{seed}.json"
                )
                command = [
                    python_bin(root),
                    "-m",
                    "ml.alphazero_lite.arena",
                    "--challenger",
                    str(row["challenger_path"]),
                    "--current",
                    str(row["current_path"]),
                    "--games",
                    str(args.arena_games),
                    "--challenger-simulations",
                    str(budget["challenger_simulations"]),
                    "--current-simulations",
                    str(budget["current_simulations"]),
                    "--workers",
                    str(args.workers),
                    "--seed",
                    str(seed),
                    "--min-score",
                    "0.55",
                    "--out",
                    str(report_file),
                    "--fpu-mode",
                    "parent_q",
                    "--reuse-subtree",
                    "--normalize-values",
                    "--root-policy-mode",
                    "deterministic",
                    "--tactical-root-bias",
                    "0.1",
                ]
                if budget["challenger_simulations"] == budget["current_simulations"]:
                    command.extend(
                        [
                            "--random-opening-plies",
                            str(args.fair_budget_random_opening_plies),
                        ]
                    )
                if row["challenger_transform"] is not None:
                    command.extend(
                        [
                            "--challenger-root-prior-transform",
                            row["challenger_transform"],
                        ]
                    )
                if row["current_transform"] is not None:
                    command.extend(
                        [
                            "--current-root-prior-transform",
                            row["current_transform"],
                        ]
                    )
                plan = run_command(command, cwd=root, dry_run=args.dry_run)
                plan["report_path"] = str(report_file)
                arena_plans.append(plan)
                if not args.dry_run:
                    arena_results.append(
                        arena_report_row(
                            row=row,
                            budget=budget,
                            seed=seed,
                            report=load_json(report_file),
                            report_path=report_file,
                        )
                    )

    local_guard_rows = []
    if not args.dry_run:
        for artifact_name, artifact_path in (
            ("current", current_path),
            ("guarded-w2", candidate_path),
        ):
            local_guard_rows.extend(
                collect_local_guard_results(
                    artifact_path=artifact_path,
                    artifact_name=artifact_name,
                    transform_name=None,
                    reference_rows=reference_rows,
                )
            )
            local_guard_rows.extend(
                collect_local_guard_results(
                    artifact_path=artifact_path,
                    artifact_name=artifact_name,
                    transform_name=args.transform_name,
                    reference_rows=reference_rows,
                )
            )

    hard_state_results = []
    hard_state_plans = []
    for artifact_name, artifact_path in (
        ("current", current_path),
        ("guarded-w2", candidate_path),
    ):
        for transform in (None, args.transform_name):
            for simulations in (384, 1200):
                report_file = out_root / (
                    f"hard_state_{artifact_name}_{transform or 'original_prior'}_{simulations}.json"
                )
                command = [
                    python_bin(root),
                    "-m",
                    "ml.alphazero_lite.hard_state_validation",
                    "--artifact-path",
                    str(artifact_path),
                    "--validation-path",
                    str(validation_path),
                    "--teacher-simulations",
                    str(simulations),
                    "--artifact-simulations",
                    str(simulations),
                    "--c-puct",
                    "1.25",
                    "--seed",
                    "42",
                    "--report-rows",
                    "--out",
                    str(report_file),
                ]
                if transform is not None:
                    command.extend(["--root-prior-transform", transform])
                plan = run_command(command, cwd=root, dry_run=args.dry_run)
                plan["report_path"] = str(report_file)
                hard_state_plans.append(plan)
                if not args.dry_run:
                    hard_state_results.append(
                        hard_state_report_row(
                            artifact=artifact_name,
                            transform=transform,
                            simulations=simulations,
                            report=load_json(report_file),
                            report_path=report_file,
                        )
                    )

    activation_telemetry = {}
    if not args.dry_run:
        for row in hard_state_results:
            activation_telemetry[
                f"{row['artifact']}::{row['transform'] or 'original_prior'}::{row['simulations']}"
            ] = row["root_prior_transform_telemetry"]

    summary = {
        "schema": SCHEMA,
        "transform_name": args.transform_name,
        "current_path": str(current_path),
        "candidate_path": str(candidate_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "out_root": str(out_root),
        "arena_supports_side_specific_transforms": True,
        "fair_budget_random_opening_plies": int(args.fair_budget_random_opening_plies),
        "seeds": seeds,
        "arena_plans": arena_plans,
        "hard_state_plans": hard_state_plans,
        "local_guard_results": local_guard_rows,
        "arena_results": arena_results,
        "hard_state_results": hard_state_results,
        "activation_telemetry": activation_telemetry,
    }
    if args.dry_run:
        summary["classification"] = "dry_run_only"
        summary["interpretation"] = "No evaluations were executed."
        summary["recommended_next_action"] = (
            "run the planned rollout validation matrix."
        )
    else:
        classification, next_action = classify(
            {**summary, "transform_name": args.transform_name}
        )
        summary["classification"] = classification
        summary["interpretation"] = classification.replace("_", " ")
        summary["recommended_next_action"] = next_action

    write_json(summary_path, summary)
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
