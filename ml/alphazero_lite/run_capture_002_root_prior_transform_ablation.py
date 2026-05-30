#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    build_row_views,
    probe_artifact_position,
    validated_diagnostic_state,
)
from ml.alphazero_lite.root_prior_transforms import (
    ALL_TRANSFORM_NAMES,
    ARENA_TRANSFORM_NAMES,
    apply_root_prior_transform,
    build_root_prior_override,
    move_feature_annotations_for,
)
from ml.alphazero_lite.run_capture_002_003_search_prior_control_experiment import (
    DEFAULT_CURRENT_PATH,
    SEARCH_CONTROL_OVERRIDES,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    MATERIAL_DEGRADE_MARGIN,
    build_probe_row,
    load_json,
    repo_root,
    row_map_from_reference,
    write_json,
)


DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_root_prior_transform_ablation"
DEFAULT_CANDIDATE_PATHS = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
SIMULATION_BUDGETS = (64, 128, 384, 1200)
PRIMARY_ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
)
EXPLICIT_GUARD_ROW_IDS = (
    "capture_available-005",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
SEED = 17
C_PUCT = 1.25
SCHEMA = "azlite_capture_002_root_prior_transform_ablation_v1"
ARENA_GAMES = 40
ARENA_WORKERS = 4
ARENA_CHALLENGER_SIMULATIONS = 640
ARENA_CURRENT_SIMULATIONS = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--candidate-paths", default=DEFAULT_CANDIDATE_PATHS)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--arena-games", type=int, default=ARENA_GAMES)
    parser.add_argument("--arena-workers", type=int, default=ARENA_WORKERS)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_artifact_paths(
    root: Path, current_path: str, candidate_paths: str
) -> list[dict[str, str | Path]]:
    resolved: list[dict[str, str | Path]] = [
        {"artifact": "current", "artifact_path": resolve_path(root, current_path)}
    ]
    for item in candidate_paths.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        path = resolve_path(root, stripped)
        resolved.append({"artifact": path.name, "artifact_path": path})
    return resolved


def capture_row_ids(reference_rows: dict[str, dict]) -> list[str]:
    return sorted(
        row_id for row_id in reference_rows if row_id.startswith("capture_available-")
    )


def guard_row_ids(all_capture_row_ids: list[str]) -> list[str]:
    return [
        row_id
        for row_id in all_capture_row_ids
        if row_id not in PRIMARY_ROW_IDS
        and row_id in set(EXPLICIT_GUARD_ROW_IDS + tuple(all_capture_row_ids))
    ]


def wrong_extra_turn_move_for(
    *, row_id: str, legal_moves: list[int], move_features: dict[int, dict]
) -> int | None:
    if (
        row_id == "capture_available-002"
        and 2 in legal_moves
        and bool((move_features.get(2) or {}).get("gives_extra_turn", False))
    ):
        return 2
    return None


def visit_share_by_move(search_view: dict) -> dict[str, float] | None:
    visit_distribution = search_view.get("visit_distribution") or {}
    if not visit_distribution:
        return None
    total = float(sum(float(value) for value in visit_distribution.values()))
    if total <= 0.0:
        return None
    return {
        str(move): round(float(value) / total, 4)
        for move, value in visit_distribution.items()
    }


def metric_for_move(
    distribution: dict[str, float] | None, move: int | None
) -> float | None:
    if distribution is None or move is None:
        return None
    value = distribution.get(str(move))
    return None if value is None else float(value)


def build_result_entry(
    *,
    artifact: str,
    artifact_path: Path,
    row_id: str,
    reference_move: int,
    wrong_extra_turn_move: int | None,
    legal_moves: list[int],
    move_features: dict[int, dict],
    transform_name: str,
    simulations: int,
    row_view: dict,
    original_prior: np.ndarray,
    transformed_prior: np.ndarray,
    transform_telemetry: dict,
) -> dict:
    search_view = row_view.get("search_view") or {}
    value_view = row_view.get("value_view") or {}
    selected_move = search_view.get("searched_selected_move")
    visit_share_distribution = visit_share_by_move(search_view)
    return {
        "artifact": artifact,
        "artifact_path": str(artifact_path),
        "row_id": row_id,
        "transform_name": transform_name,
        "simulations": int(simulations),
        "legal_moves": [int(move) for move in legal_moves],
        "move_features": {str(move): move_features[int(move)] for move in legal_moves},
        "original_prior": {
            str(move): round(float(original_prior[move]), 4) for move in legal_moves
        },
        "transformed_prior": {
            str(move): round(float(transformed_prior[move]), 4) for move in legal_moves
        },
        "selected_move": selected_move,
        "reference_move": reference_move,
        "selected_is_reference": selected_move == reference_move,
        "reference_visit_share": metric_for_move(
            visit_share_distribution, reference_move
        ),
        "wrong_extra_turn_move": wrong_extra_turn_move,
        "wrong_extra_turn_visit_share": metric_for_move(
            visit_share_distribution, wrong_extra_turn_move
        ),
        "selected_minus_reference_q": value_view.get(
            "selected_minus_reference_q_margin"
        ),
        "q_by_move": value_view.get("per_child_q_values"),
        "visit_share_by_move": visit_share_distribution,
        "root_prior_transform_mass_shift": float(transform_telemetry["mass_shift"]),
        "pass_fail_reason": "selected_reference"
        if selected_move == reference_move
        else (
            "selected_wrong_extra_turn"
            if wrong_extra_turn_move is not None
            and selected_move == wrong_extra_turn_move
            else "selected_non_reference"
        ),
    }


def indexed_results(results: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {
        (result["row_id"], result["transform_name"], int(result["simulations"])): result
        for result in results
    }


def evaluate_local_gate(
    *, artifact_results: list[dict], guard_rows: list[str]
) -> dict[str, dict]:
    indexed = indexed_results(artifact_results)
    local_gate: dict[str, dict] = {}
    for transform_name in ALL_TRANSFORM_NAMES:
        failures: list[str] = []
        row_002_384 = indexed[("capture_available-002", transform_name, 384)]
        row_002_1200 = indexed[("capture_available-002", transform_name, 1200)]
        row_003_384 = indexed[("capture_available-003", transform_name, 384)]
        row_003_1200 = indexed[("capture_available-003", transform_name, 1200)]
        row_003_base_384 = indexed[("capture_available-003", "original_prior", 384)]
        row_003_base_1200 = indexed[("capture_available-003", "original_prior", 1200)]

        if row_002_384.get("selected_move") != 4:
            failures.append("row_002_384_reference_not_selected")
        if row_002_1200.get("selected_move") == 2:
            failures.append("row_002_1200_wrong_extra_turn_reappeared")
        if row_003_384.get("selected_move") != 1:
            failures.append("row_003_384_reference_not_selected")
        if row_003_1200.get("selected_move") != 1:
            failures.append("row_003_1200_reference_not_selected")

        for budget, transformed, baseline in (
            (384, row_003_384, row_003_base_384),
            (1200, row_003_1200, row_003_base_1200),
        ):
            transformed_share = transformed.get("reference_visit_share")
            baseline_share = baseline.get("reference_visit_share")
            if (
                transformed_share is not None
                and baseline_share is not None
                and float(transformed_share)
                < float(baseline_share) - MATERIAL_DEGRADE_MARGIN
            ):
                failures.append(
                    f"row_003_{budget}_reference_visit_share_materially_degraded"
                )

        for row_id in guard_rows:
            for budget in (384, 1200):
                baseline = indexed[(row_id, "original_prior", budget)]
                transformed = indexed[(row_id, transform_name, budget)]
                if baseline.get("selected_is_reference") and not transformed.get(
                    "selected_is_reference"
                ):
                    failures.append(f"{row_id}_{budget}_reference_regressed")

        local_gate[transform_name] = {"pass": not failures, "failures": failures}
    return local_gate


def classify_results(*, artifact_summaries: list[dict]) -> tuple[str, str, str]:
    feature_based = set(ALL_TRANSFORM_NAMES) - {"original_prior", "uniform_legal_prior"}
    if any(
        summary["local_gate"][transform_name]["pass"]
        for summary in artifact_summaries
        for transform_name in feature_based
    ):
        return (
            "search_time_prior_calibration_viable",
            "add the best transform as an optional MCTS evaluation mode and run a broader arena/MCTS1200 validation",
            "a rule-conditioned feature-based transform cleared the local gate",
        )
    if any(
        summary["local_gate"]["uniform_legal_prior"]["pass"]
        for summary in artifact_summaries
    ):
        return (
            "generic_prior_overconfidence",
            "policy entropy / prior-calibration training objective",
            "only uniform legal prior cleared the local gate",
        )
    if any(
        summary["results_indexed"][("capture_available-002", name, 384)][
            "selected_move"
        ]
        == 4
        and not summary["local_gate"][name]["pass"]
        for summary in artifact_summaries
        for name in ("extra_turn_damp_050", "extra_turn_damp_025")
    ):
        return (
            "extra_turn_tradeoff",
            "more precise move-feature conditioning; do not deploy damping",
            "extra-turn damping can help row 002 but violates preservation constraints",
        )
    return (
        "non_generalizable_root_correction",
        "teacher/label audit and feature encoding review",
        "no general feature-based transform cleared the local gate",
    )


def choose_best_candidate_transform(
    candidate_summary: dict, guard_rows: list[str]
) -> str | None:
    viable = [
        name
        for name in ARENA_TRANSFORM_NAMES
        if candidate_summary["local_gate"].get(name, {}).get("pass")
    ]
    if not viable:
        return None
    indexed = candidate_summary["results_indexed"]

    def score(transform_name: str) -> tuple[float, float, float]:
        row_002_1200 = indexed[("capture_available-002", transform_name, 1200)]
        row_003_384 = indexed[("capture_available-003", transform_name, 384)]
        row_003_1200 = indexed[("capture_available-003", transform_name, 1200)]
        baseline_003_384 = indexed[("capture_available-003", "original_prior", 384)]
        baseline_003_1200 = indexed[("capture_available-003", "original_prior", 1200)]
        preservation_penalty = max(
            0.0,
            float(baseline_003_384.get("reference_visit_share") or 0.0)
            - float(row_003_384.get("reference_visit_share") or 0.0),
        ) + max(
            0.0,
            float(baseline_003_1200.get("reference_visit_share") or 0.0)
            - float(row_003_1200.get("reference_visit_share") or 0.0),
        )
        family_hits = sum(
            1
            for row_id in guard_rows
            if indexed[(row_id, transform_name, 384)].get("selected_is_reference")
            and indexed[(row_id, transform_name, 1200)].get("selected_is_reference")
        )
        return (
            float(row_002_1200.get("reference_visit_share") or 0.0),
            -preservation_penalty,
            float(family_hits),
        )

    return max(viable, key=score)


def run_arena_evaluation(
    *,
    root: Path,
    output_root: Path,
    candidate_path: Path,
    current_path: Path,
    transform_name: str,
    arena_games: int,
    arena_workers: int,
) -> dict:
    report_path = output_root / f"arena_{candidate_path.name}_{transform_name}.json"
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.arena",
        "--challenger",
        str(candidate_path),
        "--current",
        str(current_path),
        "--games",
        str(arena_games),
        "--challenger-simulations",
        str(ARENA_CHALLENGER_SIMULATIONS),
        "--current-simulations",
        str(ARENA_CURRENT_SIMULATIONS),
        "--workers",
        str(arena_workers),
        "--min-score",
        "0.55",
        "--out",
        str(report_path),
        "--root-prior-transform",
        transform_name,
        "--fpu-mode",
        str(SEARCH_CONTROL_OVERRIDES["fpu_mode"]),
        "--root-policy-mode",
        str(SEARCH_CONTROL_OVERRIDES["root_policy_mode"]),
        "--tactical-root-bias",
        str(SEARCH_CONTROL_OVERRIDES["tactical_root_bias"]),
    ]
    if bool(SEARCH_CONTROL_OVERRIDES["reuse_subtree"]):
        command.append("--reuse-subtree")
    if bool(SEARCH_CONTROL_OVERRIDES["normalize_values"]):
        command.append("--normalize-values")

    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "candidate_artifact": candidate_path.name,
        "challenger_path": str(candidate_path),
        "current_path": str(current_path),
        "transform_name": transform_name,
        "report_path": str(report_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": load_json(report_path),
    }


def build_markdown(summary: dict) -> str:
    lines = [
        "# AlphaZero-lite Capture 002 Root-Prior Transform Ablation Results",
        "",
        "## Context",
        "",
        "This run tests whether a general rule-conditioned root-prior transform can fix `capture_available-002` while preserving `capture_available-003` and nearby opening-capture rows.",
        "",
        f"- classification: `{summary['classification']}`",
        f"- reference artifact: `{summary['reference_artifact_path']}`",
        f"- output summary: `{summary['summary_path']}`",
        "",
        "## Transform definitions",
        "",
        "- `original_prior`: no transform",
        "- `uniform_legal_prior`: uniform over legal root moves",
        "- `extra_turn_damp_050`: multiply extra-turn priors by `0.50`, then renormalize",
        "- `extra_turn_damp_025`: multiply extra-turn priors by `0.25`, then renormalize",
        "- `no_extra_turn_capture_boost_2x`: multiply no-extra-turn capture priors by `2.0`, then renormalize",
        "- `hybrid_damp050_captureboost2x`: combine `extra_turn_damp_050` and `no_extra_turn_capture_boost_2x`",
        "- `prior_temperature_2x`: flatten legal priors with temperature `2.0`",
        "",
        "## Local row-pair results",
        "",
        "| artifact | row_id | transform | simulations | selected_move | reference_move | selected_is_reference | reference_visit_share | wrong_extra_turn_visit_share | selected_minus_reference_q | original_reference_prior | transformed_reference_prior | original_wrong_extra_turn_prior | transformed_wrong_extra_turn_prior | local_gate_pass | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in summary["artifacts"]:
        indexed = artifact["results_indexed"]
        for row_id in PRIMARY_ROW_IDS:
            for transform_name in ALL_TRANSFORM_NAMES:
                for simulations in SIMULATION_BUDGETS:
                    result = indexed[(row_id, transform_name, simulations)]
                    wrong_move = result.get("wrong_extra_turn_move")
                    lines.append(
                        "| {artifact} | {row_id} | {transform} | {simulations} | {selected_move} | {reference_move} | {selected_is_reference} | {reference_visit_share} | {wrong_extra_turn_visit_share} | {selected_minus_reference_q} | {original_reference_prior} | {transformed_reference_prior} | {original_wrong_extra_turn_prior} | {transformed_wrong_extra_turn_prior} | {local_gate_pass} | {notes} |".format(
                            artifact=artifact["artifact"],
                            row_id=row_id,
                            transform=transform_name,
                            simulations=simulations,
                            selected_move=result.get("selected_move"),
                            reference_move=result.get("reference_move"),
                            selected_is_reference=str(
                                result.get("selected_is_reference")
                            ).lower(),
                            reference_visit_share=result.get("reference_visit_share"),
                            wrong_extra_turn_visit_share=result.get(
                                "wrong_extra_turn_visit_share"
                            ),
                            selected_minus_reference_q=result.get(
                                "selected_minus_reference_q"
                            ),
                            original_reference_prior=result["original_prior"].get(
                                str(result.get("reference_move"))
                            ),
                            transformed_reference_prior=result["transformed_prior"].get(
                                str(result.get("reference_move"))
                            ),
                            original_wrong_extra_turn_prior=None
                            if wrong_move is None
                            else result["original_prior"].get(str(wrong_move)),
                            transformed_wrong_extra_turn_prior=None
                            if wrong_move is None
                            else result["transformed_prior"].get(str(wrong_move)),
                            local_gate_pass=str(
                                artifact["local_gate"][transform_name]["pass"]
                            ).lower(),
                            notes=result.get("pass_fail_reason"),
                        )
                    )

    lines.extend(
        [
            "",
            "## Opening-family preservation results",
            "",
            f"Evaluated opening-capture rows: `{', '.join(summary['all_capture_row_ids'])}`.",
            "",
            "| artifact | row_id | transform | simulations | selected_move | reference_move | selected_is_reference | reference_visit_share | wrong_extra_turn_visit_share | selected_minus_reference_q | original_reference_prior | transformed_reference_prior | original_wrong_extra_turn_prior | transformed_wrong_extra_turn_prior | local_gate_pass | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for artifact in summary["artifacts"]:
        indexed = artifact["results_indexed"]
        for row_id in summary["guard_row_ids"]:
            for transform_name in ALL_TRANSFORM_NAMES:
                for simulations in (384, 1200):
                    result = indexed[(row_id, transform_name, simulations)]
                    lines.append(
                        "| {artifact} | {row_id} | {transform} | {simulations} | {selected_move} | {reference_move} | {selected_is_reference} | {reference_visit_share} | {wrong_extra_turn_visit_share} | {selected_minus_reference_q} | {original_reference_prior} | {transformed_reference_prior} | {original_wrong_extra_turn_prior} | {transformed_wrong_extra_turn_prior} | {local_gate_pass} | {notes} |".format(
                            artifact=artifact["artifact"],
                            row_id=row_id,
                            transform=transform_name,
                            simulations=simulations,
                            selected_move=result.get("selected_move"),
                            reference_move=result.get("reference_move"),
                            selected_is_reference=str(
                                result.get("selected_is_reference")
                            ).lower(),
                            reference_visit_share=result.get("reference_visit_share"),
                            wrong_extra_turn_visit_share=result.get(
                                "wrong_extra_turn_visit_share"
                            ),
                            selected_minus_reference_q=result.get(
                                "selected_minus_reference_q"
                            ),
                            original_reference_prior=result["original_prior"].get(
                                str(result.get("reference_move"))
                            ),
                            transformed_reference_prior=result["transformed_prior"].get(
                                str(result.get("reference_move"))
                            ),
                            original_wrong_extra_turn_prior=None,
                            transformed_wrong_extra_turn_prior=None,
                            local_gate_pass=str(
                                artifact["local_gate"][transform_name]["pass"]
                            ).lower(),
                            notes=result.get("pass_fail_reason"),
                        )
                    )

    if summary.get("arena_results"):
        lines.extend(["", "## Optional arena results", ""])
        for arena_result in summary["arena_results"]:
            report = arena_result["report"]
            lines.append(
                f"- `{arena_result['candidate_artifact']}` with `{arena_result['transform_name']}`: score `{report['score']}`, wins `{report['wins']}`, losses `{report['losses']}`, draws `{report['draws']}`"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- overall classification: `{summary['classification']}`",
            f"- interpretation: {summary['classification_notes']}",
        ]
    )
    for artifact in summary["artifacts"]:
        viable = [name for name, gate in artifact["local_gate"].items() if gate["pass"]]
        lines.append(
            f"- `{artifact['artifact']}` viable transforms: `{', '.join(viable) if viable else 'none'}`"
        )
    lines.extend(
        [
            "",
            "## Recommended next branch",
            "",
            f"Recommendation: **{summary['recommended_next_branch']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    reference_artifact_path = resolve_path(root, args.reference_artifact)
    reference_rows = row_map_from_reference(load_json(reference_artifact_path))
    all_capture_row_ids = capture_row_ids(reference_rows)
    guards = guard_row_ids(all_capture_row_ids)
    artifact_specs = parse_artifact_paths(root, args.current_path, args.candidate_paths)
    output_root = resolve_path(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "root_prior_transform_ablation_summary.json"
    report_path = (
        root
        / "docs/alphazero-lite-capture-002-root-prior-transform-ablation-results.md"
    )

    arena_module = importlib.import_module("ml.alphazero_lite.arena")
    artifact_summaries = []
    for artifact_spec in artifact_specs:
        evaluator = arena_module.ArtifactEvaluator(Path(artifact_spec["artifact_path"]))
        results: list[dict] = []
        for row_id in all_capture_row_ids:
            reference_row = reference_rows[row_id]
            probe_row = build_probe_row(reference_row)
            state = validated_diagnostic_state(row=probe_row)
            legal_moves = list(probe_row["legal_moves"])
            reference_move = int(probe_row["reference_move"])
            move_features = move_feature_annotations_for(
                state=state, legal_moves=legal_moves
            )
            wrong_extra_turn_move = wrong_extra_turn_move_for(
                row_id=row_id,
                legal_moves=legal_moves,
                move_features=move_features,
            )
            for transform_name in ALL_TRANSFORM_NAMES:
                override = build_root_prior_override(transform_name)
                for simulations in SIMULATION_BUDGETS:
                    probe_summary = probe_artifact_position(
                        artifact_path=str(artifact_spec["artifact_path"]),
                        evaluator=evaluator,
                        state=state,
                        simulations=simulations,
                        seed=SEED,
                        c_puct=C_PUCT,
                        search_options=dict(SEARCH_CONTROL_OVERRIDES),
                        ablation_mode="full",
                        root_prior_override=override,
                    )
                    row_view = build_row_views(
                        row=probe_row, probe_summary=probe_summary
                    )
                    prior_before = np.asarray(
                        (probe_summary.get("root_prior_telemetry") or {}).get("before")
                        or probe_summary.get("policy")
                        or [],
                        dtype=np.float32,
                    )
                    if prior_before.size == 0:
                        prior_before = np.zeros(6, dtype=np.float32)
                    transformed_prior, transform_telemetry = apply_root_prior_transform(
                        state=state,
                        legal_moves=legal_moves,
                        original_root_prior=prior_before,
                        move_feature_annotations=move_features,
                        transform_name=transform_name,
                    )
                    results.append(
                        build_result_entry(
                            artifact=str(artifact_spec["artifact"]),
                            artifact_path=Path(artifact_spec["artifact_path"]),
                            row_id=row_id,
                            reference_move=reference_move,
                            wrong_extra_turn_move=wrong_extra_turn_move,
                            legal_moves=legal_moves,
                            move_features=move_features,
                            transform_name=transform_name,
                            simulations=simulations,
                            row_view=row_view,
                            original_prior=prior_before,
                            transformed_prior=transformed_prior,
                            transform_telemetry=transform_telemetry,
                        )
                    )

        results_indexed = indexed_results(results)
        local_gate = evaluate_local_gate(artifact_results=results, guard_rows=guards)
        artifact_summaries.append(
            {
                "artifact": str(artifact_spec["artifact"]),
                "artifact_path": str(artifact_spec["artifact_path"]),
                "results": results,
                "results_indexed": results_indexed,
                "local_gate": local_gate,
            }
        )

    classification, recommended_next_branch, classification_notes = classify_results(
        artifact_summaries=artifact_summaries
    )

    arena_results = []
    candidate_summaries = [
        summary for summary in artifact_summaries if summary["artifact"] != "current"
    ]
    if candidate_summaries:
        best_transform = choose_best_candidate_transform(candidate_summaries[0], guards)
        if best_transform is not None:
            arena_results.append(
                run_arena_evaluation(
                    root=root,
                    output_root=output_root,
                    candidate_path=Path(candidate_summaries[0]["artifact_path"]),
                    current_path=resolve_path(root, args.current_path),
                    transform_name=best_transform,
                    arena_games=args.arena_games,
                    arena_workers=args.arena_workers,
                )
            )

    summary = {
        "schema": SCHEMA,
        "reference_artifact_path": str(reference_artifact_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "search_options": dict(SEARCH_CONTROL_OVERRIDES),
        "simulation_budgets": list(SIMULATION_BUDGETS),
        "transforms": list(ALL_TRANSFORM_NAMES),
        "all_capture_row_ids": all_capture_row_ids,
        "guard_row_ids": guards,
        "artifacts": [
            {key: value for key, value in artifact.items() if key != "results_indexed"}
            for artifact in artifact_summaries
        ],
        "arena_results": arena_results,
        "classification": classification,
        "classification_notes": classification_notes,
        "recommended_next_branch": recommended_next_branch,
    }
    write_json(summary_path, summary)
    report_path.write_text(
        build_markdown({**summary, "artifacts": artifact_summaries}), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "classification": classification,
                "recommended_next_branch": recommended_next_branch,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
