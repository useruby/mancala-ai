#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
from ml.alphazero_lite.run_capture_002_003_search_prior_control_experiment import (
    DEFAULT_CURRENT_PATH,
    SEARCH_CONTROL_OVERRIDES,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    build_probe_row,
    load_json,
    repo_root,
    row_map_from_reference,
    write_json,
)


DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_root_prior_intervention"
DEFAULT_RUN_ID = "capture-002-root-prior-intervention"
DEFAULT_CANDIDATE_PATHS = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
ROW_IDS = ("capture_available-002", "capture_available-003")
SIMULATION_BUDGETS = (32, 64, 128, 384, 1200)
INTERVENTION_NAMES = (
    "original_prior",
    "uniform_legal_prior",
    "zero_wrong_extra_turn_prior",
    "equalize_reference_and_wrong",
    "swap_reference_and_wrong",
    "force_reference_prior_advantage",
)
SCHEMA = "azlite_capture_002_root_prior_intervention_v1"
NEAR_ZERO_PRIOR = 1e-6
FORCE_REFERENCE_PRIOR = 0.50
SEED = 17
C_PUCT = 1.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--candidate-paths", default=DEFAULT_CANDIDATE_PATHS)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_artifact_paths(
    root: Path, current_path: str, candidate_paths: str
) -> list[dict]:
    resolved = [
        {"artifact_label": "current", "artifact_path": resolve_path(root, current_path)}
    ]
    for item in candidate_paths.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        path = resolve_path(root, stripped)
        resolved.append({"artifact_label": path.name, "artifact_path": path})
    return resolved


def _distribution_for_moves(
    priors: np.ndarray, legal_moves: list[int]
) -> dict[int, float]:
    return {int(move): float(priors[move]) for move in legal_moves}


def _renormalize_legal(priors: np.ndarray, legal_moves: list[int]) -> np.ndarray:
    normalized = np.zeros_like(priors, dtype=np.float32)
    legal_total = float(np.sum(priors[legal_moves]))
    if legal_total <= 0.0:
        normalized[legal_moves] = 1.0 / len(legal_moves)
        return normalized
    normalized[legal_moves] = priors[legal_moves] / legal_total
    return normalized


def _strongest_competitor_move(
    *, row_id: str, legal_moves: list[int], reference_move: int, priors: np.ndarray
) -> int:
    if row_id == "capture_available-002" and 2 in legal_moves and 2 != reference_move:
        return 2
    competitors = [move for move in legal_moves if move != reference_move]
    if not competitors:
        return reference_move
    return max(competitors, key=lambda move: (float(priors[move]), -int(move)))


def build_root_prior_override(*, row_id: str, reference_move: int, intervention: str):
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(priors, dtype=np.float32).copy()
        competitor_move = _strongest_competitor_move(
            row_id=row_id,
            legal_moves=legal_moves,
            reference_move=reference_move,
            priors=adjusted,
        )

        if intervention == "original_prior":
            return _renormalize_legal(adjusted, legal_moves)

        if intervention == "uniform_legal_prior":
            adjusted[:] = 0.0
            adjusted[legal_moves] = 1.0 / len(legal_moves)
            return adjusted.astype(np.float32)

        if intervention == "zero_wrong_extra_turn_prior":
            adjusted[competitor_move] = NEAR_ZERO_PRIOR
            return _renormalize_legal(adjusted, legal_moves)

        if intervention == "equalize_reference_and_wrong":
            target = max(
                float(adjusted[reference_move]), float(adjusted[competitor_move])
            )
            adjusted[reference_move] = target
            adjusted[competitor_move] = target
            return _renormalize_legal(adjusted, legal_moves)

        if intervention == "swap_reference_and_wrong":
            adjusted[reference_move], adjusted[competitor_move] = (
                float(adjusted[competitor_move]),
                float(adjusted[reference_move]),
            )
            return _renormalize_legal(adjusted, legal_moves)

        if intervention == "force_reference_prior_advantage":
            adjusted[:] = 0.0
            adjusted[reference_move] = FORCE_REFERENCE_PRIOR
            other_moves = [move for move in legal_moves if move != reference_move]
            if other_moves:
                baseline = np.asarray(priors, dtype=np.float32).copy()
                baseline[reference_move] = 0.0
                other_total = float(np.sum(baseline[other_moves]))
                remaining = 1.0 - FORCE_REFERENCE_PRIOR
                if other_total > 0.0:
                    adjusted[other_moves] = remaining * (
                        baseline[other_moves] / other_total
                    )
                else:
                    adjusted[other_moves] = remaining / len(other_moves)
            return _renormalize_legal(adjusted, legal_moves)

        raise ValueError(f"unsupported intervention: {intervention}")

    return override


def metric_from_distribution(
    distribution: dict[str, float] | None, move: int
) -> float | None:
    if distribution is None:
        return None
    value = distribution.get(str(move))
    return None if value is None else float(value)


def visit_share_for_move(search_view: dict, move: int) -> float | None:
    visit_distribution = search_view.get("visit_distribution") or {}
    if not visit_distribution:
        return None
    total = float(sum(float(value) for value in visit_distribution.values()))
    if total <= 0.0:
        return None
    visits = visit_distribution.get(str(move))
    if visits is None:
        return 0.0
    return round(float(visits) / total, 4)


def child_q_by_move(row_view: dict, move: int) -> float | None:
    q_values = (row_view.get("value_view") or {}).get("per_child_q_values") or {}
    value = q_values.get(str(move))
    return None if value is None else float(value)


def selection_score_by_move(probe_summary: dict, move: int) -> float | None:
    selection_breakdown = probe_summary.get("selection_breakdown") or {}
    moves = selection_breakdown.get("moves") or []
    for entry in moves:
        if int(entry.get("move", -1)) == int(move):
            score = entry.get("selection_score")
            return None if score is None else float(score)
    return None


def first_budget_reference_selected(
    results: list[dict], reference_move: int
) -> int | None:
    for result in sorted(results, key=lambda item: int(item["simulations"])):
        if result["selected_move"] == reference_move:
            return int(result["simulations"])
    return None


def pass_fail_reason(
    *,
    row_id: str,
    selected_move: int | None,
    reference_move: int,
    competitor_move: int,
    first_budget_selected: int | None,
) -> str:
    if selected_move == reference_move:
        if row_id == "capture_available-003":
            return "pass_reference_preserved"
        return "pass_reference_selected"
    if row_id == "capture_available-002" and competitor_move == 2:
        if first_budget_selected is None:
            return "fail_move_2_persisted"
        return "fail_reference_lost_again_after_early_flip"
    return "fail_reference_not_selected"


def classify_artifact(rows: dict[str, list[dict]]) -> tuple[str, str, str]:
    by_row_and_intervention = {
        row_id: {
            intervention: sorted(values, key=lambda item: int(item["simulations"]))
            for intervention, values in _group_by_intervention(row_results).items()
        }
        for row_id, row_results in rows.items()
    }

    row_002 = by_row_and_intervention["capture_available-002"]
    row_003 = by_row_and_intervention["capture_available-003"]

    def selected(row_results: list[dict]) -> bool:
        return any(result["selected_is_reference"] for result in row_results)

    if (
        (
            selected(row_002["uniform_legal_prior"])
            or selected(row_002["equalize_reference_and_wrong"])
        )
        and all(result["selected_is_reference"] for result in row_003["original_prior"])
        and all(
            result["selected_is_reference"] for result in row_003["uniform_legal_prior"]
        )
    ):
        return (
            "policy_prior_sensitive",
            "policy-prior calibration experiment",
            "uniform/equalized root priors flipped 002 while preserving 003",
        )

    if selected(row_002["force_reference_prior_advantage"]) and not any(
        selected(row_002[name])
        for name in (
            "uniform_legal_prior",
            "zero_wrong_extra_turn_prior",
            "equalize_reference_and_wrong",
            "swap_reference_and_wrong",
        )
    ):
        return (
            "strong_prior_lock_in",
            "value/teacher audit for row 002",
            "only the positive-control reference prior advantage flipped 002",
        )

    if not selected(row_002["zero_wrong_extra_turn_prior"]):
        return (
            "value_or_backup_dominant",
            "value/teacher audit for row 002",
            "002 still rejected the reference after near-zero competitor prior",
        )

    if any(
        not result["selected_is_reference"] for result in row_003["uniform_legal_prior"]
    ) or any(
        not result["selected_is_reference"]
        for result in row_003["equalize_reference_and_wrong"]
    ):
        return (
            "row_pair_prior_tradeoff",
            "rule-conditioned prior calibration with explicit 002/003 preservation constraints",
            "002 flips only when 003 regresses",
        )

    return (
        "value_or_backup_dominant",
        "value/teacher audit for row 002",
        "root-prior interventions did not cleanly resolve 002",
    )


def _group_by_intervention(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(str(result["intervention"]), []).append(result)
    return grouped


def build_markdown(summary: dict) -> str:
    lines = [
        "# AlphaZero-lite Capture 002 Root-Prior Intervention Results",
        "",
        "## Context",
        "",
        "This branch isolates whether the persistent `capture_available-002` failure is driven by root policy-prior pressure or by downstream value/backup dynamics.",
        "",
        "## Inputs",
        "",
        f"- reference artifact: `{summary['reference_artifact_path']}`",
        f"- output summary: `{summary['summary_path']}`",
    ]
    for artifact in summary["artifacts"]:
        lines.append(
            f"- artifact `{artifact['artifact']}`: `{artifact['artifact_path']}`"
        )
    lines.extend(
        [
            "",
            "## Intervention definitions",
            "",
            "- `original_prior`: no override",
            "- `uniform_legal_prior`: uniform over legal root moves",
            "- `zero_wrong_extra_turn_prior`: near-zero competitor prior then renormalize",
            "- `equalize_reference_and_wrong`: reference and competitor priors matched then renormalized",
            "- `swap_reference_and_wrong`: reference and competitor priors swapped",
            "- `force_reference_prior_advantage`: reference prior set to `0.50` as positive control",
            "",
            "## Results table",
            "",
            "| artifact | row_id | intervention | simulations | reference_move | competitor_move | selected_move | selected_is_reference | reference_prior_before | competitor_prior_before | reference_prior_after | competitor_prior_after | reference_visit_share | competitor_visit_share | reference_q | competitor_q | selected_minus_reference_q | decision | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for artifact in summary["artifacts"]:
        for result in artifact["results_table"]:
            lines.append(
                "| {artifact} | {row_id} | {intervention} | {simulations} | {reference_move} | {competitor_move} | {selected_move} | {selected_is_reference} | {reference_prior_before:.4f} | {competitor_prior_before:.4f} | {reference_prior_after:.4f} | {competitor_prior_after:.4f} | {reference_visit_share:.4f} | {competitor_visit_share:.4f} | {reference_q:.4f} | {competitor_q:.4f} | {selected_minus_reference_q:.4f} | `{decision}` | {notes} |".format(
                    artifact=artifact["artifact"],
                    row_id=result["row_id"],
                    intervention=result["intervention"],
                    simulations=result["simulations"],
                    reference_move=result["reference_move"],
                    competitor_move=result["competitor_move"],
                    selected_move=result["selected_move"],
                    selected_is_reference=str(result["selected_is_reference"]).lower(),
                    reference_prior_before=float(
                        result["reference_prior_before"] or 0.0
                    ),
                    competitor_prior_before=float(
                        result["competitor_prior_before"] or 0.0
                    ),
                    reference_prior_after=float(result["reference_prior_after"] or 0.0),
                    competitor_prior_after=float(
                        result["competitor_prior_after"] or 0.0
                    ),
                    reference_visit_share=float(result["reference_visit_share"] or 0.0),
                    competitor_visit_share=float(
                        result["competitor_visit_share"] or 0.0
                    ),
                    reference_q=float(result["reference_q"] or 0.0),
                    competitor_q=float(result["competitor_q"] or 0.0),
                    selected_minus_reference_q=float(
                        result["selected_minus_reference_q"] or 0.0
                    ),
                    decision=result["decision"],
                    notes=result["notes"],
                )
            )
    lines.extend(["", "## Interpretation", ""])
    for artifact in summary["artifacts"]:
        lines.append(
            f"- `{artifact['artifact']}` classified as `{artifact['classification']}`: {artifact['classification_notes']}"
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
    artifact_specs = parse_artifact_paths(root, args.current_path, args.candidate_paths)

    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "root_prior_intervention_summary.json"
    report_path = (
        root / "docs/alphazero-lite-capture-002-root-prior-intervention-results.md"
    )

    artifacts_summary = []
    for artifact_spec in artifact_specs:
        artifact_results = []
        for row_id in ROW_IDS:
            reference_row = reference_rows[row_id]
            probe_row = build_probe_row(reference_row)
            state = validated_diagnostic_state(row=probe_row)
            legal_moves = list(probe_row["legal_moves"])
            reference_move = int(probe_row["reference_move"])

            for intervention in INTERVENTION_NAMES:
                override = build_root_prior_override(
                    row_id=row_id,
                    reference_move=reference_move,
                    intervention=intervention,
                )
                per_intervention_results = []
                for simulations in SIMULATION_BUDGETS:
                    probe_summary = probe_artifact_position(
                        artifact_path=str(artifact_spec["artifact_path"]),
                        state=state,
                        simulations=simulations,
                        seed=SEED,
                        c_puct=C_PUCT,
                        search_options=dict(SEARCH_CONTROL_OVERRIDES),
                        ablation_mode="full",
                        root_prior_override=None
                        if intervention == "original_prior"
                        else override,
                    )
                    row_view = build_row_views(
                        row=probe_row, probe_summary=probe_summary
                    )
                    prior_telemetry = probe_summary.get("root_prior_telemetry") or {}
                    before = np.asarray(
                        prior_telemetry.get("before")
                        or probe_summary.get("policy")
                        or [],
                        dtype=np.float32,
                    )
                    after = np.asarray(
                        prior_telemetry.get("after")
                        or probe_summary.get("policy")
                        or [],
                        dtype=np.float32,
                    )
                    competitor_move = _strongest_competitor_move(
                        row_id=row_id,
                        legal_moves=legal_moves,
                        reference_move=reference_move,
                        priors=after if after.size else before,
                    )
                    selected_move = row_view["search_view"]["searched_selected_move"]
                    reference_q = child_q_by_move(row_view, reference_move)
                    competitor_q = child_q_by_move(row_view, competitor_move)
                    selected_q = (
                        None
                        if selected_move is None
                        else child_q_by_move(row_view, selected_move)
                    )
                    reference_visit_share = visit_share_for_move(
                        row_view["search_view"], reference_move
                    )
                    competitor_visit_share = visit_share_for_move(
                        row_view["search_view"], competitor_move
                    )
                    selected_visit_share = row_view["search_view"].get(
                        "selected_move_visit_share"
                    )
                    result = {
                        "artifact": artifact_spec["artifact_label"],
                        "artifact_path": str(artifact_spec["artifact_path"]),
                        "row_id": row_id,
                        "intervention": intervention,
                        "simulations": int(simulations),
                        "reference_move": reference_move,
                        "competitor_move": competitor_move,
                        "wrong_or_competitor_move": competitor_move,
                        "selected_move": selected_move,
                        "selected_is_reference": selected_move == reference_move,
                        "selected_is_wrong_extra_turn": row_id
                        == "capture_available-002"
                        and selected_move == 2,
                        "reference_visit_share": reference_visit_share,
                        "wrong_or_competitor_visit_share": competitor_visit_share,
                        "competitor_visit_share": competitor_visit_share,
                        "reference_prior_before": None
                        if before.size == 0
                        else float(before[reference_move]),
                        "wrong_or_competitor_prior_before": None
                        if before.size == 0
                        else float(before[competitor_move]),
                        "competitor_prior_before": None
                        if before.size == 0
                        else float(before[competitor_move]),
                        "reference_prior_after": None
                        if after.size == 0
                        else float(after[reference_move]),
                        "wrong_or_competitor_prior_after": None
                        if after.size == 0
                        else float(after[competitor_move]),
                        "competitor_prior_after": None
                        if after.size == 0
                        else float(after[competitor_move]),
                        "reference_q": reference_q,
                        "wrong_or_competitor_q": competitor_q,
                        "competitor_q": competitor_q,
                        "selected_minus_reference_q": None
                        if selected_q is None or reference_q is None
                        else round(float(selected_q - reference_q), 4),
                        "selected_minus_reference_visit_share": None
                        if selected_visit_share is None or reference_visit_share is None
                        else round(
                            float(selected_visit_share) - float(reference_visit_share),
                            4,
                        ),
                        "final_root_selection_score": None
                        if selected_move is None
                        else selection_score_by_move(probe_summary, selected_move),
                        "decision": "reference_selected"
                        if selected_move == reference_move
                        else "reference_not_selected",
                        "notes": "002 positive-control"
                        if intervention == "force_reference_prior_advantage"
                        and row_id == "capture_available-002"
                        else "",
                    }
                    per_intervention_results.append(result)
                    artifact_results.append(result)

                first_budget = first_budget_reference_selected(
                    per_intervention_results, reference_move=reference_move
                )
                for result in per_intervention_results:
                    result["first_budget_where_reference_becomes_selected"] = (
                        first_budget
                    )
                    result["pass_fail_reason"] = pass_fail_reason(
                        row_id=row_id,
                        selected_move=result["selected_move"],
                        reference_move=reference_move,
                        competitor_move=result["competitor_move"],
                        first_budget_selected=first_budget,
                    )

        grouped_rows = {
            row_id: [r for r in artifact_results if r["row_id"] == row_id]
            for row_id in ROW_IDS
        }
        classification, next_branch, classification_notes = classify_artifact(
            grouped_rows
        )
        artifacts_summary.append(
            {
                "artifact": artifact_spec["artifact_label"],
                "artifact_path": str(artifact_spec["artifact_path"]),
                "classification": classification,
                "recommended_next_branch": next_branch,
                "classification_notes": classification_notes,
                "results": artifact_results,
                "results_table": [
                    {
                        **result,
                        "decision": result["pass_fail_reason"],
                        "notes": result["notes"] or result["pass_fail_reason"],
                    }
                    for result in artifact_results
                ],
            }
        )

    recommended_next_branch = artifacts_summary[0]["recommended_next_branch"]
    summary = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "reference_artifact_path": str(reference_artifact_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "search_options": dict(SEARCH_CONTROL_OVERRIDES),
        "simulation_budgets": list(SIMULATION_BUDGETS),
        "interventions": list(INTERVENTION_NAMES),
        "artifacts": artifacts_summary,
        "recommended_next_branch": recommended_next_branch,
    }
    write_json(summary_path, summary)
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(
        json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
