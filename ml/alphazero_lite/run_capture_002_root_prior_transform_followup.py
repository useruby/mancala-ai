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
from ml.alphazero_lite.root_prior_transforms import (
    FOLLOWUP_TRANSFORM_NAMES,
    apply_root_prior_transform,
    build_root_prior_override,
    move_feature_annotations_for,
)
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


DEFAULT_REFERENCE_ARTIFACT = (
    "/tmp/azlite_failure_family_diag/train_only_forensic_references.json"
)
DEFAULT_ARTIFACT = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_root_prior_transform_followup"
SIMULATION_BUDGETS = (384, 1200)
ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-005",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
SEED = 17
C_PUCT = 1.25
SCHEMA = "azlite_capture_002_root_prior_transform_followup_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def metric_for_move(
    distribution: dict[str, float] | None, move: int | None
) -> float | None:
    if distribution is None or move is None:
        return None
    value = distribution.get(str(move))
    return None if value is None else float(value)


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


def build_markdown(summary: dict) -> str:
    lines = [
        "# AlphaZero-lite Capture 002 Root-Prior Transform Follow-up",
        "",
        "## Context",
        "",
        "This follow-up narrows extra-turn damping to a tighter legal-move feature pattern derived from the first ablation.",
        "",
        f"- source branch: `{summary['recommended_next_branch']}`",
        f"- artifact: `{summary['artifact_path']}`",
        f"- summary JSON: `{summary['summary_path']}`",
        "",
        "## Results",
        "",
        "| row_id | transform | simulations | selected_move | reference_move | selected_is_reference | reference_visit_share | selected_minus_reference_q | original_reference_prior | transformed_reference_prior | gate_pass | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        lines.append(
            "| {row_id} | {transform} | {simulations} | {selected_move} | {reference_move} | {selected_is_reference} | {reference_visit_share} | {selected_minus_reference_q} | {original_reference_prior} | {transformed_reference_prior} | {gate_pass} | {notes} |".format(
                row_id=result["row_id"],
                transform=result["transform_name"],
                simulations=result["simulations"],
                selected_move=result["selected_move"],
                reference_move=result["reference_move"],
                selected_is_reference=str(result["selected_is_reference"]).lower(),
                reference_visit_share=result["reference_visit_share"],
                selected_minus_reference_q=result["selected_minus_reference_q"],
                original_reference_prior=result["original_prior"][
                    str(result["reference_move"])
                ],
                transformed_reference_prior=result["transformed_prior"][
                    str(result["reference_move"])
                ],
                gate_pass=str(result["gate_pass"]).lower(),
                notes=result["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Recommendation: **{summary['recommendation']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    artifact_path = resolve_path(root, args.artifact)
    reference_artifact_path = resolve_path(root, args.reference_artifact)
    output_root = resolve_path(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "root_prior_transform_followup_summary.json"
    report_path = (
        root / "docs/alphazero-lite-capture-002-root-prior-transform-followup.md"
    )

    arena = __import__("ml.alphazero_lite.arena", fromlist=["ArtifactEvaluator"])
    evaluator = arena.ArtifactEvaluator(artifact_path)
    reference_rows = row_map_from_reference(load_json(reference_artifact_path))

    results = []
    for row_id in ROW_IDS:
        reference_row = reference_rows[row_id]
        probe_row = build_probe_row(reference_row)
        state = validated_diagnostic_state(row=probe_row)
        legal_moves = list(probe_row["legal_moves"])
        move_features = move_feature_annotations_for(
            state=state, legal_moves=legal_moves
        )
        for transform_name in FOLLOWUP_TRANSFORM_NAMES:
            override = build_root_prior_override(transform_name)
            for simulations in SIMULATION_BUDGETS:
                probe_summary = probe_artifact_position(
                    artifact_path=str(artifact_path),
                    evaluator=evaluator,
                    state=state,
                    simulations=simulations,
                    seed=SEED,
                    c_puct=C_PUCT,
                    search_options=dict(SEARCH_CONTROL_OVERRIDES),
                    ablation_mode="full",
                    root_prior_override=override,
                )
                row_view = build_row_views(row=probe_row, probe_summary=probe_summary)
                original_prior = np.asarray(
                    (probe_summary.get("root_prior_telemetry") or {}).get("before")
                    or probe_summary.get("policy")
                    or [],
                    dtype=np.float32,
                )
                transformed_prior, _telemetry = apply_root_prior_transform(
                    state=state,
                    legal_moves=legal_moves,
                    original_root_prior=original_prior,
                    move_feature_annotations=move_features,
                    transform_name=transform_name,
                )
                visit_share = visit_share_by_move(row_view.get("search_view") or {})
                results.append(
                    {
                        "row_id": row_id,
                        "transform_name": transform_name,
                        "simulations": simulations,
                        "reference_move": int(probe_row["reference_move"]),
                        "selected_move": row_view["search_view"][
                            "searched_selected_move"
                        ],
                        "selected_is_reference": row_view["search_view"][
                            "searched_selected_move"
                        ]
                        == int(probe_row["reference_move"]),
                        "reference_visit_share": metric_for_move(
                            visit_share, int(probe_row["reference_move"])
                        ),
                        "selected_minus_reference_q": row_view["value_view"][
                            "selected_minus_reference_q_margin"
                        ],
                        "original_prior": {
                            str(move): round(float(original_prior[move]), 4)
                            for move in legal_moves
                        },
                        "transformed_prior": {
                            str(move): round(float(transformed_prior[move]), 4)
                            for move in legal_moves
                        },
                    }
                )

    indexed = {(r["row_id"], r["transform_name"], r["simulations"]): r for r in results}
    for result in results:
        failures = []
        transform_name = result["transform_name"]
        row_id = result["row_id"]
        if row_id == "capture_available-002":
            if indexed[(row_id, transform_name, 384)]["selected_move"] != 4:
                failures.append("002@384_fail")
            if indexed[(row_id, transform_name, 1200)]["selected_move"] == 2:
                failures.append("002@1200_revert_2")
        if row_id == "capture_available-003":
            baseline_384 = indexed[(row_id, FOLLOWUP_TRANSFORM_NAMES[0], 384)]
            del baseline_384
            if indexed[(row_id, transform_name, 384)]["selected_move"] != 1:
                failures.append("003@384_fail")
            if indexed[(row_id, transform_name, 1200)]["selected_move"] != 1:
                failures.append("003@1200_fail")
        if row_id in {
            "capture_available-006",
            "capture_available-007",
            "capture_available-008",
        }:
            if not indexed[(row_id, transform_name, 384)]["selected_is_reference"]:
                failures.append(f"{row_id}@384_regress")
            if not indexed[(row_id, transform_name, 1200)]["selected_is_reference"]:
                failures.append(f"{row_id}@1200_regress")
        result["gate_pass"] = not failures
        result["notes"] = ",".join(failures) if failures else "followup_local_ok"

    recommendation = "teacher/label audit and feature encoding review"
    if any(
        indexed[("capture_available-002", name, 384)]["selected_move"] == 4
        and indexed[("capture_available-002", name, 1200)]["selected_move"] != 2
        and indexed[("capture_available-003", name, 384)]["selected_move"] == 1
        and indexed[("capture_available-003", name, 1200)]["selected_move"] == 1
        for name in FOLLOWUP_TRANSFORM_NAMES
    ):
        recommendation = "broader arena/MCTS1200 validation for the best narrow feature-conditioned transform"

    summary = {
        "schema": SCHEMA,
        "artifact_path": str(artifact_path),
        "reference_artifact_path": str(reference_artifact_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "results": results,
        "recommended_next_branch": "more precise move-feature conditioning; do not deploy damping",
        "recommendation": recommendation,
    }
    write_json(summary_path, summary)
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "recommendation": recommendation,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
