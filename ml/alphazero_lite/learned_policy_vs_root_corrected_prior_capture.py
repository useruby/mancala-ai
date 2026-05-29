from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    build_row_views,
    probe_artifact_position,
    validated_diagnostic_state,
)
from ml.alphazero_lite.run_capture_002_003_search_prior_control_experiment import (
    SEARCH_CONTROL_OVERRIDES,
)
from ml.alphazero_lite.run_capture_002_root_prior_intervention import (
    C_PUCT,
    SEED,
    build_root_prior_override,
    selection_score_by_move,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    build_probe_row,
    load_json,
    row_map_from_reference,
)


SCHEMA = "azlite_learned_policy_vs_root_corrected_prior_capture_v1"
PRIMARY_ROW_ID = "capture_available-002"
PRESERVATION_ROW_ID = "capture_available-003"
INTERVENTIONS = (
    "original_prior",
    "zero_wrong_extra_turn_prior",
    "swap_reference_and_wrong",
)
SIMULATIONS = 384


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture learned-policy versus root-corrected prior mismatch telemetry"
    )
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _trace_excerpt(probe_summary: dict) -> dict:
    selection_breakdown = probe_summary.get("selection_breakdown") or {}
    visit_snapshots = probe_summary.get("visit_snapshots") or []
    selected_move = probe_summary.get("selected_move")
    selected_selection_score = None
    if selected_move is not None:
        selected_selection_score = selection_score_by_move(
            probe_summary, int(selected_move)
        )
    return {
        "selected_move": selected_move,
        "selected_move_selection_score": selected_selection_score,
        "selection_breakdown": copy.deepcopy(selection_breakdown),
        "visit_snapshots": copy.deepcopy(visit_snapshots),
    }


def _capture_one(
    *, artifact_path: str, reference_row: dict, intervention: str, seed_offset: int
) -> dict:
    probe_row = build_probe_row(reference_row)
    state = validated_diagnostic_state(row=probe_row)
    reference_move = int(reference_row["reference_move"])
    override = None
    if intervention != "original_prior":
        override = build_root_prior_override(
            row_id=str(reference_row["id"]),
            reference_move=reference_move,
            intervention=intervention,
        )
    probe_summary = probe_artifact_position(
        artifact_path=artifact_path,
        state=state,
        simulations=SIMULATIONS,
        seed=SEED + seed_offset,
        c_puct=C_PUCT,
        search_options=dict(SEARCH_CONTROL_OVERRIDES),
        root_prior_override=override,
    )
    row_views = build_row_views(row=probe_row, probe_summary=probe_summary)
    policy_view = row_views.get("policy_view") or {}
    search_view = row_views.get("search_view") or {}
    value_view = row_views.get("value_view") or {}
    selected_move = search_view.get("searched_selected_move")
    reference_selection_score = selection_score_by_move(probe_summary, reference_move)
    selected_selection_score = None
    if selected_move is not None:
        selected_selection_score = selection_score_by_move(probe_summary, int(selected_move))
    return {
        "intervention": intervention,
        "reference_move": reference_move,
        "searched_selected_move": selected_move,
        "policy_reference_probability": policy_view.get("reference_move_probability"),
        "policy_selected_minus_reference_margin": policy_view.get(
            "selected_minus_reference_margin"
        ),
        "reference_move_visit_share": search_view.get("reference_move_visit_share"),
        "selected_move_visit_share": search_view.get("selected_move_visit_share"),
        "selected_minus_reference_q_margin": value_view.get(
            "selected_minus_reference_q_margin"
        ),
        "reference_move_selection_score": reference_selection_score,
        "selected_move_selection_score": selected_selection_score,
        "row_views": copy.deepcopy(row_views),
        "trace_excerpt": _trace_excerpt(probe_summary),
    }


def build_payload(*, artifact_path: str, reference_artifact: dict) -> dict:
    reference_rows = row_map_from_reference(reference_artifact)
    primary_row = reference_rows[PRIMARY_ROW_ID]
    preservation_row = reference_rows[PRESERVATION_ROW_ID]

    primary_interventions = {}
    for index, intervention in enumerate(INTERVENTIONS):
        primary_interventions[intervention] = _capture_one(
            artifact_path=artifact_path,
            reference_row=primary_row,
            intervention=intervention,
            seed_offset=index,
        )

    preservation_interventions = {}
    for index, intervention in enumerate(INTERVENTIONS):
        preservation_interventions[intervention] = _capture_one(
            artifact_path=artifact_path,
            reference_row=preservation_row,
            intervention=intervention,
            seed_offset=100 + index,
        )

    primary_original = primary_interventions["original_prior"]
    primary_zero_wrong = primary_interventions["zero_wrong_extra_turn_prior"]
    primary_swap = primary_interventions["swap_reference_and_wrong"]

    if (
        primary_original["searched_selected_move"] != primary_original["reference_move"]
        and primary_zero_wrong["searched_selected_move"] == primary_zero_wrong["reference_move"]
        and primary_swap["searched_selected_move"] == primary_swap["reference_move"]
    ):
        classification = "learned_prior_underweights_reference_and_overweights_wrong_extra_turn"
        recommendation = "write_learned_policy_vs_root_corrected_prior_review_spec"
    else:
        classification = "mismatch_not_isolated"
        recommendation = "stop_learned_policy_vs_root_corrected_prior_capture_inconclusive"

    return {
        "schema": SCHEMA,
        "artifact_path": artifact_path,
        "focus_row_id": PRIMARY_ROW_ID,
        "preservation_row_id": PRESERVATION_ROW_ID,
        "settings": {
            "simulations": SIMULATIONS,
            "seed": SEED,
            "c_puct": C_PUCT,
            "interventions": list(INTERVENTIONS),
            "search_options": dict(SEARCH_CONTROL_OVERRIDES),
        },
        "classification": classification,
        "decision": recommendation,
        "focus_row": {
            "row_id": PRIMARY_ROW_ID,
            "interventions": primary_interventions,
        },
        "preservation_row": {
            "row_id": PRESERVATION_ROW_ID,
            "interventions": preservation_interventions,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        artifact_path=str(args.artifact_path),
        reference_artifact=load_json(args.reference_artifact),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": payload["schema"],
                "classification": payload["classification"],
                "decision": payload["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
