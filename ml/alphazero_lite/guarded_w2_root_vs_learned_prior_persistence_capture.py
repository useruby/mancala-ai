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
from ml.alphazero_lite.run_capture_002_root_prior_intervention import (
    C_PUCT,
    SEED,
    build_root_prior_override,
    selection_score_by_move,
    visit_share_for_move,
)
from ml.alphazero_lite.run_capture_002_003_search_prior_control_experiment import (
    SEARCH_CONTROL_OVERRIDES,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    build_probe_row,
    load_json,
    row_map_from_reference,
)


SCHEMA = "azlite_guarded_w2_root_vs_learned_prior_persistence_capture_v1"
ROW_IDS = ("capture_available-002", "capture_available-003")
INTERVENTIONS = (
    "original_prior",
    "zero_wrong_extra_turn_prior",
    "swap_reference_and_wrong",
)
SIMULATIONS = 384


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture guarded w2 root-vs-learned prior persistence telemetry"
    )
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _selection_score_for_selected(probe_summary: dict) -> float | None:
    selected_move = probe_summary.get("selected_move")
    if selected_move is None:
        return None
    return selection_score_by_move(probe_summary, int(selected_move))


def _trace_excerpt(probe_summary: dict) -> dict:
    selection_breakdown = probe_summary.get("selection_breakdown") or {}
    visit_snapshots = probe_summary.get("visit_snapshots") or []
    moves = selection_breakdown.get("moves") or []
    return {
        "selected_move": probe_summary.get("selected_move"),
        "selected_move_selection_score": _selection_score_for_selected(probe_summary),
        "move_selection_breakdown": copy.deepcopy(moves),
        "visit_snapshots": copy.deepcopy(visit_snapshots),
    }


def _capture_intervention(
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
    selected_selection_score = (
        None
        if selected_move is None
        else selection_score_by_move(probe_summary, int(selected_move))
    )
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
        "reference_move_visit_share_from_distribution": visit_share_for_move(
            search_view, reference_move
        ),
        "trace_excerpt": _trace_excerpt(probe_summary),
        "row_views": copy.deepcopy(row_views),
    }


def _row_decision(row_payload: dict) -> str:
    original = row_payload["interventions"]["original_prior"]
    zero_wrong = row_payload["interventions"]["zero_wrong_extra_turn_prior"]
    swapped = row_payload["interventions"]["swap_reference_and_wrong"]
    row_id = row_payload["row_id"]

    if row_id == "capture_available-002":
        if (
            zero_wrong["searched_selected_move"] == row_payload["reference_move"]
            and swapped["searched_selected_move"] == row_payload["reference_move"]
            and original["searched_selected_move"] != row_payload["reference_move"]
        ):
            if original["policy_reference_probability"] is not None and float(
                original["policy_reference_probability"]
            ) < 0.1:
                return "non_root_policy_mismatch"
            return "backup_or_search_stage_gap"
    return "reference_preserved_under_root_override"


def build_payload(*, artifact_path: str, reference_artifact: dict) -> dict:
    reference_rows = row_map_from_reference(reference_artifact)
    rows = {}
    decisions = {}
    for index, row_id in enumerate(ROW_IDS):
        reference_row = reference_rows[row_id]
        intervention_payloads = {}
        for intervention_index, intervention in enumerate(INTERVENTIONS):
            intervention_payloads[intervention] = _capture_intervention(
                artifact_path=artifact_path,
                reference_row=reference_row,
                intervention=intervention,
                seed_offset=(index * 100) + intervention_index,
            )
        row_payload = {
            "row_id": row_id,
            "reference_move": int(reference_row["reference_move"]),
            "interventions": intervention_payloads,
        }
        row_payload["decision"] = _row_decision(row_payload)
        rows[row_id] = row_payload
        decisions[row_payload["decision"]] = decisions.get(row_payload["decision"], 0) + 1

    if rows["capture_available-002"]["decision"] == "non_root_policy_mismatch":
        next_branch = "learned_policy_vs_root_corrected_prior_capture"
    else:
        next_branch = "search_value_persistence_gap_capture"

    return {
        "schema": SCHEMA,
        "artifact_path": artifact_path,
        "reference_artifact_path": reference_artifact.get("suite_path"),
        "settings": {
            "simulations": SIMULATIONS,
            "seed": SEED,
            "c_puct": C_PUCT,
            "interventions": list(INTERVENTIONS),
        },
        "rows": rows,
        "summary": {
            "decision_counts": decisions,
            "next_branch": next_branch,
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
                "next_branch": payload["summary"]["next_branch"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
