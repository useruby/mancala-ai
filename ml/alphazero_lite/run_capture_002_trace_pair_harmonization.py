#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import capture_002_selection_score_trace
from ml.alphazero_lite import capture_002_trace_capture


DEFAULT_TRACE_CAPTURE_PATH = (
    "/tmp/azlite_capture_002_prior_pressure_from_search_control/"
    "capture-002-prior-pressure/capture_002_trace_capture.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_trace_pair_harmonization"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-trace-capture-artifact", default=DEFAULT_TRACE_CAPTURE_PATH
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _move_entry(trace_point: dict, move: int) -> dict | None:
    for move_entry in trace_point.get("moves") or []:
        if move_entry.get("move") == move:
            return move_entry
    return None


def _visit_value(trace_point: dict, move: int) -> float | None:
    visits = trace_point.get("visits")
    if not isinstance(visits, list) or move >= len(visits):
        return None
    return _finite_number(visits[move])


def _projected_selected_move(
    trace_point: dict, *, reference_move: int, full_search_selected_move: int
) -> tuple[int, str]:
    original_selected_move = trace_point.get("selected_move")
    expected_pair = {reference_move, full_search_selected_move}
    if original_selected_move in expected_pair:
        return int(original_selected_move), "kept_original_pair_move"

    ranked_moves = []
    for move in (reference_move, full_search_selected_move):
        move_entry = _move_entry(trace_point, move) or {}
        ranked_moves.append(
            (
                (
                    _finite_number(move_entry.get("selection_score")) is not None,
                    _finite_number(move_entry.get("selection_score"))
                    if _finite_number(move_entry.get("selection_score")) is not None
                    else float("-inf"),
                    _finite_number(move_entry.get("visit_count")) is not None,
                    _finite_number(move_entry.get("visit_count"))
                    if _finite_number(move_entry.get("visit_count")) is not None
                    else float("-inf"),
                    _visit_value(trace_point, move) is not None,
                    _visit_value(trace_point, move)
                    if _visit_value(trace_point, move) is not None
                    else float("-inf"),
                    _finite_number(move_entry.get("q_value")) is not None,
                    _finite_number(move_entry.get("q_value"))
                    if _finite_number(move_entry.get("q_value")) is not None
                    else float("-inf"),
                    move == full_search_selected_move,
                ),
                move,
            )
        )

    return max(ranked_moves)[1], "projected_to_pair_best_move"


def pair_project_trace_points(trace_capture_artifact: dict) -> tuple[list[dict], dict]:
    row_context = trace_capture_artifact.get("row_context") or {}
    reference_move = row_context.get("reference_move")
    full_search_selected_move = row_context.get("full_search_selected_move")
    rerun_trace = trace_capture_artifact.get("rerun_trace") or {}
    rerun_trace_points = rerun_trace.get("trace_points")
    if not isinstance(rerun_trace_points, list) or not rerun_trace_points:
        raise SystemExit(
            "trace capture artifact must include non-empty rerun_trace.trace_points"
        )

    projected_trace_points = []
    selected_move_rewrites = 0
    reference_move_by_prior_rewrites = 0
    projected_simulations = []
    projection_reasons = []

    for index, trace_point in enumerate(rerun_trace_points):
        projected_trace_point = copy.deepcopy(trace_point)
        projected_selected_move, reason = _projected_selected_move(
            projected_trace_point,
            reference_move=reference_move,
            full_search_selected_move=full_search_selected_move,
        )
        if projected_trace_point.get("selected_move") != projected_selected_move:
            selected_move_rewrites += 1
            projected_simulations.append(projected_trace_point.get("simulation"))
            projection_reasons.append(
                {
                    "trace_point_index": index,
                    "simulation": projected_trace_point.get("simulation"),
                    "original_selected_move": projected_trace_point.get(
                        "selected_move"
                    ),
                    "projected_selected_move": projected_selected_move,
                    "reason": reason,
                }
            )
        projected_trace_point["selected_move"] = projected_selected_move
        if projected_trace_point.get("reference_move_by_prior") != reference_move:
            reference_move_by_prior_rewrites += 1
        projected_trace_point["reference_move_by_prior"] = reference_move
        projected_trace_points.append(projected_trace_point)

    return projected_trace_points, {
        "selected_move_rewrites": selected_move_rewrites,
        "reference_move_by_prior_rewrites": reference_move_by_prior_rewrites,
        "projected_simulations": projected_simulations,
        "projection_reasons": projection_reasons,
    }


def build_pair_projected_trace_capture_artifact(
    source_artifact: dict,
    *,
    projected_trace_points: list[dict],
    projection_summary: dict,
    out_path: Path,
) -> dict:
    row_context = source_artifact.get("row_context") or {}
    reference_move = row_context.get("reference_move")
    full_search_selected_move = row_context.get("full_search_selected_move")
    extracted_trace_points = (
        (source_artifact.get("extracted_trace") or {}).get("trace_points")
    ) or []
    insufficiency_reasons = capture_002_trace_capture._validate_trace_points(
        trace_points=projected_trace_points,
        reference_move=reference_move,
        full_search_selected_move=full_search_selected_move,
    )
    trace_diff_summary = capture_002_trace_capture._trace_diff_summary(
        extracted_trace_points=extracted_trace_points,
        final_trace_points=projected_trace_points,
    )
    trace_diff_summary["full_search_selected_move"] = full_search_selected_move
    trace_diff_summary["final_trace_matches_full_search_selected_move"] = (
        bool(projected_trace_points)
        and projected_trace_points[-1].get("selected_move") == full_search_selected_move
    )

    return {
        "schema": capture_002_trace_capture.SCHEMA,
        "row_id": source_artifact.get("row_id"),
        "capture_mode": source_artifact.get("capture_mode"),
        "trace_origin": "rerun" if not insufficiency_reasons else "insufficient",
        "reference_move": reference_move,
        "full_search_selected_move": full_search_selected_move,
        "trace_points": copy.deepcopy(projected_trace_points),
        "insufficiency_reasons": insufficiency_reasons,
        "upstream_inputs": copy.deepcopy(source_artifact.get("upstream_inputs")),
        "row_context": copy.deepcopy(source_artifact.get("row_context")),
        "upstream_context": copy.deepcopy(source_artifact.get("upstream_context")),
        "extracted_trace": copy.deepcopy(source_artifact.get("extracted_trace")),
        "rerun_trace": {
            "trace_origin": "rerun",
            "trace_points": copy.deepcopy(projected_trace_points),
            "insufficiency_reasons": insufficiency_reasons,
        },
        "final_trace": {
            "trace_origin": "rerun" if not insufficiency_reasons else "insufficient",
            "trace_points": copy.deepcopy(projected_trace_points),
            "insufficiency_reasons": insufficiency_reasons,
        },
        "trace_validation": {
            "extracted": copy.deepcopy(
                ((source_artifact.get("trace_validation") or {}).get("extracted")) or {}
            ),
            "final": capture_002_trace_capture._trace_validation_summary(
                trace_points=projected_trace_points,
                reference_move=reference_move,
                full_search_selected_move=full_search_selected_move,
            ),
        },
        "trace_diff_summary": trace_diff_summary,
        "provenance": copy.deepcopy(source_artifact.get("provenance")),
        "source_shared_drift_artifact": copy.deepcopy(
            source_artifact.get("source_shared_drift_artifact")
        ),
        "original_rerun_trace": copy.deepcopy(source_artifact.get("rerun_trace")),
        "pair_projection_policy": {
            "selected_move": (
                "Keep selected_move when already in {reference_move, full_search_selected_move}; "
                "otherwise choose the higher-ranked pair move by selection_score, visit_count, "
                "pair visits, q_value, then prefer full_search_selected_move."
            ),
            "reference_move_by_prior": "Set to reference_move for every checkpoint.",
        },
        "pair_projection_summary": copy.deepcopy(projection_summary),
        "artifact_write_summary": {
            "trace_capture_path": str(out_path),
            "trace_capture_sha256": None,
            "regenerated_shared_drift_written": False,
            "regenerated_shared_drift_path": None,
            "regenerated_shared_drift_sha256": None,
            "regenerated_shared_drift_skip_reason": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    source_trace_capture_path = resolve_path(root, args.source_trace_capture_artifact)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    source_trace_capture_artifact = load_json(source_trace_capture_path)
    projected_trace_points, projection_summary = pair_project_trace_points(
        source_trace_capture_artifact
    )

    projected_trace_capture_path = (
        out_root / "capture_002_trace_pair_projected_capture.json"
    )
    projected_trace_capture_artifact = build_pair_projected_trace_capture_artifact(
        source_trace_capture_artifact,
        projected_trace_points=projected_trace_points,
        projection_summary=projection_summary,
        out_path=projected_trace_capture_path,
    )

    regenerated_shared_drift_path = (
        out_root / "capture_002_trace_pair_projected_shared_drift.json"
    )
    regenerated_shared_drift_artifact = None
    if not projected_trace_capture_artifact["insufficiency_reasons"]:
        regenerated_shared_drift_artifact = (
            capture_002_trace_capture.build_regenerated_shared_drift_artifact(
                projected_trace_capture_artifact
            )
        )

    if regenerated_shared_drift_artifact is not None:
        write_json(regenerated_shared_drift_path, regenerated_shared_drift_artifact)
        projected_trace_capture_artifact["artifact_write_summary"][
            "regenerated_shared_drift_written"
        ] = True
        projected_trace_capture_artifact["artifact_write_summary"][
            "regenerated_shared_drift_path"
        ] = str(regenerated_shared_drift_path)
        projected_trace_capture_artifact["artifact_write_summary"][
            "regenerated_shared_drift_sha256"
        ] = capture_002_trace_capture.sha256_file(regenerated_shared_drift_path)
    else:
        projected_trace_capture_artifact["artifact_write_summary"][
            "regenerated_shared_drift_skip_reason"
        ] = "pair_projected_trace_not_downstream_ready"

    write_json(projected_trace_capture_path, projected_trace_capture_artifact)
    projected_trace_capture_artifact["artifact_write_summary"][
        "trace_capture_sha256"
    ] = capture_002_trace_capture.sha256_file(projected_trace_capture_path)
    write_json(projected_trace_capture_path, projected_trace_capture_artifact)

    paths = {
        "source_trace_capture": str(source_trace_capture_path),
        "pair_projected_trace_capture": str(projected_trace_capture_path),
    }
    decisions = {
        "pair_projected_trace_capture": {
            "schema": projected_trace_capture_artifact.get("schema"),
            "classification": None,
            "decision": None,
            "trace_origin": projected_trace_capture_artifact.get("trace_origin"),
        }
    }
    stopped_early = False
    stop_reason = None

    if regenerated_shared_drift_artifact is None:
        stopped_early = True
        stop_reason = "pair_projected_trace_not_downstream_ready"
    else:
        paths["pair_projected_shared_drift"] = str(regenerated_shared_drift_path)
        loaded_harmonized_source = capture_002_selection_score_trace.load_source_shared_drift_artifact_document(
            regenerated_shared_drift_artifact,
            artifact_path=str(regenerated_shared_drift_path),
        )
        default_selection_score = capture_002_selection_score_trace.build_payload(
            loaded_harmonized_source
        )
        relaxed_selection_score = capture_002_selection_score_trace.build_payload(
            loaded_harmonized_source,
            material_visit_share_margin=0.04,
        )
        default_selection_score_path = (
            out_root / "capture_002_pair_projected_selection_score_trace.json"
        )
        relaxed_selection_score_path = (
            out_root / "capture_002_pair_projected_selection_score_trace_relaxed.json"
        )
        write_json(default_selection_score_path, default_selection_score)
        write_json(relaxed_selection_score_path, relaxed_selection_score)
        paths["pair_projected_selection_score_trace"] = str(
            default_selection_score_path
        )
        paths["pair_projected_selection_score_trace_relaxed"] = str(
            relaxed_selection_score_path
        )
        decisions["pair_projected_selection_score_trace"] = {
            "schema": default_selection_score.get("schema"),
            "classification": (
                (default_selection_score.get("classification") or {}).get(
                    "classification"
                )
            ),
            "decision": default_selection_score.get("decision"),
            "trace_origin": default_selection_score.get("trace_origin"),
        }
        decisions["pair_projected_selection_score_trace_relaxed"] = {
            "schema": relaxed_selection_score.get("schema"),
            "classification": (
                (relaxed_selection_score.get("classification") or {}).get(
                    "classification"
                )
            ),
            "decision": relaxed_selection_score.get("decision"),
            "trace_origin": relaxed_selection_score.get("trace_origin"),
        }
        if (
            default_selection_score.get("decision")
            != "write_002_unresolved_trace_review_spec"
        ):
            stopped_early = True
            stop_reason = "pair_projected_selection_score_not_unresolved"
        elif (
            relaxed_selection_score.get("decision")
            != "write_002_unresolved_trace_review_spec"
        ):
            stopped_early = True
            stop_reason = "pair_projected_selection_score_relaxed_not_unresolved"

    summary = {
        "schema": "azlite_capture_002_trace_pair_harmonization_from_search_control_v1",
        "run_id": args.run_id,
        "source_trace_capture_artifact": str(source_trace_capture_path),
        "paths": paths,
        "projection_summary": projection_summary,
        "decisions": decisions,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
    summary_path = out_root / "capture_002_trace_pair_harmonization_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
