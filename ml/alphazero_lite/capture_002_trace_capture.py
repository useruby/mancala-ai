from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path

from ml.alphazero_lite import capture_002_003_search_policy_arbitration as search_policy_arbitration
from ml.alphazero_lite import capture_002_003_row_split_followup as row_split_followup
from ml.alphazero_lite import capture_002_selection_score_trace as selection_score_trace

SCHEMA = "azlite_capture_002_trace_capture_v1"
SOURCE_SHARED_DRIFT_SCHEMA = "azlite_shared_full_search_drift_diagnostic_v1"
ROW_ID = "capture_available-002"
CAPTURE_MODES = {"extract_only", "extract_then_rerun"}
TRACE_ORIGINS = {"extracted", "rerun", "insufficient"}


def sha256_file(path: Path) -> str:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
        artifact_write_summary = payload.get("artifact_write_summary")
        if isinstance(artifact_write_summary, dict) and "trace_capture_sha256" in artifact_write_summary:
            normalized_payload = copy.deepcopy(payload)
            normalized_payload["artifact_write_summary"]["trace_capture_sha256"] = None
            if "regenerated_shared_drift_sha256" in normalized_payload["artifact_write_summary"]:
                normalized_payload["artifact_write_summary"]["regenerated_shared_drift_sha256"] = None
            raw_bytes = (json.dumps(normalized_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    digest = hashlib.sha256()
    digest.update(raw_bytes)
    return digest.hexdigest()


def _source_payload_snapshot(path: Path) -> tuple[dict, str]:
    raw_bytes = path.read_bytes()
    source_payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("source shared drift artifact must be a JSON object")
    artifact_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    return source_payload, artifact_sha256


def _source_provenance_type_from_payload(source_payload: dict) -> str:
    provenance_source = ((source_payload.get("selected_artifact") or {}).get("provenance_source"))
    if provenance_source == "materialized_fixture":
        return "fixture"
    if provenance_source in {"selection_manifest.selected_target", "selection_manifest.selected_artifact"}:
        return "artifact"
    raise ValueError("source shared drift artifact selected_artifact provenance_source is unsupported")


def _source_provenance_type(path: Path) -> str:
    source_payload, _ = _source_payload_snapshot(path)
    return _source_provenance_type_from_payload(source_payload)


def _validated_source_artifact_from_payload(source_payload: dict) -> dict:
    if source_payload.get("schema") != SOURCE_SHARED_DRIFT_SCHEMA:
        raise ValueError(
            f"source shared drift artifact has wrong schema: expected {SOURCE_SHARED_DRIFT_SCHEMA}"
        )

    classification = source_payload.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("source shared drift artifact must include classification")
    if classification.get("classification") != "shared_mechanism_disproved":
        raise ValueError("source shared drift artifact classification must be shared_mechanism_disproved")

    rows = source_payload.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("source shared drift artifact rows must be a dict")
    normalized_rows = {
        row_id: row_split_followup._normalize_row(row_id, rows.get(row_id))
        for row_id in row_split_followup.ROW_IDS
    }

    paired_summary = source_payload.get("paired_summary")
    if not isinstance(paired_summary, dict):
        raise ValueError("source shared drift artifact must include paired_summary")
    normalized_paired_summary = dict(paired_summary)
    normalized_paired_summary["probe_mode_failure_paths"] = row_split_followup._validate_probe_mode_failure_paths(
        paired_summary
    )
    normalized_paired_summary["probe_mode_selected_moves"] = row_split_followup._validate_probe_mode_selected_moves(
        paired_summary=paired_summary,
        rows=rows,
    )
    normalized_paired_summary["shared_mechanism_supported"] = row_split_followup._validate_shared_mechanism_supported(
        paired_summary=paired_summary
    )

    for row_id in row_split_followup.ROW_IDS:
        for probe_mode in row_split_followup.PROBE_MODE_KEYS:
            paired_selected_move = normalized_paired_summary["probe_mode_selected_moves"][row_id][probe_mode]
            row_selected_move = normalized_rows[row_id]["probe_mode_traces"][probe_mode]["selected_move"]
            if paired_selected_move != row_selected_move:
                raise ValueError(
                    f"source shared drift artifact paired_summary probe_mode_selected_moves {row_id} {probe_mode} must match row trace"
                )
        if normalized_paired_summary["probe_mode_selected_moves"][row_id]["full_search"] != normalized_rows[row_id][
            "full_search_selected_move"
        ]:
            raise ValueError(
                f"source shared drift artifact paired_summary probe_mode_selected_moves {row_id} full_search must match full_search_selected_move"
            )

    selected_artifact = row_split_followup._validate_selected_artifact(source_payload.get("selected_artifact"))
    settings = row_split_followup._validate_settings(source_payload.get("settings"))

    decision = source_payload.get("decision")
    if not isinstance(decision, str) or not decision:
        raise ValueError("source shared drift artifact must include decision")
    if decision != row_split_followup.EXPECTED_SOURCE_DECISION:
        raise ValueError(
            f"source shared drift artifact decision must be {row_split_followup.EXPECTED_SOURCE_DECISION}"
        )

    return {
        "schema": source_payload["schema"],
        "classification": classification,
        "decision": decision,
        "selected_artifact": selected_artifact,
        "settings": settings,
        "paired_summary": normalized_paired_summary,
        "rows": normalized_rows,
    }


def load_source_shared_drift_artifact(path: Path, *, allow_fixture_provenance: bool) -> dict:
    source_payload, artifact_sha256 = _source_payload_snapshot(path)
    source_provenance_type = _source_provenance_type_from_payload(source_payload)
    if source_provenance_type == "fixture" and not allow_fixture_provenance:
        raise ValueError("source shared drift artifact fixture provenance requires allow_fixture_provenance")

    loaded_artifact = _validated_source_artifact_from_payload(source_payload)

    return {
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "schema": loaded_artifact["schema"],
        "classification": copy.deepcopy(loaded_artifact["classification"]),
        "decision": loaded_artifact["decision"],
        "selected_artifact": copy.deepcopy(loaded_artifact["selected_artifact"]),
        "settings": copy.deepcopy(loaded_artifact["settings"]),
        "paired_summary": copy.deepcopy(loaded_artifact["paired_summary"]),
        "row": copy.deepcopy(loaded_artifact["rows"][ROW_ID]),
        "source_payload": json.loads(json.dumps(source_payload)),
        "source_provenance": {
            "type": source_provenance_type,
            "artifact_path": str(path),
            "artifact_sha256": artifact_sha256,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and validate the 002 full-search trace from a shared drift artifact"
    )
    parser.add_argument("--source-shared-drift-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-fixture-provenance", action="store_true")
    parser.add_argument(
        "--capture-mode",
        choices=sorted(CAPTURE_MODES),
        default="extract_then_rerun",
    )
    return parser.parse_args(argv)


def _normalize_trace_points(source_artifact: dict) -> list[dict]:
    row = source_artifact.get("row") or {}
    reference_move = row.get("reference_move")
    trace_points = []
    raw_trace_points = []
    root_start = row.get("root_start")
    if root_start is not None:
        raw_trace_points.append(root_start)
    raw_trace_points.extend(row.get("snapshots") or [])

    legal_moves = list(row.get("legal_moves") or [])
    for index, trace_point in enumerate(raw_trace_points):
        if not isinstance(trace_point, dict):
            raise ValueError(f"source shared drift artifact trace_points[{index}] must be a dict")

        selected_move = trace_point.get("selected_move")
        reference_move_by_prior = trace_point.get("reference_move_by_prior")
        simulation = trace_point.get("simulation")
        visits = trace_point.get("visits")
        moves = trace_point.get("moves")

        if isinstance(selected_move, bool) or not isinstance(selected_move, int):
            raise ValueError(f"source shared drift artifact trace_points[{index}] selected_move must be int")
        if selected_move not in legal_moves:
            raise ValueError(f"source shared drift artifact trace_points[{index}] selected_move must be legal")
        if reference_move_by_prior is None:
            reference_move_by_prior = reference_move
        if isinstance(reference_move_by_prior, bool) or not isinstance(reference_move_by_prior, int):
            raise ValueError(
                f"source shared drift artifact trace_points[{index}] reference_move_by_prior must be int"
            )
        if reference_move_by_prior not in legal_moves:
            raise ValueError(
                f"source shared drift artifact trace_points[{index}] reference_move_by_prior must be legal"
            )
        if isinstance(simulation, bool) or not isinstance(simulation, (int, float)) or not math.isfinite(simulation):
            raise ValueError(f"source shared drift artifact trace_points[{index}] simulation must be finite numeric")
        if not isinstance(visits, list):
            raise ValueError(f"source shared drift artifact trace_points[{index}] visits must be a list")
        if not isinstance(moves, list):
            raise ValueError(f"source shared drift artifact trace_points[{index}] moves must be a list")

        normalized_moves = []
        for move_index, move_entry in enumerate(moves):
            if not isinstance(move_entry, dict):
                raise ValueError(
                    f"source shared drift artifact trace_points[{index}] moves[{move_index}] must be a dict"
                )
            move = move_entry.get("move")
            if isinstance(move, bool) or not isinstance(move, int) or move not in legal_moves:
                raise ValueError(
                    f"source shared drift artifact trace_points[{index}] moves[{move_index}] move must be legal int"
                )
            normalized_move_entry = {
                "move": int(move),
                "prior": float(move_entry["prior"]),
                "q_value": float(move_entry["q_value"]),
                "selection_score": float(move_entry["selection_score"]),
                "used_fpu": bool(move_entry["used_fpu"]),
                "visit_count": float(move_entry["visit_count"]),
            }
            normalized_moves.append(normalized_move_entry)

        trace_points.append(
            {
                "selected_move": int(selected_move),
                "reference_move_by_prior": int(reference_move_by_prior),
                "simulation": float(simulation),
                "visits": [float(visit) for visit in visits],
                "moves": normalized_moves,
            }
        )

    return trace_points


def _validate_trace_points(*, trace_points: list[dict], reference_move, full_search_selected_move) -> list[str]:
    insufficiency_reasons = []

    if isinstance(reference_move, bool) or not isinstance(reference_move, int):
        insufficiency_reasons.append("missing_reference_move")
    if isinstance(full_search_selected_move, bool) or not isinstance(full_search_selected_move, int):
        insufficiency_reasons.append("missing_full_search_selected_move")
    if insufficiency_reasons:
        return insufficiency_reasons

    if not trace_points:
        insufficiency_reasons.append("missing_trace_points")
        return insufficiency_reasons
    if len(trace_points) < 2:
        insufficiency_reasons.append("too_few_trace_points")

    if full_search_selected_move not in [trace_point.get("selected_move") for trace_point in trace_points]:
        insufficiency_reasons.append("trace_never_reaches_full_search_selected_move")

    final_trace_point = trace_points[-1]
    if final_trace_point.get("selected_move") != full_search_selected_move:
        insufficiency_reasons.append("final_trace_selected_move_mismatch")

    expected_pair = {reference_move, full_search_selected_move}
    previous_simulation = None
    for trace_point in trace_points:
        if trace_point.get("selected_move") not in expected_pair:
            insufficiency_reasons.append("trace_points_pair_mismatch")
            break
        if trace_point.get("reference_move_by_prior") != reference_move:
            insufficiency_reasons.append("trace_points_pair_mismatch")
            break

    for trace_point in trace_points:
        simulation = trace_point.get("simulation")
        if previous_simulation is not None and simulation < previous_simulation:
            insufficiency_reasons.append("trace_points_out_of_order")
            break
        previous_simulation = simulation

    downstream_sufficiency = selection_score_trace._trace_sufficiency(
        trace_points,
        reference_move=reference_move,
        full_search_selected_move=full_search_selected_move,
    )
    insufficiency_reasons.extend(downstream_sufficiency["insufficiency_reasons"])

    return insufficiency_reasons


def _source_block(source_artifact: dict) -> dict:
    return {
        "artifact_path": source_artifact.get("artifact_path"),
        "artifact_sha256": source_artifact.get("artifact_sha256"),
        "schema": source_artifact.get("schema"),
        "decision": source_artifact.get("decision"),
        "classification": copy.deepcopy(source_artifact.get("classification")),
        "selected_artifact": copy.deepcopy(source_artifact.get("selected_artifact")),
        "source_provenance": copy.deepcopy(source_artifact.get("source_provenance")),
        "source_payload": copy.deepcopy(source_artifact.get("source_payload")),
    }


def _row_context(source_artifact: dict) -> dict:
    row = source_artifact.get("row") or {}
    return {
        "row_id": row.get("row_id"),
        "canonical_state": row.get("canonical_state"),
        "legal_moves": copy.deepcopy(row.get("legal_moves")),
        "reference_move": row.get("reference_move"),
        "full_search_selected_move": row.get("full_search_selected_move"),
    }


def _upstream_inputs(source_artifact: dict, *, reason: str | None = None) -> dict:
    if reason is not None:
        return {
            "seed": None,
            "simulation_count": None,
            "search_settings": None,
            "reason": reason,
        }

    settings = source_artifact.get("settings") or {}
    return {
        "seed": settings.get("seed"),
        "simulation_count": settings.get("simulation_count"),
        "search_settings": copy.deepcopy(settings.get("search_settings")),
        "reason": None,
    }


def _rerun_trace_points_or_empty(rerun_payload: dict | None) -> list[dict]:
    if not isinstance(rerun_payload, dict):
        return []

    rerun_trace_points = rerun_payload.get("trace_points")
    if not isinstance(rerun_trace_points, list):
        return []
    return copy.deepcopy(rerun_trace_points)


def _rerun_insufficiency_reasons(rerun_payload: dict | None) -> list[str]:
    if not isinstance(rerun_payload, dict):
        return []
    rerun_insufficiency_reasons = rerun_payload.get("insufficiency_reasons")
    if not isinstance(rerun_insufficiency_reasons, list):
        return []
    return [reason for reason in rerun_insufficiency_reasons if isinstance(reason, str) and reason]


def _normalize_rerun_trace_points(trace_points: list[dict], *, reference_move: int) -> list[dict]:
    normalized_trace_points = []
    for trace_point in trace_points:
        normalized_trace_point = copy.deepcopy(trace_point)
        if normalized_trace_point.get("reference_move_by_prior") is None:
            normalized_trace_point["reference_move_by_prior"] = reference_move
        normalized_trace_points.append(normalized_trace_point)
    return normalized_trace_points


def _null_backed_rerun_context(reason: str) -> dict:
    return _upstream_inputs({}, reason=reason)


def _trace_section(*, trace_origin: str, trace_points: list[dict], insufficiency_reasons: list[str]) -> dict:
    return {
        "trace_origin": trace_origin,
        "trace_points": copy.deepcopy(trace_points),
        "insufficiency_reasons": list(insufficiency_reasons),
    }


def _trace_validation_summary(*, trace_points: list[dict], reference_move, full_search_selected_move) -> dict:
    return {
        "insufficiency_reasons": _validate_trace_points(
            trace_points=trace_points,
            reference_move=reference_move,
            full_search_selected_move=full_search_selected_move,
        )
    }


def _collect_field_paths(value, *, prefix: str = "") -> set[str]:
    field_paths = set()
    if isinstance(value, dict):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            field_paths.add(next_prefix)
            field_paths.update(_collect_field_paths(nested_value, prefix=next_prefix))
    elif isinstance(value, list):
        for nested_value in value:
            field_paths.update(_collect_field_paths(nested_value, prefix=prefix))
    return field_paths


def _compare_field_changes(*, extracted_value, final_value, prefix: str, added_fields: set[str], removed_fields: set[str], changed_fields: set[str]) -> None:
    if isinstance(extracted_value, dict) and isinstance(final_value, dict):
        for key in extracted_value.keys() - final_value.keys():
            removed_fields.add(f"{prefix}.{key}" if prefix else str(key))
        for key in final_value.keys() - extracted_value.keys():
            added_fields.add(f"{prefix}.{key}" if prefix else str(key))
        for key in extracted_value.keys() & final_value.keys():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _compare_field_changes(
                extracted_value=extracted_value[key],
                final_value=final_value[key],
                prefix=next_prefix,
                added_fields=added_fields,
                removed_fields=removed_fields,
                changed_fields=changed_fields,
            )
        return

    if (
        isinstance(extracted_value, list)
        and isinstance(final_value, list)
        and all(isinstance(item, dict) and isinstance(item.get("move"), int) for item in extracted_value)
        and all(isinstance(item, dict) and isinstance(item.get("move"), int) for item in final_value)
    ):
        extracted_moves = {item["move"]: item for item in extracted_value}
        final_moves = {item["move"]: item for item in final_value}
        for move in extracted_moves.keys() & final_moves.keys():
            _compare_field_changes(
                extracted_value=extracted_moves[move],
                final_value=final_moves[move],
                prefix=prefix,
                added_fields=added_fields,
                removed_fields=removed_fields,
                changed_fields=changed_fields,
            )
        return

    if extracted_value != final_value:
        changed_fields.add(prefix)


def _field_change_summary(*, extracted_trace_points: list[dict], final_trace_points: list[dict]) -> dict:
    added_fields = set()
    removed_fields = set()
    changed_fields = set()

    if extracted_trace_points and final_trace_points:
        _compare_field_changes(
            extracted_value=extracted_trace_points[0],
            final_value=final_trace_points[0],
            prefix="",
            added_fields=added_fields,
            removed_fields=removed_fields,
            changed_fields=changed_fields,
        )

    extracted_snapshots = extracted_trace_points[1:]
    final_snapshots = final_trace_points[1:]
    for extracted_trace_point, final_trace_point in zip(reversed(extracted_snapshots), reversed(final_snapshots)):
        _compare_field_changes(
            extracted_value=extracted_trace_point,
            final_value=final_trace_point,
            prefix="",
            added_fields=added_fields,
            removed_fields=removed_fields,
            changed_fields=changed_fields,
        )

    return {
        "field_change_counts": {
            "added_fields": len(added_fields),
            "removed_fields": len(removed_fields),
            "changed_fields": len(changed_fields),
        },
        "field_changes": {
            "added_fields": sorted(added_fields),
            "removed_fields": sorted(removed_fields),
            "changed_fields": sorted(changed_fields),
        },
    }


def _trace_diff_summary(*, extracted_trace_points: list[dict], final_trace_points: list[dict]) -> dict:
    extracted_selected_move = None if not extracted_trace_points else extracted_trace_points[-1].get("selected_move")
    final_selected_move = None if not final_trace_points else final_trace_points[-1].get("selected_move")
    extracted_simulations = [trace_point.get("simulation") for trace_point in extracted_trace_points]
    final_simulations = [trace_point.get("simulation") for trace_point in final_trace_points]
    field_change_summary = _field_change_summary(
        extracted_trace_points=extracted_trace_points,
        final_trace_points=final_trace_points,
    )
    return {
        "trace_origin_changed": extracted_trace_points != final_trace_points,
        "trace_points_changed": extracted_trace_points != final_trace_points,
        "selected_move_changed": extracted_selected_move != final_selected_move,
        "simulation_sequence_changed": extracted_simulations != final_simulations,
        "root_start_changed": bool(extracted_trace_points and final_trace_points)
        and extracted_trace_points[0] != final_trace_points[0],
        "snapshots_changed": extracted_trace_points[1:] != final_trace_points[1:],
        "extracted_trace_point_count": len(extracted_trace_points),
        "final_trace_point_count": len(final_trace_points),
        "trace_point_count_delta": len(final_trace_points) - len(extracted_trace_points),
        "extracted_first_simulation": None if not extracted_trace_points else extracted_trace_points[0].get("simulation"),
        "final_first_simulation": None if not final_trace_points else final_trace_points[0].get("simulation"),
        "extracted_final_selected_move": extracted_selected_move,
        "final_final_selected_move": final_selected_move,
        "extracted_final_simulation": None if not extracted_trace_points else extracted_trace_points[-1].get("simulation"),
        "final_final_simulation": None if not final_trace_points else final_trace_points[-1].get("simulation"),
        "field_change_counts": field_change_summary["field_change_counts"],
        "field_changes": field_change_summary["field_changes"],
    }


def build_trace_capture_artifact(source_artifact: dict, *, capture_mode: str, rerun_capture=None) -> dict:
    if capture_mode not in CAPTURE_MODES:
        raise ValueError(f"capture_mode must be one of {sorted(CAPTURE_MODES)}")

    row = source_artifact.get("row") or {}
    extracted_trace_points = _normalize_trace_points(source_artifact)
    trace_points = extracted_trace_points
    insufficiency_reasons = _validate_trace_points(
        trace_points=extracted_trace_points,
        reference_move=row.get("reference_move"),
        full_search_selected_move=row.get("full_search_selected_move"),
    )
    trace_origin = "extracted" if not insufficiency_reasons else "insufficient"
    upstream_inputs = _upstream_inputs(source_artifact)
    rerun_trace = None

    if trace_origin == "insufficient" and capture_mode == "extract_then_rerun":
        if rerun_capture is None:
            upstream_inputs = _null_backed_rerun_context("rerun_blocked")
        else:
            rerun_payload = rerun_capture(source_artifact)
            rerun_trace_points = _rerun_trace_points_or_empty(rerun_payload)
            rerun_payload_reasons = _rerun_insufficiency_reasons(rerun_payload)
            if rerun_trace_points:
                normalized_rerun_trace_points = _normalize_rerun_trace_points(
                    rerun_trace_points,
                    reference_move=row.get("reference_move"),
                )
                rerun_insufficiency_reasons = _validate_trace_points(
                    trace_points=normalized_rerun_trace_points,
                    reference_move=row.get("reference_move"),
                    full_search_selected_move=row.get("full_search_selected_move"),
                )
                rerun_trace = _trace_section(
                    trace_origin="rerun",
                    trace_points=normalized_rerun_trace_points,
                    insufficiency_reasons=rerun_insufficiency_reasons,
                )
                trace_points = normalized_rerun_trace_points
                insufficiency_reasons = rerun_insufficiency_reasons
                trace_origin = "rerun" if not rerun_insufficiency_reasons else "insufficient"
                if rerun_insufficiency_reasons:
                    upstream_inputs = _null_backed_rerun_context("rerun_unusable_trace_insufficient")
            else:
                rerun_trace = _trace_section(
                    trace_origin="insufficient",
                    trace_points=[],
                    insufficiency_reasons=rerun_payload_reasons,
                )
                upstream_inputs = _null_backed_rerun_context(
                    rerun_payload_reasons[0] if rerun_payload_reasons else "rerun_unusable_missing_trace_points"
                )

    extracted_trace = _trace_section(
        trace_origin="extracted" if not _validate_trace_points(
            trace_points=extracted_trace_points,
            reference_move=row.get("reference_move"),
            full_search_selected_move=row.get("full_search_selected_move"),
        ) else "insufficient",
        trace_points=extracted_trace_points,
        insufficiency_reasons=_validate_trace_points(
            trace_points=extracted_trace_points,
            reference_move=row.get("reference_move"),
            full_search_selected_move=row.get("full_search_selected_move"),
        ),
    )
    final_trace = _trace_section(
        trace_origin=trace_origin,
        trace_points=trace_points,
        insufficiency_reasons=insufficiency_reasons,
    )
    trace_diff_summary = _trace_diff_summary(
        extracted_trace_points=extracted_trace_points,
        final_trace_points=trace_points,
    )
    trace_diff_summary["full_search_selected_move"] = row.get("full_search_selected_move")
    trace_diff_summary["final_trace_matches_full_search_selected_move"] = (
        bool(trace_points) and trace_points[-1].get("selected_move") == row.get("full_search_selected_move")
    )

    return {
        "schema": SCHEMA,
        "row_id": row.get("row_id"),
        "capture_mode": capture_mode,
        "trace_origin": trace_origin,
        "reference_move": row.get("reference_move"),
        "full_search_selected_move": row.get("full_search_selected_move"),
        "trace_points": trace_points,
        "insufficiency_reasons": insufficiency_reasons,
        "upstream_inputs": upstream_inputs,
        "row_context": _row_context(source_artifact),
        "upstream_context": copy.deepcopy(upstream_inputs),
        "extracted_trace": extracted_trace,
        "rerun_trace": rerun_trace,
        "final_trace": final_trace,
        "trace_validation": {
            "extracted": _trace_validation_summary(
                trace_points=extracted_trace_points,
                reference_move=row.get("reference_move"),
                full_search_selected_move=row.get("full_search_selected_move"),
            ),
            "final": _trace_validation_summary(
                trace_points=trace_points,
                reference_move=row.get("reference_move"),
                full_search_selected_move=row.get("full_search_selected_move"),
            ),
        },
        "trace_diff_summary": trace_diff_summary,
        "provenance": {
            "trace_capture_schema": SCHEMA,
            "source_shared_drift_artifact_path": source_artifact.get("artifact_path"),
            "source_shared_drift_artifact_sha256": source_artifact.get("artifact_sha256"),
            "source_shared_drift_artifact_schema": source_artifact.get("schema"),
            "source_shared_drift_artifact_decision": source_artifact.get("decision"),
        },
        "source_shared_drift_artifact": _source_block(source_artifact),
    }


def build_regenerated_shared_drift_artifact(trace_capture_artifact: dict) -> dict | None:
    if not isinstance(trace_capture_artifact, dict):
        return None

    if trace_capture_artifact.get("trace_origin") not in {"extracted", "rerun"}:
        return None

    trace_points = trace_capture_artifact.get("trace_points")
    if not isinstance(trace_points, list) or len(trace_points) < 2:
        return None

    source_artifact = (trace_capture_artifact.get("source_shared_drift_artifact") or {}).get("source_payload")
    if not isinstance(source_artifact, dict):
        return None

    regenerated = json.loads(json.dumps(source_artifact))
    rows = regenerated.get("rows")
    if not isinstance(rows, dict):
        return None

    row = rows.get(ROW_ID)
    if not isinstance(row, dict):
        return None

    legal_moves = row.get("legal_moves")
    if not isinstance(legal_moves, list):
        return None

    expected_full_search_selected_move = trace_capture_artifact.get("full_search_selected_move")
    if trace_points[-1].get("selected_move") != expected_full_search_selected_move:
        return None

    row["root_start"] = copy.deepcopy(trace_points[0])
    row["snapshots"] = copy.deepcopy(trace_points[1:])

    final_trace_point = trace_points[-1]
    reference_move = row.get("reference_move")
    if (
        isinstance(reference_move, bool)
        or not isinstance(reference_move, int)
        or reference_move >= len(final_trace_point.get("visits") or [])
        or expected_full_search_selected_move >= len(final_trace_point.get("visits") or [])
    ):
        return None

    root_start_trace_point = trace_points[0]
    root_start_visits = root_start_trace_point.get("visits")
    if (
        not isinstance(root_start_visits, list)
        or reference_move >= len(root_start_visits)
        or expected_full_search_selected_move >= len(root_start_visits)
    ):
        return None

    final_deltas = {
        "selected_visits": float(final_trace_point["visits"][expected_full_search_selected_move])
        - float(root_start_visits[expected_full_search_selected_move]),
        "reference_visits": float(final_trace_point["visits"][reference_move]) - float(root_start_visits[reference_move]),
    }
    row["final_deltas"] = final_deltas
    row["full_search_selected_move"] = expected_full_search_selected_move

    probe_mode_traces = row.get("probe_mode_traces")
    if isinstance(probe_mode_traces, dict) and isinstance(probe_mode_traces.get("full_search"), dict):
        probe_mode_traces["full_search"]["root_start"] = copy.deepcopy(trace_points[0])
        probe_mode_traces["full_search"]["snapshots"] = copy.deepcopy(trace_points[1:])
        probe_mode_traces["full_search"]["final_deltas"] = copy.deepcopy(final_deltas)
        probe_mode_traces["full_search"]["selected_move"] = expected_full_search_selected_move

    paired_summary = regenerated.get("paired_summary")
    if isinstance(paired_summary, dict):
        selected_moves = paired_summary.get("probe_mode_selected_moves")
        if isinstance(selected_moves, dict) and isinstance(selected_moves.get(ROW_ID), dict):
            selected_moves[ROW_ID]["full_search"] = expected_full_search_selected_move

    regenerated["trace_capture_provenance"] = {
        "trace_capture_schema": trace_capture_artifact.get("schema"),
        "trace_origin": trace_capture_artifact.get("trace_origin"),
        "row_id": trace_capture_artifact.get("row_id"),
        "trace_capture_artifact_path": ((trace_capture_artifact.get("artifact_write_summary") or {}).get("trace_capture_path")),
        "trace_capture_artifact_sha256": ((trace_capture_artifact.get("artifact_write_summary") or {}).get("trace_capture_sha256")),
        "source_shared_drift_artifact_path": (trace_capture_artifact.get("source_shared_drift_artifact") or {}).get(
            "artifact_path"
        ),
        "source_shared_drift_artifact_sha256": (trace_capture_artifact.get("source_shared_drift_artifact") or {}).get(
            "artifact_sha256"
        ),
    }
    regenerated["trace_capture_diff_summary"] = copy.deepcopy(trace_capture_artifact.get("trace_diff_summary"))

    try:
        downstream_payload = selection_score_trace.classify_source_shared_drift_artifact_document(
            regenerated,
            artifact_path="<regenerated-shared-drift-artifact>",
        )
    except ValueError:
        return None

    classification = ((downstream_payload.get("classification") or {}).get("classification"))
    if classification not in {
        "selection_score_pressure_confirmed",
        "q_support_precedes_selection_score",
        "unresolved",
    }:
        return None

    if downstream_payload.get("trace_origin") == "insufficient":
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            canonical_validation_path = Path(tmp) / "regenerated_shared_drift.json"
            canonical_validation_path.write_text(json.dumps(regenerated), encoding="utf-8")
            row_split_followup.load_source_shared_drift_artifact(canonical_validation_path)
    except ValueError:
        return None

    return regenerated


def _default_rerun_result(loaded_source_artifact: dict, *, capture_mode: str) -> dict | None:
    if capture_mode != "extract_then_rerun":
        return None

    row = loaded_source_artifact.get("row") or {}
    settings = loaded_source_artifact.get("settings") or {}
    search_settings = settings.get("search_settings") or {}
    artifact_path = ((loaded_source_artifact.get("selected_artifact") or {}).get("path"))
    canonical_state = row.get("canonical_state")

    try:
        state = _state_from_canonical_state(canonical_state)
    except ValueError:
        return {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]}

    if not isinstance(artifact_path, str) or not artifact_path:
        return {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]}
    if isinstance(settings.get("simulation_count"), bool) or not isinstance(settings.get("simulation_count"), int):
        return {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]}
    if isinstance(settings.get("seed"), bool) or not isinstance(settings.get("seed"), int):
        return {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]}
    if isinstance(search_settings.get("c_puct"), bool) or not isinstance(search_settings.get("c_puct"), (int, float)):
        return {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]}

    try:
        probe_summary = search_policy_arbitration.probe_artifact_position(
            artifact_path=artifact_path,
            state=state,
            simulations=settings["simulation_count"],
            seed=settings["seed"],
            c_puct=float(search_settings["c_puct"]),
            search_options=copy.deepcopy(search_settings),
            ablation_mode="full",
        )
    except Exception:
        return {"insufficiency_reasons": ["deterministic_rerun_failed"]}

    rerun_trace_points = copy.deepcopy(probe_summary.get("visit_snapshots") or [])

    if not rerun_trace_points:
        return {"insufficiency_reasons": ["deterministic_rerun_inputs_incomplete"]}

    return {"trace_points": rerun_trace_points}


def _state_from_canonical_state(canonical_state: str) -> dict:
    try:
        state = json.loads(canonical_state)
    except json.JSONDecodeError as exc:
        raise ValueError("canonical_state must be parseable JSON") from exc

    if not isinstance(state, dict):
        raise ValueError("canonical_state must be a JSON object")

    player_pits = state.get("player_pits")
    opponent_pits = state.get("opponent_pits")
    if not isinstance(player_pits, list) or not isinstance(opponent_pits, list):
        raise ValueError("canonical_state must include player_pits and opponent_pits")

    return {
        "player_pits": list(player_pits),
        "opponent_pits": list(opponent_pits),
        "player_store": int(state.get("player_store", 0)),
        "opponent_store": int(state.get("opponent_store", 0)),
        "current_player": int(state.get("current_player", 0)),
    }


def _derived_regenerated_path(out_path: Path) -> Path:
    regenerated_path = out_path.with_name("capture_002_trace_rehydrated_shared_drift.json")
    if regenerated_path == out_path:
        return out_path.with_name("capture_002_trace_rehydrated_shared_drift.artifact_2.json")
    return regenerated_path


def _stable_trace_capture_sha256(artifact: dict) -> str:
    normalized_artifact = copy.deepcopy(artifact)
    artifact_write_summary = normalized_artifact.get("artifact_write_summary") or {}
    artifact_write_summary["trace_capture_sha256"] = None
    if "regenerated_shared_drift_sha256" in artifact_write_summary:
        artifact_write_summary["regenerated_shared_drift_sha256"] = None
    normalized_artifact["artifact_write_summary"] = artifact_write_summary
    return hashlib.sha256((json.dumps(normalized_artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded_source_artifact = load_source_shared_drift_artifact(
        args.source_shared_drift_artifact,
        allow_fixture_provenance=args.allow_fixture_provenance,
    )

    rerun_result = _default_rerun_result(loaded_source_artifact, capture_mode=args.capture_mode)
    rerun_capture = None if rerun_result is None else (lambda _source_artifact: copy.deepcopy(rerun_result))
    artifact = build_trace_capture_artifact(
        loaded_source_artifact,
        capture_mode=args.capture_mode,
        rerun_capture=rerun_capture,
    )
    artifact["artifact_write_summary"] = {
        "trace_capture_path": None,
        "trace_capture_sha256": None,
        "regenerated_shared_drift_written": False,
        "regenerated_shared_drift_path": None,
        "regenerated_shared_drift_sha256": None,
        "regenerated_shared_drift_skip_reason": "trace_capture_not_downstream_ready",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact["artifact_write_summary"]["trace_capture_path"] = str(args.out)

    regenerated_path = _derived_regenerated_path(args.out)
    artifact["artifact_write_summary"]["regenerated_shared_drift_written"] = True
    artifact["artifact_write_summary"]["regenerated_shared_drift_path"] = str(regenerated_path)
    artifact["artifact_write_summary"]["regenerated_shared_drift_skip_reason"] = None

    regenerated = build_regenerated_shared_drift_artifact(artifact)
    if regenerated is not None:
        artifact["artifact_write_summary"]["trace_capture_sha256"] = _stable_trace_capture_sha256(artifact)
        regenerated["trace_capture_provenance"]["trace_capture_artifact_sha256"] = artifact["artifact_write_summary"][
            "trace_capture_sha256"
        ]
        regenerated_path.write_text(json.dumps(regenerated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact["artifact_write_summary"]["regenerated_shared_drift_sha256"] = sha256_file(regenerated_path)
    else:
        artifact["artifact_write_summary"]["regenerated_shared_drift_written"] = False
        artifact["artifact_write_summary"]["regenerated_shared_drift_path"] = None
        artifact["artifact_write_summary"]["regenerated_shared_drift_skip_reason"] = "trace_capture_not_downstream_ready"
        artifact["artifact_write_summary"]["trace_capture_sha256"] = _stable_trace_capture_sha256(artifact)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": artifact["schema"],
                "trace_origin": artifact["trace_origin"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
