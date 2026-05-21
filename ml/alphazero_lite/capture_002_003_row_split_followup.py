from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ml.alphazero_lite.kalah_rules import PITS_PER_PLAYER


SCHEMA = "azlite_capture_002_003_row_split_followup_v1"
SOURCE_SHARED_DRIFT_SCHEMA = "azlite_shared_full_search_drift_diagnostic_v1"
ROW_IDS = ["capture_available-002", "capture_available-003"]
THRESHOLDS = {
    "meaningful_q_margin": 0.03,
    "small_q_margin": 0.03,
    "meaningful_visit_share_overtake": 0.05,
    "early_snapshot_fraction": 0.10,
    "minimum_early_snapshot_count": 1,
    "material_selection_score_margin": 0.05,
    "material_prior_margin": 0.05,
}
LANE_002_DECISIONS = {
    "selection_score_overtake": "write_002_selection_score_trace_spec",
    "early_fpu_pressure": "write_002_fpu_pressure_ablation_spec",
    "early_child_value_override": "write_002_child_value_audit_spec",
    "backup_accumulation_drift": "write_002_backup_accumulation_spec",
    "prior_pressure_with_small_q": "write_002_root_pressure_spec",
    "unresolved": "stop_002_unresolved",
}
LANE_003_DECISIONS = {
    "value_only_child_q_prefers_wrong_move": "write_003_child_value_audit_spec",
    "value_only_visit_amplification_without_q": "write_003_value_only_visit_trace_spec",
    "policy_value_conflict": "write_003_policy_value_conflict_spec",
    "rule_feature_value_collision": "write_003_rule_value_collision_spec",
    "insufficient_value_trace": "write_003_value_trace_capture_spec",
    "unresolved": "stop_003_unresolved",
}
TOP_LEVEL_DECISIONS = [
    "write_002_selection_score_trace_spec",
    "write_002_fpu_pressure_ablation_spec",
    "write_002_child_value_audit_spec",
    "write_002_backup_accumulation_spec",
    "write_002_root_pressure_spec",
    "write_003_child_value_audit_spec",
    "write_003_value_only_visit_trace_spec",
    "write_003_policy_value_conflict_spec",
    "write_003_rule_value_collision_spec",
    "write_003_value_trace_capture_spec",
    "write_parallel_row_followup_specs",
    "stop_row_split_unresolved",
]
REQUIRED_FULL_SEARCH_SETTINGS_KEYS = [
    "c_puct",
    "fpu_mode",
    "normalize_values",
    "reuse_subtree",
    "root_policy_mode",
    "tactical_root_bias",
]
PROBE_MODE_KEYS = ["policy_only", "value_only", "full_search"]
ALLOWED_FAILURE_PATHS = {
    "reference_kept",
    "diverged_before_full_search",
    "full_search_drift",
    "not_applicable",
}
EXPECTED_SOURCE_DECISION = "write_row_split_followup_spec"
CONSUMED_TRACE_KEYS = {
    "capture_available-002": {"full_search"},
    "capture_available-003": {"policy_only", "value_only", "full_search"},
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_selected_artifact(selected_artifact: dict) -> dict:
    if not isinstance(selected_artifact, dict):
        raise ValueError("source shared drift artifact must include selected_artifact")

    path = selected_artifact.get("path")
    provenance_source = selected_artifact.get("provenance_source")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "source shared drift artifact selected_artifact must include non-empty path"
        )
    if not isinstance(provenance_source, str) or not provenance_source:
        raise ValueError(
            "source shared drift artifact selected_artifact must include non-empty provenance_source"
        )

    for optional_key in ["selected_artifact", "selected_target"]:
        optional_value = selected_artifact.get(optional_key)
        if optional_value is not None and not isinstance(optional_value, str):
            raise ValueError(
                f"source shared drift artifact selected_artifact {optional_key} must be string or null"
            )

    return dict(selected_artifact)


def _validate_probe_mode_failure_paths(paired_summary: dict) -> dict:
    failure_paths = paired_summary.get("probe_mode_failure_paths")
    if not isinstance(failure_paths, dict):
        raise ValueError(
            "source shared drift artifact must include paired_summary probe_mode_failure_paths"
        )

    normalized_failure_paths = {}
    for row_id in ROW_IDS:
        row_failure_paths = failure_paths.get(row_id)
        if not isinstance(row_failure_paths, dict):
            raise ValueError(
                f"source shared drift artifact paired_summary missing row {row_id}"
            )
        normalized_row_failure_paths = {}
        for probe_mode in PROBE_MODE_KEYS:
            failure_path = row_failure_paths.get(probe_mode)
            if not isinstance(failure_path, str) or not failure_path:
                raise ValueError(
                    f"source shared drift artifact paired_summary probe_mode_failure_paths missing {probe_mode} for {row_id}"
                )
            if failure_path not in ALLOWED_FAILURE_PATHS:
                raise ValueError(
                    f"source shared drift artifact paired_summary failure path {failure_path} is not allowed"
                )
            normalized_row_failure_paths[probe_mode] = failure_path
        normalized_failure_paths[row_id] = normalized_row_failure_paths

    return normalized_failure_paths


def _validate_probe_mode_selected_moves(*, paired_summary: dict, rows: dict) -> dict:
    selected_moves = paired_summary.get("probe_mode_selected_moves")
    if not isinstance(selected_moves, dict):
        raise ValueError(
            "source shared drift artifact must include paired_summary probe_mode_selected_moves"
        )

    normalized_selected_moves = {}
    for row_id in ROW_IDS:
        row_selected_moves = selected_moves.get(row_id)
        if not isinstance(row_selected_moves, dict):
            raise ValueError(
                f"source shared drift artifact paired_summary probe_mode_selected_moves missing row {row_id}"
            )
        legal_moves = list((rows.get(row_id) or {}).get("legal_moves") or [])
        normalized_row_selected_moves = {}
        for probe_mode in PROBE_MODE_KEYS:
            selected_move = row_selected_moves.get(probe_mode)
            if (
                isinstance(selected_move, bool)
                or not isinstance(selected_move, int)
                or selected_move not in legal_moves
            ):
                raise ValueError(
                    f"source shared drift artifact paired_summary probe_mode_selected_moves {row_id} {probe_mode} must be legal int"
                )
            normalized_row_selected_moves[probe_mode] = int(selected_move)
        normalized_selected_moves[row_id] = normalized_row_selected_moves

    return normalized_selected_moves


def _validate_shared_mechanism_supported(*, paired_summary: dict) -> bool:
    shared_mechanism_supported = paired_summary.get("shared_mechanism_supported")
    if not isinstance(shared_mechanism_supported, bool):
        raise ValueError(
            "source shared drift artifact paired_summary shared_mechanism_supported must be bool"
        )
    if shared_mechanism_supported:
        raise ValueError(
            "source shared drift artifact paired_summary shared_mechanism_supported must be false"
        )
    return False


def _finite_number(value, *, context: str, allow_none: bool = False):
    if value is None and allow_none:
        return value
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"source shared drift artifact {context} must be finite numeric"
        )
    return value


def _validate_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("source shared drift artifact must include settings")

    search_settings = settings.get("search_settings")
    if not isinstance(search_settings, dict):
        raise ValueError(
            "source shared drift artifact must include settings search_settings"
        )
    for key in REQUIRED_FULL_SEARCH_SETTINGS_KEYS:
        if key not in search_settings:
            raise ValueError(
                f"source shared drift artifact search_settings missing required key: {key}"
            )
    for key in ["c_puct", "tactical_root_bias"]:
        _finite_number(search_settings[key], context=f"search_settings {key}")

    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("source shared drift artifact settings seed must be an int")

    simulation_count = settings.get("simulation_count")
    if (
        isinstance(simulation_count, bool)
        or not isinstance(simulation_count, int)
        or simulation_count <= 0
    ):
        raise ValueError(
            "source shared drift artifact settings simulation_count must be positive int"
        )

    return {
        "search_settings": dict(search_settings),
        "seed": seed,
        "simulation_count": simulation_count,
    }


def _validate_required_move(*, value, legal_moves: list[int], context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"source shared drift artifact {context} must be an int")
    if not 0 <= value < PITS_PER_PLAYER:
        raise ValueError(
            f"source shared drift artifact {context} must be between 0 and {PITS_PER_PLAYER - 1}"
        )
    if value not in legal_moves:
        raise ValueError(
            f"source shared drift artifact {context} must be present in legal_moves"
        )
    return int(value)


def _validate_canonical_state(
    *, canonical_state: str, reference_move: int, row_id: str
) -> str:
    try:
        state = json.loads(canonical_state)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"source shared drift artifact row {row_id} canonical_state must be parseable JSON"
        ) from exc

    if not isinstance(state, dict):
        raise ValueError(
            f"source shared drift artifact row {row_id} canonical_state must be a JSON object"
        )

    for key in ["player_pits", "opponent_pits"]:
        pits = state.get(key)
        if not isinstance(pits, list):
            raise ValueError(
                f"source shared drift artifact row {row_id} canonical_state {key} must be a list"
            )
        if len(pits) <= reference_move:
            raise ValueError(
                f"source shared drift artifact row {row_id} canonical_state {key} must cover reference_move"
            )

    return canonical_state


def _validate_required_alias(
    *, row_id: str, field_name: str, row_value, nested_value
) -> None:
    if row_value != nested_value:
        raise ValueError(
            f"source shared drift artifact row {row_id} {field_name} must match full_search alias"
        )


def _require_row_alias_field(*, row: dict, row_id: str, field_name: str):
    if field_name not in row:
        raise ValueError(
            f"source shared drift artifact row {row_id} {field_name} is required"
        )
    return row.get(field_name)


def _validate_missing_fields(*, missing_fields, context: str) -> list[str]:
    if missing_fields is None:
        return []
    if not isinstance(missing_fields, list):
        raise ValueError(f"{context} missing_fields must be a list")
    if any(not isinstance(field, str) for field in missing_fields):
        raise ValueError(f"{context} missing_fields entries must be strings")
    return list(missing_fields)


def _require_trace_presence(
    *,
    row_id: str,
    probe_mode: str,
    trace: dict,
    missing_fields: list[str],
    legal_moves: list[int],
) -> dict:
    if not isinstance(trace, dict):
        raise ValueError(
            f"source shared drift artifact row {row_id} missing probe_mode_traces.{probe_mode}"
        )

    normalized_trace = dict(trace)
    snapshots = normalized_trace.get("snapshots")
    root_start = normalized_trace.get("root_start")
    final_deltas = normalized_trace.get("final_deltas")

    if probe_mode == "full_search":
        if not isinstance(root_start, dict):
            raise ValueError(
                f"source shared drift artifact row {row_id} full_search root_start must be a dict"
            )
        if not isinstance(snapshots, list):
            raise ValueError(
                f"source shared drift artifact row {row_id} full_search snapshots must be a list"
            )
        if not snapshots:
            raise ValueError(
                f"source shared drift artifact row {row_id} full_search snapshots are required"
            )
        if not isinstance(final_deltas, dict):
            raise ValueError(
                f"source shared drift artifact row {row_id} full_search final_deltas must be a dict"
            )
        for key in ["selected_visits", "reference_visits"]:
            if key not in final_deltas:
                raise ValueError(
                    f"source shared drift artifact row {row_id} full_search final_deltas missing {key}"
                )

    if probe_mode == "value_only" and row_id == "capture_available-003":
        missing_field_set = set(missing_fields or [])

        if root_start is not None and not isinstance(root_start, dict):
            raise ValueError(
                "source shared drift artifact row capture_available-003 value_only.root_start must be a dict when present"
            )
        if snapshots is not None and not isinstance(snapshots, list):
            raise ValueError(
                "source shared drift artifact row capture_available-003 value_only.snapshots must be a list when present"
            )
        if root_start is not None and "value_only.root_start" in missing_field_set:
            raise ValueError(
                "source shared drift artifact row capture_available-003 missing_fields cannot include value_only.root_start when trace data is present"
            )
        if (
            isinstance(snapshots, list)
            and snapshots
            and "value_only.snapshots" in missing_field_set
        ):
            raise ValueError(
                "source shared drift artifact row capture_available-003 missing_fields cannot include value_only.snapshots when trace data is present"
            )

        root_start_missing = root_start is None
        snapshots_missing = snapshots is None or (
            isinstance(snapshots, list) and not snapshots
        )
        allow_upstream_empty_value_trace = (
            root_start is None and snapshots == [] and not missing_field_set
        )
        if (
            root_start_missing
            and "value_only.root_start" not in missing_field_set
            and not allow_upstream_empty_value_trace
        ):
            raise ValueError(
                "source shared drift artifact row capture_available-003 value_only.root_start must be present or declared in missing_fields"
            )
        if (
            snapshots_missing
            and "value_only.snapshots" not in missing_field_set
            and not allow_upstream_empty_value_trace
        ):
            raise ValueError(
                "source shared drift artifact row capture_available-003 value_only.snapshots must be present or declared in missing_fields"
            )
        if snapshots is None and "value_only.snapshots" in missing_field_set:
            normalized_trace["snapshots"] = []

    selected_move = normalized_trace.get("selected_move")
    if isinstance(selected_move, bool) or not isinstance(selected_move, int):
        raise ValueError(
            f"source shared drift artifact row {row_id} {probe_mode} selected_move must be an int"
        )
    if selected_move not in legal_moves:
        raise ValueError(
            f"source shared drift artifact row {row_id} {probe_mode} selected_move must be legal"
        )

    if probe_mode in CONSUMED_TRACE_KEYS.get(row_id, set()):
        if not isinstance(final_deltas, dict):
            raise ValueError(
                f"source shared drift artifact row {row_id} {probe_mode} final_deltas must be a dict"
            )
        for key in ["selected_visits", "reference_visits"]:
            if key not in final_deltas:
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} final_deltas missing {key}"
                )
            _finite_number(
                final_deltas[key],
                context=f"row {row_id} {probe_mode} final_deltas {key}",
                allow_none=False,
            )

        def validate_trace_point(point: dict, *, context: str) -> None:
            for required_key in ["selected_move", "simulation", "visits", "moves"]:
                if required_key not in point:
                    raise ValueError(
                        f"source shared drift artifact row {row_id} {probe_mode} {context} missing {required_key}"
                    )
            point_selected_move = point["selected_move"]
            if isinstance(point_selected_move, bool) or not isinstance(
                point_selected_move, int
            ):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} {context} selected_move must be an int"
                )
            if point_selected_move not in legal_moves:
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} {context} selected_move must be legal"
                )
            simulation = point["simulation"]
            _finite_number(
                simulation, context=f"row {row_id} {probe_mode} {context} simulation"
            )
            visits = point["visits"]
            if not isinstance(visits, list):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} {context} visits must be a list"
                )
            for visit in visits:
                _finite_number(
                    visit, context=f"row {row_id} {probe_mode} {context} visits entry"
                )
            if len(visits) <= max(legal_moves, default=-1):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} {context} visits must cover legal_moves"
                )
            moves = point["moves"]
            if not isinstance(moves, list):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} {context} moves must be a list"
                )
            seen_moves = []
            for move_index, move in enumerate(moves):
                if not isinstance(move, dict):
                    raise ValueError(
                        f"source shared drift artifact row {row_id} {probe_mode} {context} moves[{move_index}] must be a dict"
                    )
                for required_key in [
                    "move",
                    "prior",
                    "q_value",
                    "selection_score",
                    "used_fpu",
                    "visit_count",
                ]:
                    if required_key not in move:
                        raise ValueError(
                            f"source shared drift artifact row {row_id} {probe_mode} {context} moves[{move_index}] missing {required_key}"
                        )
                move_value = move["move"]
                if (
                    isinstance(move_value, bool)
                    or not isinstance(move_value, int)
                    or move_value not in legal_moves
                ):
                    raise ValueError(
                        f"source shared drift artifact row {row_id} {probe_mode} {context} moves[{move_index}] move must be legal int"
                    )
                seen_moves.append(move_value)
                for numeric_key in [
                    "prior",
                    "q_value",
                    "selection_score",
                    "visit_count",
                ]:
                    _finite_number(
                        move[numeric_key],
                        context=f"row {row_id} {probe_mode} {context} moves[{move_index}] {numeric_key}",
                    )
                if not isinstance(move["used_fpu"], bool):
                    raise ValueError(
                        f"source shared drift artifact row {row_id} {probe_mode} {context} moves[{move_index}] used_fpu must be bool"
                    )
            if set(seen_moves) != set(legal_moves) or len(seen_moves) != len(
                legal_moves
            ):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} {context} moves must cover legal_moves exactly"
                )

        if root_start is not None:
            if not isinstance(root_start, dict):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} root_start must be a dict"
                )
            validate_trace_point(root_start, context="root_start")

        if not isinstance(normalized_trace.get("snapshots"), list):
            raise ValueError(
                f"source shared drift artifact row {row_id} {probe_mode} snapshots must be a list"
            )
        for snapshot_index, snapshot in enumerate(
            normalized_trace.get("snapshots") or []
        ):
            if not isinstance(snapshot, dict):
                raise ValueError(
                    f"source shared drift artifact row {row_id} {probe_mode} snapshots[{snapshot_index}] must be a dict"
                )
            validate_trace_point(snapshot, context=f"snapshots[{snapshot_index}]")

    return normalized_trace


def _normalize_row(row_id: str, row: dict) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"missing required row id: {row_id}")

    missing_fields = _validate_missing_fields(
        missing_fields=row.get("missing_fields"),
        context=f"source shared drift artifact row {row_id}",
    )

    actual_row_id = row.get("row_id")
    if not isinstance(actual_row_id, str) or actual_row_id != row_id:
        raise ValueError(
            f"source shared drift artifact row {row_id} row_id must match row key"
        )

    canonical_state = row.get("canonical_state")
    if not isinstance(canonical_state, str) or not canonical_state:
        raise ValueError(
            f"source shared drift artifact row {row_id} canonical_state must be a non-empty string"
        )

    legal_moves = row.get("legal_moves")
    if not isinstance(legal_moves, list) or not legal_moves:
        raise ValueError(
            f"source shared drift artifact row {row_id} legal_moves must be a non-empty list"
        )
    if any(isinstance(move, bool) or not isinstance(move, int) for move in legal_moves):
        raise ValueError(
            f"source shared drift artifact row {row_id} legal_moves must contain ints"
        )
    if any(move < 0 for move in legal_moves):
        raise ValueError(
            f"source shared drift artifact row {row_id} legal_moves must be non-negative"
        )
    if any(move >= PITS_PER_PLAYER for move in legal_moves):
        raise ValueError(
            f"source shared drift artifact row {row_id} legal_moves must be between 0 and {PITS_PER_PLAYER - 1}"
        )
    if len(set(legal_moves)) != len(legal_moves):
        raise ValueError(
            f"source shared drift artifact row {row_id} legal_moves must be unique"
        )
    normalized_legal_moves = [int(move) for move in legal_moves]

    reference_move = _validate_required_move(
        value=row.get("reference_move"),
        legal_moves=normalized_legal_moves,
        context=f"row {row_id} reference_move",
    )

    full_search_selected_move = _validate_required_move(
        value=row.get("full_search_selected_move"),
        legal_moves=normalized_legal_moves,
        context=f"row {row_id} full_search_selected_move",
    )

    canonical_state = _validate_canonical_state(
        canonical_state=canonical_state,
        reference_move=reference_move,
        row_id=row_id,
    )

    probe_mode_traces = row.get("probe_mode_traces")
    if not isinstance(probe_mode_traces, dict):
        raise ValueError(
            f"source shared drift artifact row {row_id} must include probe_mode_traces"
        )

    normalized_probe_mode_traces = {
        probe_mode: _require_trace_presence(
            row_id=row_id,
            probe_mode=probe_mode,
            trace=probe_mode_traces.get(probe_mode),
            missing_fields=missing_fields,
            legal_moves=normalized_legal_moves,
        )
        for probe_mode in PROBE_MODE_KEYS
    }
    full_search_trace = normalized_probe_mode_traces["full_search"]
    _validate_required_alias(
        row_id=row_id,
        field_name="full_search_selected_move",
        row_value=full_search_selected_move,
        nested_value=full_search_trace.get("selected_move"),
    )
    row_root_start = _require_row_alias_field(
        row=row, row_id=row_id, field_name="root_start"
    )
    row_snapshots = _require_row_alias_field(
        row=row, row_id=row_id, field_name="snapshots"
    )
    row_final_deltas = _require_row_alias_field(
        row=row, row_id=row_id, field_name="final_deltas"
    )
    _validate_required_alias(
        row_id=row_id,
        field_name="root_start",
        row_value=row_root_start,
        nested_value=full_search_trace.get("root_start"),
    )
    _validate_required_alias(
        row_id=row_id,
        field_name="snapshots",
        row_value=row_snapshots,
        nested_value=full_search_trace.get("snapshots"),
    )
    _validate_required_alias(
        row_id=row_id,
        field_name="final_deltas",
        row_value=row_final_deltas,
        nested_value=full_search_trace.get("final_deltas"),
    )

    normalized_row = {
        "canonical_state": canonical_state,
        "legal_moves": normalized_legal_moves,
        "reference_move": reference_move,
        "full_search_selected_move": full_search_selected_move,
        "probe_mode_traces": normalized_probe_mode_traces,
        "missing_fields": list(missing_fields),
        "root_start": row_root_start,
        "snapshots": row_snapshots,
        "final_deltas": row_final_deltas,
    }

    for optional_key in ["row_id"]:
        if optional_key in row:
            normalized_row[optional_key] = row.get(optional_key)

    return normalized_row


def load_source_shared_drift_artifact(path: Path) -> dict:
    artifact = load_json(path)

    if not isinstance(artifact, dict):
        raise ValueError("source shared drift artifact must be a JSON object")

    if artifact.get("schema") != SOURCE_SHARED_DRIFT_SCHEMA:
        raise ValueError(
            f"source shared drift artifact has wrong schema: expected {SOURCE_SHARED_DRIFT_SCHEMA}"
        )

    classification = artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("source shared drift artifact must include classification")
    if classification.get("classification") != "shared_mechanism_disproved":
        raise ValueError(
            "source shared drift artifact classification must be shared_mechanism_disproved"
        )

    rows = artifact.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("source shared drift artifact rows must be a dict")
    normalized_rows = {
        row_id: _normalize_row(row_id, rows.get(row_id)) for row_id in ROW_IDS
    }

    paired_summary = artifact.get("paired_summary")
    if not isinstance(paired_summary, dict):
        raise ValueError("source shared drift artifact must include paired_summary")
    normalized_paired_summary = dict(paired_summary)
    normalized_paired_summary["probe_mode_failure_paths"] = (
        _validate_probe_mode_failure_paths(paired_summary)
    )
    normalized_paired_summary["probe_mode_selected_moves"] = (
        _validate_probe_mode_selected_moves(
            paired_summary=paired_summary,
            rows=rows,
        )
    )
    normalized_paired_summary["shared_mechanism_supported"] = (
        _validate_shared_mechanism_supported(paired_summary=paired_summary)
    )

    for row_id in ROW_IDS:
        for probe_mode in PROBE_MODE_KEYS:
            paired_selected_move = normalized_paired_summary[
                "probe_mode_selected_moves"
            ][row_id][probe_mode]
            row_selected_move = normalized_rows[row_id]["probe_mode_traces"][
                probe_mode
            ]["selected_move"]
            if paired_selected_move != row_selected_move:
                raise ValueError(
                    f"source shared drift artifact paired_summary probe_mode_selected_moves {row_id} {probe_mode} must match row trace"
                )
        if (
            normalized_paired_summary["probe_mode_selected_moves"][row_id][
                "full_search"
            ]
            != normalized_rows[row_id]["full_search_selected_move"]
        ):
            raise ValueError(
                f"source shared drift artifact paired_summary probe_mode_selected_moves {row_id} full_search must match full_search_selected_move"
            )

    selected_artifact = _validate_selected_artifact(artifact.get("selected_artifact"))
    settings = _validate_settings(artifact.get("settings"))

    decision = artifact.get("decision")
    if not isinstance(decision, str) or not decision:
        raise ValueError("source shared drift artifact must include decision")
    if decision != EXPECTED_SOURCE_DECISION:
        raise ValueError(
            f"source shared drift artifact decision must be {EXPECTED_SOURCE_DECISION}"
        )

    return {
        "artifact_path": str(path),
        "schema": artifact["schema"],
        "classification": classification,
        "decision": decision,
        "selected_artifact": selected_artifact,
        "settings": settings,
        "paired_summary": normalized_paired_summary,
        "rows": normalized_rows,
    }


def _move_entry(trace_point: dict | None, move: int) -> dict | None:
    if not isinstance(trace_point, dict):
        return None
    for move_entry in trace_point.get("moves") or []:
        if move_entry.get("move") == move:
            return move_entry
    return None


def _selection_score_rank(trace_point: dict | None, move: int) -> int | None:
    if not isinstance(trace_point, dict):
        return None
    ranked_moves = sorted(
        trace_point.get("moves") or [],
        key=lambda entry: (-entry["selection_score"], entry["move"]),
    )
    for index, move_entry in enumerate(ranked_moves, start=1):
        if move_entry["move"] == move:
            return index
    return None


def _visit_share_from_trace(
    trace_point: dict | None, *, selected_move: int, reference_move: int
) -> float | None:
    selected_entry = _move_entry(trace_point, selected_move)
    reference_entry = _move_entry(trace_point, reference_move)
    if selected_entry is None or reference_entry is None:
        return None
    moves = trace_point.get("moves") if isinstance(trace_point, dict) else None
    if not isinstance(moves, list):
        return None
    total_visits = sum(move.get("visit_count", 0.0) for move in moves)
    if not total_visits:
        return None
    return round(
        (selected_entry["visit_count"] - reference_entry["visit_count"]) / total_visits,
        10,
    )


def _selected_minus_reference_metric(
    trace_point: dict | None, *, selected_move: int, reference_move: int, key: str
):
    selected_entry = _move_entry(trace_point, selected_move)
    reference_entry = _move_entry(trace_point, reference_move)
    if selected_entry is None or reference_entry is None:
        return None
    return round(selected_entry[key] - reference_entry[key], 10)


def _first_overtake_snapshot(
    snapshots: list[dict] | None,
    *,
    selected_move: int,
    reference_move: int,
    key: str,
    root_start: dict | None = None,
):
    if root_start is not None:
        margin = _selected_minus_reference_metric(
            root_start,
            selected_move=selected_move,
            reference_move=reference_move,
            key=key,
        )
        if margin is not None and margin > 0:
            return root_start.get("simulation")

    for snapshot in snapshots or []:
        margin = _selected_minus_reference_metric(
            snapshot,
            selected_move=selected_move,
            reference_move=reference_move,
            key=key,
        )
        if margin is not None and margin > 0:
            return snapshot.get("simulation")
    return None


def _parse_rule_features(canonical_state: str, *, reference_move: int) -> dict:
    state = json.loads(canonical_state)
    player_pits = list(state.get("player_pits") or [])
    opponent_pits = list(state.get("opponent_pits") or [])
    return {
        "player_empty_pits": [
            index for index, stones in enumerate(player_pits) if stones == 0
        ],
        "opponent_empty_pits": [
            index for index, stones in enumerate(opponent_pits) if stones == 0
        ],
        "reference_pit_stones": player_pits[reference_move],
    }


def _build_lane_base(*, source_artifact: dict, row_id: str) -> tuple[dict, dict, dict]:
    row = source_artifact["rows"][row_id]
    paired_summary = source_artifact["paired_summary"]
    return (
        row,
        paired_summary,
        {
            "row_id": row_id,
            "failure_path": paired_summary["probe_mode_failure_paths"][row_id][
                "full_search"
            ],
            "canonical_state": row["canonical_state"],
            "legal_moves": list(row["legal_moves"]),
            "reference_move": row["reference_move"],
            "selected_moves_by_probe_mode": dict(
                paired_summary["probe_mode_selected_moves"][row_id]
            ),
            "trace_inputs": dict(row["probe_mode_traces"]),
            "derived_metrics": {},
            "missing_fields": list(row["missing_fields"]),
        },
    )


def build_lane_002(source_artifact: dict) -> dict:
    row, _, lane = _build_lane_base(
        source_artifact=source_artifact, row_id="capture_available-002"
    )
    reference_move = row["reference_move"]
    selected_move = row["full_search_selected_move"]
    root_start = row["probe_mode_traces"]["full_search"].get("root_start")
    snapshots = row["probe_mode_traces"]["full_search"].get("snapshots") or []
    final_snapshot = snapshots[-1] if snapshots else None
    lane["simulation_count"] = source_artifact["settings"]["simulation_count"]

    lane["derived_metrics"] = {
        "reference_move": reference_move,
        "full_search_selected_move": selected_move,
        "root_start_reference_rank_by_selection_score": _selection_score_rank(
            root_start, reference_move
        ),
        "root_start_selected_rank_by_selection_score": _selection_score_rank(
            root_start, selected_move
        ),
        "root_start_reference_q": (_move_entry(root_start, reference_move) or {}).get(
            "q_value"
        ),
        "root_start_selected_q": (_move_entry(root_start, selected_move) or {}).get(
            "q_value"
        ),
        "root_start_reference_selection_score": (
            _move_entry(root_start, reference_move) or {}
        ).get("selection_score"),
        "root_start_selected_selection_score": (
            _move_entry(root_start, selected_move) or {}
        ).get("selection_score"),
        "first_selected_visit_overtake_snapshot": _first_overtake_snapshot(
            snapshots,
            selected_move=selected_move,
            reference_move=reference_move,
            key="visit_count",
            root_start=root_start,
        ),
        "first_selected_q_overtake_snapshot": _first_overtake_snapshot(
            snapshots,
            selected_move=selected_move,
            reference_move=reference_move,
            key="q_value",
            root_start=root_start,
        ),
        "first_selected_selection_score_overtake_snapshot": _first_overtake_snapshot(
            snapshots,
            selected_move=selected_move,
            reference_move=reference_move,
            key="selection_score",
            root_start=root_start,
        ),
        "final_selected_minus_reference_visit_share": _visit_share_from_trace(
            final_snapshot,
            selected_move=selected_move,
            reference_move=reference_move,
        ),
        "final_selected_minus_reference_q_margin": _selected_minus_reference_metric(
            final_snapshot,
            selected_move=selected_move,
            reference_move=reference_move,
            key="q_value",
        ),
        "final_selected_minus_reference_prior_margin": _selected_minus_reference_metric(
            final_snapshot,
            selected_move=selected_move,
            reference_move=reference_move,
            key="prior",
        ),
        "missing_fields": list(row["missing_fields"]),
    }
    return lane


def build_lane_003(source_artifact: dict) -> dict:
    row, _, lane = _build_lane_base(
        source_artifact=source_artifact, row_id="capture_available-003"
    )
    reference_move = row["reference_move"]
    value_only_trace = row["probe_mode_traces"]["value_only"]
    full_search_trace = row["probe_mode_traces"]["full_search"]
    policy_only_trace = row["probe_mode_traces"]["policy_only"]
    value_only_snapshots = value_only_trace.get("snapshots") or []
    value_only_final_snapshot = (
        value_only_snapshots[-1] if value_only_snapshots else None
    )
    full_search_snapshots = full_search_trace.get("snapshots") or []
    full_search_final_snapshot = (
        full_search_snapshots[-1] if full_search_snapshots else None
    )
    policy_root_start = policy_only_trace.get("root_start")

    lane["derived_metrics"] = {
        "reference_move": reference_move,
        "value_only_selected_move": value_only_trace.get("selected_move"),
        "full_search_selected_move": full_search_trace.get("selected_move"),
        "policy_only_selected_move": policy_only_trace.get("selected_move"),
        "value_only_selected_minus_reference_q_margin": _selected_minus_reference_metric(
            value_only_final_snapshot,
            selected_move=value_only_trace.get("selected_move"),
            reference_move=reference_move,
            key="q_value",
        ),
        "value_only_selected_minus_reference_visit_share": _visit_share_from_trace(
            value_only_final_snapshot,
            selected_move=value_only_trace.get("selected_move"),
            reference_move=reference_move,
        ),
        "full_search_selected_minus_reference_q_margin": _selected_minus_reference_metric(
            full_search_final_snapshot,
            selected_move=full_search_trace.get("selected_move"),
            reference_move=reference_move,
            key="q_value",
        ),
        "full_search_selected_minus_reference_visit_share": _visit_share_from_trace(
            full_search_final_snapshot,
            selected_move=full_search_trace.get("selected_move"),
            reference_move=reference_move,
        ),
        "policy_reference_prior": (
            _move_entry(policy_root_start, reference_move) or {}
        ).get("prior"),
        "policy_value_selected_prior": (
            _move_entry(policy_root_start, value_only_trace.get("selected_move")) or {}
        ).get("prior"),
        "rule_features": _parse_rule_features(
            row["canonical_state"], reference_move=reference_move
        ),
        "value_only_first_selected_visit_overtake_snapshot": _first_overtake_snapshot(
            value_only_snapshots,
            selected_move=value_only_trace.get("selected_move"),
            reference_move=reference_move,
            key="visit_count",
        ),
        "value_only_first_selected_q_overtake_snapshot": _first_overtake_snapshot(
            value_only_snapshots,
            selected_move=value_only_trace.get("selected_move"),
            reference_move=reference_move,
            key="q_value",
        ),
        "value_only_first_selected_selection_score_overtake_snapshot": _first_overtake_snapshot(
            value_only_snapshots,
            selected_move=value_only_trace.get("selected_move"),
            reference_move=reference_move,
            key="selection_score",
        ),
        "missing_fields": list(row["missing_fields"]),
    }
    return lane


def _lane_002_snapshot_is_early(
    *, lane: dict, snapshot: float | None, thresholds: dict
) -> bool:
    if snapshot is None:
        return False

    snapshots = ((lane.get("trace_inputs") or {}).get("full_search") or {}).get(
        "snapshots"
    ) or []
    minimum_count = int(thresholds["minimum_early_snapshot_count"])
    if len(snapshots) < minimum_count:
        return False

    if len(snapshots) == minimum_count == 1:
        simulation_horizon = lane.get("simulation_count")
        if (
            isinstance(simulation_horizon, bool)
            or not isinstance(simulation_horizon, int)
            or simulation_horizon <= 0
        ):
            return False
        return (float(snapshot) / float(simulation_horizon)) <= float(
            thresholds["early_snapshot_fraction"]
        )

    last_snapshot = snapshots[-1] if snapshots else None
    last_simulation = (
        (last_snapshot or {}).get("simulation")
        if isinstance(last_snapshot, dict)
        else None
    )
    if (
        isinstance(last_simulation, bool)
        or not isinstance(last_simulation, (int, float))
        or last_simulation <= 0
    ):
        return False

    return (float(snapshot) / float(last_simulation)) <= float(
        thresholds["early_snapshot_fraction"]
    )


def classify_lane_002(*, lane: dict, thresholds: dict) -> dict:
    derived_metrics = dict(lane.get("derived_metrics") or {})
    meaningful_q_margin = float(thresholds["meaningful_q_margin"])
    small_q_margin = float(thresholds["small_q_margin"])
    meaningful_visit_share_overtake = float(
        thresholds["meaningful_visit_share_overtake"]
    )
    material_selection_score_margin = float(
        thresholds["material_selection_score_margin"]
    )
    material_prior_margin = float(thresholds["material_prior_margin"])

    final_q_margin = derived_metrics.get("final_selected_minus_reference_q_margin")
    final_visit_share = derived_metrics.get(
        "final_selected_minus_reference_visit_share"
    )
    final_prior_margin = derived_metrics.get(
        "final_selected_minus_reference_prior_margin"
    )
    selection_score_gap = None
    if (
        derived_metrics.get("root_start_reference_selection_score") is not None
        and derived_metrics.get("root_start_selected_selection_score") is not None
    ):
        selection_score_gap = float(
            derived_metrics["root_start_reference_selection_score"]
        ) - float(derived_metrics["root_start_selected_selection_score"])

    first_selection_score_overtake = derived_metrics.get(
        "first_selected_selection_score_overtake_snapshot"
    )
    first_visit_overtake = derived_metrics.get("first_selected_visit_overtake_snapshot")
    first_q_overtake = derived_metrics.get("first_selected_q_overtake_snapshot")

    if (
        selection_score_gap is not None
        and selection_score_gap >= material_selection_score_margin
        and first_selection_score_overtake is not None
        and (
            first_q_overtake is None
            or float(first_selection_score_overtake) < float(first_q_overtake)
        )
    ):
        classification = "selection_score_overtake"
        evidence_summary = "The selected move starts materially behind on root selection score and overtakes on selection score before Q does."
    elif (
        _lane_002_snapshot_is_early(
            lane=lane, snapshot=first_visit_overtake, thresholds=thresholds
        )
        and final_q_margin is not None
        and float(final_q_margin) < small_q_margin
    ):
        classification = "early_fpu_pressure"
        evidence_summary = "The selected move overtakes visits early while Q stays small, consistent with early FPU pressure rather than value evidence."
    elif (
        _lane_002_snapshot_is_early(
            lane=lane, snapshot=first_q_overtake, thresholds=thresholds
        )
        and final_q_margin is not None
        and float(final_q_margin) >= meaningful_q_margin
    ):
        classification = "early_child_value_override"
        evidence_summary = "The selected move gains an early Q overtake and finishes with a meaningful Q margin over the reference move."
    elif (
        final_q_margin is not None
        and float(final_q_margin) < small_q_margin
        and final_visit_share is not None
        and float(final_visit_share) >= meaningful_visit_share_overtake
    ):
        classification = "backup_accumulation_drift"
        evidence_summary = "The selected move only pulls ahead later in search, visit-share drift is meaningful, and the final Q margin stays small."
    elif (
        final_q_margin is not None
        and float(final_q_margin) < small_q_margin
        and final_prior_margin is not None
        and float(final_prior_margin) >= material_prior_margin
    ):
        classification = "prior_pressure_with_small_q"
        evidence_summary = "The selected move retains material prior pressure at the root while the final Q margin stays below the meaningful threshold."
    else:
        classification = "unresolved"
        evidence_summary = "Lane 002 evidence does not isolate one supported mechanism."

    return {
        "classification": classification,
        "decision": LANE_002_DECISIONS[classification],
        "evidence_summary": evidence_summary,
    }


def _lane_003_missing_fields(lane: dict) -> list[str]:
    missing_fields = lane.get("missing_fields")
    if missing_fields is None:
        missing_fields = (lane.get("derived_metrics") or {}).get("missing_fields") or []
    return _validate_missing_fields(missing_fields=missing_fields, context="lane 003")


def _lane_003_trace_is_insufficient(*, lane: dict) -> bool:
    derived_metrics = dict(lane.get("derived_metrics") or {})
    missing_fields = set(_lane_003_missing_fields(lane))
    if missing_fields.intersection({"value_only.root_start", "value_only.snapshots"}):
        return True
    return (
        derived_metrics.get("value_only_selected_minus_reference_q_margin") is None
        or derived_metrics.get("value_only_selected_minus_reference_visit_share")
        is None
    )


def classify_lane_003(*, lane: dict, thresholds: dict) -> dict:
    derived_metrics = dict(lane.get("derived_metrics") or {})
    meaningful_q_margin = float(thresholds["meaningful_q_margin"])
    small_q_margin = float(thresholds["small_q_margin"])
    meaningful_visit_share_overtake = float(
        thresholds["meaningful_visit_share_overtake"]
    )
    material_prior_margin = float(thresholds["material_prior_margin"])

    value_only_selected_move = derived_metrics.get("value_only_selected_move")
    full_search_selected_move = derived_metrics.get("full_search_selected_move")
    policy_only_selected_move = derived_metrics.get("policy_only_selected_move")
    reference_move = derived_metrics.get("reference_move")
    value_only_q_margin = derived_metrics.get(
        "value_only_selected_minus_reference_q_margin"
    )
    value_only_visit_share = derived_metrics.get(
        "value_only_selected_minus_reference_visit_share"
    )
    reference_prior = derived_metrics.get("policy_reference_prior")
    value_selected_prior = derived_metrics.get("policy_value_selected_prior")
    rule_features = dict(derived_metrics.get("rule_features") or {})
    prior_margin = None
    if reference_prior is not None and value_selected_prior is not None:
        prior_margin = float(reference_prior) - float(value_selected_prior)

    if _lane_003_trace_is_insufficient(lane=lane):
        classification = "insufficient_value_trace"
        evidence_summary = "003 is missing value-only trace fields needed to interpret whether value evidence or visit amplification caused the divergence."
    elif (
        value_only_selected_move != reference_move
        and policy_only_selected_move == reference_move
        and prior_margin is not None
        and prior_margin >= material_prior_margin
        and (
            (
                value_only_q_margin is not None
                and float(value_only_q_margin) >= meaningful_q_margin
            )
            or (
                value_only_visit_share is not None
                and float(value_only_visit_share) >= meaningful_visit_share_overtake
            )
        )
    ):
        classification = "policy_value_conflict"
        evidence_summary = "Policy keeps the reference move on top while value-only evidence prefers a different move, indicating a policy-value conflict in 003."
    elif (
        value_only_selected_move != reference_move
        and policy_only_selected_move == reference_move
        and full_search_selected_move == value_only_selected_move
        and value_only_q_margin is not None
        and float(value_only_q_margin) >= meaningful_q_margin
        and rule_features.get("reference_pit_stones") == 0
        and reference_prior is not None
        and value_selected_prior is not None
    ):
        classification = "rule_feature_value_collision"
        evidence_summary = "003 carries state-specific rule features that can collide with value evidence around the reference move."
    elif (
        value_only_selected_move != reference_move
        and policy_only_selected_move == reference_move
        and full_search_selected_move == value_only_selected_move
        and value_only_q_margin is not None
        and float(value_only_q_margin) >= meaningful_q_margin
    ):
        classification = "value_only_child_q_prefers_wrong_move"
        evidence_summary = "Value-only child Q evidence prefers the wrong move over the reference move strongly enough to explain the 003 divergence."
    elif (
        value_only_selected_move != reference_move
        and value_only_visit_share is not None
        and float(value_only_visit_share) >= meaningful_visit_share_overtake
        and value_only_q_margin is not None
        and float(value_only_q_margin) < small_q_margin
    ):
        classification = "value_only_visit_amplification_without_q"
        evidence_summary = "Value-only search amplifies visits toward the wrong move without a meaningful Q advantage over the reference move."
    else:
        classification = "unresolved"
        evidence_summary = "Lane 003 evidence does not isolate one supported mechanism."

    return {
        "classification": classification,
        "decision": LANE_003_DECISIONS[classification],
        "evidence_summary": evidence_summary,
    }


def _is_actionable_decision(decision: str | None) -> bool:
    return isinstance(decision, str) and decision.startswith("write_")


def top_level_decision(*, lane_002: dict, lane_003: dict) -> str:
    lane_002_decision = lane_002.get("decision")
    lane_003_decision = lane_003.get("decision")
    lane_002_actionable = _is_actionable_decision(lane_002_decision)
    lane_003_actionable = _is_actionable_decision(lane_003_decision)

    if lane_003.get(
        "classification"
    ) == "insufficient_value_trace" and _lane_003_trace_is_insufficient(lane=lane_003):
        return "write_003_value_trace_capture_spec"
    if lane_002_actionable and lane_003_actionable:
        return "write_parallel_row_followup_specs"
    if lane_002_actionable:
        return lane_002_decision
    if lane_003_actionable:
        return lane_003_decision
    return "stop_row_split_unresolved"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose row split follow-up for capture_available-002 and -003"
    )
    parser.add_argument("--source-shared-drift-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def build_payload(
    *, source_artifact: dict, lane_002: dict, lane_003: dict, decision: str
) -> dict:
    lanes = {
        lane_002["row_id"]: dict(lane_002),
        lane_003["row_id"]: dict(lane_003),
    }
    evidence_summary = " ".join(
        f"{row_id}: {lanes[row_id]['evidence_summary']}" for row_id in ROW_IDS
    )
    return {
        "schema": SCHEMA,
        "source_shared_drift_artifact": source_artifact["artifact_path"],
        "selected_artifact": dict(source_artifact["selected_artifact"]),
        "thresholds": dict(THRESHOLDS),
        "settings": dict(source_artifact["settings"]),
        "lanes": lanes,
        "summary": {
            "row_ids": list(ROW_IDS),
            "shared_mechanism_supported": False,
            "lane_classifications": {
                row_id: lanes[row_id]["classification"] for row_id in ROW_IDS
            },
            "lane_decisions": {row_id: lanes[row_id]["decision"] for row_id in ROW_IDS},
            "next_safe_branch": decision,
            "evidence_summary": evidence_summary,
        },
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_artifact = load_source_shared_drift_artifact(
        args.source_shared_drift_artifact
    )
    lane_002 = build_lane_002(source_artifact)
    lane_002.update(classify_lane_002(lane=lane_002, thresholds=THRESHOLDS))
    lane_003 = build_lane_003(source_artifact)
    lane_003.update(classify_lane_003(lane=lane_003, thresholds=THRESHOLDS))
    decision = top_level_decision(lane_002=lane_002, lane_003=lane_003)
    payload = build_payload(
        source_artifact=source_artifact,
        lane_002=lane_002,
        lane_003=lane_003,
        decision=decision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"artifact_path": str(args.out), "schema": SCHEMA, "decision": decision},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
