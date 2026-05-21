from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.alphazero_lite.kalah_rules import PITS_PER_PLAYER


SCHEMA = "azlite_shared_full_search_drift_diagnostic_v1"
SOURCE_ARBITRATION_SCHEMA = "azlite_capture_002_003_search_policy_arbitration_v1"
ROW_IDS = ["capture_available-002", "capture_available-003"]
THRESHOLDS = {
    "meaningful_q_margin": 0.03,
    "small_q_margin": 0.03,
    "meaningful_visit_share_overtake": 0.05,
    "material_prior_distortion": 0.05,
    "early_snapshot_fraction": 0.10,
    "minimum_early_snapshot_count": 1,
}
CLASSIFICATION_DECISIONS = {
    "shared_mechanism_disproved": "write_row_split_followup_spec",
    "root_prior_decay": "write_fpu_root_pressure_spec",
    "child_value_override": "write_child_value_override_spec",
    "backup_accumulation_drift": "write_backup_accumulation_spec",
    "fpu_or_unvisited_child_pressure": "write_fpu_root_pressure_spec",
    "tactical_root_bias_interaction": "write_tactical_root_bias_spec",
    "unresolved": "stop_unresolved",
}
REQUIRED_FULL_SEARCH_SETTINGS_KEYS = [
    "c_puct",
    "fpu_mode",
    "normalize_values",
    "reuse_subtree",
    "root_policy_mode",
    "tactical_root_bias",
]
PROBE_MODE_KEYS = ["policy_only", "value_only", "full_search"]
PROBE_MODE_FAILURE_PATHS = {
    "reference_kept",
    "diverged_before_full_search",
    "full_search_drift",
    "not_applicable",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_selected_artifact(selected_artifact: dict) -> dict:
    path = selected_artifact.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(
            "source arbitration artifact selected_artifact must include non-empty string path"
        )

    provenance_source = selected_artifact.get("provenance_source")
    if not isinstance(provenance_source, str) or not provenance_source:
        raise ValueError(
            "source arbitration artifact selected_artifact must include non-empty string provenance_source"
        )

    selected_target = selected_artifact.get("selected_target")
    if selected_target is not None and not isinstance(selected_target, str):
        raise ValueError(
            "source arbitration artifact selected_artifact selected_target must be string or null"
        )

    selected_artifact_path = selected_artifact.get("selected_artifact")
    if selected_artifact_path is not None and not isinstance(
        selected_artifact_path, str
    ):
        raise ValueError(
            "source arbitration artifact selected_artifact selected_artifact must be string or null"
        )

    return dict(selected_artifact)


def _validate_numeric_visits(*, visits, context: str) -> None:
    for index, visit in enumerate(visits):
        if isinstance(visit, bool) or not isinstance(visit, (int, float)):
            raise ValueError(f"{context}[{index}] must be numeric")


def _validate_move_index(*, move: int | None, visits, context: str) -> None:
    if move is None or visits is None:
        return
    if move < 0 or move >= len(visits):
        raise ValueError(f"{context} must be within visits bounds")


def _validate_optional_move_field(*, value, context: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{context} must be an integer")
    if value is not None and value < 0:
        raise ValueError(f"{context} must be non-negative")


def _validate_optional_numeric_field(*, value, context: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{context} must be numeric")


def _validate_optional_boolean_field(*, value, context: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")


def _validate_optional_canonical_state(*, value, context: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{context} must be a string")


def _validate_optional_legal_moves(*, value, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list")
    for index, move in enumerate(value):
        if isinstance(move, bool) or not isinstance(move, int):
            raise ValueError(f"{context}[{index}] must be an integer")
        if not 0 <= move < PITS_PER_PLAYER:
            raise ValueError(
                f"{context}[{index}] must be between 0 and {PITS_PER_PLAYER - 1}"
            )


def _validate_nested_move_field(*, value, context: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{context} must be an integer")
    if value is not None and not 0 <= value < PITS_PER_PLAYER:
        raise ValueError(f"{context} must be between 0 and {PITS_PER_PLAYER - 1}")


def _validate_nested_numeric_field(*, value, context: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{context} must be numeric")


def _validate_nested_visit_distribution(*, visit_distribution, context: str) -> None:
    if visit_distribution is None:
        return
    if not isinstance(visit_distribution, dict):
        raise ValueError(f"{context} must be a dict")
    for move, visit_count in visit_distribution.items():
        try:
            move_index = int(move)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} move keys must be integer-like") from exc
        if str(move_index) != str(move):
            raise ValueError(f"{context} move keys must be integer-like")
        if not 0 <= move_index < PITS_PER_PLAYER:
            raise ValueError(
                f"{context} move keys must be between 0 and {PITS_PER_PLAYER - 1}"
            )
        _validate_nested_numeric_field(value=visit_count, context=f"{context}[{move}]")


def _validate_visit_snapshots(
    *,
    row_id: str,
    probe_mode: str,
    reference_move: int,
    selected_move: int | None,
    visit_snapshots,
) -> None:
    if visit_snapshots is None:
        return
    if not isinstance(visit_snapshots, list):
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots must be a list"
        )

    for index, snapshot in enumerate(visit_snapshots):
        if not isinstance(snapshot, dict):
            raise ValueError(
                f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] must be a dict"
            )

        simulation = snapshot.get("simulation")
        if simulation is not None and (
            isinstance(simulation, bool) or not isinstance(simulation, (int, float))
        ):
            raise ValueError(
                f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] simulation must be numeric"
            )

        _validate_optional_move_field(
            value=snapshot.get("selected_move"),
            context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] selected_move",
        )
        _validate_optional_move_field(
            value=snapshot.get("reference_move_by_prior"),
            context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] reference_move_by_prior",
        )
        _validate_optional_move_field(
            value=snapshot.get("reference_move_rank_by_visits"),
            context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] reference_move_rank_by_visits",
        )
        _validate_optional_move_field(
            value=snapshot.get("reference_move_rank_by_q"),
            context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] reference_move_rank_by_q",
        )
        _validate_optional_move_field(
            value=snapshot.get("reference_move_rank_by_selection_score"),
            context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] reference_move_rank_by_selection_score",
        )

        visits = snapshot.get("visits")
        if visits is not None and not isinstance(visits, (list, tuple)):
            raise ValueError(
                f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] visits must be a list or tuple"
            )
        if visits is not None:
            visits_context = f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] visits"
            _validate_numeric_visits(visits=visits, context=visits_context)
            _validate_move_index(
                move=reference_move,
                visits=visits,
                context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] reference_move",
            )
            _validate_move_index(
                move=selected_move,
                visits=visits,
                context=f"source arbitration artifact row {row_id} {probe_mode} visit_snapshots[{index}] selected_move",
            )


def _validate_child_stats(*, row_id: str, probe_mode: str, child_stats) -> None:
    if child_stats is None:
        return
    if not isinstance(child_stats, list):
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} child_stats must be a list"
        )

    for index, child in enumerate(child_stats):
        if not isinstance(child, dict):
            raise ValueError(
                f"source arbitration artifact row {row_id} {probe_mode} child_stats[{index}] must be a dict"
            )
        move = child.get("move")
        if isinstance(move, bool) or not isinstance(move, int):
            raise ValueError(
                f"source arbitration artifact row {row_id} {probe_mode} child_stats[{index}] move must be an integer"
            )
        if not 0 <= move < PITS_PER_PLAYER:
            raise ValueError(
                f"source arbitration artifact row {row_id} {probe_mode} child_stats[{index}] move must be between 0 and {PITS_PER_PLAYER - 1}"
            )
        _validate_optional_numeric_field(
            value=child.get("visits"),
            context=f"source arbitration artifact row {row_id} {probe_mode} child_stats[{index}] visits",
        )
        _validate_optional_numeric_field(
            value=child.get("q_value"),
            context=f"source arbitration artifact row {row_id} {probe_mode} child_stats[{index}] q_value",
        )


def _validate_probe_mode_summary(
    *, row_id: str, probe_mode: str, reference_move: int, probe_mode_summary: dict
) -> None:
    selected_move = probe_mode_summary.get("selected_move")
    if selected_move is not None and (
        isinstance(selected_move, bool) or not isinstance(selected_move, int)
    ):
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} selected_move must be an integer"
        )
    if selected_move is not None and selected_move < 0:
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} selected_move must be non-negative"
        )

    visits = probe_mode_summary.get("visits")
    if visits is not None and not isinstance(visits, (list, tuple)):
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} visits must be a list or tuple"
        )
    if visits is not None:
        visits_context = f"source arbitration artifact row {row_id} {probe_mode} visits"
        _validate_numeric_visits(visits=visits, context=visits_context)
        _validate_move_index(
            move=reference_move,
            visits=visits,
            context=f"source arbitration artifact row {row_id} {probe_mode} reference_move",
        )
        _validate_move_index(
            move=selected_move,
            visits=visits,
            context=f"source arbitration artifact row {row_id} {probe_mode} selected_move",
        )

    _validate_optional_move_field(
        value=probe_mode_summary.get("prior_reference_move"),
        context=f"source arbitration artifact row {row_id} {probe_mode} prior_reference_move",
    )
    _validate_optional_move_field(
        value=probe_mode_summary.get("prior_selected_move"),
        context=f"source arbitration artifact row {row_id} {probe_mode} prior_selected_move",
    )
    _validate_optional_numeric_field(
        value=probe_mode_summary.get("selected_minus_reference_q_margin"),
        context=f"source arbitration artifact row {row_id} {probe_mode} selected_minus_reference_q_margin",
    )
    _validate_optional_numeric_field(
        value=probe_mode_summary.get("selected_minus_reference_visit_share"),
        context=f"source arbitration artifact row {row_id} {probe_mode} selected_minus_reference_visit_share",
    )
    _validate_optional_boolean_field(
        value=probe_mode_summary.get("tactical_bias_applied"),
        context=f"source arbitration artifact row {row_id} {probe_mode} tactical_bias_applied",
    )
    _validate_child_stats(
        row_id=row_id,
        probe_mode=probe_mode,
        child_stats=probe_mode_summary.get("child_stats"),
    )

    _validate_visit_snapshots(
        row_id=row_id,
        probe_mode=probe_mode,
        reference_move=reference_move,
        selected_move=selected_move,
        visit_snapshots=probe_mode_summary.get("visit_snapshots"),
    )


def _normalized_visit_distribution(
    *,
    visit_distribution: dict | None,
    child_stats,
    reference_move: int,
    selected_move: int | None,
):
    if visit_distribution is None:
        return None
    _validate_nested_visit_distribution(
        visit_distribution=visit_distribution,
        context="search_view.visit_distribution",
    )

    moves = [reference_move]
    if selected_move is not None:
        moves.append(selected_move)
    for move in visit_distribution:
        moves.append(int(move))
    for child in child_stats or []:
        if isinstance(child, dict) and "move" in child:
            moves.append(int(child["move"]))

    max_move = max(moves, default=-1)
    if max_move < 0:
        return []

    visits = [0.0] * (max_move + 1)
    for move, visit_count in visit_distribution.items():
        visits[int(move)] = float(visit_count)
    return visits


def _normalize_probe_mode_summary(
    *, row: dict, row_id: str, probe_mode: str, reference_move: int
) -> dict | None:
    probe_mode_summary = row.get(probe_mode)
    if isinstance(probe_mode_summary, dict):
        return probe_mode_summary

    probe_views = row.get("probe_views")
    if not isinstance(probe_views, dict):
        return probe_mode_summary

    probe_view = probe_views.get(probe_mode)
    if not isinstance(probe_view, dict):
        return probe_view

    search_view = probe_view.get("search_view")
    value_view = probe_view.get("value_view")
    if not isinstance(search_view, dict):
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} search_view must be a dict"
        )
    if not isinstance(value_view, dict):
        raise ValueError(
            f"source arbitration artifact row {row_id} {probe_mode} value_view must be a dict"
        )

    selected_move = search_view.get("searched_selected_move")
    _validate_nested_move_field(
        value=selected_move,
        context=f"source arbitration artifact row {row_id} {probe_mode} selected_move",
    )
    child_stats = search_view.get("child_stats")
    _validate_child_stats(
        row_id=row_id,
        probe_mode=probe_mode,
        child_stats=child_stats,
    )
    visits = _normalized_visit_distribution(
        visit_distribution=search_view.get("visit_distribution"),
        child_stats=child_stats,
        reference_move=reference_move,
        selected_move=selected_move,
    )
    reference_visit_share = search_view.get("reference_move_visit_share")
    selected_visit_share = search_view.get("selected_move_visit_share")
    _validate_nested_numeric_field(
        value=reference_visit_share,
        context=f"source arbitration artifact row {row_id} {probe_mode} search_view.reference_move_visit_share",
    )
    _validate_nested_numeric_field(
        value=selected_visit_share,
        context=f"source arbitration artifact row {row_id} {probe_mode} search_view.selected_move_visit_share",
    )
    selected_minus_reference_visit_share = None
    if reference_visit_share is not None and selected_visit_share is not None:
        selected_minus_reference_visit_share = float(selected_visit_share) - float(
            reference_visit_share
        )

    selected_minus_reference_q_margin = value_view.get(
        "selected_minus_reference_q_margin"
    )
    _validate_nested_numeric_field(
        value=selected_minus_reference_q_margin,
        context=f"source arbitration artifact row {row_id} {probe_mode} selected_minus_reference_q_margin",
    )

    normalized_summary = {
        "selected_move": selected_move,
        "visits": visits,
        "selected_minus_reference_q_margin": selected_minus_reference_q_margin,
        "selected_minus_reference_visit_share": selected_minus_reference_visit_share,
        "child_stats": child_stats,
    }
    if "visit_snapshots" in search_view:
        normalized_summary["visit_snapshots"] = search_view.get("visit_snapshots")
    return normalized_summary


def load_source_arbitration_artifact(artifact_path: Path) -> dict:
    artifact = load_json(artifact_path)

    if not isinstance(artifact, dict):
        raise ValueError("source arbitration artifact must be a JSON object")

    if artifact.get("schema") != SOURCE_ARBITRATION_SCHEMA:
        raise ValueError(
            f"source arbitration artifact has wrong schema: expected {SOURCE_ARBITRATION_SCHEMA}"
        )

    rows = artifact.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("source arbitration artifact rows must be a dict")

    resolved_rows = {}
    for row_id in ROW_IDS:
        row = rows.get(row_id)
        if not isinstance(row, dict):
            raise ValueError(f"missing required row id: {row_id}")
        reference_move = row.get("reference_move")
        if isinstance(reference_move, bool) or not isinstance(reference_move, int):
            raise ValueError(
                f"source arbitration artifact row {row_id} must include integer reference_move"
            )
        if reference_move < 0:
            raise ValueError(
                f"source arbitration artifact row {row_id} reference_move must be non-negative"
            )
        normalized_row = dict(row)
        _validate_optional_canonical_state(
            value=normalized_row.get("canonical_state"),
            context=f"source arbitration artifact row {row_id} canonical_state",
        )
        _validate_optional_legal_moves(
            value=normalized_row.get("legal_moves"),
            context=f"source arbitration artifact row {row_id} legal_moves",
        )
        for probe_mode in PROBE_MODE_KEYS:
            probe_mode_summary = _normalize_probe_mode_summary(
                row=row,
                row_id=row_id,
                probe_mode=probe_mode,
                reference_move=reference_move,
            )
            if not isinstance(probe_mode_summary, dict):
                raise ValueError(
                    f"source arbitration artifact row {row_id} must include dict {probe_mode}"
                )
            _validate_probe_mode_summary(
                row_id=row_id,
                probe_mode=probe_mode,
                reference_move=reference_move,
                probe_mode_summary=probe_mode_summary,
            )
            normalized_row[probe_mode] = probe_mode_summary
        resolved_rows[row_id] = normalized_row

    settings = artifact.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("source arbitration artifact must include search_settings")

    search_settings = settings.get("search_settings")
    if not isinstance(search_settings, dict):
        raise ValueError("source arbitration artifact must include search_settings")

    for key in REQUIRED_FULL_SEARCH_SETTINGS_KEYS:
        if key not in search_settings:
            raise ValueError(
                f"source arbitration artifact search_settings missing required key: {key}"
            )

    seeds = settings.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("source arbitration artifact must include non-empty seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("source arbitration artifact seeds must be integers")
    if len(set(seeds)) != 1:
        raise ValueError(
            "source arbitration artifact seeds must be identical to derive a single seed"
        )
    seed = seeds[0]

    simulation_count = settings.get("simulation_count")
    if (
        isinstance(simulation_count, bool)
        or not isinstance(simulation_count, int)
        or simulation_count <= 0
    ):
        raise ValueError(
            "source arbitration artifact must include positive simulation_count"
        )

    selected_artifact = artifact.get("selected_artifact")
    if not isinstance(selected_artifact, dict):
        raise ValueError("source arbitration artifact must include selected_artifact")
    selected_artifact = _validate_selected_artifact(selected_artifact)

    classification = artifact.get("classification")
    if not isinstance(classification, dict):
        raise ValueError("source arbitration artifact must include classification")

    classification_label = classification.get("classification")
    if classification_label not in CLASSIFICATION_DECISIONS:
        raise ValueError(
            f"source arbitration artifact has unsupported classification: {classification_label}"
        )

    decision = artifact.get("decision")
    if decision is None:
        raise ValueError("source arbitration artifact must include decision")

    expected_decision = CLASSIFICATION_DECISIONS[classification_label]
    if decision != expected_decision:
        raise ValueError(
            "source arbitration artifact decision does not match classification contract"
        )

    return {
        "artifact_path": str(artifact_path),
        "schema": artifact["schema"],
        "row_ids": list(ROW_IDS),
        "rows": resolved_rows,
        "selected_artifact": selected_artifact,
        "source_settings": {
            "search_settings": {
                key: search_settings[key] for key in REQUIRED_FULL_SEARCH_SETTINGS_KEYS
            },
            "seeds": list(seeds),
            "simulation_count": simulation_count,
        },
        "search_settings": {
            key: search_settings[key] for key in REQUIRED_FULL_SEARCH_SETTINGS_KEYS
        },
        "seed": seed,
        "simulation_count": simulation_count,
        "classification": classification,
        "decision": decision,
    }


def selected_minus_reference_visit_share(
    *, visits, selected_move: int | None, reference_move: int | None
) -> float | None:
    if visits is None or selected_move is None or reference_move is None:
        return None

    _validate_move_index(move=selected_move, visits=visits, context="selected_move")
    _validate_move_index(move=reference_move, visits=visits, context="reference_move")

    total_visits = float(sum(float(visit) for visit in visits))
    if total_visits <= 0.0:
        return 0.0

    selected_share = float(visits[int(selected_move)]) / total_visits
    reference_share = float(visits[int(reference_move)]) / total_visits
    return selected_share - reference_share


def _visit_count(visits, move: int | None) -> float | None:
    if visits is None or move is None:
        return None
    _validate_move_index(move=move, visits=visits, context="move")
    return float(visits[int(move)])


def _build_mode_trace(*, reference_move: int, probe_mode_summary: dict | None) -> dict:
    probe_mode_summary = dict(probe_mode_summary or {})
    if reference_move < 0:
        raise ValueError("reference_move must be non-negative")

    selected_move = probe_mode_summary.get("selected_move")
    if selected_move is not None:
        selected_move = int(selected_move)
        if selected_move < 0:
            raise ValueError("selected_move must be non-negative")

    snapshots = list(probe_mode_summary.get("visit_snapshots") or [])
    root_start = snapshots[0] if snapshots else None
    final_visits = probe_mode_summary.get("visits")
    if final_visits is None and snapshots:
        final_visits = snapshots[-1].get("visits")

    root_start_visits = None if root_start is None else root_start.get("visits")
    selected_final_visits = _visit_count(final_visits, selected_move)
    selected_root_start_visits = _visit_count(root_start_visits, selected_move)
    reference_final_visits = _visit_count(final_visits, reference_move)
    reference_root_start_visits = _visit_count(root_start_visits, reference_move)

    final_deltas = {
        "selected_visits": (
            None
            if selected_final_visits is None or selected_root_start_visits is None
            else selected_final_visits - selected_root_start_visits
        ),
        "reference_visits": (
            None
            if reference_final_visits is None or reference_root_start_visits is None
            else reference_final_visits - reference_root_start_visits
        ),
    }

    return {
        "selected_move": selected_move,
        "root_start": root_start,
        "snapshots": snapshots,
        "final_deltas": final_deltas,
        "selected_minus_reference_visit_share": selected_minus_reference_visit_share(
            visits=final_visits,
            selected_move=selected_move,
            reference_move=reference_move,
        ),
    }


def _normalize_legal_moves(legal_moves) -> list[int] | None:
    if legal_moves is None:
        return None
    _validate_optional_legal_moves(value=legal_moves, context="legal_moves")
    return [int(move) for move in legal_moves]


def build_row_trace(
    *, row_id: str, canonical_state, legal_moves, reference_move: int, probe_modes: dict
) -> dict:
    canonical_state_value = (
        canonical_state if isinstance(canonical_state, str) else None
    )
    normalized_legal_moves = _normalize_legal_moves(legal_moves)
    full_search_trace = _build_mode_trace(
        reference_move=int(reference_move),
        probe_mode_summary=probe_modes.get("full_search"),
    )
    missing_fields = []
    if canonical_state_value is None:
        missing_fields.append("canonical_state")
    if normalized_legal_moves is None:
        missing_fields.append("legal_moves")
    if full_search_trace.get("selected_move") is None:
        missing_fields.append("full_search_selected_move")

    return {
        "row_id": row_id,
        "canonical_state": canonical_state_value,
        "legal_moves": [] if normalized_legal_moves is None else normalized_legal_moves,
        "reference_move": int(reference_move),
        "full_search_selected_move": full_search_trace.get("selected_move"),
        "root_start": full_search_trace.get("root_start"),
        "snapshots": list(full_search_trace.get("snapshots") or []),
        "final_deltas": dict(full_search_trace.get("final_deltas") or {}),
        "missing_fields": missing_fields,
        "probe_mode_traces": {
            probe_mode: (
                full_search_trace
                if probe_mode == "full_search"
                else _build_mode_trace(
                    reference_move=int(reference_move),
                    probe_mode_summary=probe_modes.get(probe_mode),
                )
            )
            for probe_mode in PROBE_MODE_KEYS
        },
    }


def _failure_path_for_probe_mode(
    *,
    probe_mode: str,
    reference_move: int | None,
    selected_move: int | None,
    prior_probe_paths: dict[str, str] | None = None,
) -> str:
    if reference_move is None or selected_move is None:
        return "not_applicable"
    if probe_mode == "full_search":
        if (prior_probe_paths or {}).get(
            "policy_only"
        ) == "diverged_before_full_search":
            return "diverged_before_full_search"
        if (prior_probe_paths or {}).get("value_only") == "diverged_before_full_search":
            return "diverged_before_full_search"
        return (
            "reference_kept" if selected_move == reference_move else "full_search_drift"
        )
    return (
        "reference_kept"
        if selected_move == reference_move
        else "diverged_before_full_search"
    )


def build_paired_summary(row_traces: list[dict]) -> dict:
    row_ids = [row_trace.get("row_id") for row_trace in row_traces]
    if sorted(row_ids) != sorted(ROW_IDS):
        raise ValueError(
            f"paired summary requires exact required row ids once each: {ROW_IDS}"
        )

    probe_mode_selected_moves = {
        row_trace["row_id"]: {
            probe_mode: (
                (row_trace.get("probe_mode_traces") or {}).get(probe_mode) or {}
            ).get("selected_move")
            for probe_mode in PROBE_MODE_KEYS
        }
        for row_trace in row_traces
    }
    probe_mode_failure_paths = {}
    for row_trace in row_traces:
        row_paths = {}
        for probe_mode in PROBE_MODE_KEYS:
            row_paths[probe_mode] = _failure_path_for_probe_mode(
                probe_mode=probe_mode,
                reference_move=row_trace.get("reference_move"),
                selected_move=(
                    (row_trace.get("probe_mode_traces") or {}).get(probe_mode) or {}
                ).get("selected_move"),
                prior_probe_paths=row_paths,
            )
        probe_mode_failure_paths[row_trace["row_id"]] = row_paths

    invalid_paths = {
        failure_path
        for row_paths in probe_mode_failure_paths.values()
        for failure_path in row_paths.values()
    } - PROBE_MODE_FAILURE_PATHS
    if invalid_paths:
        raise ValueError(
            f"unsupported probe mode failure paths: {sorted(invalid_paths)}"
        )

    full_search_comparable_paths = [
        row_paths["full_search"]
        for row_paths in probe_mode_failure_paths.values()
        if row_paths["full_search"] != "not_applicable"
    ]
    pre_full_search_kept = all(
        row_paths.get("policy_only") == "reference_kept"
        and row_paths.get("value_only") == "reference_kept"
        for row_paths in probe_mode_failure_paths.values()
    )

    return {
        "probe_mode_selected_moves": probe_mode_selected_moves,
        "probe_mode_failure_paths": probe_mode_failure_paths,
        "shared_mechanism_supported": (
            pre_full_search_kept
            and len(full_search_comparable_paths) == len(probe_mode_failure_paths)
            and len(set(full_search_comparable_paths)) == 1
        ),
    }


def _full_search_row(rows: dict, row_id: str) -> dict:
    return dict(((rows.get(row_id) or {}).get("full_search") or {}))


def _snapshot_is_early(
    *, snapshots: list[dict], thresholds: dict, simulation_count: int | None = None
) -> bool:
    minimum_count = int(thresholds["minimum_early_snapshot_count"])
    if len(snapshots) < minimum_count:
        return False

    first_snapshot = snapshots[0] if snapshots else None
    last_snapshot = snapshots[-1] if snapshots else None
    if not isinstance(first_snapshot, dict) or not isinstance(last_snapshot, dict):
        return False

    first_simulation = first_snapshot.get("simulation")
    last_simulation = last_snapshot.get("simulation")
    if not isinstance(first_simulation, (int, float)) or not isinstance(
        last_simulation, (int, float)
    ):
        return False
    if first_simulation <= 0 or last_simulation <= 0:
        return False
    if len(snapshots) == minimum_count == 1:
        if (
            isinstance(simulation_count, bool)
            or not isinstance(simulation_count, int)
            or simulation_count <= 0
        ):
            return False
        return (float(first_simulation) / float(simulation_count)) <= float(
            thresholds["early_snapshot_fraction"]
        )

    return (float(first_simulation) / float(last_simulation)) <= float(
        thresholds["early_snapshot_fraction"]
    )


def _prior_is_materially_distorted(*, row: dict, thresholds: dict) -> bool:
    prior_reference_move = row.get("prior_reference_move")
    prior_selected_move = row.get("prior_selected_move")
    if prior_reference_move is None or prior_selected_move is None:
        return False
    prior_distortion = (
        1.0 if int(prior_reference_move) != int(prior_selected_move) else 0.0
    )
    return prior_distortion >= float(thresholds["material_prior_distortion"])


def _q_margin(row: dict) -> float | None:
    margin = row.get("selected_minus_reference_q_margin")
    return None if margin is None else float(margin)


def _visit_share_overtake(row: dict) -> float | None:
    overtake = row.get("selected_minus_reference_visit_share")
    return None if overtake is None else float(overtake)


def _drifting_child_has_no_q_or_visits(row: dict) -> bool:
    selected_move = row.get("selected_move")
    if selected_move is None:
        return False

    for child in row.get("child_stats") or []:
        if int(child.get("move")) != int(selected_move):
            continue
        visits = child.get("visits")
        q_value = child.get("q_value")
        return q_value is None or float(visits or 0.0) <= 0.0
    return False


def _all_rows_match(row_ids: list[str], rows: dict, predicate) -> bool:
    return all(predicate(_full_search_row(rows, row_id)) for row_id in row_ids)


def _all_rows_have_numeric_q_margin(row_ids: list[str], rows: dict) -> bool:
    return all(
        _q_margin(_full_search_row(rows, row_id)) is not None for row_id in row_ids
    )


def _all_rows_have_numeric_visit_share_overtake(row_ids: list[str], rows: dict) -> bool:
    return all(
        _visit_share_overtake(_full_search_row(rows, row_id)) is not None
        for row_id in row_ids
    )


def classify_paired_summary(
    *,
    rows: dict,
    paired_summary: dict,
    thresholds: dict,
    simulation_count: int | None = None,
) -> dict:
    row_ids = list(ROW_IDS)
    if not paired_summary.get("shared_mechanism_supported"):
        return {
            "classification": "shared_mechanism_disproved",
            "evidence_summary": "The paired rows do not share the same failure path, so a single shared full-search drift mechanism is not supported.",
        }

    failure_paths = paired_summary.get("probe_mode_failure_paths") or {}

    def full_search_failure_path(row_id: str) -> str | None:
        row_failure_path = failure_paths.get(row_id)
        if isinstance(row_failure_path, dict):
            return row_failure_path.get("full_search")
        return row_failure_path

    if any(
        full_search_failure_path(row_id) != "full_search_drift" for row_id in row_ids
    ):
        return {
            "classification": "unresolved",
            "evidence_summary": "Shared full-search drift is supported, but the paired evidence does not isolate one approved mechanism.",
        }

    meaningful_q_margin = float(thresholds["meaningful_q_margin"])
    small_q_margin = float(thresholds["small_q_margin"])
    meaningful_visit_share_overtake = float(
        thresholds["meaningful_visit_share_overtake"]
    )

    def row_has_early_snapshot(row: dict) -> bool:
        return _snapshot_is_early(
            snapshots=list(row.get("visit_snapshots") or []),
            thresholds=thresholds,
            simulation_count=simulation_count,
        )

    def row_is_early_prior_drift(row: dict) -> bool:
        return row_has_early_snapshot(row) and _prior_is_materially_distorted(
            row=row, thresholds=thresholds
        )

    def row_preserves_prior_early(row: dict) -> bool:
        return row_has_early_snapshot(row) and not _prior_is_materially_distorted(
            row=row, thresholds=thresholds
        )

    if _all_rows_match(row_ids, rows, row_is_early_prior_drift):
        if _all_rows_match(
            row_ids, rows, lambda row: bool(row.get("tactical_bias_applied"))
        ) and _all_rows_match(
            row_ids,
            rows,
            lambda row: (
                (_q_margin(row) is not None) and _q_margin(row) <= small_q_margin
            ),
        ):
            return {
                "classification": "tactical_root_bias_interaction",
                "evidence_summary": "Both rows drift from the first search snapshot with small Q margins while tactical root bias is explicitly active.",
            }

        if _all_rows_match(
            row_ids, rows, _drifting_child_has_no_q_or_visits
        ) and _all_rows_match(
            row_ids,
            rows,
            lambda row: (
                (_q_margin(row) is not None) and _q_margin(row) <= small_q_margin
            ),
        ):
            return {
                "classification": "fpu_or_unvisited_child_pressure",
                "evidence_summary": "Both rows drift from the first search snapshot with small Q margins, and the drifting child remains unvisited or lacks an observed Q value.",
            }

        if _all_rows_match(
            row_ids,
            rows,
            lambda row: (
                (_q_margin(row) is not None) and _q_margin(row) <= small_q_margin
            ),
        ):
            return {
                "classification": "root_prior_decay",
                "evidence_summary": "Both rows drift only after full search, early visits already tilt away from the prior reference move, and child Q margins stay below the meaningful threshold.",
            }

    if _all_rows_match(row_ids, rows, row_preserves_prior_early):
        if _all_rows_match(
            row_ids,
            rows,
            lambda row: (
                (_q_margin(row) is not None) and _q_margin(row) >= meaningful_q_margin
            ),
        ):
            return {
                "classification": "child_value_override",
                "evidence_summary": "Both rows preserve the prior reference move early, but child Q margins exceed the meaningful threshold and ultimately override it.",
            }

        if (
            _all_rows_have_numeric_q_margin(row_ids, rows)
            and _all_rows_have_numeric_visit_share_overtake(row_ids, rows)
            and _all_rows_match(
                row_ids,
                rows,
                lambda row: (
                    _q_margin(row) <= small_q_margin
                    and _visit_share_overtake(row) >= meaningful_visit_share_overtake
                ),
            )
        ):
            return {
                "classification": "backup_accumulation_drift",
                "evidence_summary": "Both rows hold the prior reference move early, child Q margins stay small, and drift appears only after backup accumulation later in search.",
            }

    return {
        "classification": "unresolved",
        "evidence_summary": "Shared full-search drift is supported, but the paired evidence does not isolate one approved mechanism.",
    }


def decision_for_classification(classification: str) -> str:
    return CLASSIFICATION_DECISIONS[classification]


def build_rows_payload(*, source_arbitration_artifact: dict) -> dict[str, dict]:
    return {
        row_id: build_row_trace(
            row_id=row_id,
            canonical_state=(source_arbitration_artifact["rows"][row_id] or {}).get(
                "canonical_state"
            ),
            legal_moves=(source_arbitration_artifact["rows"][row_id] or {}).get(
                "legal_moves"
            ),
            reference_move=int(
                (source_arbitration_artifact["rows"][row_id] or {})["reference_move"]
            ),
            probe_modes=source_arbitration_artifact["rows"][row_id],
        )
        for row_id in ROW_IDS
    }


def build_payload(*, source_arbitration_artifact: dict, rows: dict[str, dict]) -> dict:
    row_traces = [rows[row_id] for row_id in ROW_IDS]
    paired_summary = build_paired_summary(row_traces)
    classification = classify_paired_summary(
        rows=source_arbitration_artifact["rows"],
        paired_summary=paired_summary,
        thresholds=THRESHOLDS,
        simulation_count=source_arbitration_artifact["simulation_count"],
    )
    return {
        "schema": SCHEMA,
        "source_arbitration_artifact": {
            "artifact_path": source_arbitration_artifact["artifact_path"],
            "schema": source_arbitration_artifact["schema"],
            "row_ids": list(source_arbitration_artifact["row_ids"]),
            "selected_artifact": source_arbitration_artifact["selected_artifact"],
            "settings": dict(source_arbitration_artifact["source_settings"]),
            "classification": source_arbitration_artifact["classification"],
            "decision": source_arbitration_artifact["decision"],
        },
        "selected_artifact": source_arbitration_artifact["selected_artifact"],
        "thresholds": dict(THRESHOLDS),
        "settings": {
            "search_settings": dict(source_arbitration_artifact["search_settings"]),
            "seed": source_arbitration_artifact["seed"],
            "simulation_count": source_arbitration_artifact["simulation_count"],
        },
        "rows": rows,
        "paired_summary": paired_summary,
        "classification": classification,
        "decision": decision_for_classification(classification["classification"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose shared full-search drift from a source arbitration artifact"
    )
    parser.add_argument("--source-arbitration-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_arbitration_artifact = load_source_arbitration_artifact(
        args.source_arbitration_artifact
    )
    rows = build_rows_payload(source_arbitration_artifact=source_arbitration_artifact)
    payload = build_payload(
        source_arbitration_artifact=source_arbitration_artifact, rows=rows
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": SCHEMA,
                "decision": payload["decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
