from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from ml.alphazero_lite.forensic_suite import canonical_state_key


REQUIRED_CANDIDATE_FIELDS = (
    "state",
    "side_to_move",
    "legal_moves",
    "selection_reason",
    "source_artifact",
    "source_run",
    "consequence",
    "metrics",
)

PLY_TAG_PATTERN = re.compile(r"^ply_(\d+)$")

REQUIRED_STATE_FIELDS = (
    "player_pits",
    "opponent_pits",
    "player_store",
    "opponent_store",
    "current_player",
)

REASON_WEIGHTS = {
    "challenger_loss_vs_current": 10.0,
    "challenger_loss_vs_mcts1200": 10.0,
    "student_teacher_disagreement": 6.0,
    "large_value_error": 6.0,
    "high_search_entropy": 4.0,
    "large_best_second_gap": 4.0,
}

CONSEQUENCE_WEIGHTS = {
    "caused_loss": 4,
    "move_disagreement_in_loss": 3,
    "high_value_miscalibration": 2,
    "high_entropy_decision": 1,
    "forced_decision_gap": 0,
    "promotion_risk": 0,
}

LOSS_ARTIFACT_REASONS = {
    "arena_loss_vs_current": "challenger_loss_vs_current",
    "arena_loss_vs_mcts1200": "challenger_loss_vs_mcts1200",
}

SUPPORTED_REPORT_KINDS = {
    "forensic_suite",
    "arena_loss_vs_current",
    "arena_loss_vs_mcts1200",
}

REQUIRED_SOURCE_COVERAGE = {
    "challenger_loss_vs_current",
    "challenger_loss_vs_mcts1200",
    "student_teacher_disagreement",
    "high_search_entropy",
    "large_best_second_gap",
    "large_value_error",
}

NUMERIC_CANDIDATE_METRICS = (
    "regret",
    "value_error",
    "entropy",
    "best_second_gap",
)


def _coerce_metric_value(metrics: dict[str, Any], field: str) -> float:
    value = metrics.get(field, 0.0)
    if value is None:
        raise ValueError(f"candidate metric {field} must be numeric")

    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"candidate metric {field} must be numeric") from error


def _normalize_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("candidate state must be a dictionary")

    for field in REQUIRED_STATE_FIELDS:
        if field not in state:
            raise ValueError(f"candidate state is missing required field: {field}")

    player_pits = state["player_pits"]
    opponent_pits = state["opponent_pits"]
    if not isinstance(player_pits, list) or len(player_pits) != 6:
        raise ValueError("candidate state player_pits must be a list of 6 integers")
    if not isinstance(opponent_pits, list) or len(opponent_pits) != 6:
        raise ValueError("candidate state opponent_pits must be a list of 6 integers")

    canonical_player_pits: list[int] = []
    for value in player_pits:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("candidate state player_pits must be a list of 6 integers")
        canonical_player_pits.append(value)

    canonical_opponent_pits: list[int] = []
    for value in opponent_pits:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("candidate state opponent_pits must be a list of 6 integers")
        canonical_opponent_pits.append(value)

    for field in ("player_store", "opponent_store", "current_player"):
        if isinstance(state[field], bool) or not isinstance(state[field], int):
            raise ValueError(f"candidate state {field} must be an integer")

    current_player = int(state["current_player"])
    if current_player not in (0, 1):
        raise ValueError("candidate state.current_player must be 0 or 1")

    return {
        "player_pits": canonical_player_pits,
        "opponent_pits": canonical_opponent_pits,
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
        "current_player": current_player,
    }


def _normalize_optional_ply(candidate: dict[str, Any]) -> int | None:
    raw_ply = candidate.get("ply")
    raw_move_index = candidate.get("move_index")
    if raw_ply is not None and (isinstance(raw_ply, bool) or not isinstance(raw_ply, int)):
        raise ValueError("candidate ply must be an integer")
    if raw_move_index is not None and (isinstance(raw_move_index, bool) or not isinstance(raw_move_index, int)):
        raise ValueError("candidate ply must be an integer")
    if raw_ply is not None and raw_move_index is not None and raw_ply != raw_move_index:
        raise ValueError("candidate ply and move_index must match")
    raw_ply = raw_ply if raw_ply is not None else raw_move_index
    if raw_ply is None:
        return None
    if raw_ply < 0:
        raise ValueError("candidate ply must be non-negative")
    return int(raw_ply)


def _forensic_row_ply(row: dict[str, Any]) -> int | None:
    tags = row.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str):
                continue
            match = PLY_TAG_PATTERN.fullmatch(tag)
            if match is not None:
                return int(match.group(1))

    bucket = str(row.get("bucket", ""))
    if bucket == "opening_plies_1_8":
        return 8
    return None


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_CANDIDATE_FIELDS:
        if field not in candidate:
            raise ValueError(f"candidate is missing required field: {field}")

    normalized_candidate = dict(candidate)
    normalized_candidate["state"] = _normalize_state(candidate["state"])

    side_to_move = candidate["side_to_move"]
    if isinstance(side_to_move, bool) or not isinstance(side_to_move, int):
        raise ValueError("candidate side_to_move must be 0 or 1")
    normalized_candidate["side_to_move"] = int(side_to_move)

    legal_moves = candidate["legal_moves"]
    if not isinstance(legal_moves, list) or not legal_moves:
        raise ValueError("candidate legal_moves must contain unique moves in range 0..5")
    normalized_legal_moves: list[int] = []
    for move in legal_moves:
        if isinstance(move, bool) or not isinstance(move, int) or move < 0 or move >= 6:
            raise ValueError("candidate legal_moves must contain unique moves in range 0..5")
        normalized_legal_moves.append(move)
    if len(set(normalized_legal_moves)) != len(normalized_legal_moves):
        raise ValueError("candidate legal_moves must contain unique moves in range 0..5")
    normalized_candidate["legal_moves"] = normalized_legal_moves

    normalized_candidate["selection_reason"] = str(candidate["selection_reason"])
    normalized_candidate["source_artifact"] = str(candidate["source_artifact"])

    source_run = candidate["source_run"]
    if not isinstance(source_run, dict):
        raise ValueError("candidate source_run must be a dictionary")
    normalized_candidate["source_run"] = dict(source_run)

    normalized_candidate["consequence"] = str(candidate["consequence"])

    metrics = candidate["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("candidate metrics must be a dictionary")
    normalized_candidate["metrics"] = dict(metrics)

    ply = _normalize_optional_ply(candidate)
    if ply is not None:
        normalized_candidate["ply"] = ply

    if normalized_candidate["consequence"] not in CONSEQUENCE_WEIGHTS:
        raise ValueError(f"unknown consequence: {normalized_candidate['consequence']}")
    if normalized_candidate["side_to_move"] not in (0, 1):
        raise ValueError("candidate side_to_move must be 0 or 1")
    if normalized_candidate["side_to_move"] != normalized_candidate["state"]["current_player"]:
        raise ValueError("candidate side_to_move must match state.current_player")

    normalized_candidate["canonical_state"] = canonical_state_key(normalized_candidate["state"])
    return normalized_candidate


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    merged_fields = {"selection_reason", "source_artifact", "source_run", "metrics"}

    for candidate in candidates:
        normalized_candidate = normalize_candidate(candidate)
        canonical_state = normalized_candidate["canonical_state"]

        if canonical_state not in deduplicated:
            merged_candidate = dict(normalized_candidate)
            merged_candidate.pop("selection_reason")
            merged_candidate.pop("source_artifact")
            merged_candidate.pop("source_run")
            metrics = merged_candidate.pop("metrics")
            merged_candidate["selection_reasons"] = [normalized_candidate["selection_reason"]]
            merged_candidate["source_artifacts"] = [normalized_candidate["source_artifact"]]
            merged_candidate["source_runs"] = [normalized_candidate["source_run"]]
            merged_candidate["metadata"] = {
                "max_regret": _coerce_metric_value(metrics, "regret"),
                "max_value_error": _coerce_metric_value(metrics, "value_error"),
                "max_entropy": _coerce_metric_value(metrics, "entropy"),
                "max_best_second_gap": _coerce_metric_value(metrics, "best_second_gap"),
            }
            deduplicated[canonical_state] = merged_candidate
            continue

        merged_candidate = deduplicated[canonical_state]

        for field, value in normalized_candidate.items():
            if field in merged_fields:
                continue

            if field == "consequence":
                if CONSEQUENCE_WEIGHTS.get(str(value), 0) > CONSEQUENCE_WEIGHTS.get(str(merged_candidate[field]), 0):
                    merged_candidate[field] = value
                continue

            if field == "ply":
                merged_candidate[field] = max(int(merged_candidate.get(field, value)), int(value))
                continue

            if merged_candidate[field] != value:
                raise ValueError(f"conflicting candidate field: {field}")

        if normalized_candidate["selection_reason"] not in merged_candidate["selection_reasons"]:
            merged_candidate["selection_reasons"].append(normalized_candidate["selection_reason"])

        if normalized_candidate["source_artifact"] not in merged_candidate["source_artifacts"]:
            merged_candidate["source_artifacts"].append(normalized_candidate["source_artifact"])

        if normalized_candidate["source_run"] not in merged_candidate["source_runs"]:
            merged_candidate["source_runs"].append(normalized_candidate["source_run"])

        metrics = normalized_candidate["metrics"]
        merged_candidate["metadata"]["max_regret"] = max(
            merged_candidate["metadata"]["max_regret"],
            _coerce_metric_value(metrics, "regret"),
        )
        merged_candidate["metadata"]["max_value_error"] = max(
            merged_candidate["metadata"]["max_value_error"],
            _coerce_metric_value(metrics, "value_error"),
        )
        merged_candidate["metadata"]["max_entropy"] = max(
            merged_candidate["metadata"]["max_entropy"],
            _coerce_metric_value(metrics, "entropy"),
        )
        merged_candidate["metadata"]["max_best_second_gap"] = max(
            merged_candidate["metadata"]["max_best_second_gap"],
            _coerce_metric_value(metrics, "best_second_gap"),
        )

    return list(deduplicated.values())


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []

    for row in rows:
        metadata = row["metadata"]
        for reason in row["selection_reasons"]:
            if reason not in REASON_WEIGHTS:
                raise ValueError(f"unknown selection reason: {reason}")

        reason_score = sum(REASON_WEIGHTS[reason] for reason in row["selection_reasons"])
        regret_boost = float(metadata.get("max_regret", 0.0)) * 5.0
        value_error_boost = float(metadata.get("max_value_error", 0.0)) * 5.0
        entropy_boost = float(metadata.get("max_entropy", 0.0)) * 3.0
        gap_boost = float(metadata.get("max_best_second_gap", 0.0)) * 3.0
        source_classes = {str(source_run.get("kind", "")) for source_run in row["source_runs"]}
        multi_source_boost = max(0, len(source_classes) - 1) * 2.0
        priority_score = round(
            reason_score
            + regret_boost
            + value_error_boost
            + entropy_boost
            + gap_boost
            + multi_source_boost,
            4,
        )

        scored_rows.append(
            {
                **row,
                "priority_score": priority_score,
                "priority_breakdown": {
                    "reason_score": round(reason_score, 4),
                    "regret_boost": round(regret_boost, 4),
                    "value_error_boost": round(value_error_boost, 4),
                    "entropy_boost": round(entropy_boost, 4),
                    "gap_boost": round(gap_boost, 4),
                    "multi_source_boost": round(multi_source_boost, 4),
                },
            }
        )

    return sorted(
        scored_rows,
        key=lambda row: (-float(row["priority_score"]), str(row["canonical_state"])),
    )


def classify_report(payload: dict[str, Any], path: str) -> str:
    if payload.get("schema") == "azlite_forensic_suite_v1":
        return "forensic_suite"

    kind = payload.get("kind")
    if kind == "forensic_suite":
        raise ValueError(f"unsupported hard-state mining report: {path}")
    if kind in SUPPORTED_REPORT_KINDS:
        return str(kind)
    raise ValueError(f"unsupported hard-state mining report: {path}")


def _forensic_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for field in (
        "regret",
        "teacher_value",
        "system_value",
        "value_error",
        "entropy",
        "best_second_gap",
        "selected_move",
        "reference_move",
        "bucket",
        "id",
    ):
        if field in row:
            metrics[field] = row[field]

    if "agrees_top1" in row:
        metrics["agrees_top1"] = bool(row["agrees_top1"])

    return metrics


def _positive_metric(row: dict[str, Any], field: str) -> float:
    metrics = _forensic_row_metrics(row)
    return _coerce_metric_value(metrics, field)


def _extract_forensic_candidates(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source_artifact = str(artifact["path"])
    source_run_base = {
        "kind": "forensic_suite",
        "schema": str(artifact.get("schema", "")),
    }

    for system_name, system_payload in dict(artifact.get("systems", {})).items():
        if system_name != "challenger":
            continue

        source_run = {
            **source_run_base,
            "system": str(system_name),
        }

        for row in system_payload.get("rows", []):
            if not isinstance(row, dict):
                raise ValueError("forensic row must be a dictionary")

            for field in ("state", "side_to_move", "legal_moves"):
                if field not in row:
                    raise ValueError(f"missing required row field: {field}")

            metrics = _forensic_row_metrics(row)
            row_ply = _forensic_row_ply(row)

            if row.get("agrees_top1") is False:
                candidates.append(
                    normalize_candidate(
                        {
                            "state": row["state"],
                            "side_to_move": row["side_to_move"],
                            "legal_moves": row["legal_moves"],
                            "selection_reason": "student_teacher_disagreement",
                            "source_artifact": source_artifact,
                            "source_run": source_run,
                            "consequence": "move_disagreement_in_loss",
                            "metrics": metrics,
                            **({"ply": row_ply} if row_ply is not None else {}),
                        }
                    )
                )

            if _positive_metric(row, "value_error") > 0.0:
                candidates.append(
                    normalize_candidate(
                        {
                            "state": row["state"],
                            "side_to_move": row["side_to_move"],
                            "legal_moves": row["legal_moves"],
                            "selection_reason": "large_value_error",
                            "source_artifact": source_artifact,
                            "source_run": source_run,
                            "consequence": "high_value_miscalibration",
                            "metrics": metrics,
                            **({"ply": row_ply} if row_ply is not None else {}),
                        }
                    )
                )

            if _positive_metric(row, "entropy") > 0.0:
                candidates.append(
                    normalize_candidate(
                        {
                            "state": row["state"],
                            "side_to_move": row["side_to_move"],
                            "legal_moves": row["legal_moves"],
                            "selection_reason": "high_search_entropy",
                            "source_artifact": source_artifact,
                            "source_run": source_run,
                            "consequence": "high_entropy_decision",
                            "metrics": metrics,
                            **({"ply": row_ply} if row_ply is not None else {}),
                        }
                    )
                )

            if _positive_metric(row, "best_second_gap") > 0.0:
                candidates.append(
                    normalize_candidate(
                        {
                            "state": row["state"],
                            "side_to_move": row["side_to_move"],
                            "legal_moves": row["legal_moves"],
                            "selection_reason": "large_best_second_gap",
                            "source_artifact": source_artifact,
                            "source_run": source_run,
                            "consequence": "forced_decision_gap",
                            "metrics": metrics,
                            **({"ply": row_ply} if row_ply is not None else {}),
                        }
                    )
                )

    return candidates


def extract_candidates(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for artifact in artifacts:
        source_artifact = str(artifact["path"])
        source_kind = classify_report(artifact, source_artifact)

        if source_kind == "forensic_suite":
            candidates.extend(_extract_forensic_candidates(artifact))
            continue

        for row in artifact["rows"]:
            if not isinstance(row, dict):
                raise ValueError("artifact row must be a dictionary")

            for field in ("state", "side_to_move", "legal_moves", "selection_reason", "consequence"):
                if field not in row:
                    raise ValueError(f"missing required row field: {field}")

            expected_reason = LOSS_ARTIFACT_REASONS.get(source_kind)
            if expected_reason is not None and row.get("selection_reason") != expected_reason:
                raise ValueError(
                    f"selection_reason does not match artifact kind: {source_kind}"
                )

            candidates.append(
                normalize_candidate(
                    {
                        "state": row["state"],
                        "side_to_move": row["side_to_move"],
                        "legal_moves": row["legal_moves"],
                        "selection_reason": row["selection_reason"],
                        "source_artifact": source_artifact,
                        "source_run": {"kind": source_kind},
                        "consequence": row["consequence"],
                        "metrics": dict(row.get("metrics", {})),
                        **({"ply": row["ply"]} if "ply" in row else {}),
                        **({"move_index": row["move_index"]} if "move_index" in row else {}),
                    }
                )
            )

    return candidates


def _load_supported_artifact(path: Path) -> dict[str, Any] | None:
    try:
        raw_payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read hard-state mining artifact: {path}") from error

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in hard-state mining artifact: {path}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"hard-state mining artifact must be a JSON object: {path}")

    try:
        kind = classify_report(payload, str(path))
    except ValueError:
        return None

    artifact = {
        "path": str(path),
        "kind": kind,
    }

    if kind == "forensic_suite":
        systems = payload.get("systems", {})
        if not isinstance(systems, dict):
            raise ValueError(f"invalid forensic-suite artifact: {path}: 'systems' must be an object")

        normalized_systems: dict[str, dict[str, Any]] = {}
        for system_name, system_payload in systems.items():
            if not isinstance(system_payload, dict):
                raise ValueError(
                    f"invalid forensic-suite artifact: {path}: system '{system_name}' must be an object"
                )
            normalized_systems[str(system_name)] = dict(system_payload)

        artifact["schema"] = payload.get("schema")
        artifact["systems"] = normalized_systems
        return artifact

    if "rows" not in payload:
        raise ValueError(f"summary-only loss artifact is not supported: {path}")

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"malformed hard-state mining artifact rows: {path}")

    artifact["rows"] = list(rows)
    return artifact


def load_artifacts(inputs: list[str]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []

    for raw_input in inputs:
        input_path = Path(raw_input)
        if input_path.is_dir():
            for candidate_path in sorted(input_path.rglob("*.json")):
                artifact = _load_supported_artifact(candidate_path)
                if artifact is not None:
                    artifacts.append(artifact)
            continue

        artifact = _load_supported_artifact(input_path)
        if artifact is not None:
            artifacts.append(artifact)

    if not artifacts:
        raise ValueError("no supported hard-state mining artifacts found")

    return artifacts


def build_summary(
    artifacts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_counts_by_reason: dict[str, int] = {}
    candidate_counts_by_source_class: dict[str, int] = {}
    contribution_totals_by_reason: dict[str, int] = {}

    for candidate in candidates:
        reason = str(candidate["selection_reason"])
        candidate_counts_by_reason[reason] = candidate_counts_by_reason.get(reason, 0) + 1
        source_class = reason
        candidate_counts_by_source_class[source_class] = candidate_counts_by_source_class.get(source_class, 0) + 1

    for row in rows:
        for reason in row["selection_reasons"]:
            contribution_totals_by_reason[reason] = contribution_totals_by_reason.get(reason, 0) + 1

    source_coverage = sorted(candidate_counts_by_reason)
    return {
        "schema": "hard_state_mining_report_v1",
        "inputs": [artifact["path"] for artifact in artifacts],
        "candidate_counts_by_reason": candidate_counts_by_reason,
        "candidate_counts_by_source_class": candidate_counts_by_source_class,
        "contribution_totals_by_reason": contribution_totals_by_reason,
        "deduplicated_rows": len(rows),
        "top_priority": rows[:10],
        "source_coverage": source_coverage,
        "all_required_source_classes_observed": all(
            reason in candidate_counts_by_reason for reason in REQUIRED_SOURCE_COVERAGE
        ),
    }


def run_pipeline(inputs: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifacts = load_artifacts(inputs)
    candidates = extract_candidates(artifacts)
    rows = score_rows(deduplicate_candidates(candidates))
    return rows, build_summary(artifacts, candidates, rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_summary_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
