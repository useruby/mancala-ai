#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options


DEFAULT_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_FALLBACK_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_BUDGETS = (384, 1200)
DEFAULT_SEED = 17
DEFAULT_C_PUCT = 1.25
SCHEMA = "azlite_corrected_guard_kill_gate_v1"
GUARD_ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
REFERENCE_MOVES = {
    "capture_available-002": 2,
    "capture_available-003": 2,
    "capture_available-006": 2,
    "capture_available-007": 2,
    "capture_available-008": 1,
}
STALE_REFERENCE_MARKERS = (
    "train_only_forensic_references",
    "incumbent_train_only_forensic_references",
)


def parse_budgets(raw_value: str) -> tuple[int, ...]:
    budgets = tuple(
        int(value.strip()) for value in raw_value.split(",") if value.strip()
    )
    if not budgets:
        raise ValueError("guard budgets must not be empty")
    return budgets


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_reference_rows(
    reference_artifact: Path,
    fallback_reference_artifact: Path | None = DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = load_json(reference_artifact).get("rows") or []
    fallback_rows = []
    if fallback_reference_artifact is not None and fallback_reference_artifact.exists():
        fallback_rows = load_json(fallback_reference_artifact).get("rows") or []
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in fallback_rows:
        row_id = row.get("id")
        if isinstance(row_id, str):
            rows_by_id.setdefault(row_id, []).append(row)
    for row in rows:
        row_id = row.get("id")
        if isinstance(row_id, str):
            rows_by_id[row_id] = [row]

    missing_metadata = False
    duplicate_conflicts = 0
    notes: list[str] = []
    normalized_rows: dict[str, dict[str, Any]] = {}

    reference_path_text = str(reference_artifact)
    stale_reference_artifact = any(
        marker in reference_path_text for marker in STALE_REFERENCE_MARKERS
    )
    if stale_reference_artifact:
        notes.append("stale_reference_artifact")

    for row_id in GUARD_ROW_IDS:
        matches = list(rows_by_id.get(row_id, []))
        if not matches:
            missing_metadata = True
            notes.append(f"missing_{row_id}")
            continue
        if len(matches) > 1:
            duplicate_conflicts += 1
        row = matches[0]
        child_stats = list(row.get("child_stats") or [])
        legal_moves = [int(child["move"]) for child in child_stats if "move" in child]
        state = row.get("state")
        reference_move = row.get("reference_move")
        if (
            not isinstance(state, dict)
            or not legal_moves
            or reference_move is None
            or int(reference_move) != REFERENCE_MOVES[row_id]
        ):
            missing_metadata = True
            notes.append(f"invalid_metadata_{row_id}")
            continue
        normalized_rows[row_id] = {
            "row_id": row_id,
            "state": dict(state),
            "legal_moves": legal_moves,
            "corrected_reference_move": int(reference_move),
        }

    status = "ok"
    if missing_metadata or stale_reference_artifact or duplicate_conflicts > 0:
        status = "invalid"
    return (
        {
            "reference_artifact": str(reference_artifact),
            "fallback_reference_artifact": None
            if fallback_reference_artifact is None
            else str(fallback_reference_artifact),
            "missing_metadata": missing_metadata,
            "stale_reference_artifact": stale_reference_artifact,
            "duplicate_conflicts": duplicate_conflicts,
            "status": status,
            "notes": ",".join(notes) if notes else "ok",
        },
        normalized_rows,
    )


def probe_view(
    *, probe: dict[str, Any], legal_moves: list[int], reference_move: int
) -> dict[str, Any]:
    child_stats = {
        int(child["move"]): child for child in list(probe.get("child_stats") or [])
    }
    visits = list(probe.get("visits") or [])
    total_visits = sum(
        float(visits[move]) for move in legal_moves if move < len(visits)
    )
    selected_move = probe.get("selected_move")
    reference_visit_share = None
    if total_visits > 0.0 and reference_move < len(visits):
        reference_visit_share = round(float(visits[reference_move]) / total_visits, 4)
    selected_q = None
    if selected_move is not None and int(selected_move) in child_stats:
        selected_q = round(
            float(child_stats[int(selected_move)].get("q_value", 0.0)), 4
        )
    reference_q = None
    if reference_move in child_stats:
        reference_q = round(float(child_stats[reference_move].get("q_value", 0.0)), 4)
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "selected_is_reference": selected_move == reference_move,
        "reference_visit_share": reference_visit_share,
        "selected_minus_reference_q_margin": None
        if selected_q is None or reference_q is None
        else round(selected_q - reference_q, 4),
    }


def run_corrected_guard_kill_gate(
    *,
    candidate_path: Path,
    reference_artifact: Path = DEFAULT_REFERENCE_ARTIFACT,
    fallback_reference_artifact: Path | None = DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    seed: int = DEFAULT_SEED,
    c_puct: float = DEFAULT_C_PUCT,
) -> dict[str, Any]:
    reference_validation, reference_rows = validate_reference_rows(
        reference_artifact,
        fallback_reference_artifact=fallback_reference_artifact,
    )
    search_options = dict(build_eval_search_options())
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_path": str(candidate_path),
        "reference_artifact": str(reference_artifact),
        "fallback_reference_artifact": None
        if fallback_reference_artifact is None
        else str(fallback_reference_artifact),
        "budgets": list(budgets),
        "c_puct": float(c_puct),
        "reference_validation": reference_validation,
        "rows": [],
        "pass": False,
        "decision": "reject_guard_regression",
        "reason": reference_validation["notes"],
    }
    if reference_validation["status"] != "ok":
        return payload

    evaluator = ArtifactEvaluator(candidate_path)
    row_payloads: list[dict[str, Any]] = []
    gate_pass = True
    for row_index, row_id in enumerate(GUARD_ROW_IDS):
        row = reference_rows.get(row_id)
        if row is None:
            gate_pass = False
            row_payloads.append(
                {
                    "row_id": row_id,
                    "corrected_reference_move": REFERENCE_MOVES[row_id],
                    "selected_move_384": None,
                    "selected_move_1200": None,
                    "reference_visit_share_384": None,
                    "reference_visit_share_1200": None,
                    "pass": False,
                    "notes": "missing_corrected_reference_metadata",
                }
            )
            continue

        results_by_budget: dict[int, dict[str, Any]] = {}
        for budget in budgets:
            probe = evaluate_artifact_position(
                artifact_path=candidate_path,
                evaluator=evaluator,
                state=row["state"],
                simulations=int(budget),
                seed=seed + row_index + int(budget),
                c_puct=float(c_puct),
                search_options=search_options,
                ablation_mode="full",
            )
            results_by_budget[int(budget)] = probe_view(
                probe=probe,
                legal_moves=row["legal_moves"],
                reference_move=row["corrected_reference_move"],
            )

        row_pass = all(
            results_by_budget[int(budget)]["selected_is_reference"] is True
            for budget in budgets
        )
        if not row_pass:
            gate_pass = False
        row_payloads.append(
            {
                "row_id": row_id,
                "corrected_reference_move": row["corrected_reference_move"],
                "selected_move_384": results_by_budget.get(384, {}).get(
                    "selected_move"
                ),
                "selected_move_1200": results_by_budget.get(1200, {}).get(
                    "selected_move"
                ),
                "reference_visit_share_384": results_by_budget.get(384, {}).get(
                    "reference_visit_share"
                ),
                "reference_visit_share_1200": results_by_budget.get(1200, {}).get(
                    "reference_visit_share"
                ),
                "selected_minus_reference_q_margin_384": results_by_budget.get(
                    384, {}
                ).get("selected_minus_reference_q_margin"),
                "selected_minus_reference_q_margin_1200": results_by_budget.get(
                    1200, {}
                ).get("selected_minus_reference_q_margin"),
                "pass": row_pass,
                "notes": "pass_reference_selected"
                if row_pass
                else "selected_move_not_reference",
            }
        )

    payload["rows"] = row_payloads
    payload["pass"] = gate_pass
    payload["decision"] = "pass" if gate_pass else "reject_guard_regression"
    payload["reason"] = (
        "all_corrected_reference_rows_selected"
        if gate_pass
        else "guard_row_selected_non_reference"
    )
    return payload
