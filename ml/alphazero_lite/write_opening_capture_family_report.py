#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import canonical_state_key

REPORT_SCHEMA = "opening_capture_family_report_v1"
REFERENCE_SCHEMA = "azlite_forensic_references_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--current-artifact", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--artifact-simulations", type=int, default=384)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_suite_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("forensic suite must be a JSON list")

    suite_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("forensic suite rows must be dictionaries")
        suite_rows.append(
            {
                "id": row.get("id"),
                "state": row.get("state"),
                "legal_moves": list(row.get("legal_moves") or []),
                "phase": row.get("phase"),
                "bucket": row.get("bucket"),
                "reference_move": row.get("reference_move"),
            }
        )
    return suite_rows


def load_reference_moves(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    if not path.exists():
        raise ValueError("reference artifact does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("reference artifact is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("reference artifact must be a JSON object")
    if payload.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"reference artifact must use schema {REFERENCE_SCHEMA}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("reference artifact must contain a rows list")
    reference_moves: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        canonical_state = row.get("canonical_state")
        reference_move = row.get("reference_move")
        if not isinstance(canonical_state, str):
            continue
        if isinstance(reference_move, int) and not isinstance(reference_move, bool):
            reference_moves[canonical_state] = reference_move
    return reference_moves


def enrich_suite_rows_with_reference_moves(rows: list[dict], reference_moves: dict[str, int]) -> list[dict]:
    enriched_rows = []
    for row in rows:
        enriched_row = dict(row)
        raw_state = enriched_row.get("state")
        if isinstance(raw_state, dict):
            canonical_state = canonical_state_key(raw_state)
            if canonical_state in reference_moves:
                enriched_row["reference_move"] = reference_moves[canonical_state]
        enriched_rows.append(enriched_row)
    return enriched_rows


def is_opening_capture_family_row(row: dict) -> bool:
    row_id = row.get("id")
    return (
        isinstance(row_id, str)
        and row_id.startswith("capture_available-")
        and row.get("bucket") == "capture_available"
        and row.get("phase") == "opening"
        and row.get("legal_moves") == [0, 1, 2, 3, 4]
    )


def has_stable_reference_move(row: dict) -> bool:
    reference_move = row.get("reference_move")
    return isinstance(reference_move, int) and not isinstance(reference_move, bool)


def is_tracked_family_row(row: dict) -> bool:
    return is_opening_capture_family_row(row) and has_stable_reference_move(row)


def _tracked_family_reference_status(row: dict) -> str | None:
    if not is_opening_capture_family_row(row):
        return None
    reference_move = row.get("reference_move")
    if reference_move is None:
        return "missing"
    return "tracked" if has_stable_reference_move(row) else "invalid"


def select_tracked_family_rows(rows: list[dict]) -> list[dict]:
    return [dict(row) for row in rows if is_tracked_family_row(row)]


def missing_tracked_family_references(rows: list[dict]) -> list[dict]:
    missing = []
    for row in rows:
        status = _tracked_family_reference_status(row)
        if status is None:
            continue
        row_id = row.get("id")
        if status == "missing":
            missing.append({"code": "missing_reference_move", "id": row_id})
        elif status == "invalid":
            missing.append(
                {
                    "code": "invalid_reference_move",
                    "id": row_id,
                    "reference_move": row.get("reference_move"),
                }
            )
    return missing


def load_arena_module():
    return importlib.import_module("ml.alphazero_lite.arena")


def _select_from_distribution(distribution: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(distribution[move]), move))


def _summarize_distribution(
    distribution: list[float], *, selected_move: int | None, reference_move: int, value: float
) -> dict:
    early_mass = float(distribution[0]) + float(distribution[1])
    reference_mass = float(distribution[reference_move]) if 0 <= reference_move < len(distribution) else 0.0
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "value": float(value),
        "early_mass": round(early_mass, 4),
        "reference_mass": round(reference_mass, 4),
        "reference_margin": round(reference_mass - early_mass, 4),
        "reference_move": int(reference_move),
    }


def summarize_prior(position_summary: dict, *, reference_move: int, legal_moves: list[int]) -> dict:
    policy = list(position_summary.get("policy") or [])
    distribution = [0.0] * 6
    for move in legal_moves:
        if move < len(policy):
            distribution[move] = float(policy[move])
    return _summarize_distribution(
        distribution,
        selected_move=_select_from_distribution(distribution, legal_moves),
        reference_move=reference_move,
        value=float(position_summary.get("value", 0.0)),
    )


def summarize_search(position_summary: dict, *, reference_move: int, legal_moves: list[int]) -> dict:
    visits = list(position_summary.get("visits") or [])
    distribution = [0.0] * 6
    legal_total = sum(float(visits[move]) for move in legal_moves if move < len(visits))
    if legal_total > 0:
        for move in legal_moves:
            if move < len(visits):
                distribution[move] = float(visits[move]) / legal_total
    else:
        distribution = [0.0] * 6
        policy = list(position_summary.get("policy") or [])
        for move in legal_moves:
            if move < len(policy):
                distribution[move] = float(policy[move])
    return _summarize_distribution(
        distribution,
        selected_move=position_summary.get("selected_move")
        if position_summary.get("selected_move") is not None
        else _select_from_distribution(distribution, legal_moves),
        reference_move=reference_move,
        value=float(position_summary.get("value", 0.0)),
    )


def build_report(
    *,
    suite_path: Path,
    reference_path: Path | None = None,
    current_artifact_path: Path,
    candidate_artifact_path: Path,
    artifact_simulations: int,
    c_puct: float,
    seed: int,
) -> dict:
    suite_rows = load_suite_rows(suite_path)
    enriched_rows = enrich_suite_rows_with_reference_moves(suite_rows, load_reference_moves(reference_path))
    tracked_rows = select_tracked_family_rows(enriched_rows)
    missing_references = missing_tracked_family_references(enriched_rows)

    report_rows = []
    if tracked_rows:
        arena = load_arena_module()
        search_options = arena.build_eval_search_options()
        current_evaluator = arena.ArtifactEvaluator(current_artifact_path)
        candidate_evaluator = arena.ArtifactEvaluator(candidate_artifact_path)

        for index, row in enumerate(tracked_rows):
            state = row["state"]
            legal_moves = list(row["legal_moves"])
            reference_move = int(row["reference_move"])
            current_summary = arena.evaluate_artifact_position(
                artifact_path=current_artifact_path,
                evaluator=current_evaluator,
                state=state,
                simulations=artifact_simulations,
                seed=seed + index,
                c_puct=c_puct,
                search_options=search_options,
            )
            candidate_summary = arena.evaluate_artifact_position(
                artifact_path=candidate_artifact_path,
                evaluator=candidate_evaluator,
                state=state,
                simulations=artifact_simulations,
                seed=seed + index,
                c_puct=c_puct,
                search_options=search_options,
            )
            report_rows.append(
                {
                    "id": row["id"],
                    "bucket": row["bucket"],
                    "phase": row["phase"],
                    "legal_moves": legal_moves,
                    "reference_move": reference_move,
                    "current_prior_summary": summarize_prior(
                        current_summary,
                        reference_move=reference_move,
                        legal_moves=legal_moves,
                    ),
                    "candidate_prior_summary": summarize_prior(
                        candidate_summary,
                        reference_move=reference_move,
                        legal_moves=legal_moves,
                    ),
                    "current_searched_summary": summarize_search(
                        current_summary,
                        reference_move=reference_move,
                        legal_moves=legal_moves,
                    ),
                    "candidate_searched_summary": summarize_search(
                        candidate_summary,
                        reference_move=reference_move,
                        legal_moves=legal_moves,
                    ),
                }
            )

    return {
        "schema": REPORT_SCHEMA,
        "suite_path": str(suite_path),
        "reference_path": None if reference_path is None else str(reference_path),
        "current_artifact_path": str(current_artifact_path),
        "candidate_artifact_path": str(candidate_artifact_path),
        "artifact_simulations": int(artifact_simulations),
        "c_puct": float(c_puct),
        "seed": int(seed),
        "missing_references": missing_references,
        "rows": report_rows,
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(
        suite_path=Path(args.suite),
        reference_path=None if args.reference is None else Path(args.reference),
        current_artifact_path=Path(args.current_artifact),
        candidate_artifact_path=Path(args.candidate_artifact),
        artifact_simulations=int(args.artifact_simulations),
        c_puct=float(args.c_puct),
        seed=int(args.seed),
    )
    write_report(Path(args.out), report)


if __name__ == "__main__":
    main()
