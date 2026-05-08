from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.alphazero_lite.diagnose_search_interaction import build_matrix_from_runs, choose_next_branch


SEARCH_INTERACTION_SCHEMA = "azlite_search_interaction_diagnostic_v1"
SEARCH_VALUE_INTERACTION_SCHEMA = "azlite_search_value_interaction_diagnostic_v1"
SEARCH_VALUE_PRIMARY_ROW_IDS = [
    "capture_available-002",
    "capture_available-003",
    "incumbent_proxy_disagreement-031",
    "incumbent_proxy_disagreement-033",
]
SEARCH_VALUE_COMPARATOR_ROW_IDS = ["opening_plies_1_8-057"]
DECISION_LABELS = [
    "bad_priors",
    "search_overrides_prior",
    "q_value_backup_issue",
    "insufficient_child_stats",
    "mixed",
]
EXCLUDED_ROW_PREFIXES = ("high_imbalance-",)
EXCLUDED_ROW_IDS = {"sparse_endgame-009"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def build_matrix_payload(*, original_run_dir: Path, rebalanced_run_dir: Path) -> dict:
    matrix = build_matrix_from_runs(original_run=original_run_dir, rebalanced_run=rebalanced_run_dir)
    return {"matrix": matrix, "summary": choose_next_branch(matrix)}


def _is_excluded_row(row_id: str) -> bool:
    return row_id.startswith(EXCLUDED_ROW_PREFIXES) or row_id in EXCLUDED_ROW_IDS


def resolve_target_rows(*, original_run_dir: Path, rebalanced_run_dir: Path, explicit_rows: list[str] | None) -> list[str]:
    if explicit_rows:
        resolved = [row_id for row_id in explicit_rows if not _is_excluded_row(row_id)]
        if not resolved:
            raise ValueError("search interaction diagnostic resolved no target rows")
        return resolved

    payload = build_matrix_payload(original_run_dir=original_run_dir, rebalanced_run_dir=rebalanced_run_dir)
    rows = payload["summary"].get("priority_rows") or []
    resolved = [row_id for row_id in rows if not _is_excluded_row(row_id)]
    if not resolved:
        raise ValueError("search interaction diagnostic resolved no target rows")
    return resolved


def load_selected_artifact_path(run_dir: Path) -> str:
    manifest = _load_json(run_dir / "selection" / "selection_manifest.json")
    selected_target = manifest.get("selected_target")
    if isinstance(selected_target, str) and selected_target:
        return selected_target

    selected_artifact = manifest.get("selected_artifact")
    if not isinstance(selected_artifact, str) or not selected_artifact:
        raise ValueError(f"selection manifest missing selected_artifact for {run_dir}")
    return selected_artifact


def load_row_context(*, row_id: str, original_run_dir: Path, rebalanced_run_dir: Path) -> dict:
    original_forensics = _load_json(original_run_dir / "final" / "selected_candidate_forensics.json")
    rebalanced_forensics = _load_json(rebalanced_run_dir / "final" / "selected_candidate_forensics.json")
    original_opening = _load_json(original_run_dir / "final" / "opening_capture_family_report.json")
    rebalanced_opening = _load_json(rebalanced_run_dir / "final" / "opening_capture_family_report.json")

    current_rows = _rows_by_id(original_forensics["systems"]["current"]["rows"])
    original_rows = _rows_by_id(original_forensics["systems"]["challenger"]["rows"])
    rebalanced_rows = _rows_by_id(rebalanced_forensics["systems"]["challenger"]["rows"])
    original_opening_rows = _rows_by_id(original_opening.get("rows") or [])
    rebalanced_opening_rows = _rows_by_id(rebalanced_opening.get("rows") or [])

    current_row = current_rows[row_id]
    return {
        "row_id": row_id,
        "bucket": current_row["bucket"],
        "phase": current_row["phase"],
        "reference_move": current_row["reference_move"],
        "teacher_value": current_row["teacher_value"],
        "current_row": current_row,
        "original_row": original_rows[row_id],
        "rebalanced_row": rebalanced_rows[row_id],
        "original_opening_row": original_opening_rows.get(row_id),
        "rebalanced_opening_row": rebalanced_opening_rows.get(row_id),
    }


def diagnostic_out_path(*, rebalanced_run_dir: Path) -> Path:
    return rebalanced_run_dir / "final" / "search_interaction_diagnostic.json"


def search_value_interaction_diagnostic_out_path(*, rebalanced_run_dir: Path) -> Path:
    return rebalanced_run_dir / "final" / "search_value_interaction_diagnostic.json"


def _distribution_from_policy(policy: list[float], legal_moves: list[int]) -> dict[str, float]:
    return {str(move): round(float(policy[move]) if move < len(policy) else 0.0, 4) for move in legal_moves}


def _distribution_from_visits(visits: list[float], legal_moves: list[int]) -> dict[str, float] | None:
    legal_total = sum(float(visits[move]) for move in legal_moves if move < len(visits))
    if legal_total <= 0:
        return None
    return {
        str(move): round((float(visits[move]) if move < len(visits) else 0.0) / legal_total, 4)
        for move in legal_moves
    }


def _per_move_child_stats(child_stats: list[dict]) -> tuple[dict[str, int] | None, dict[str, float] | None, list[dict] | None]:
    if not child_stats:
        return None, None, None
    visits = {str(int(child["move"])): int(child.get("visits", 0)) for child in child_stats}
    q_values = {
        str(int(child["move"])): round(float(child["q_value"]), 4)
        for child in child_stats
        if "q_value" in child
    }
    normalized_children = [
        {
            "move": int(child["move"]),
            "visits": int(child.get("visits", 0)),
            **({"q_value": round(float(child["q_value"]), 4)} if "q_value" in child else {}),
            **({"win_rate": round(float(child["win_rate"]), 4)} if "win_rate" in child else {}),
        }
        for child in child_stats
    ]
    return visits, q_values or None, normalized_children


def _top_distribution_key(distribution: dict[str, float] | None) -> int | None:
    if not distribution:
        return None
    move, _ = max(distribution.items(), key=lambda item: (float(item[1]), -int(item[0])))
    return int(move)


def _top_q_value_key(q_values: dict[str, float] | None) -> int | None:
    if not q_values:
        return None
    move, _ = max(q_values.items(), key=lambda item: (float(item[1]), -int(item[0])))
    return int(move)


def _snapshot_status(visit_snapshots) -> str:
    if visit_snapshots is None:
        return "missing"
    if len(visit_snapshots) == 0:
        return "empty"
    return "available"


def _selection_breakdown_top_move(artifact: dict, key: str, fallback) -> int | None:
    breakdown = artifact.get("selection_breakdown") or {}
    top_move = breakdown.get(key)
    if top_move is None:
        return fallback
    return int(top_move)


def summarize_row_mechanism(row_payload: dict) -> str:
    artifact = row_payload["rebalanced_challenger"]
    policy_top_move = _selection_breakdown_top_move(
        artifact,
        "policy_top_move",
        _top_distribution_key(artifact.get("raw_policy_distribution")),
    )
    visit_top_move = _selection_breakdown_top_move(
        artifact,
        "visit_top_move",
        _top_distribution_key(artifact.get("searched_visit_distribution")),
    )
    q_top_move = _selection_breakdown_top_move(
        artifact,
        "q_top_move",
        _top_q_value_key(artifact.get("per_move_q_values")),
    )
    snapshot_status = _snapshot_status(artifact.get("visit_snapshots"))
    return (
        f"policy leans to {policy_top_move}, visits finish on {visit_top_move}, "
        f"q-values favor {q_top_move}, snapshots {snapshot_status}"
    )


def _search_value_row_ids(resolved_rows: list[str]) -> tuple[list[str], list[str]]:
    primary = [row_id for row_id in SEARCH_VALUE_PRIMARY_ROW_IDS if row_id in resolved_rows]
    comparator = [row_id for row_id in SEARCH_VALUE_COMPARATOR_ROW_IDS if row_id in resolved_rows]
    return primary, comparator


def _build_search_value_rows(rows: dict[str, dict]) -> dict[str, dict]:
    def sibling_decision(row_payload: dict) -> str:
        rebalanced = row_payload["rebalanced_challenger"]
        reference_key = str(row_payload["reference_move"])
        selected_move = rebalanced.get("selected_move")
        if selected_move is None:
            return "insufficient_child_stats"
        selected_key = str(selected_move)
        raw_policy = rebalanced.get("raw_policy_distribution") or {}
        searched = rebalanced.get("searched_visit_distribution") or {}
        q_values = rebalanced.get("per_move_q_values") or {}
        if {reference_key, selected_key}.issubset(raw_policy) and {reference_key, selected_key}.issubset(searched) and {reference_key, selected_key}.issubset(q_values):
            return row_payload["decision"]
        return "insufficient_child_stats"

    search_value_rows = {}
    for row_id, row_payload in rows.items():
        decision = sibling_decision(row_payload)
        search_value_rows[row_id] = {
            **row_payload,
            "decision": decision,
            "notes": {
                "search_overrides_prior": ["prior corrected but searched move still wrong"],
                "q_value_backup_issue": ["backup values favor the wrong child"],
                "bad_priors": ["reference move remains underweighted in raw policy"],
                "insufficient_child_stats": ["child stats unavailable for one or more compared artifacts"],
                "mixed": ["multiple mechanisms remain plausible"],
            }[decision],
            "row_mechanism_summary": summarize_row_mechanism(row_payload),
        }
    return search_value_rows


def build_search_value_interaction_payload_from_source_payload(*, payload: dict, source_diagnostic_path: str) -> dict:
    primary_row_ids, comparator_row_ids = _search_value_row_ids(payload["row_source"]["resolved_rows"])
    sibling_rows = _build_search_value_rows(payload["rows"])
    return {
        **payload,
        "schema": SEARCH_VALUE_INTERACTION_SCHEMA,
        "source_diagnostic_path": source_diagnostic_path,
        "primary_row_ids": primary_row_ids,
        "comparator_row_ids": comparator_row_ids,
        "rows": sibling_rows,
        "summary": {
            **build_summary(sibling_rows),
            "primary_row_count": len(primary_row_ids),
            "comparator_row_count": len(comparator_row_ids),
        },
    }


def build_artifact_row(
    *,
    artifact_path: str,
    row_id: str,
    bucket: str,
    phase: str,
    reference_move: int,
    system_value: float,
    teacher_value: float,
    value_error: float,
    probe_summary: dict,
    legal_moves: list[int],
) -> dict:
    probe_value = probe_summary.get("value")
    normalized_probe_value = round(float(probe_value), 4) if probe_value is not None else None
    normalized_probe_value_error = (
        round(abs(float(probe_value) - float(teacher_value)), 4) if probe_value is not None else None
    )
    raw_policy_distribution = _distribution_from_policy(list(probe_summary.get("policy") or []), legal_moves)
    searched_visit_distribution = _distribution_from_visits(list(probe_summary.get("visits") or []), legal_moves)
    per_move_visits, per_move_q_values, child_stats = _per_move_child_stats(list(probe_summary.get("child_stats") or []))

    missing_fields = [
        field_name
        for field_name, value in {
            "probe_value": normalized_probe_value,
            "probe_value_error": normalized_probe_value_error,
            "searched_visit_distribution": searched_visit_distribution,
            "per_move_visits": per_move_visits,
            "per_move_q_values": per_move_q_values,
            "child_stats": child_stats,
        }.items()
        if value is None
    ]

    return {
        "artifact_path": artifact_path,
        "row_id": row_id,
        "bucket": bucket,
        "phase": phase,
        "reference_move": int(reference_move),
        "selected_move": probe_summary.get("selected_move"),
        "forensic_system_value": round(float(system_value), 4),
        "teacher_value": round(float(teacher_value), 4),
        "forensic_value_error": round(float(value_error), 4),
        "probe_value": normalized_probe_value,
        "probe_value_error": normalized_probe_value_error,
        "raw_policy_distribution": raw_policy_distribution,
        "searched_visit_distribution": searched_visit_distribution,
        "per_move_visits": per_move_visits,
        "per_move_q_values": per_move_q_values,
        "child_stats_available": child_stats is not None,
        "child_stats": child_stats,
        "selection_breakdown": probe_summary.get("selection_breakdown"),
        "visit_snapshots": (
            None if "visit_snapshots" not in probe_summary else list(probe_summary.get("visit_snapshots") or [])
        ),
        "missing_fields": missing_fields,
    }


def load_arena_module():
    import importlib

    return importlib.import_module("ml.alphazero_lite.arena")


def probe_artifact_position(*, artifact_path: str, state: dict, simulations: int, seed: int, c_puct: float, evaluator=None) -> dict:
    arena = load_arena_module()
    if evaluator is None:
        evaluator = arena.ArtifactEvaluator(Path(artifact_path))
    return arena.evaluate_artifact_position(
        artifact_path=Path(artifact_path),
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=c_puct,
        search_options=arena.build_eval_search_options(),
    )


def decide_row(row_payload: dict) -> str:
    rebalanced = row_payload["rebalanced_challenger"]
    reference_key = str(row_payload["reference_move"])
    compared = [row_payload["current"], row_payload["original_challenger"], rebalanced]
    if not all(artifact.get("child_stats_available", False) for artifact in compared):
        return "insufficient_child_stats"

    raw_policy = rebalanced.get("raw_policy_distribution") or {}
    searched = rebalanced.get("searched_visit_distribution") or {}
    q_values = rebalanced.get("per_move_q_values") or {}
    selected_move = rebalanced.get("selected_move")
    if selected_move == row_payload["reference_move"]:
        return "mixed"

    selected_key = str(selected_move)
    q_gap = q_values.get(selected_key, -1.0) - q_values.get(reference_key, -1.0)

    if raw_policy.get(reference_key, 0.0) >= 0.4 and searched.get(reference_key, 0.0) < 0.3:
        if q_gap > 0.3:
            return "q_value_backup_issue"
        return "search_overrides_prior"

    if q_gap > 0.3:
        return "q_value_backup_issue"

    if raw_policy.get(reference_key, 0.0) < 0.2:
        return "bad_priors"

    return "mixed"


def build_summary(rows: dict[str, dict]) -> dict:
    decision_counts = {label: 0 for label in DECISION_LABELS}
    for row in rows.values():
        decision_counts[row["decision"]] += 1

    if decision_counts["search_overrides_prior"] or decision_counts["q_value_backup_issue"]:
        next_branch = "search_value_interaction_investigation"
    elif decision_counts["bad_priors"]:
        next_branch = "replay_source_coverage_investigation"
    else:
        next_branch = "broaden_failure_surface_review"
    return {"decision_counts": decision_counts, "next_branch": next_branch}


def build_search_interaction_payload(
    *,
    original_run_dir: Path,
    rebalanced_run_dir: Path,
    current_artifact_path: str,
    explicit_rows: list[str] | None,
    artifact_simulations: int = 384,
    c_puct: float = 1.25,
    seed: int = 42,
) -> dict:
    resolved_rows = resolve_target_rows(
        original_run_dir=original_run_dir,
        rebalanced_run_dir=rebalanced_run_dir,
        explicit_rows=explicit_rows,
    )
    arena = load_arena_module()
    original_selected_artifact = load_selected_artifact_path(original_run_dir)
    rebalanced_selected_artifact = load_selected_artifact_path(rebalanced_run_dir)
    evaluators = {
        current_artifact_path: arena.ArtifactEvaluator(Path(current_artifact_path)),
        original_selected_artifact: arena.ArtifactEvaluator(Path(original_selected_artifact)),
        rebalanced_selected_artifact: arena.ArtifactEvaluator(Path(rebalanced_selected_artifact)),
    }
    rows = {}
    for index, row_id in enumerate(resolved_rows):
        context = load_row_context(row_id=row_id, original_run_dir=original_run_dir, rebalanced_run_dir=rebalanced_run_dir)
        state = context["current_row"]["state"]
        legal_moves = list(context["current_row"]["legal_moves"])
        current = build_artifact_row(
            artifact_path=current_artifact_path,
            row_id=row_id,
            bucket=context["bucket"],
            phase=context["phase"],
            reference_move=context["reference_move"],
            system_value=context["current_row"]["system_value"],
            teacher_value=context["teacher_value"],
            value_error=context["current_row"]["value_error"],
            probe_summary=probe_artifact_position(
                artifact_path=current_artifact_path,
                state=state,
                simulations=artifact_simulations,
                seed=seed + index,
                c_puct=c_puct,
                evaluator=evaluators[current_artifact_path],
            ),
            legal_moves=legal_moves,
        )
        original_challenger = build_artifact_row(
            artifact_path=original_selected_artifact,
            row_id=row_id,
            bucket=context["bucket"],
            phase=context["phase"],
            reference_move=context["reference_move"],
            system_value=context["original_row"]["system_value"],
            teacher_value=context["teacher_value"],
            value_error=context["original_row"]["value_error"],
            probe_summary=probe_artifact_position(
                artifact_path=original_selected_artifact,
                state=state,
                simulations=artifact_simulations,
                seed=seed + 1000 + index,
                c_puct=c_puct,
                evaluator=evaluators[original_selected_artifact],
            ),
            legal_moves=legal_moves,
        )
        rebalanced_challenger = build_artifact_row(
            artifact_path=rebalanced_selected_artifact,
            row_id=row_id,
            bucket=context["bucket"],
            phase=context["phase"],
            reference_move=context["reference_move"],
            system_value=context["rebalanced_row"]["system_value"],
            teacher_value=context["teacher_value"],
            value_error=context["rebalanced_row"]["value_error"],
            probe_summary=probe_artifact_position(
                artifact_path=rebalanced_selected_artifact,
                state=state,
                simulations=artifact_simulations,
                seed=seed + 2000 + index,
                c_puct=c_puct,
                evaluator=evaluators[rebalanced_selected_artifact],
            ),
            legal_moves=legal_moves,
        )
        row_payload = {
            "bucket": context["bucket"],
            "phase": context["phase"],
            "reference_move": context["reference_move"],
            "teacher_value": round(float(context["teacher_value"]), 4),
            "notes": [],
            "current": current,
            "original_challenger": original_challenger,
            "rebalanced_challenger": rebalanced_challenger,
        }
        row_payload["decision"] = decide_row(row_payload)
        row_payload["notes"] = {
            "search_overrides_prior": ["prior corrected but searched move still wrong"],
            "q_value_backup_issue": ["backup values favor the wrong child"],
            "bad_priors": ["reference move remains underweighted in raw policy"],
            "insufficient_child_stats": ["child stats unavailable for one or more compared artifacts"],
            "mixed": ["multiple mechanisms remain plausible"],
        }[row_payload["decision"]]
        rows[row_id] = row_payload

    return {
        "schema": SEARCH_INTERACTION_SCHEMA,
        "original_run_dir": str(original_run_dir),
        "rebalanced_run_dir": str(rebalanced_run_dir),
        "current_artifact_path": current_artifact_path,
        "row_source": {
            "kind": "explicit_rows" if explicit_rows else "matrix_summary_priority_rows",
            "matrix_summary_path": None,
            "explicit_rows": explicit_rows,
            "resolved_rows": resolved_rows,
        },
        "rows": rows,
        "summary": build_summary(rows),
    }


def build_search_value_interaction_payload(
    *,
    original_run_dir: Path,
    rebalanced_run_dir: Path,
    current_artifact_path: str,
    explicit_rows: list[str] | None,
    artifact_simulations: int = 384,
    c_puct: float = 1.25,
    seed: int = 42,
) -> dict:
    payload = build_search_interaction_payload(
        original_run_dir=original_run_dir,
        rebalanced_run_dir=rebalanced_run_dir,
        current_artifact_path=current_artifact_path,
        explicit_rows=explicit_rows,
        artifact_simulations=artifact_simulations,
        c_puct=c_puct,
        seed=seed,
    )
    return build_search_value_interaction_payload_from_source_payload(
        payload=payload,
        source_diagnostic_path=str(diagnostic_out_path(rebalanced_run_dir=rebalanced_run_dir)),
    )


def _write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--rebalanced-run", required=True)
    parser.add_argument("--current-artifact", required=True)
    parser.add_argument("--row", action="append", dest="rows")
    parser.add_argument("--artifact-simulations", type=int, default=384)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rebalanced_run_dir = Path(args.rebalanced_run)
    payload = build_search_interaction_payload(
        original_run_dir=Path(args.original_run),
        rebalanced_run_dir=rebalanced_run_dir,
        current_artifact_path=args.current_artifact,
        explicit_rows=args.rows,
        artifact_simulations=args.artifact_simulations,
        c_puct=args.c_puct,
        seed=args.seed,
    )
    out_path = diagnostic_out_path(rebalanced_run_dir=rebalanced_run_dir)
    _write_payload(out_path, payload)

    sibling_payload = build_search_value_interaction_payload_from_source_payload(
        payload=payload,
        source_diagnostic_path=str(out_path),
    )
    sibling_out_path = search_value_interaction_diagnostic_out_path(rebalanced_run_dir=rebalanced_run_dir)
    _write_payload(sibling_out_path, sibling_payload)
    print(
        json.dumps(
            {
                "artifact_path": str(out_path),
                "schema": payload["schema"],
                "rows": payload["row_source"]["resolved_rows"],
                "search_value_interaction_artifact_path": str(sibling_out_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
