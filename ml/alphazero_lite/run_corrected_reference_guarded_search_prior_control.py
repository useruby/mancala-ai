#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import evaluate_artifact_position
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
    row_map_from_reference,
    write_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options


DEFAULT_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_LEGACY_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_OPENING_ARTIFACT = Path(
    "/tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact.jsonl"
)
DEFAULT_OPENING_SUBFAMILY_DIAGNOSTIC = Path(
    "/tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/tmp/azlite_corrected_reference_guarded_search_prior_control"
)
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-corrected-reference-guarded-search-prior-control-results.md"
)
DEFAULT_BUDGETS = (64, 128, 384, 1200)
DEFAULT_SEED = 17
DEFAULT_C_PUCT = 1.25
REQUIRED_ROW_IDS = ("capture_available-002", "capture_available-003")
CONTROL_ROW_IDS = (
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
OPENING_SUBFAMILIES = (
    "opening_extra_turn_overbias",
    "opening_edge_move_5_preference",
    "opening_missed_extra_turn_continuation",
)
OPENING_SAMPLE_SIZE = 3
SCHEMA = "azlite_corrected_reference_guarded_search_prior_control_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default="corrected-reference-guarded-search-prior-control"
    )
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--legacy-reference-artifact",
        type=Path,
        default=DEFAULT_LEGACY_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument(
        "--opening-artifact", type=Path, default=DEFAULT_OPENING_ARTIFACT
    )
    parser.add_argument(
        "--opening-subfamily-diagnostic",
        type=Path,
        default=DEFAULT_OPENING_SUBFAMILY_DIAGNOSTIC,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--opening-sample-size", type=int, default=OPENING_SAMPLE_SIZE)
    return parser.parse_args(argv)


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_csv_ints(raw_value: str) -> list[int]:
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise SystemExit("--budgets must not be empty")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def normalized_row_from_reference(
    row: dict[str, Any], *, source: str
) -> dict[str, Any]:
    child_stats = list(row.get("child_stats") or [])
    legal_moves = [int(child["move"]) for child in child_stats if "move" in child]
    if not legal_moves:
        legal_moves = [int(move) for move in list(row.get("legal_moves") or [])]
    return {
        "id": str(row["id"]),
        "canonical_state": str(row.get("canonical_state", "")),
        "state": dict(row.get("state") or {}),
        "legal_moves": legal_moves,
        "reference_move": int(row["reference_move"]),
        "child_stats": child_stats,
        "reference_source": source,
    }


def normalized_row_from_opening_artifact(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("raw_state") if isinstance(row.get("raw_state"), dict) else None
    if state is None:
        raise ValueError("opening artifact row is missing raw_state")
    child_stats = [
        {
            "move": int(child["move"]),
            "visits": int(child.get("visits", 0)),
            "q_value": float((2.0 * float(child.get("win_rate", 0.0))) - 1.0),
        }
        for child in list(row.get("teacher_child_stats") or [])
    ]
    legal_moves = [int(move) for move in list(row.get("legal_moves") or [])]
    if not legal_moves:
        legal_moves = [int(child["move"]) for child in child_stats]
    row_id = None
    for source_run in list(row.get("source_runs") or []):
        if source_run.get("kind") == "opening_plies_family":
            row_id = source_run.get("id")
            break
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("opening artifact row is missing source row id")
    return {
        "id": row_id,
        "canonical_state": str(row.get("canonical_state", "")),
        "state": dict(state),
        "legal_moves": legal_moves,
        "reference_move": int(row["reference_move"]),
        "child_stats": child_stats,
        "reference_source": "opening_guarded_artifact",
    }


def build_row_catalog(
    *,
    reference_artifact: Path,
    legacy_reference_artifact: Path,
    opening_artifact: Path,
) -> dict[str, dict[str, Any]]:
    tracked_rows = row_map_from_reference(load_json(reference_artifact))
    legacy_rows = row_map_from_reference(load_json(legacy_reference_artifact))
    opening_rows = {}
    if opening_artifact.exists():
        for row in read_jsonl(opening_artifact):
            try:
                normalized = normalized_row_from_opening_artifact(row)
            except (KeyError, TypeError, ValueError):
                continue
            opening_rows[normalized["id"]] = normalized

    catalog = {}
    for row_id, row in tracked_rows.items():
        catalog[row_id] = normalized_row_from_reference(
            row, source="tracked_corrected_reference"
        )
    for row_id, row in legacy_rows.items():
        catalog.setdefault(
            row_id,
            normalized_row_from_reference(row, source="legacy_reference_fallback"),
        )
    for row_id, row in opening_rows.items():
        catalog.setdefault(row_id, row)
    return catalog


def available_artifacts(root: Path, current_path: Path) -> list[dict[str, Any]]:
    del root
    specs = [
        {
            "label": "current",
            "path": current_path,
            "family": "current",
        },
        {
            "label": "corrected_replay_w1",
            "path": Path(
                "/tmp/azlite_corrected_reference_targeted_hard_state_replay/"
                "corrected-reference-targeted-hard-state-replay/variants/w1/versions/"
                "aggressive-v3-targeted-hard-state-replay-hard-state-replay-w1-iter1"
            ),
            "family": "corrected_replay",
        },
        {
            "label": "corrected_replay_w2",
            "path": Path(
                "/tmp/azlite_corrected_reference_targeted_hard_state_replay/"
                "corrected-reference-targeted-hard-state-replay/variants/w2/versions/"
                "aggressive-v3-targeted-hard-state-replay-hard-state-replay-w2-iter1"
            ),
            "family": "corrected_replay",
        },
        {
            "label": "corrected_replay_w4",
            "path": Path(
                "/tmp/azlite_corrected_reference_targeted_hard_state_replay/"
                "corrected-reference-targeted-hard-state-replay/variants/w4/versions/"
                "aggressive-v3-targeted-hard-state-replay-hard-state-replay-w4-iter1"
            ),
            "family": "corrected_replay",
        },
        {
            "label": "opening_extra_turn_overbias_corrected_w1",
            "path": Path(
                "/tmp/azlite_rule_conditioned_opening_full_guarded/"
                "opening-extra-turn-overbias-corrected/w1/versions/"
                "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w1-iter1"
            ),
            "family": "opening_replay",
        },
        {
            "label": "opening_extra_turn_overbias_corrected_w2",
            "path": Path(
                "/tmp/azlite_rule_conditioned_opening_full_guarded/"
                "opening-extra-turn-overbias-corrected/w2/versions/"
                "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
            ),
            "family": "opening_replay",
        },
    ]
    return [spec for spec in specs if spec["path"].exists()]


def build_search_setting_matrix() -> list[dict[str, Any]]:
    baseline = dict(build_eval_search_options())
    return [
        {
            "label": "baseline_eval_search",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": dict(baseline),
        },
        {
            "label": "no_subtree_reuse",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": {**baseline, "reuse_subtree": False},
        },
        {
            "label": "parent_q_fpu",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": {**baseline, "fpu_mode": "parent_q"},
        },
        {
            "label": "normalized_values",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": {**baseline, "normalize_values": True},
        },
        {
            "label": "deterministic_root_policy",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": {**baseline, "root_policy_mode": "deterministic"},
        },
        {
            "label": "low_cpuct",
            "c_puct": 0.75,
            "search_options": dict(baseline),
        },
        {
            "label": "high_cpuct",
            "c_puct": 1.75,
            "search_options": dict(baseline),
        },
        {
            "label": "no_tactical_root_bias",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": {**baseline, "tactical_root_bias": 0.0},
        },
        {
            "label": "full_search_control",
            "c_puct": DEFAULT_C_PUCT,
            "search_options": {
                **baseline,
                "fpu_mode": "parent_q",
                "reuse_subtree": True,
                "normalize_values": True,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.1,
            },
        },
    ]


def metric_rank(distribution: dict[int, float], move: int) -> int | None:
    if move not in distribution:
        return None
    ranked = sorted(distribution, key=lambda key: (-distribution[key], key))
    for index, candidate in enumerate(ranked, start=1):
        if candidate == move:
            return index
    return None


def visit_share(visits: list[float], move: int) -> float | None:
    if move >= len(visits):
        return None
    total = float(sum(visits))
    if total <= 0.0:
        return None
    return round(float(visits[move]) / total, 4)


def child_stat_map(child_stats: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(child["move"]): child for child in child_stats}


def selection_map(
    selection_breakdown: dict[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    if not isinstance(selection_breakdown, dict):
        return {}
    return {
        int(entry["move"]): entry
        for entry in list(selection_breakdown.get("moves") or [])
        if isinstance(entry, dict) and "move" in entry
    }


def policy_distribution(
    probe_summary: dict[str, Any], legal_moves: list[int]
) -> dict[int, float]:
    raw_policy = list(probe_summary.get("policy") or [])
    distribution = {}
    for move in legal_moves:
        if move < len(raw_policy):
            distribution[move] = round(float(raw_policy[move]), 4)
    return distribution


def score_notes(
    *,
    selected_move: int | None,
    reference_move: int,
    reference_policy: float | None,
    selected_policy: float | None,
    reference_q: float | None,
    selected_q: float | None,
    reference_u: float | None,
    selected_u: float | None,
) -> str:
    if selected_move is None:
        return "no_selected_move"
    if selected_move == reference_move:
        return "pass_reference_selected"
    if reference_q is not None and selected_q is not None and selected_q > reference_q:
        if (
            reference_u is not None
            and selected_u is not None
            and selected_u > reference_u
        ):
            return "selected_move_beats_reference_on_q_and_u"
        return "q_favors_selected"
    if reference_u is not None and selected_u is not None and selected_u > reference_u:
        return "u_favors_selected"
    if (
        reference_policy is not None
        and selected_policy is not None
        and selected_policy > reference_policy
    ):
        return "policy_prior_favors_selected"
    return "selected_move_not_reference"


def probe_row(
    *,
    artifact: dict[str, Any],
    row: dict[str, Any],
    budget: int,
    search_setting: dict[str, Any],
    seed: int,
    root_prior_override=None,
) -> dict[str, Any]:
    probe_summary = evaluate_artifact_position(
        artifact_path=artifact["path"],
        state=dict(row["state"]),
        simulations=budget,
        seed=seed,
        c_puct=float(search_setting["c_puct"]),
        search_options=dict(search_setting["search_options"]),
        ablation_mode="full",
        root_prior_override=root_prior_override,
    )
    reference_move = int(row["reference_move"])
    selected_move = probe_summary.get("selected_move")
    legal_moves = list(row["legal_moves"])
    policy = policy_distribution(probe_summary, legal_moves)
    visits = [float(value) for value in list(probe_summary.get("visits") or [])]
    selection_breakdown = probe_summary.get("selection_breakdown") or {}
    selection_entries = selection_map(selection_breakdown)
    child_stats = child_stat_map(list(probe_summary.get("child_stats") or []))

    reference_policy = policy.get(reference_move)
    selected_policy = None if selected_move is None else policy.get(int(selected_move))
    reference_visit_share = visit_share(visits, reference_move)
    selected_visit_share = (
        None if selected_move is None else visit_share(visits, int(selected_move))
    )
    reference_q = None
    if reference_move in child_stats:
        reference_q = round(float(child_stats[reference_move].get("q_value", 0.0)), 4)
    selected_q = None
    if selected_move is not None and int(selected_move) in child_stats:
        selected_q = round(
            float(child_stats[int(selected_move)].get("q_value", 0.0)), 4
        )
    reference_entry = selection_entries.get(reference_move) or {}
    selected_entry = (
        selection_entries.get(int(selected_move)) or {}
        if selected_move is not None
        else {}
    )
    reference_u = reference_entry.get("u_component")
    selected_u = selected_entry.get("u_component")
    reference_score = reference_entry.get("selection_score")
    selected_score = selected_entry.get("selection_score")
    selected_minus_reference_q_margin = None
    if reference_q is not None and selected_q is not None:
        selected_minus_reference_q_margin = round(selected_q - reference_q, 4)

    result = {
        "artifact": artifact["label"],
        "artifact_path": str(artifact["path"]),
        "artifact_family": artifact["family"],
        "row_id": row["id"],
        "budget": int(budget),
        "search_setting": search_setting["label"],
        "search_options": dict(search_setting["search_options"]),
        "c_puct": float(search_setting["c_puct"]),
        "corrected_reference_move": reference_move,
        "selected_move": None if selected_move is None else int(selected_move),
        "selected_is_corrected_reference": selected_move == reference_move,
        "reference_policy_probability": reference_policy,
        "reference_policy_rank": metric_rank(policy, reference_move),
        "reference_visit_share": reference_visit_share,
        "selected_visit_share": selected_visit_share,
        "q_reference": reference_q,
        "q_selected": selected_q,
        "selected_minus_reference_q_margin": selected_minus_reference_q_margin,
        "u_reference": None if reference_u is None else round(float(reference_u), 4),
        "u_selected": None if selected_u is None else round(float(selected_u), 4),
        "selection_score_reference": None
        if reference_score is None
        else round(float(reference_score), 4),
        "selection_score_selected": None
        if selected_score is None
        else round(float(selected_score), 4),
        "pass": selected_move == reference_move,
        "pass_fail_reason": score_notes(
            selected_move=None if selected_move is None else int(selected_move),
            reference_move=reference_move,
            reference_policy=reference_policy,
            selected_policy=selected_policy,
            reference_q=reference_q,
            selected_q=selected_q,
            reference_u=None if reference_u is None else float(reference_u),
            selected_u=None if selected_u is None else float(selected_u),
        ),
        "reference_source": row.get("reference_source"),
        "selection_breakdown": selection_breakdown,
        "visit_snapshots": list(probe_summary.get("visit_snapshots") or []),
        "root_prior_telemetry": probe_summary.get("root_prior_telemetry") or {},
    }
    return result


def probe_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    return {
        (row["artifact"], row["row_id"], int(row["budget"]), row["search_setting"]): row
        for row in rows
    }


def classify_current_vs_candidate(
    current_row: dict[str, Any], candidate_row: dict[str, Any]
) -> str:
    current_pass = bool(current_row["pass"])
    candidate_pass = bool(candidate_row["pass"])
    if current_pass and candidate_pass:
        return "current_pass_candidate_pass"
    if current_pass and not candidate_pass:
        return "current_pass_candidate_fail"
    if not current_pass and candidate_pass:
        return "current_fail_candidate_pass"
    return "current_fail_candidate_fail"


def snapshot_entry(snapshot: dict[str, Any], move: int) -> dict[str, Any] | None:
    for entry in list(snapshot.get("moves") or []):
        if int(entry.get("move", -1)) == move:
            return entry
    return None


def snapshot_selection_score(snapshot: dict[str, Any], move: int) -> float | None:
    entry = snapshot_entry(snapshot, move)
    if entry is None or entry.get("selection_score") is None:
        return None
    return float(entry["selection_score"])


def first_snapshot_where(
    snapshots: list[dict[str, Any]], predicate
) -> dict[str, Any] | None:
    for snapshot in snapshots:
        if predicate(snapshot):
            return snapshot
    return None


def trace_diagnosis(
    *,
    final_result: dict[str, Any],
    low_cpuct_result: dict[str, Any] | None,
    no_bias_result: dict[str, Any] | None,
) -> str:
    if low_cpuct_result is not None and low_cpuct_result["pass"]:
        return "failure_disappears_under_low_cpuct"
    if no_bias_result is not None and no_bias_result["pass"]:
        return "failure_disappears_under_no_tactical_root_bias"
    q_reference = final_result.get("q_reference")
    q_selected = final_result.get("q_selected")
    u_reference = final_result.get("u_reference")
    u_selected = final_result.get("u_selected")
    if q_reference is not None and q_selected is not None and q_reference >= q_selected:
        if (
            u_reference is not None
            and u_selected is not None
            and u_selected > u_reference
        ):
            return "u_or_prior_pressure_explains_selection"
        return "q_supports_reference_but_selection_still_loses"
    if q_reference is not None and q_selected is not None and q_selected > q_reference:
        return "q_supports_selected_move"
    return "trace_inconclusive"


def build_trace_row(
    *,
    failing_result: dict[str, Any],
    low_cpuct_result: dict[str, Any] | None,
    no_bias_result: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_move = int(failing_result["selected_move"])
    reference_move = int(failing_result["corrected_reference_move"])
    snapshots = list(failing_result.get("visit_snapshots") or [])
    first_visit_overtake = first_snapshot_where(
        snapshots,
        lambda snapshot: (
            selected_move < len(snapshot.get("visits") or [])
            and reference_move < len(snapshot.get("visits") or [])
            and float(snapshot["visits"][selected_move])
            > float(snapshot["visits"][reference_move])
        ),
    )
    first_score_overtake = first_snapshot_where(
        snapshots,
        lambda snapshot: (
            snapshot_selection_score(snapshot, selected_move) is not None
            and snapshot_selection_score(snapshot, reference_move) is not None
            and float(snapshot_selection_score(snapshot, selected_move) or 0.0)
            > float(snapshot_selection_score(snapshot, reference_move) or 0.0)
        ),
    )
    final_breakdown = selection_map(failing_result.get("selection_breakdown") or {})
    reference_entry = final_breakdown.get(reference_move) or {}
    selected_entry = final_breakdown.get(selected_move) or {}
    diagnosis = trace_diagnosis(
        final_result=failing_result,
        low_cpuct_result=low_cpuct_result,
        no_bias_result=no_bias_result,
    )
    return {
        "artifact": failing_result["artifact"],
        "row_id": failing_result["row_id"],
        "search_setting": failing_result["search_setting"],
        "budget": int(failing_result["budget"]),
        "corrected_reference_move": reference_move,
        "selected_move": selected_move,
        "first_simulation_selected_overtakes_reference_by_visits": None
        if first_visit_overtake is None
        else int(first_visit_overtake["simulation"]),
        "first_simulation_selected_overtakes_reference_by_score": None
        if first_score_overtake is None
        else int(first_score_overtake["simulation"]),
        "p_reference": None
        if reference_entry.get("prior") is None
        else round(float(reference_entry["prior"]), 4),
        "p_selected": None
        if selected_entry.get("prior") is None
        else round(float(selected_entry["prior"]), 4),
        "q_reference": failing_result.get("q_reference"),
        "q_selected": failing_result.get("q_selected"),
        "u_reference": failing_result.get("u_reference"),
        "u_selected": failing_result.get("u_selected"),
        "n_reference": None
        if reference_entry.get("visit_count") is None
        else int(reference_entry["visit_count"]),
        "n_selected": None
        if selected_entry.get("visit_count") is None
        else int(selected_entry["visit_count"]),
        "score_margin_selected_minus_reference": None
        if selected_entry.get("selection_score") is None
        or reference_entry.get("selection_score") is None
        else round(
            float(selected_entry["selection_score"])
            - float(reference_entry["selection_score"]),
            4,
        ),
        "q_support": (
            "reference"
            if (
                failing_result.get("q_reference") is not None
                and failing_result.get("q_selected") is not None
                and float(failing_result["q_reference"])
                > float(failing_result["q_selected"])
            )
            else "selected"
        ),
        "u_pressure_explains_selection": bool(
            failing_result.get("q_reference") is not None
            and failing_result.get("q_selected") is not None
            and float(failing_result["q_reference"])
            >= float(failing_result["q_selected"])
            and failing_result.get("u_reference") is not None
            and failing_result.get("u_selected") is not None
            and float(failing_result["u_selected"])
            > float(failing_result["u_reference"])
        ),
        "failure_disappears_under_low_cpuct": bool(
            low_cpuct_result is not None and low_cpuct_result["pass"]
        ),
        "failure_disappears_under_no_tactical_root_bias": bool(
            no_bias_result is not None and no_bias_result["pass"]
        ),
        "diagnosis": diagnosis,
    }


def equalize_reference_and_selected_override(reference_move: int, selected_move: int):
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(priors, dtype=np.float32).copy()
        target = max(float(adjusted[reference_move]), float(adjusted[selected_move]))
        adjusted[reference_move] = target
        adjusted[selected_move] = target
        normalized = np.zeros_like(adjusted)
        total = float(np.sum(adjusted[legal_moves]))
        if total <= 0.0:
            normalized[legal_moves] = 1.0 / len(legal_moves)
        else:
            normalized[legal_moves] = adjusted[legal_moves] / total
        return normalized.astype(np.float32)

    return override


def clamp_selected_prior_to_reference_override(reference_move: int, selected_move: int):
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        adjusted = np.asarray(priors, dtype=np.float32).copy()
        adjusted[selected_move] = min(
            float(adjusted[selected_move]), float(adjusted[reference_move])
        )
        normalized = np.zeros_like(adjusted)
        total = float(np.sum(adjusted[legal_moves]))
        if total <= 0.0:
            normalized[legal_moves] = 1.0 / len(legal_moves)
        else:
            normalized[legal_moves] = adjusted[legal_moves] / total
        return normalized.astype(np.float32)

    return override


def uniform_legal_prior_override():
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        adjusted = np.zeros_like(np.asarray(priors, dtype=np.float32))
        adjusted[legal_moves] = 1.0 / len(legal_moves)
        return adjusted.astype(np.float32)

    return override


def intervention_results_for_failing_row(
    *,
    artifact: dict[str, Any],
    row: dict[str, Any],
    failing_result: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    selected_move = int(failing_result["selected_move"])
    reference_move = int(failing_result["corrected_reference_move"])
    baseline_options = dict(build_eval_search_options())
    interventions = [
        {
            "label": "equalize_priors",
            "c_puct": float(failing_result["c_puct"]),
            "search_options": dict(failing_result["search_options"]),
            "root_prior_override": equalize_reference_and_selected_override(
                reference_move, selected_move
            ),
        },
        {
            "label": "zero_tactical_root_bias",
            "c_puct": float(failing_result["c_puct"]),
            "search_options": {
                **failing_result["search_options"],
                "tactical_root_bias": 0.0,
            },
            "root_prior_override": None,
        },
        {
            "label": "clamp_selected_prior_to_reference",
            "c_puct": float(failing_result["c_puct"]),
            "search_options": dict(failing_result["search_options"]),
            "root_prior_override": clamp_selected_prior_to_reference_override(
                reference_move, selected_move
            ),
        },
        {
            "label": "equal_initialize_q",
            "c_puct": float(failing_result["c_puct"]),
            "search_options": {**failing_result["search_options"], "fpu_mode": "zero"},
            "root_prior_override": None,
        },
        {
            "label": "uniform_legal_prior",
            "c_puct": float(failing_result["c_puct"]),
            "search_options": dict(baseline_options),
            "root_prior_override": uniform_legal_prior_override(),
        },
    ]
    results = []
    for intervention in interventions:
        result = probe_row(
            artifact=artifact,
            row=row,
            budget=int(failing_result["budget"]),
            search_setting={
                "label": intervention["label"],
                "c_puct": intervention["c_puct"],
                "search_options": intervention["search_options"],
            },
            seed=seed,
            root_prior_override=intervention["root_prior_override"],
        )
        result["intervention"] = intervention["label"]
        results.append(result)
    return results


def opening_samples(
    *,
    subfamily_diagnostic: Path,
    sample_size: int,
) -> dict[str, list[str]]:
    payload = load_json(subfamily_diagnostic)
    rows = list(payload.get("rows") or [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("subfamily"), str):
            grouped[str(row["subfamily"])].append(row)
    selections = {}
    for subfamily in OPENING_SUBFAMILIES:
        chosen = sorted(
            grouped.get(subfamily, []),
            key=lambda row: (
                -float(row.get("regret", 0.0)),
                str(row.get("row_id", "")),
            ),
        )[:sample_size]
        selections[subfamily] = [str(row["row_id"]) for row in chosen]
    return selections


def summarize_opening_compatibility(
    *,
    probe_results: list[dict[str, Any]],
    sample_rows: dict[str, list[str]],
    best_setting: str,
) -> list[dict[str, Any]]:
    result_index = probe_index(probe_results)
    summaries = []
    for subfamily, row_ids in sample_rows.items():
        baseline_rows = []
        full_rows = []
        best_rows = []
        for row_id in row_ids:
            baseline_rows.append(
                result_index.get(("current", row_id, 384, "baseline_eval_search"))
            )
            full_rows.append(
                result_index.get(("current", row_id, 384, "full_search_control"))
            )
            best_rows.append(result_index.get(("current", row_id, 384, best_setting)))

        def pass_rate(rows: list[dict[str, Any] | None]) -> float | None:
            present = [row for row in rows if row is not None]
            if not present:
                return None
            return round(sum(1 for row in present if row["pass"]) / len(present), 4)

        regressions = []
        for baseline_row, best_row in zip(baseline_rows, best_rows, strict=False):
            if baseline_row is None or best_row is None:
                continue
            if baseline_row["pass"] and not best_row["pass"]:
                regressions.append(best_row["row_id"])
        summaries.append(
            {
                "subfamily": subfamily,
                "rows_sampled": len(row_ids),
                "baseline_pass_rate": pass_rate(baseline_rows),
                "full_search_control_pass_rate": pass_rate(full_rows),
                "best_guard_setting": best_setting,
                "best_guard_setting_pass_rate": pass_rate(best_rows),
                "regressions": regressions,
                "notes": "helps"
                if (pass_rate(best_rows) or 0.0) > (pass_rate(baseline_rows) or 0.0)
                else "hurts"
                if (pass_rate(best_rows) or 0.0) < (pass_rate(baseline_rows) or 0.0)
                else "neutral",
            }
        )
    return summaries


def best_002_003_setting(probe_results: list[dict[str, Any]]) -> str:
    score_by_setting: dict[str, int] = defaultdict(int)
    for row in probe_results:
        if row["artifact"] == "current":
            continue
        if row["row_id"] not in REQUIRED_ROW_IDS:
            continue
        if int(row["budget"]) not in {384, 1200}:
            continue
        if row["pass"]:
            score_by_setting[row["search_setting"]] += 1
    if not score_by_setting:
        return "full_search_control"
    return max(sorted(score_by_setting), key=lambda key: score_by_setting[key])


def overall_classification(
    *,
    current_vs_candidate_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    opening_compatibility_rows: list[dict[str, Any]],
    best_setting: str,
) -> tuple[str, str, str]:
    replay_regressions = [
        row
        for row in current_vs_candidate_rows
        if row["row_id"] in REQUIRED_ROW_IDS
        and row["candidate"] != "current"
        and row["classification"] == "current_pass_candidate_fail"
    ]
    if replay_regressions:
        return (
            "replay_induced_guard_regression",
            "abandon corrected-reference replay branches and rebuild candidate datasets without rows that perturb corrected guards.",
            "Current already satisfies the corrected 002/003 gate at baseline while replay candidates regress it.",
        )

    opening_regressions = any(row["regressions"] for row in opening_compatibility_rows)
    if best_setting and opening_regressions:
        return (
            "guard_fix_opening_regression",
            "do not deploy; split gate/search settings by family only as diagnostics.",
            "A setting helps 002/003 but regresses sampled opening subfamilies.",
        )

    if any(trace["failure_disappears_under_low_cpuct"] for trace in trace_rows) or any(
        row["intervention"] == "equalize_priors" and row["pass"]
        for row in intervention_rows
    ):
        return (
            "prior_pressure_causal",
            "run a cpuct/root-prior calibration experiment, not replay.",
            "Low-cpuct or prior equalization clears the failing corrected guard row.",
        )

    if any(
        trace["failure_disappears_under_no_tactical_root_bias"] for trace in trace_rows
    ):
        return (
            "tactical_bias_collision",
            "run a tactical-root-bias ablation across the corrected forensic suite.",
            "Zeroing tactical root bias clears a failing corrected guard row.",
        )

    value_dominant = [
        trace
        for trace in trace_rows
        if trace["q_support"] == "selected"
        and not trace["failure_disappears_under_low_cpuct"]
    ]
    if value_dominant:
        return (
            "value_or_backup_dominant",
            "run a child-afterstate value/backup audit under corrected references.",
            "Failing rows keep favoring the non-reference move in child Q values after search-control probes.",
        )

    return (
        "search_control_candidate",
        f"run a small equal-budget PUCT arena with `{best_setting}` only, no training.",
        "Current and candidates need search control and one setting looks locally clean.",
    )


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def render_markdown(
    *,
    summary: dict[str, Any],
    guard_table_rows: list[dict[str, Any]],
    current_vs_candidate_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    opening_compatibility_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# AlphaZero-lite Corrected-Reference Guarded Search-Prior Control Results",
        "",
        "## 1. Context",
        "",
        "- PR #32 established that corrected-reference replay cleaned up labels but did not improve arena strength.",
        "- The active blocker remained the corrected `capture_available-002/003` gate, especially on replay/opening replay branches.",
        f"- Default corrected reference artifact: `{summary['reference_artifact']}`.",
        "",
        "## 2. Why replay is paused",
        "",
        f"- Overall classification: `{summary['classification']}`.",
        f"- Classification rationale: {summary['classification_notes']}",
        "- No training, broad arena, promotion, or new replay dataset construction was run in this control branch.",
        "",
        "## 3. Corrected-reference baseline behavior",
        "",
        "| artifact | row_id | budget | search_setting | corrected_reference_move | selected_move | selected_is_corrected_reference | reference_policy_probability | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | pass | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in guard_table_rows:
        lines.append(
            "| {artifact} | {row_id} | {budget} | {search_setting} | {corrected_reference_move} | {selected_move} | {selected_is_corrected_reference} | {reference_policy_probability} | {reference_visit_share} | {selected_visit_share} | {selected_minus_reference_q_margin} | {pass_value} | {notes} |".format(
                artifact=row["artifact"],
                row_id=row["row_id"],
                budget=row["budget"],
                search_setting=row["search_setting"],
                corrected_reference_move=row["corrected_reference_move"],
                selected_move=row["selected_move"],
                selected_is_corrected_reference=str(
                    bool(row["selected_is_corrected_reference"])
                ).lower(),
                reference_policy_probability=format_float(
                    row["reference_policy_probability"]
                ),
                reference_visit_share=format_float(row["reference_visit_share"]),
                selected_visit_share=format_float(row["selected_visit_share"]),
                selected_minus_reference_q_margin=format_float(
                    row["selected_minus_reference_q_margin"]
                ),
                pass_value=str(bool(row["pass"])).lower(),
                notes=row["pass_fail_reason"],
            )
        )
    lines.extend(
        [
            "",
            "## 4. Search-setting matrix",
            "",
            "- `baseline_eval_search`: current `build_eval_search_options()` defaults.",
            "- `no_subtree_reuse`: explicit `reuse_subtree=false`.",
            "- `parent_q_fpu`: `fpu_mode=parent_q`.",
            "- `normalized_values`: `normalize_values=true`.",
            "- `deterministic_root_policy`: explicit deterministic root tie-break.",
            "- `low_cpuct`: `c_puct=0.75`.",
            "- `high_cpuct`: `c_puct=1.75`.",
            "- `no_tactical_root_bias`: `tactical_root_bias=0.0`.",
            "- `full_search_control`: `parent_q` FPU, subtree reuse, normalized values, deterministic root, tactical bias `0.1`.",
            "",
            "## 5. Current vs replay-candidate guard comparison",
            "",
            "| row_id | current_selected | current_pass | candidate | candidate_selected | candidate_pass | classification | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in current_vs_candidate_rows:
        lines.append(
            f"| {row['row_id']} | {row['current_selected']} | {str(bool(row['current_pass'])).lower()} | {row['candidate']} | {row['candidate_selected']} | {str(bool(row['candidate_pass'])).lower()} | `{row['classification']}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Selection-pressure trace",
            "",
            "| artifact | row_id | search_setting | budget | corrected_reference_move | selected_move | p_reference | p_selected | q_reference | q_selected | u_reference | u_selected | n_reference | n_selected | score_margin_selected_minus_reference | diagnosis |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in trace_rows:
        lines.append(
            "| {artifact} | {row_id} | {search_setting} | {budget} | {corrected_reference_move} | {selected_move} | {p_reference} | {p_selected} | {q_reference} | {q_selected} | {u_reference} | {u_selected} | {n_reference} | {n_selected} | {score_margin} | {diagnosis} |".format(
                artifact=row["artifact"],
                row_id=row["row_id"],
                search_setting=row["search_setting"],
                budget=row["budget"],
                corrected_reference_move=row["corrected_reference_move"],
                selected_move=row["selected_move"],
                p_reference=format_float(row["p_reference"]),
                p_selected=format_float(row["p_selected"]),
                q_reference=format_float(row["q_reference"]),
                q_selected=format_float(row["q_selected"]),
                u_reference=format_float(row["u_reference"]),
                u_selected=format_float(row["u_selected"]),
                n_reference=row["n_reference"]
                if row["n_reference"] is not None
                else "-",
                n_selected=row["n_selected"] if row["n_selected"] is not None else "-",
                score_margin=format_float(row["score_margin_selected_minus_reference"]),
                diagnosis=row["diagnosis"],
            )
        )
    lines.extend(
        [
            "",
            "## 7. Guard-specific diagnostic interventions",
            "",
            "- Diagnostic interventions were only run on rows that still failed corrected reference at `384` or `1200` simulations.",
            "- `equal_initialize_q` was approximated with `fpu_mode=zero`, which equalizes root FPU initialization across unvisited children without changing production code.",
        ]
    )
    for row in intervention_rows:
        lines.append(
            f"- `{row['artifact']}` `{row['row_id']}` budget `{row['budget']}` intervention `{row['intervention']}`: selected `{row['selected_move']}`, pass `{str(bool(row['pass'])).lower()}`, reason `{row['pass_fail_reason']}`."
        )
    lines.extend(
        [
            "",
            "## 8. Opening-subfamily compatibility check",
            "",
            "| subfamily | rows_sampled | baseline_pass_rate | best_guard_setting_pass_rate | regressions | notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in opening_compatibility_rows:
        lines.append(
            f"| {row['subfamily']} | {row['rows_sampled']} | {format_float(row['baseline_pass_rate'])} | {format_float(row['best_guard_setting_pass_rate'])} | `{json.dumps(row['regressions'])}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 9. Interpretation",
            "",
            f"- Best 002/003 search setting by replay-row pass count: `{summary['best_setting']}`.",
            f"- Corrected-guard decision: `{summary['classification']}`.",
            "- Recommended next action is driven by the explicit decision rules and the current-vs-candidate comparison.",
            "",
            "## 10. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    reference_artifact = resolve_path(root, args.reference_artifact)
    legacy_reference_artifact = resolve_path(root, args.legacy_reference_artifact)
    current_path = resolve_path(root, args.current_path)
    opening_artifact = resolve_path(root, args.opening_artifact)
    opening_subfamily_diagnostic = resolve_path(root, args.opening_subfamily_diagnostic)
    output_root = resolve_path(root, args.output_root) / args.run_id
    report_path = resolve_path(root, args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)

    row_catalog = build_row_catalog(
        reference_artifact=reference_artifact,
        legacy_reference_artifact=legacy_reference_artifact,
        opening_artifact=opening_artifact,
    )
    artifacts = available_artifacts(root, current_path)
    if not artifacts:
        raise SystemExit("no artifacts available to evaluate")

    search_settings = build_search_setting_matrix()
    budgets = parse_csv_ints(args.budgets)
    probe_rows = [
        row_catalog[row_id]
        for row_id in (*REQUIRED_ROW_IDS, *CONTROL_ROW_IDS)
        if row_id in row_catalog
    ]
    if len(probe_rows) < len(REQUIRED_ROW_IDS) + len(CONTROL_ROW_IDS):
        missing = sorted(
            row_id
            for row_id in (*REQUIRED_ROW_IDS, *CONTROL_ROW_IDS)
            if row_id not in row_catalog
        )
        raise SystemExit(f"missing required corrected rows: {missing}")

    probe_results = []
    for artifact in artifacts:
        for row in probe_rows:
            for budget in budgets:
                for search_setting in search_settings:
                    probe_results.append(
                        probe_row(
                            artifact=artifact,
                            row=row,
                            budget=budget,
                            search_setting=search_setting,
                            seed=int(args.seed),
                        )
                    )

    result_index = probe_index(probe_results)
    current_vs_candidate_rows = []
    for artifact in artifacts:
        if artifact["label"] == "current":
            continue
        for row_id in (*REQUIRED_ROW_IDS, *CONTROL_ROW_IDS):
            current_row = result_index[("current", row_id, 384, "baseline_eval_search")]
            candidate_row = result_index[
                (artifact["label"], row_id, 384, "baseline_eval_search")
            ]
            baseline_1200 = result_index.get(
                (artifact["label"], row_id, 1200, "baseline_eval_search")
            )
            classification = classify_current_vs_candidate(current_row, candidate_row)
            current_vs_candidate_rows.append(
                {
                    "row_id": row_id,
                    "current_selected": current_row["selected_move"],
                    "current_pass": current_row["pass"],
                    "candidate": artifact["label"],
                    "candidate_selected": candidate_row["selected_move"],
                    "candidate_pass": candidate_row["pass"],
                    "classification": classification,
                    "notes": "384 primary"
                    if baseline_1200 is None
                    else f"1200 selected={baseline_1200['selected_move']} pass={str(bool(baseline_1200['pass'])).lower()}",
                }
            )

    trace_rows = []
    intervention_rows = []
    for artifact in artifacts:
        if artifact["label"] == "current":
            continue
        for row in probe_rows:
            if row["id"] not in REQUIRED_ROW_IDS:
                continue
            for budget in (384, 1200):
                failing = result_index[
                    (artifact["label"], row["id"], budget, "baseline_eval_search")
                ]
                if failing["pass"] or failing["selected_move"] is None:
                    continue
                low_cpuct_row = result_index.get(
                    (artifact["label"], row["id"], budget, "low_cpuct")
                )
                no_bias_row = result_index.get(
                    (artifact["label"], row["id"], budget, "no_tactical_root_bias")
                )
                trace_rows.append(
                    build_trace_row(
                        failing_result=failing,
                        low_cpuct_result=low_cpuct_row,
                        no_bias_result=no_bias_row,
                    )
                )
                intervention_rows.extend(
                    intervention_results_for_failing_row(
                        artifact=artifact,
                        row=row,
                        failing_result=failing,
                        seed=int(args.seed),
                    )
                )

    best_setting_for_opening = best_002_003_setting(probe_results)
    opening_sample_rows = opening_samples(
        subfamily_diagnostic=opening_subfamily_diagnostic,
        sample_size=int(args.opening_sample_size),
    )
    for row_id in sorted(
        {row_id for row_ids in opening_sample_rows.values() for row_id in row_ids}
    ):
        row = row_catalog.get(row_id)
        if row is None:
            continue
        for budget in (384,):
            for setting_name in {
                "baseline_eval_search",
                "full_search_control",
                best_setting_for_opening,
            }:
                setting = next(
                    item for item in search_settings if item["label"] == setting_name
                )
                probe_results.append(
                    probe_row(
                        artifact=next(
                            item for item in artifacts if item["label"] == "current"
                        ),
                        row=row,
                        budget=budget,
                        search_setting=setting,
                        seed=int(args.seed),
                    )
                )

    best_setting = best_002_003_setting(probe_results)
    opening_compatibility_rows = summarize_opening_compatibility(
        probe_results=probe_results,
        sample_rows=opening_sample_rows,
        best_setting=best_setting,
    )
    classification, recommended_next_action, classification_notes = (
        overall_classification(
            current_vs_candidate_rows=current_vs_candidate_rows,
            trace_rows=trace_rows,
            intervention_rows=intervention_rows,
            opening_compatibility_rows=opening_compatibility_rows,
            best_setting=best_setting,
        )
    )

    guard_table_rows = [
        row
        for row in probe_results
        if row["row_id"] in (*REQUIRED_ROW_IDS, *CONTROL_ROW_IDS)
    ]
    summary = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "reference_artifact": str(reference_artifact),
        "legacy_reference_artifact": str(legacy_reference_artifact),
        "current_path": str(current_path),
        "artifacts": [
            {
                "label": artifact["label"],
                "path": str(artifact["path"]),
                "family": artifact["family"],
            }
            for artifact in artifacts
        ],
        "budgets": budgets,
        "search_settings": [item["label"] for item in search_settings],
        "best_setting": best_setting,
        "classification": classification,
        "classification_notes": classification_notes,
        "recommended_next_action": recommended_next_action,
        "opening_samples": opening_sample_rows,
        "guard_table_rows": guard_table_rows,
        "current_vs_candidate_rows": current_vs_candidate_rows,
        "trace_rows": trace_rows,
        "intervention_rows": intervention_rows,
        "opening_compatibility_rows": opening_compatibility_rows,
    }
    summary_path = (
        output_root / "corrected_reference_guarded_search_prior_control_summary.json"
    )
    write_json(summary_path, summary)
    report_path.write_text(
        render_markdown(
            summary=summary,
            guard_table_rows=guard_table_rows,
            current_vs_candidate_rows=current_vs_candidate_rows,
            trace_rows=trace_rows,
            intervention_rows=intervention_rows,
            opening_compatibility_rows=opening_compatibility_rows,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "classification": classification,
                "recommended_next_action": recommended_next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
