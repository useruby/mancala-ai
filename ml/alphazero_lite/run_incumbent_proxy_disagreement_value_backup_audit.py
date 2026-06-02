#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.self_play import (
    Node,
    PUCT,
    build_eval_search_options,
    terminal_value,
)


DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_SELECTED_ROWS_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining/selected_non_opening_family_rows.jsonl"
)
DEFAULT_MINING_SUMMARY_PATH = Path(
    "/tmp/azlite_corrected_non_opening_failure_mining/non_opening_failure_family_summary.json"
)
DEFAULT_SUMMARY_OUT = Path(
    "/tmp/azlite_incumbent_proxy_value_backup_audit/"
    "incumbent_proxy_value_backup_audit_summary.json"
)
DEFAULT_REPORT_OUT = Path(
    "docs/alphazero-lite-incumbent-proxy-disagreement-value-backup-audit-results.md"
)
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
ROOT_BUDGETS = (384, 1200)
CHILD_PUCT_BUDGETS = (384, 1200)
CHILD_TEACHER_BASE_BUDGETS = (1200, 2400)
OPTIONAL_CHILD_TEACHER_BUDGET = 5000
CHILD_TEACHER_SEEDS = (11, 23, 37, 42, 101)
COUNTERFACTUAL_BUDGETS = (384, 1200)
C_PUCT = 1.25
FAMILY = "incumbent_proxy_disagreement"
SCHEMA = "azlite_incumbent_proxy_disagreement_value_backup_audit_v1"
REQUIRED_TARGET_ROW_IDS = (
    "incumbent_proxy_disagreement-009",
    "incumbent_proxy_disagreement-035",
    "incumbent_proxy_disagreement-022",
    "incumbent_proxy_disagreement-024",
    "incumbent_proxy_disagreement-011",
    "incumbent_proxy_disagreement-014",
    "incumbent_proxy_disagreement-007",
    "incumbent_proxy_disagreement-025",
    "incumbent_proxy_disagreement-023",
    "incumbent_proxy_disagreement-021",
)
REQUIRED_CONTROL_ROW_IDS = (
    "incumbent_proxy_disagreement-026",
    "incumbent_proxy_disagreement-028",
)
SEVERITY_SCORE = {"high": 2, "medium": 1, "low": 0, "none": 0}
MAX_PROJECTED_5000_SECONDS = 1200.0


@dataclass(frozen=True)
class AuditRow:
    row_id: str
    role: str
    severity: str
    failure_mode: str
    corrected_reference_move: int
    current_selected_move: int
    suite_state: dict[str, Any]
    legal_moves: list[int]
    canonical_state_hash: str
    reference_teacher_value: float | None
    reference_unstable: bool
    reference_integrity_error: bool
    inventory_failure_status: str | None
    mining_metrics: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument(
        "--selected-rows-path", type=Path, default=DEFAULT_SELECTED_ROWS_PATH
    )
    parser.add_argument(
        "--mining-summary-path", type=Path, default=DEFAULT_MINING_SUMMARY_PATH
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpuct", type=float, default=C_PUCT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def round_float(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def state_to_root_perspective_value(
    *, raw_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(raw_value)
        if int(state["current_player"]) == int(root_player)
        else -float(raw_value)
    )


def root_to_state_perspective_value(
    *, root_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(root_value)
        if int(state["current_player"]) == int(root_player)
        else -float(root_value)
    )


def child_state_from_move(root_state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(int(move)))
    if not succeeded:
        raise ValueError(f"illegal move {move} for state")
    return game.to_state()


def selected_move_from_policy(
    policy: list[float], legal_moves: list[int]
) -> int | None:
    if not legal_moves:
        return None
    return max(legal_moves, key=lambda move: (float(policy[move]), -int(move)))


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(value) for value in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / float(total))


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -SEVERITY_SCORE.get(str(row.get("severity") or "none"), 0),
        -abs(float(row.get("selected_minus_reference_q_margin_384") or 0.0)),
        float(row.get("reference_visit_share_384") or 1.0),
        str(row["row_id"]),
    )


def selection_entry_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selection_breakdown = summary.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(selection_breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def rerun_family_mining_if_needed(root: Path, args: argparse.Namespace) -> None:
    if args.selected_rows_path.exists():
        return
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_corrected_non_opening_failure_family_mining",
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"missing selected rows file and rerun failed with exit code {completed.returncode}"
        )
    if not args.selected_rows_path.exists():
        raise SystemExit(
            f"selected rows file still missing after rerun: {args.selected_rows_path}"
        )


def load_reference_maps(
    reference_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = load_json(reference_path)
    rows = payload.get("rows") or []
    by_id: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        by_id[str(row["id"])] = row
        canonical = str(row.get("canonical_state") or canonical_state_key(row["state"]))
        by_canonical[canonical] = row
    return by_id, by_canonical


def mining_inventory_map(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    payload = load_json(summary_path)
    inventory = payload.get("inventory_rows") or []
    return {
        str(row["row_id"]): row
        for row in inventory
        if isinstance(row, dict) and row.get("row_id") is not None
    }


def representative_metric_map(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    payload = load_json(summary_path)
    rows = payload.get("representative_rows") or []
    return {
        str(row["row_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("row_id") is not None
    }


def validate_and_enrich_rows(
    args: argparse.Namespace,
) -> tuple[list[AuditRow], list[dict[str, Any]]]:
    suite = load_suite(args.suite_path)
    suite_by_id = {position.id: position for position in suite}
    reference_by_id, reference_by_canonical = load_reference_maps(args.reference_path)
    inventory_by_id = mining_inventory_map(args.mining_summary_path)
    representative_by_id = representative_metric_map(args.mining_summary_path)
    selected_rows = load_jsonl(args.selected_rows_path)
    roles_seen: Counter[str] = Counter()
    audit_rows: list[AuditRow] = []
    validation_rows: list[dict[str, Any]] = []

    for row in selected_rows:
        row_id = str(row["row_id"])
        role = str(row["recommended_role"])
        roles_seen[role] += 1
        if str(row.get("family")) != FAMILY:
            raise ValueError(f"{row_id} family mismatch: expected {FAMILY}")
        suite_row = suite_by_id.get(row_id)
        if suite_row is None:
            raise ValueError(f"{row_id} missing from forensic suite")
        legal_moves = list(suite_row.legal_moves)
        corrected_reference_move = int(row["corrected_reference_move"])
        if corrected_reference_move not in legal_moves:
            raise ValueError(
                f"{row_id} corrected reference move {corrected_reference_move} is illegal"
            )
        canonical = canonical_state_key(suite_row.state)
        if canonical != str(row["canonical_state_hash"]):
            raise ValueError(
                f"{row_id} canonical_state_hash does not match suite canonical state"
            )
        reference_row = reference_by_id.get(row_id) or reference_by_canonical.get(
            canonical
        )
        if reference_row is None:
            raise ValueError(f"{row_id} missing from corrected reference data")
        reference_unstable = bool(reference_row.get("reference_unstable", False))
        if reference_unstable:
            raise ValueError(f"{row_id} is marked reference_unstable")
        inventory_row = inventory_by_id.get(row_id, {})
        failure_status = inventory_row.get("failure_status")
        reference_integrity_error = failure_status == "reference_integrity_error"
        if reference_integrity_error:
            raise ValueError(f"{row_id} is marked reference_integrity_error")
        mining_metrics = representative_by_id.get(row_id, {})
        audit_row = AuditRow(
            row_id=row_id,
            role=role,
            severity=str(row.get("severity") or "none"),
            failure_mode=str(row.get("failure_mode") or "unknown"),
            corrected_reference_move=corrected_reference_move,
            current_selected_move=int(row["current_selected_move"]),
            suite_state=dict(suite_row.state),
            legal_moves=legal_moves,
            canonical_state_hash=canonical,
            reference_teacher_value=round_float(reference_row.get("teacher_value")),
            reference_unstable=reference_unstable,
            reference_integrity_error=reference_integrity_error,
            inventory_failure_status=None
            if failure_status is None
            else str(failure_status),
            mining_metrics=dict(mining_metrics),
        )
        audit_rows.append(audit_row)
        validation_rows.append(
            {
                "row_id": row_id,
                "role": role,
                "corrected_reference_move": corrected_reference_move,
                "reference_stable": True,
                "status": "ok",
                "notes": "reference and suite canonical state validated",
            }
        )

    for expected_role in (
        "target_candidate",
        "preservation_control",
        "holdout_candidate",
    ):
        if roles_seen[expected_role] == 0:
            raise ValueError(
                f"selected rows are missing required role bucket {expected_role}"
            )
    return audit_rows, validation_rows


def root_baseline_for_row(
    *,
    evaluator: ArtifactEvaluator,
    row: AuditRow,
    budget: int,
    seed: int,
    cpuct: float,
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=row.suite_state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=float(cpuct),
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    selected_move = (
        None if result.get("selected_move") is None else int(result["selected_move"])
    )
    legal_moves = [int(move) for move in result.get("legal_moves") or row.legal_moves]
    move_consequences = []
    for move in legal_moves:
        consequence = move_consequence_for_state(row.suite_state, move)
        child_state = child_state_from_move(row.suite_state, move)
        move_consequences.append(
            {
                "move": int(move),
                "gives_extra_turn": bool(consequence["gives_extra_turn"]),
                "produces_capture": bool(consequence["produces_capture"]),
                "capture_count": int(consequence["capture_count"]),
                "immediate_store_delta": int(consequence["store_delta_immediate"]),
                "side_to_move_after": int(child_state["current_player"]),
                "game_over_after_move": bool(consequence["game_over_after_move"]),
                "remaining_seed_count": int(
                    sum(KalahGame.from_state(child_state).pits)
                ),
            }
        )
    selection_map = selection_entry_map(result)
    reference_entry = selection_map.get(row.corrected_reference_move, {})
    selected_entry = (
        selection_map.get(selected_move, {}) if selected_move is not None else {}
    )
    root_value = round_float(float(result.get("value", 0.0)))
    return {
        "row_id": row.row_id,
        "role": row.role,
        "budget": int(budget),
        "legal_moves": legal_moves,
        "corrected_reference_move": int(row.corrected_reference_move),
        "selected_move": selected_move,
        "pass": selected_move == row.corrected_reference_move,
        "root_value": root_value,
        "root_policy": {
            str(move): round_float(float(result["policy"][move]))
            for move in legal_moves
        },
        "reference_prior": round_float(float(reference_entry.get("prior", 0.0))),
        "selected_prior": round_float(float(selected_entry.get("prior", 0.0)))
        if selected_move is not None
        else None,
        "reference_visit_share": visit_share(
            list(result.get("visits") or []), row.corrected_reference_move
        ),
        "selected_visit_share": visit_share(
            list(result.get("visits") or []), selected_move
        )
        if selected_move is not None
        else None,
        "reference_q": round_float(float(reference_entry.get("q_value", 0.0))),
        "selected_q": round_float(float(selected_entry.get("q_value", 0.0)))
        if selected_move is not None
        else None,
        "reference_u": round_float(float(reference_entry.get("u_component", 0.0))),
        "selected_u": round_float(float(selected_entry.get("u_component", 0.0)))
        if selected_move is not None
        else None,
        "reference_selection_score": round_float(
            float(reference_entry.get("selection_score", 0.0))
        ),
        "selected_selection_score": round_float(
            float(selected_entry.get("selection_score", 0.0))
        )
        if selected_move is not None
        else None,
        "selected_minus_reference_q_margin": round_float(
            float(selected_entry.get("q_value", 0.0))
            - float(reference_entry.get("q_value", 0.0))
        )
        if selected_move is not None
        else None,
        "move_consequences": move_consequences,
        "selection_breakdown": result.get("selection_breakdown") or {},
        "notes": "deterministic PUCT baseline",
    }


def estimate_teacher_5000_budget(
    failing_rows: list[AuditRow],
    sample_state: dict[str, Any],
    seeds: tuple[int, ...],
) -> tuple[bool, str | None]:
    started = time.perf_counter()
    ClassicMCTS(
        KalahGame.from_state(sample_state),
        simulations=CHILD_TEACHER_BASE_BUDGETS[0],
        seed=int(seeds[0]),
    ).root_summary()
    elapsed = max(time.perf_counter() - started, 1e-6)
    projected = (
        elapsed
        * (OPTIONAL_CHILD_TEACHER_BUDGET / CHILD_TEACHER_BASE_BUDGETS[0])
        * len(failing_rows)
        * len(seeds)
        * 2.0
    )
    if projected > MAX_PROJECTED_5000_SECONDS:
        return False, (
            f"skipped 5000 teacher budget: projected ~{projected:.1f}s across "
            f"{len(failing_rows)} failing rows, 2 children, and {len(seeds)} seeds"
        )
    return True, None


def child_neural_audit(
    *, evaluator: ArtifactEvaluator, row: AuditRow, baseline_1200: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_player = int(row.suite_state["current_player"])
    selected_move = int(baseline_1200["selected_move"])
    child_states = {
        "corrected_reference_child": child_state_from_move(
            row.suite_state, row.corrected_reference_move
        ),
        "selected_child": child_state_from_move(row.suite_state, selected_move),
    }
    child_results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for child_name, child_state in child_states.items():
        game = KalahGame.from_state(child_state)
        policy, raw_value = evaluator.evaluate(game)
        legal_moves = game.possible_moves()
        root_value = state_to_root_perspective_value(
            raw_value=float(raw_value), state=child_state, root_player=root_player
        )
        child_results[child_name] = {
            "raw_value": float(raw_value),
            "root_value": root_value,
            "policy_top_move": selected_move_from_policy(policy.tolist(), legal_moves),
            "legal_moves": legal_moves,
            "state": child_state,
        }
    diff = (
        child_results["corrected_reference_child"]["root_value"]
        - child_results["selected_child"]["root_value"]
    )
    agrees = diff > 0.0
    for child_name, move in (
        ("corrected_reference_child", row.corrected_reference_move),
        ("selected_child", selected_move),
    ):
        child_state = child_results[child_name]["state"]
        rows.append(
            {
                "row_id": row.row_id,
                "corrected_reference_move": int(row.corrected_reference_move),
                "selected_move": int(selected_move),
                "child": child_name,
                "child_move": int(move),
                "raw_value": round_float(child_results[child_name]["raw_value"]),
                "root_perspective_value": round_float(
                    child_results[child_name]["root_value"]
                ),
                "child_ref_minus_child_selected": round_float(diff),
                "agrees_with_corrected_reference": agrees,
                "child_policy_top_move": child_results[child_name]["policy_top_move"],
                "child_legal_moves": list(child_results[child_name]["legal_moves"]),
                "notes": (
                    "identity conversion"
                    if int(child_state["current_player"]) == root_player
                    else "sign flip to root perspective"
                ),
            }
        )
    return rows, {
        "selected_move": selected_move,
        "child_ref_root": child_results["corrected_reference_child"]["root_value"],
        "child_selected_root": child_results["selected_child"]["root_value"],
        "agrees": agrees,
        "child_ref_state": child_results["corrected_reference_child"]["state"],
        "child_selected_state": child_results["selected_child"]["state"],
    }


def teacher_summary_for_state(
    state: dict[str, Any], *, budget: int, seed: int
) -> dict[str, Any]:
    summary = ClassicMCTS(
        KalahGame.from_state(state), simulations=int(budget), seed=int(seed)
    ).root_summary()
    child_stats = summary.get("child_stats") or []
    selected_move = summary.get("selected_move")
    value = 0.0
    for child in child_stats:
        if int(child["move"]) == int(selected_move):
            visits = int(child.get("visits", 0))
            q_value = float(child.get("win_rate", 0.5)) if "win_rate" in child else None
            if q_value is not None:
                value = (2.0 * q_value) - 1.0
            elif visits > 0:
                value = float(child.get("q_value", 0.0))
            break
    if selected_move is not None and value == 0.0:
        for child in child_stats:
            if (
                int(child["move"]) == int(selected_move)
                and int(child.get("visits", 0)) > 0
            ):
                value = float((2.0 * float(child.get("win_rate", 0.5))) - 1.0)
                break
    return {
        "selected_move": None if selected_move is None else int(selected_move),
        "child_stats": child_stats,
        "value_raw": float(value),
    }


def teacher_child_audit(
    *,
    row: AuditRow,
    child_summary: dict[str, Any],
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    root_player = int(row.suite_state["current_player"])
    rows: list[dict[str, Any]] = []
    aggregate: dict[int, list[dict[str, Any]]] = {int(budget): [] for budget in budgets}
    for budget in budgets:
        for seed in seeds:
            ref_summary = teacher_summary_for_state(
                child_summary["child_ref_state"], budget=int(budget), seed=int(seed)
            )
            selected_summary = teacher_summary_for_state(
                child_summary["child_selected_state"],
                budget=int(budget),
                seed=int(seed),
            )
            ref_root = state_to_root_perspective_value(
                raw_value=float(ref_summary["value_raw"]),
                state=child_summary["child_ref_state"],
                root_player=root_player,
            )
            selected_root = state_to_root_perspective_value(
                raw_value=float(selected_summary["value_raw"]),
                state=child_summary["child_selected_state"],
                root_player=root_player,
            )
            diff = ref_root - selected_root
            row_payload = {
                "row_id": row.row_id,
                "budget": int(budget),
                "seed": int(seed),
                "child_ref_selected_move": ref_summary["selected_move"],
                "child_selected_selected_move": selected_summary["selected_move"],
                "child_ref_value_root": round_float(ref_root),
                "child_selected_value_root": round_float(selected_root),
                "child_ref_minus_child_selected": round_float(diff),
                "teacher_prefers_corrected_reference": diff > 0.0,
                "stable": None,
                "notes": "classic MCTS child-afterstate teacher audit",
            }
            aggregate[int(budget)].append(row_payload)
            rows.append(row_payload)
    budget_summary: dict[int, dict[str, Any]] = {}
    for budget, entries in aggregate.items():
        preferences = {
            bool(entry["teacher_prefers_corrected_reference"]) for entry in entries
        }
        stable = len(preferences) == 1
        diffs = [float(entry["child_ref_minus_child_selected"]) for entry in entries]
        ref_values = [float(entry["child_ref_value_root"]) for entry in entries]
        selected_values = [
            float(entry["child_selected_value_root"]) for entry in entries
        ]
        budget_summary[int(budget)] = {
            "stable": stable,
            "mean_diff": round_float(sum(diffs) / len(diffs)),
            "median_diff": round_float(sorted(diffs)[len(diffs) // 2]),
            "mean_child_ref_value_root": round_float(sum(ref_values) / len(ref_values)),
            "mean_child_selected_value_root": round_float(
                sum(selected_values) / len(selected_values)
            ),
            "teacher_prefers_corrected_reference": sum(
                1 for entry in entries if entry["teacher_prefers_corrected_reference"]
            )
            > (len(entries) / 2.0),
        }
        for entry in entries:
            entry["stable"] = stable
    return rows, budget_summary


def child_puct_audit(
    *,
    evaluator: ArtifactEvaluator,
    row: AuditRow,
    child_summary: dict[str, Any],
    budgets: tuple[int, ...],
    seed: int,
    cpuct: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    root_player = int(row.suite_state["current_player"])
    rows: list[dict[str, Any]] = []
    budget_summary: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        ref_result = evaluate_artifact_position(
            artifact_path=None,
            evaluator=evaluator,
            state=child_summary["child_ref_state"],
            simulations=int(budget),
            seed=int(seed),
            c_puct=float(cpuct),
            search_options=dict(SEARCH_OPTIONS),
            ablation_mode="full",
        )
        selected_result = evaluate_artifact_position(
            artifact_path=None,
            evaluator=evaluator,
            state=child_summary["child_selected_state"],
            simulations=int(budget),
            seed=int(seed),
            c_puct=float(cpuct),
            search_options=dict(SEARCH_OPTIONS),
            ablation_mode="full",
        )
        ref_raw = float(
            (ref_result.get("selection_breakdown") or {}).get("parent_q_value", 0.0)
        )
        selected_raw = float(
            (selected_result.get("selection_breakdown") or {}).get(
                "parent_q_value", 0.0
            )
        )
        ref_root = state_to_root_perspective_value(
            raw_value=ref_raw,
            state=child_summary["child_ref_state"],
            root_player=root_player,
        )
        selected_root = state_to_root_perspective_value(
            raw_value=selected_raw,
            state=child_summary["child_selected_state"],
            root_player=root_player,
        )
        diff = ref_root - selected_root
        budget_summary[int(budget)] = {
            "child_ref_value_root": round_float(ref_root),
            "child_selected_value_root": round_float(selected_root),
            "child_ref_minus_child_selected": round_float(diff),
            "puct_prefers_corrected_reference": diff > 0.0,
            "child_ref_selected_move": ref_result.get("selected_move"),
            "child_selected_selected_move": selected_result.get("selected_move"),
        }
        rows.append(
            {
                "row_id": row.row_id,
                "budget": int(budget),
                "child_ref_value_root": round_float(ref_root),
                "child_selected_value_root": round_float(selected_root),
                "child_ref_minus_child_selected": round_float(diff),
                "puct_prefers_corrected_reference": diff > 0.0,
                "child_ref_selected_move": ref_result.get("selected_move"),
                "child_selected_selected_move": selected_result.get("selected_move"),
                "notes": "artifact PUCT child-afterstate audit",
            }
        )
    return rows, budget_summary


class DiagnosticPUCT(PUCT):
    def __init__(
        self, *args, child_value_overrides: dict[int, float] | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.child_value_overrides = dict(child_value_overrides or {})
        self.root_move_by_state: dict[str, int] = {}

    def register_root_children(self, root: Node) -> None:
        for move, child in root.children.items():
            self.root_move_by_state[self._state_key(child.game)] = int(move)

    def _state_key(self, game: KalahGame) -> str:
        return json.dumps(game.to_state(), sort_keys=True, separators=(",", ":"))

    def _search(self, node: Node) -> float:
        terminal = terminal_value(node.game)
        if terminal is not None:
            return terminal
        if not node.expanded:
            _, value = self._expand(
                node,
                apply_dirichlet=False,
                dirichlet_alpha=None,
                dirichlet_epsilon=0.0,
                is_root=False,
            )
            root_move = self.root_move_by_state.get(self._state_key(node.game))
            if root_move is not None and root_move in self.child_value_overrides:
                value = float(self.child_value_overrides[root_move])
            return value
        if not node.children:
            return 0.0
        child = self._select_child(node)
        value = self._search(child)
        if child.game.current_player != node.game.current_player:
            value = -value
        child.visit_count += 1
        child.value_sum += value
        return value


def make_equalize_root_priors_override(
    reference_move: int, selected_move: int
) -> Callable[..., np.ndarray]:
    def override(*, game, legal_moves: list[int], priors: np.ndarray) -> np.ndarray:
        del game
        adjusted = np.asarray(priors, dtype=np.float32).copy()
        if reference_move in legal_moves and selected_move in legal_moves:
            target = max(
                float(adjusted[reference_move]), float(adjusted[selected_move])
            )
            adjusted[reference_move] = target
            adjusted[selected_move] = target
        normalized = np.zeros_like(adjusted)
        normalized[legal_moves] = adjusted[legal_moves]
        total = float(np.sum(normalized[legal_moves]))
        if total > 0.0:
            normalized[legal_moves] /= total
        else:
            normalized[legal_moves] = 1.0 / len(legal_moves)
        return normalized.astype(np.float32)

    return override


def uniform_legal_prior_override(
    *, game, legal_moves: list[int], priors: np.ndarray
) -> np.ndarray:
    del game, priors
    adjusted = np.zeros(6, dtype=np.float32)
    if legal_moves:
        adjusted[legal_moves] = 1.0 / len(legal_moves)
    return adjusted


def run_counterfactual(
    *,
    evaluator: ArtifactEvaluator,
    row: AuditRow,
    budget: int,
    seed: int,
    cpuct: float,
    root_prior_override,
    child_value_overrides: dict[int, float] | None,
) -> dict[str, Any]:
    root_game = KalahGame.from_state(row.suite_state)
    search = DiagnosticPUCT(
        evaluator=evaluator,
        simulations=int(budget),
        c_puct=float(cpuct),
        rng=random.Random(int(seed)),
        fpu_mode=str(SEARCH_OPTIONS["fpu_mode"]),
        reuse_subtree=bool(SEARCH_OPTIONS["reuse_subtree"]),
        normalize_values=bool(SEARCH_OPTIONS["normalize_values"]),
        root_policy_mode=str(SEARCH_OPTIONS["root_policy_mode"]),
        tactical_root_bias=float(SEARCH_OPTIONS["tactical_root_bias"]),
        ablation_mode="full",
        root_prior_override=root_prior_override,
        child_value_overrides=child_value_overrides,
    )
    root = search._root_for(root_game)
    search._expand(
        root,
        apply_dirichlet=False,
        dirichlet_alpha=None,
        dirichlet_epsilon=0.0,
        is_root=True,
    )
    search.register_root_children(root)
    for _ in range(int(budget)):
        value = search._search(root)
        root.visit_count += 1
        root.value_sum += value
    search._last_root = root
    return search.root_summary()


def counterfactual_rows(
    *,
    evaluator: ArtifactEvaluator,
    row: AuditRow,
    child_summary: dict[str, Any],
    teacher_budget_summary: dict[int, dict[str, Any]],
    neural_summary: dict[str, Any],
    seed: int,
    cpuct: float,
) -> list[dict[str, Any]]:
    selected_move = int(child_summary["selected_move"])
    root_player = int(row.suite_state["current_player"])
    teacher_budget = max(teacher_budget_summary)
    teacher_values_root = {
        row.corrected_reference_move: float(
            teacher_budget_summary[teacher_budget]["mean_child_ref_value_root"] or 0.0
        ),
        selected_move: float(
            teacher_budget_summary[teacher_budget]["mean_child_selected_value_root"]
            or 0.0
        ),
    }
    interventions = {
        "original": {
            "root_prior_override": None,
            "child_value_overrides": None,
            "notes": "no intervention",
        },
        "equalize_root_priors": {
            "root_prior_override": make_equalize_root_priors_override(
                row.corrected_reference_move, selected_move
            ),
            "child_value_overrides": None,
            "notes": "equalize corrected reference and selected root priors",
        },
        "uniform_legal_prior": {
            "root_prior_override": uniform_legal_prior_override,
            "child_value_overrides": None,
            "notes": "uniform legal prior positive control",
        },
        "teacher_child_value_override": {
            "root_prior_override": None,
            "child_value_overrides": {
                row.corrected_reference_move: root_to_state_perspective_value(
                    root_value=float(
                        neural_summary["child_ref_root"]
                        if teacher_values_root[row.corrected_reference_move] == 0.0
                        else teacher_values_root[row.corrected_reference_move]
                    ),
                    state=child_summary["child_ref_state"],
                    root_player=root_player,
                ),
                selected_move: root_to_state_perspective_value(
                    root_value=float(teacher_values_root[selected_move]),
                    state=child_summary["child_selected_state"],
                    root_player=root_player,
                ),
            },
            "notes": "override first child expansion values with teacher child values",
        },
        "neural_child_value_swap": {
            "root_prior_override": None,
            "child_value_overrides": {
                row.corrected_reference_move: root_to_state_perspective_value(
                    root_value=float(neural_summary["child_selected_root"]),
                    state=child_summary["child_ref_state"],
                    root_player=root_player,
                ),
                selected_move: root_to_state_perspective_value(
                    root_value=float(neural_summary["child_ref_root"]),
                    state=child_summary["child_selected_state"],
                    root_player=root_player,
                ),
            },
            "notes": "swap child neural values as diagnostic sanity check",
        },
    }
    rows: list[dict[str, Any]] = []
    for intervention, config in interventions.items():
        for budget in COUNTERFACTUAL_BUDGETS:
            summary = run_counterfactual(
                evaluator=evaluator,
                row=row,
                budget=int(budget),
                seed=int(seed),
                cpuct=float(cpuct),
                root_prior_override=config["root_prior_override"],
                child_value_overrides=config["child_value_overrides"],
            )
            selection_map = {
                int(entry["move"]): entry
                for entry in list(
                    (summary.get("selection_breakdown") or {}).get("moves") or []
                )
                if isinstance(entry, dict) and entry.get("move") is not None
            }
            ref_entry = selection_map.get(row.corrected_reference_move, {})
            selected_summary_move = summary.get("selected_move")
            chosen_entry = (
                selection_map.get(int(selected_summary_move), {})
                if selected_summary_move is not None
                else {}
            )
            rows.append(
                {
                    "row_id": row.row_id,
                    "intervention": intervention,
                    "budget": int(budget),
                    "selected_move": selected_summary_move,
                    "selected_is_corrected_reference": selected_summary_move
                    == row.corrected_reference_move,
                    "reference_visit_share": round_float(
                        float(ref_entry.get("visit_count", 0))
                        / max(
                            1.0,
                            sum(
                                float(entry.get("visit_count", 0))
                                for entry in selection_map.values()
                            ),
                        )
                    )
                    if selection_map
                    else None,
                    "selected_visit_share": round_float(
                        float(chosen_entry.get("visit_count", 0))
                        / max(
                            1.0,
                            sum(
                                float(entry.get("visit_count", 0))
                                for entry in selection_map.values()
                            ),
                        )
                    )
                    if selection_map and selected_summary_move is not None
                    else None,
                    "reference_q": round_float(float(ref_entry.get("q_value", 0.0))),
                    "selected_q": round_float(float(chosen_entry.get("q_value", 0.0)))
                    if selected_summary_move is not None
                    else None,
                    "selected_minus_reference_q_margin": round_float(
                        float(chosen_entry.get("q_value", 0.0))
                        - float(ref_entry.get("q_value", 0.0))
                    )
                    if selected_summary_move is not None
                    else None,
                    "flipped": selected_summary_move == row.corrected_reference_move,
                    "notes": config["notes"],
                }
            )
    return rows


def classify_row(
    *,
    row: AuditRow,
    neural_summary: dict[str, Any] | None,
    teacher_budget_summary: dict[int, dict[str, Any]] | None,
    child_puct_budget_summary: dict[int, dict[str, Any]] | None,
    root_baseline_1200: dict[str, Any],
    counterfactual: list[dict[str, Any]],
) -> dict[str, Any]:
    if row.role == "preservation_control":
        return {
            "row_id": row.row_id,
            "role": row.role,
            "row_classification": "inconclusive",
            "supporting_evidence": "passing preservation control under baseline",
            "next_use": "preserve_control",
            "notes": "control row retained to guard against regressions",
        }
    if row.role == "holdout_candidate" and bool(root_baseline_1200["pass"]):
        return {
            "row_id": row.row_id,
            "role": row.role,
            "row_classification": "inconclusive",
            "supporting_evidence": "holdout row currently passes at root; retained for out-of-sample monitoring",
            "next_use": "holdout",
            "notes": "baseline root pass",
        }
    assert neural_summary is not None
    assert teacher_budget_summary is not None
    assert child_puct_budget_summary is not None
    deepest_teacher_budget = max(teacher_budget_summary)
    teacher_prefers_reference = bool(
        teacher_budget_summary[deepest_teacher_budget][
            "teacher_prefers_corrected_reference"
        ]
    )
    teacher_stable = bool(teacher_budget_summary[deepest_teacher_budget]["stable"])
    child_puct_prefers_reference = bool(
        child_puct_budget_summary[max(child_puct_budget_summary)][
            "puct_prefers_corrected_reference"
        ]
    )
    neural_prefers_reference = bool(neural_summary["agrees"])
    root_prefers_reference = bool(root_baseline_1200["pass"])
    same_player_ref = int(neural_summary["child_ref_state"]["current_player"]) == int(
        row.suite_state["current_player"]
    )
    same_player_selected = int(
        neural_summary["child_selected_state"]["current_player"]
    ) == int(row.suite_state["current_player"])
    flip_intervention = any(
        entry["intervention"] == "teacher_child_value_override" and entry["flipped"]
        for entry in counterfactual
    )
    prior_flip = any(
        entry["intervention"] == "equalize_root_priors" and entry["flipped"]
        for entry in counterfactual
    )
    if not teacher_prefers_reference:
        classification = "corrected_reference_suspicious"
        evidence = "deepest child teacher does not prefer corrected reference"
    elif (
        same_player_ref == same_player_selected
        and row.failure_mode == "search_selection"
        and not neural_prefers_reference
        and child_puct_prefers_reference
    ):
        classification = "backup_perspective_suspect"
        evidence = "mixed same-player child signs with search-selection failure"
    elif teacher_prefers_reference and not neural_prefers_reference:
        classification = "value_head_miscalibration"
        evidence = "teacher child values prefer corrected reference while raw neural child values prefer selected move"
    elif (
        teacher_prefers_reference
        and neural_prefers_reference
        and not child_puct_prefers_reference
    ):
        classification = "puct_child_search_value_mismatch"
        evidence = "teacher and raw neural child values prefer corrected reference while child PUCT does not"
    elif (
        teacher_prefers_reference
        and neural_prefers_reference
        and child_puct_prefers_reference
        and not root_prefers_reference
    ):
        classification = "root_prior_or_selection_pressure"
        evidence = "child teacher, neural, and child PUCT all support corrected reference but root still selects away"
    elif flip_intervention and not root_prefers_reference:
        classification = "puct_child_search_value_mismatch"
        evidence = "teacher child value override flips root decision"
    elif prior_flip and not root_prefers_reference:
        classification = "root_prior_or_selection_pressure"
        evidence = "equalized root priors flip the root decision"
    elif not teacher_stable:
        classification = "inconclusive"
        evidence = "teacher child preference is unstable across seeds"
    else:
        classification = "inconclusive"
        evidence = "evidence remains mixed after neural, teacher, child PUCT, and counterfactual audits"
    return {
        "row_id": row.row_id,
        "role": row.role,
        "row_classification": classification,
        "supporting_evidence": evidence,
        "next_use": "holdout"
        if row.role == "holdout_candidate"
        else "target_candidate",
        "notes": "baseline root pass"
        if root_prefers_reference
        else "persistent root disagreement",
    }


def classify_family(
    row_classifications: list[dict[str, Any]],
) -> tuple[str, str, dict[str, int]]:
    target_rows = [
        row for row in row_classifications if row["role"] == "target_candidate"
    ]
    counts = Counter(str(row["row_classification"]) for row in target_rows)
    total = max(1, len(target_rows))
    if counts["value_head_miscalibration"] > total / 2.0:
        return (
            "value_head_family_gap",
            "build a value-calibration experiment using child-afterstate targets from incumbent_proxy_disagreement, with preservation controls and no arena until local value metrics improve.",
            dict(counts),
        )
    if (
        counts["puct_child_search_value_mismatch"]
        + counts["backup_perspective_suspect"]
    ) > total / 2.0:
        return (
            "puct_backup_family_gap",
            "audit PUCT backup/perspective implementation before any training.",
            dict(counts),
        )
    if counts["root_prior_or_selection_pressure"] > total / 2.0:
        return (
            "root_selection_family_gap",
            "run root cpuct/prior calibration diagnostics for incumbent_proxy_disagreement.",
            dict(counts),
        )
    if counts["corrected_reference_suspicious"] >= max(2, total // 3):
        return (
            "reference_family_uncertain",
            "adjudicate incumbent_proxy_disagreement references with deeper multi-seed teacher search before training.",
            dict(counts),
        )
    return (
        "mixed_family_gap",
        "split the family into mechanism-specific subfamilies before training.",
        dict(counts),
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def compact_json(value: Any) -> str:
    return "`" + json.dumps(value, sort_keys=True) + "`"


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Incumbent Proxy Disagreement Value/Backup Audit Results",
        "",
        "## 1. Context",
        "",
        "- No training was run.",
        "- No arena was run.",
        "- No model was promoted.",
        f"- Corrected references: `{summary['inputs']['reference_path']}`.",
        f"- Current artifact: `{summary['inputs']['current_artifact']}`.",
        f"- Selected family rows: `{summary['inputs']['selected_rows_path']}`.",
        "",
        "## 2. Why incumbent_proxy_disagreement was selected",
        "",
        f"- PR #42 selected `{summary['selected_family']}` as the highest-ranked corrected non-opening family still classified as a value/backup issue.",
        f"- Family stats: `{compact_json(summary['family_context'])}`.",
        "",
        "## 3. Row selection and validation",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "corrected_reference_move",
                "current_selected_384",
                "current_selected_1200",
                "reference_stable",
                "status",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    str(row["corrected_reference_move"]),
                    str(
                        row.get("current_selected_384")
                        if row.get("current_selected_384") is not None
                        else "-"
                    ),
                    str(
                        row.get("current_selected_1200")
                        if row.get("current_selected_1200") is not None
                        else "-"
                    ),
                    format_bool(row["reference_stable"]),
                    row["status"],
                    row["notes"],
                ]
                for row in summary["row_validation_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- Audited target rows: `{summary['row_selection']['target_row_ids']}`.",
            f"- Audited holdout rows: `{summary['row_selection']['holdout_row_ids']}`.",
            f"- Audited control rows: `{summary['row_selection']['control_row_ids']}`.",
            f"- Perspective conversion: `{summary['perspective_and_backup_check']['conversion_rule']}`.",
            "",
            "## 4. Root PUCT baseline",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "row_id",
                "corrected_reference_move",
                "selected_move",
                "budget",
                "reference_visit_share",
                "selected_visit_share",
                "reference_q",
                "selected_q",
                "selected_minus_reference_q_margin",
                "reference_prior",
                "selected_prior",
                "pass",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["corrected_reference_move"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    str(row["budget"]),
                    format_float(row["reference_visit_share"]),
                    format_float(row["selected_visit_share"]),
                    format_float(row["reference_q"]),
                    format_float(row["selected_q"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    format_float(row["reference_prior"]),
                    format_float(row["selected_prior"]),
                    format_bool(row["pass"]),
                    row["notes"],
                ]
                for row in summary["root_baseline_table"]
            ],
        )
    )
    lines.append("")
    for row in summary["root_consequence_notes"]:
        lines.append(
            f"- `{row['row_id']}` legal moves and consequences: {compact_json(row['move_consequences'])}."
        )
    lines.extend(["", "## 5. Child-afterstate neural value audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "corrected_reference_move",
                "selected_move",
                "child",
                "raw_value",
                "root_perspective_value",
                "child_ref_minus_child_selected",
                "agrees_with_corrected_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["corrected_reference_move"]),
                    str(row["selected_move"]),
                    row["child"],
                    format_float(row["raw_value"]),
                    format_float(row["root_perspective_value"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["agrees_with_corrected_reference"]),
                    row["notes"],
                ]
                for row in summary["child_neural_table"]
            ],
        )
    )
    lines.extend(["", "## 6. Child-afterstate teacher audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "seed",
                "child_ref_value_root",
                "child_selected_value_root",
                "child_ref_minus_child_selected",
                "teacher_prefers_corrected_reference",
                "stable",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    str(row["seed"]),
                    format_float(row["child_ref_value_root"]),
                    format_float(row["child_selected_value_root"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["teacher_prefers_corrected_reference"]),
                    format_bool(row["stable"]),
                    row["notes"],
                ]
                for row in summary["child_teacher_table"]
            ],
        )
    )
    if summary.get("teacher_5000_skip_reason"):
        lines.append("")
        lines.append(f"- {summary['teacher_5000_skip_reason']}")
    lines.extend(["", "## 7. Child-afterstate PUCT audit", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "budget",
                "child_ref_value_root",
                "child_selected_value_root",
                "child_ref_minus_child_selected",
                "puct_prefers_corrected_reference",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    str(row["budget"]),
                    format_float(row["child_ref_value_root"]),
                    format_float(row["child_selected_value_root"]),
                    format_float(row["child_ref_minus_child_selected"]),
                    format_bool(row["puct_prefers_corrected_reference"]),
                    row["notes"],
                ]
                for row in summary["child_puct_table"]
            ],
        )
    )
    lines.extend(["", "## 8. Root counterfactual diagnostics", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "intervention",
                "budget",
                "selected_move",
                "selected_is_corrected_reference",
                "reference_visit_share",
                "selected_visit_share",
                "selected_minus_reference_q_margin",
                "flipped",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["intervention"],
                    str(row["budget"]),
                    str(
                        row["selected_move"]
                        if row["selected_move"] is not None
                        else "-"
                    ),
                    format_bool(row["selected_is_corrected_reference"]),
                    format_float(row["reference_visit_share"]),
                    format_float(row["selected_visit_share"]),
                    format_float(row["selected_minus_reference_q_margin"]),
                    format_bool(row["flipped"]),
                    row["notes"],
                ]
                for row in summary["counterfactual_table"]
            ],
        )
    )
    lines.extend(["", "## 9. Row classifications", ""])
    lines.extend(
        markdown_table(
            [
                "row_id",
                "role",
                "row_classification",
                "supporting_evidence",
                "next_use",
                "notes",
            ],
            [
                [
                    row["row_id"],
                    row["role"],
                    row["row_classification"],
                    row["supporting_evidence"],
                    row["next_use"],
                    row["notes"],
                ]
                for row in summary["classification_table"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 10. Family-level interpretation",
            "",
            f"- Family classification: `{summary['family_classification']['label']}`.",
            f"- Mechanism counts: `{compact_json(summary['family_classification']['counts'])}`.",
            f"- Interpretation: {summary['family_classification']['interpretation']}",
            "",
            "## 11. Exactly one recommended next action",
            "",
            f"Recommendation: **{summary['family_classification']['recommended_next_action']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    rerun_family_mining_if_needed(root, args)
    audit_rows, validation_rows = validate_and_enrich_rows(args)
    evaluator = ArtifactEvaluator(args.current_artifact)
    family_context = {}
    if args.mining_summary_path.exists():
        mining_summary = load_json(args.mining_summary_path)
        family_context = next(
            (
                row
                for row in list(mining_summary.get("family_rankings") or [])
                if isinstance(row, dict) and str(row.get("family")) == FAMILY
            ),
            {},
        )

    root_baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for row in audit_rows:
        for budget in ROOT_BUDGETS:
            root_baselines[(row.row_id, int(budget))] = root_baseline_for_row(
                evaluator=evaluator,
                row=row,
                budget=int(budget),
                seed=int(args.seed),
                cpuct=float(args.cpuct),
            )

    for row in validation_rows:
        row["current_selected_384"] = root_baselines[(row["row_id"], 384)][
            "selected_move"
        ]
        row["current_selected_1200"] = root_baselines[(row["row_id"], 1200)][
            "selected_move"
        ]

    target_candidates = []
    holdout_candidates = []
    control_candidates = []
    for row in audit_rows:
        representative = dict(row.mining_metrics)
        if not representative:
            representative = {
                "row_id": row.row_id,
                "severity": row.severity,
                "reference_visit_share_384": root_baselines[(row.row_id, 384)][
                    "reference_visit_share"
                ],
                "selected_minus_reference_q_margin_384": root_baselines[
                    (row.row_id, 384)
                ]["selected_minus_reference_q_margin"],
            }
        else:
            representative.setdefault("severity", row.severity)
            representative.setdefault("row_id", row.row_id)
        if row.role == "target_candidate":
            target_candidates.append((row, representative))
        elif row.role == "holdout_candidate":
            holdout_candidates.append(row)
        elif row.role == "preservation_control":
            control_candidates.append(row)
    target_candidates.sort(key=lambda item: row_sort_key(item[1]))
    selected_target_rows = [row for row, _metrics in target_candidates[:10]]
    selected_target_ids = {row.row_id for row in selected_target_rows}
    for required_row_id in REQUIRED_TARGET_ROW_IDS:
        if required_row_id in {row.row_id for row, _ in target_candidates}:
            selected_target_ids.add(required_row_id)
    audited_rows = [
        row
        for row in audit_rows
        if row.row_id in selected_target_ids or row.role != "target_candidate"
    ]
    audited_rows.sort(key=lambda row: (row.role, row.row_id))
    control_row_ids = [
        row.row_id for row in audited_rows if row.role == "preservation_control"
    ]
    holdout_row_ids = [
        row.row_id for row in audited_rows if row.role == "holdout_candidate"
    ]
    target_row_ids = [
        row.row_id for row in audited_rows if row.role == "target_candidate"
    ]
    failing_rows = [row for row in audited_rows if row.role != "preservation_control"]

    teacher_budgets: list[int] = list(CHILD_TEACHER_BASE_BUDGETS)
    teacher_5000_skip_reason = None
    if failing_rows:
        can_run_5000, teacher_5000_skip_reason = estimate_teacher_5000_budget(
            failing_rows,
            child_state_from_move(
                failing_rows[0].suite_state, failing_rows[0].corrected_reference_move
            ),
            CHILD_TEACHER_SEEDS,
        )
        if can_run_5000:
            teacher_budgets.append(OPTIONAL_CHILD_TEACHER_BUDGET)

    child_neural_rows: list[dict[str, Any]] = []
    child_teacher_rows: list[dict[str, Any]] = []
    child_puct_rows: list[dict[str, Any]] = []
    counterfactual_table: list[dict[str, Any]] = []
    classification_table: list[dict[str, Any]] = []
    neural_summary_by_row: dict[str, dict[str, Any]] = {}
    teacher_summary_by_row: dict[str, dict[int, dict[str, Any]]] = {}
    child_puct_summary_by_row: dict[str, dict[int, dict[str, Any]]] = {}

    for row in audited_rows:
        baseline_1200 = root_baselines[(row.row_id, 1200)]
        if row.role != "preservation_control":
            neural_rows, neural_summary = child_neural_audit(
                evaluator=evaluator,
                row=row,
                baseline_1200=baseline_1200,
            )
            child_neural_rows.extend(neural_rows)
            neural_summary_by_row[row.row_id] = neural_summary
            teacher_rows, teacher_budget_summary = teacher_child_audit(
                row=row,
                child_summary=neural_summary,
                budgets=tuple(teacher_budgets),
                seeds=CHILD_TEACHER_SEEDS,
            )
            child_teacher_rows.extend(teacher_rows)
            teacher_summary_by_row[row.row_id] = teacher_budget_summary
            puct_rows, puct_budget_summary = child_puct_audit(
                evaluator=evaluator,
                row=row,
                child_summary=neural_summary,
                budgets=CHILD_PUCT_BUDGETS,
                seed=int(args.seed),
                cpuct=float(args.cpuct),
            )
            child_puct_rows.extend(puct_rows)
            child_puct_summary_by_row[row.row_id] = puct_budget_summary
            counterfactual_table.extend(
                counterfactual_rows(
                    evaluator=evaluator,
                    row=row,
                    child_summary=neural_summary,
                    teacher_budget_summary=teacher_budget_summary,
                    neural_summary=neural_summary,
                    seed=int(args.seed),
                    cpuct=float(args.cpuct),
                )
            )
        classification_table.append(
            classify_row(
                row=row,
                neural_summary=neural_summary_by_row.get(row.row_id),
                teacher_budget_summary=teacher_summary_by_row.get(row.row_id),
                child_puct_budget_summary=child_puct_summary_by_row.get(row.row_id),
                root_baseline_1200=baseline_1200,
                counterfactual=[
                    entry
                    for entry in counterfactual_table
                    if entry["row_id"] == row.row_id
                ],
            )
        )

    family_label, family_action, family_counts = classify_family(classification_table)
    row_validation_table = []
    for row in validation_rows:
        row_validation_table.append(dict(row))
    root_baseline_table = [
        root_baselines[key]
        for key in sorted(root_baselines, key=lambda item: (item[0], item[1]))
        if key[0] in {row.row_id for row in audited_rows}
    ]
    root_consequence_notes = [
        {
            "row_id": row.row_id,
            "move_consequences": root_baselines[(row.row_id, 1200)][
                "move_consequences"
            ],
        }
        for row in audited_rows
    ]
    summary = {
        "schema": SCHEMA,
        "selected_family": FAMILY,
        "inputs": {
            "reference_path": str(args.reference_path),
            "suite_path": str(args.suite_path),
            "current_artifact": str(args.current_artifact),
            "selected_rows_path": str(args.selected_rows_path),
            "mining_summary_path": str(args.mining_summary_path),
        },
        "guardrails": {
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
            "mutated_corrected_references": False,
        },
        "family_context": family_context,
        "row_selection": {
            "target_row_ids": target_row_ids,
            "holdout_row_ids": holdout_row_ids,
            "control_row_ids": control_row_ids,
            "required_target_rows_present": sorted(
                row_id
                for row_id in REQUIRED_TARGET_ROW_IDS
                if row_id in target_row_ids or row_id in holdout_row_ids
            ),
            "required_control_rows_present": sorted(
                row_id
                for row_id in REQUIRED_CONTROL_ROW_IDS
                if row_id in control_row_ids
            ),
        },
        "perspective_and_backup_check": {
            "conversion_rule": "+1 when child current_player == root current_player, else -1",
            "implementation_rule": "PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player",
            "notes": "extra turns keep sign; turn handoff flips sign back to root-player perspective",
        },
        "row_validation_table": row_validation_table,
        "root_baseline_table": root_baseline_table,
        "root_consequence_notes": root_consequence_notes,
        "child_neural_table": child_neural_rows,
        "child_teacher_table": child_teacher_rows,
        "child_puct_table": child_puct_rows,
        "counterfactual_table": counterfactual_table,
        "classification_table": classification_table,
        "teacher_5000_skip_reason": teacher_5000_skip_reason,
        "family_classification": {
            "label": family_label,
            "counts": family_counts,
            "interpretation": (
                "most failing rows stay value-head dominated"
                if family_label == "value_head_family_gap"
                else "child search / backup disagreement dominates"
                if family_label == "puct_backup_family_gap"
                else "root selection pressure dominates"
                if family_label == "root_selection_family_gap"
                else "corrected references need adjudication"
                if family_label == "reference_family_uncertain"
                else "multiple mechanisms are present across the audited rows"
            ),
            "recommended_next_action": family_action,
        },
    }
    write_json(args.summary_out, summary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(args.summary_out),
                "report_path": str(args.report_out),
                "family_classification": family_label,
                "recommended_next_action": family_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
