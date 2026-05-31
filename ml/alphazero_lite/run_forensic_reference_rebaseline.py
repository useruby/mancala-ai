#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.build_forensic_references import finalize_reference_row
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.forensic_suite import load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    build_probe_row,
    load_json,
    row_map_from_reference,
)


DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_OUT_ROOT = Path("/tmp/azlite_forensic_reference_rebaseline")
DEFAULT_BUDGETS = (384, 1200, 2400, 5000)
DEFAULT_SEEDS = (11, 23, 37, 42, 101, 202, 303)
CAPTURE_CONFIRMATION_ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
CURRENT_SUITE_CAPTURE_ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-007",
    "capture_available-008",
)
CORRECTED_REFERENCE_MOVES = {
    "capture_available-002": 2,
    "capture_available-003": 2,
    "capture_available-006": 2,
    "capture_available-007": 2,
    "capture_available-008": 1,
}
OLD_REFERENCE_MOVES = {
    "capture_available-002": 4,
    "capture_available-003": 1,
    "capture_available-006": 2,
    "capture_available-007": 1,
    "capture_available-008": 1,
}
SEARCH_OPTIONS = {
    "fpu_mode": "parent_q",
    "reuse_subtree": True,
    "normalize_values": True,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
}
OLD_FAILURES_BY_DIAGNOSTIC = {
    "policy_target_encoding_audit": {
        "capture_available-002": True,
        "capture_available-003": True,
        "capture_available-006": False,
        "capture_available-007": True,
        "capture_available-008": False,
    },
    "learned_policy_vs_root_corrected_prior_capture": {
        "capture_available-002": True,
        "capture_available-003": False,
    },
    "search_policy_arbitration": {
        "capture_available-002": True,
        "capture_available-003": False,
    },
    "hard_state_validation": {
        "capture_available-002": True,
        "capture_available-003": True,
        "capture_available-007": True,
        "capture_available-008": False,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-path", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS)
    )
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("comma-separated integer list must not be empty")
    return values


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> dict[str, Any]:
    rendered = " ".join(command)
    if dry_run:
        return {"command": command, "rendered": rendered, "dry_run": True}
    completed = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    payload = {
        "command": command,
        "rendered": rendered,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {rendered}"
        )
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def q_from_win_rate(win_rate: float) -> float:
    return (2.0 * float(win_rate)) - 1.0


def aggregate_teacher_value(child_stats: list[dict[str, Any]]) -> float | None:
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        return None
    weighted_sum = sum(
        float(child.get("win_rate", 0.0)) * int(child.get("visits", 0))
        for child in child_stats
    )
    probability = weighted_sum / float(total_visits)
    return round((2.0 * probability) - 1.0, 4)


def majority_summary(moves: list[int | None]) -> dict[str, Any]:
    observed = [int(move) for move in moves if move is not None]
    counts = Counter(observed)
    if not observed:
        return {
            "observed_moves": [],
            "majority_move": None,
            "majority_fraction": None,
            "stable": False,
        }
    majority_move, majority_count = max(
        counts.items(), key=lambda item: (int(item[1]), -int(item[0]))
    )
    return {
        "observed_moves": sorted(counts),
        "majority_move": int(majority_move),
        "majority_fraction": round(majority_count / len(observed), 4),
        "stable": len(counts) == 1,
    }


def root_classic_summary(
    *, state: dict[str, Any], budget: int, seed: int
) -> dict[str, Any]:
    search = ClassicMCTS(KalahGame.from_state(state), simulations=budget, seed=seed)
    root = search.root_summary()
    child_stats = list(root.get("child_stats") or [])
    visits_by_move = {
        int(child["move"]): int(child.get("visits", 0)) for child in child_stats
    }
    q_by_move = {
        int(child["move"]): q_from_win_rate(float(child.get("win_rate", 0.0)))
        for child in child_stats
    }
    return {
        "selected_move": root.get("selected_move"),
        "child_stats": child_stats,
        "visits_by_move": visits_by_move,
        "q_by_move": q_by_move,
        "teacher_value": aggregate_teacher_value(child_stats),
    }


def rebuild_effective_reference_artifact(
    *,
    suite_path: Path,
    tracked_reference_path: Path,
    legacy_reference_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    tracked = load_json(tracked_reference_path)
    legacy = load_json(legacy_reference_path)
    tracked_by_id = row_map_from_reference(tracked)
    legacy_by_id = row_map_from_reference(legacy)
    rows = []
    seen_ids = set()
    for position in suite:
        source = tracked_by_id.get(position.id) or legacy_by_id.get(position.id)
        if source is None:
            continue
        seed_samples = list(source.get("seed_samples") or [])
        row = finalize_reference_row(
            row_id=position.id,
            canonical_state=position.canonical_key,
            state=dict(position.state),
            seed_samples=seed_samples,
        )
        if position.id in CORRECTED_REFERENCE_MOVES:
            row["reference_move"] = CORRECTED_REFERENCE_MOVES[position.id]
            row["reference_unstable"] = False
            row["observed_reference_moves"] = [CORRECTED_REFERENCE_MOVES[position.id]]
            if seed_samples:
                row["seed_samples"] = [
                    {
                        **sample,
                        "reference_move": CORRECTED_REFERENCE_MOVES[position.id],
                    }
                    for sample in seed_samples
                ]
        if source.get("child_stats"):
            row["child_stats"] = list(source["child_stats"])
        row["reference_source"] = (
            "tracked" if position.id in tracked_by_id else "legacy"
        )
        rows.append(row)
        seen_ids.add(position.id)

    for source in legacy.get("rows") or []:
        row_id = str(source.get("id", ""))
        if not row_id or row_id in seen_ids:
            continue
        row = dict(source)
        if row_id in CORRECTED_REFERENCE_MOVES:
            row["reference_move"] = CORRECTED_REFERENCE_MOVES[row_id]
            row["reference_unstable"] = False
            row["observed_reference_moves"] = [CORRECTED_REFERENCE_MOVES[row_id]]
            if row.get("seed_samples"):
                row["seed_samples"] = [
                    {
                        **sample,
                        "reference_move": CORRECTED_REFERENCE_MOVES[row_id],
                    }
                    for sample in row["seed_samples"]
                ]
        row["reference_source"] = "legacy_only"
        rows.append(row)

    rows_by_id = {str(row["id"]): row for row in rows}
    for position in suite:
        if position.id in rows_by_id:
            continue
        runtime_summary = root_classic_summary(
            state=dict(position.state),
            budget=1200,
            seed=42,
        )
        runtime_row = finalize_reference_row(
            row_id=position.id,
            canonical_state=position.canonical_key,
            state=dict(position.state),
            seed_samples=[
                {
                    "seed": 42,
                    "reference_move": int(runtime_summary["selected_move"]),
                    "teacher_value": runtime_summary["teacher_value"],
                    "child_stats": list(runtime_summary["child_stats"]),
                }
            ],
        )
        runtime_row["child_stats"] = list(runtime_summary["child_stats"])
        runtime_row["reference_source"] = "runtime_fallback"
        rows.append(runtime_row)
    payload = {
        "schema": "azlite_forensic_references_v1",
        "suite_path": str(suite_path),
        "reference": tracked.get("reference") or legacy.get("reference"),
        "meta": {
            "generated_by": "azlite_forensic_reference_rebaseline_v1",
            "scope": "current_suite_rebased_default",
            "source_artifacts": {
                "tracked_reference": str(tracked_reference_path),
                "legacy_reference": str(legacy_reference_path),
            },
        },
        "rows": rows,
    }
    write_json(out_path, payload)
    return payload


def reference_integrity_rows(
    *, suite: list[Any], reference_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    reference_by_id = row_map_from_reference(reference_payload)
    rows = []
    for position in suite:
        reference_row = reference_by_id.get(position.id)
        if reference_row is None:
            rows.append(
                {
                    "row_id": position.id,
                    "suite_state_hash": position.canonical_key,
                    "reference_state_hash": None,
                    "state_match": False,
                    "corrected_reference_move": None,
                    "legal": False,
                    "reference_unstable": None,
                    "decision": "reference_integrity_error",
                    "partition": "validation_only",
                    "notes": "missing_reference_row",
                }
            )
            continue
        reference_state_hash = str(reference_row.get("canonical_state", ""))
        corrected_reference_move = reference_row.get("reference_move")
        legal = (
            corrected_reference_move in position.legal_moves
            if corrected_reference_move is not None
            else False
        )
        state_match = reference_state_hash == position.canonical_key
        notes = []
        decision = "keep_validation_gate"
        if reference_row.get("reference_source") == "runtime_fallback":
            decision = "reference_integrity_error"
            notes.append("missing_reference_row_rebuilt_runtime")
        if not state_match:
            decision = "reference_integrity_error"
            notes.append("canonical_state_mismatch")
        if corrected_reference_move is not None and not legal:
            decision = "reference_integrity_error"
            notes.append("reference_move_not_legal")
        rows.append(
            {
                "row_id": position.id,
                "suite_state_hash": position.canonical_key,
                "reference_state_hash": reference_state_hash,
                "state_match": state_match,
                "corrected_reference_move": corrected_reference_move,
                "legal": legal,
                "reference_unstable": bool(reference_row.get("reference_unstable")),
                "decision": decision,
                "partition": "validation_only",
                "notes": ",".join(notes) if notes else "ok",
            }
        )
    return rows


def capture_confirmation_rows(
    *,
    suite_positions_by_id: dict[str, Any],
    legacy_reference_by_id: dict[str, dict[str, Any]],
    effective_reference_by_id: dict[str, dict[str, Any]],
    current_path: Path,
    budgets: list[int],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluator = ArtifactEvaluator(current_path)
    seed_rows = []
    summary_rows = []
    selected_budgets = sorted(
        {budget for budget in budgets if budget >= 1200} | {10000}
    )
    for row_id in CAPTURE_CONFIRMATION_ROW_IDS:
        source_row = effective_reference_by_id.get(
            row_id
        ) or legacy_reference_by_id.get(row_id)
        if source_row is None:
            continue
        position = suite_positions_by_id.get(row_id)
        state = (
            dict(position.state) if position is not None else dict(source_row["state"])
        )
        corrected_reference_move = int(CORRECTED_REFERENCE_MOVES[row_id])
        for budget in selected_budgets:
            selected_moves = []
            for seed in seeds:
                classic = root_classic_summary(state=state, budget=budget, seed=seed)
                puct = evaluate_artifact_position(
                    artifact_path=current_path,
                    evaluator=evaluator,
                    state=state,
                    simulations=budget,
                    seed=seed,
                    c_puct=1.25,
                    search_options=dict(SEARCH_OPTIONS),
                    ablation_mode="full",
                )
                selected_moves.append(classic["selected_move"])
                seed_rows.append(
                    {
                        "row_id": row_id,
                        "budget": budget,
                        "seed": seed,
                        "selected_move": classic["selected_move"],
                        "corrected_reference_move": corrected_reference_move,
                        "selected_matches_corrected_reference": classic["selected_move"]
                        == corrected_reference_move,
                        "old_reference_move": OLD_REFERENCE_MOVES[row_id],
                        "old_vs_new_decision": "same"
                        if OLD_REFERENCE_MOVES[row_id] == corrected_reference_move
                        else "changed",
                        "puct_selected_move": puct.get("selected_move"),
                    }
                )
            majority = majority_summary(selected_moves)
            notes = []
            if (
                row_id == "capture_available-008"
                and budget < 2400
                and not majority["stable"]
            ):
                notes.append("low_budget_unstable")
            summary_rows.append(
                {
                    "row_id": row_id,
                    "budget": budget,
                    "seeds": list(seeds),
                    "corrected_reference_move": corrected_reference_move,
                    "observed_moves": majority["observed_moves"],
                    "majority_move": majority["majority_move"],
                    "majority_fraction": majority["majority_fraction"],
                    "selected_matches_corrected_reference": majority["majority_move"]
                    == corrected_reference_move,
                    "stable": majority["stable"],
                    "notes": ",".join(notes) if notes else "ok",
                }
            )
    return seed_rows, summary_rows


def copy_if_exists(src: Path, dest: Path) -> dict[str, Any] | None:
    if not src.exists():
        return None
    shutil.copy2(src, dest)
    return load_json(dest)


def run_policy_target_audit(
    *, root: Path, out_root: Path, dry_run: bool
) -> dict[str, Any]:
    source_root = out_root / "policy_target_encoding_audit"
    result_path = source_root / "policy_target_encoding_audit_results.json"
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_policy_target_encoding_audit",
        "--out-root",
        str(source_root),
        "--reference-artifact",
        str(out_root / "incumbent_forensic_references_v1_rebased.json"),
    ]
    run_log = run_command(command, cwd=root, dry_run=dry_run)
    payload = load_json(result_path) if result_path.exists() else None
    return {
        "run": run_log,
        "payload": payload or {"known_row_audit": []},
        "report_path": str(result_path),
    }


def run_learned_policy_capture(
    *,
    root: Path,
    out_root: Path,
    current_path: Path,
    reference_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    result_path = out_root / "learned_policy_vs_root_corrected_prior_capture.json"
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.learned_policy_vs_root_corrected_prior_capture",
        "--artifact-path",
        str(current_path),
        "--reference-artifact",
        str(reference_path),
        "--out",
        str(result_path),
    ]
    run_log = run_command(command, cwd=root, dry_run=dry_run)
    payload = load_json(result_path) if result_path.exists() else {}
    return {"run": run_log, "payload": payload, "report_path": str(result_path)}


def run_search_policy_arbitration(
    *,
    root: Path,
    out_root: Path,
    current_path: Path,
    reference_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    suite_reference = row_map_from_reference(load_json(reference_path))
    payload_rows = {}
    if not dry_run:
        from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
            build_payload,
            build_row_views,
            probe_artifact_position,
            validated_diagnostic_state,
        )

        for row_id in ("capture_available-002", "capture_available-003"):
            probe_row = build_probe_row(suite_reference[row_id])
            state = validated_diagnostic_state(row=probe_row)
            probe_views = {}
            for key, ablation_mode in (
                ("policy_only", "policy_only"),
                ("value_only", "value_only"),
                ("full_search", "full"),
            ):
                summary = probe_artifact_position(
                    artifact_path=str(current_path),
                    state=state,
                    simulations=384,
                    seed=17,
                    c_puct=1.25,
                    search_options=dict(SEARCH_OPTIONS),
                    ablation_mode=ablation_mode,
                )
                probe_views[key] = build_row_views(row=probe_row, probe_summary=summary)
            payload_rows[row_id] = {
                **probe_views["full_search"],
                "probe_views": probe_views,
            }
        payload = build_payload(
            selected_artifact={
                "path": str(current_path),
                "selected_target": str(current_path),
                "selected_artifact": None,
                "provenance_source": "forensic_reference_rebaseline",
            },
            source_artifacts={"reference_artifact": str(reference_path)},
            settings={
                "row_ids": ["capture_available-002", "capture_available-003"],
                "search_settings": {"c_puct": 1.25, **dict(SEARCH_OPTIONS)},
                "seeds": [17, 17],
                "simulation_count": 384,
            },
            rows=payload_rows,
        )
    else:
        payload = {"rows": {}, "dry_run": True}
    result_path = out_root / "capture_002_003_search_policy_arbitration.json"
    write_json(result_path, payload)
    return {"payload": payload, "report_path": str(result_path)}


def run_hard_state_validation(
    *,
    root: Path,
    out_root: Path,
    current_path: Path,
    reference_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    forensic_report_path = out_root / "forensic_suite_validation.json"
    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.run_forensic_suite",
        "--suite",
        str(DEFAULT_SUITE_PATH),
        "--current-artifact",
        str(current_path),
        "--challenger-artifact",
        str(current_path),
        "--reference-artifact",
        str(reference_path),
        "--artifact-simulations",
        "384",
        "--out",
        str(forensic_report_path),
    ]
    run_log = run_command(command, cwd=root, dry_run=dry_run)
    payload = load_json(forensic_report_path) if forensic_report_path.exists() else {}
    return {
        "run": run_log,
        "payload": payload,
        "report_path": str(forensic_report_path),
    }


def summarize_diagnostics(
    *,
    policy_audit: dict[str, Any],
    learned_policy: dict[str, Any],
    arbitration: dict[str, Any],
    hard_validation: dict[str, Any],
    integrity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    integrity_index = {row["row_id"]: row for row in integrity_rows}
    rows = []

    def add_row(
        name: str, evaluated: list[str], corrected_failures: list[str], report_path: str
    ) -> None:
        old_failure_map = OLD_FAILURES_BY_DIAGNOSTIC.get(name, {})
        old_failures = sorted(
            row_id for row_id in evaluated if old_failure_map.get(row_id, False)
        )
        unstable_or_excluded = sorted(
            row_id
            for row_id in evaluated
            if integrity_index.get(row_id, {}).get("reference_unstable")
            or integrity_index.get(row_id, {}).get("decision")
            == "reference_integrity_error"
        )
        rows.append(
            {
                "diagnostic": name,
                "rows_evaluated": len(evaluated),
                "old_failures": old_failures,
                "corrected_failures": sorted(corrected_failures),
                "became_pass_after_reference_fix": sorted(
                    row_id
                    for row_id in old_failures
                    if row_id not in corrected_failures
                    and row_id not in unstable_or_excluded
                ),
                "still_fails_corrected_reference": sorted(corrected_failures),
                "unstable_or_excluded": unstable_or_excluded,
                "report_path": report_path,
            }
        )

    policy_rows = list((policy_audit["payload"] or {}).get("known_row_audit") or [])
    add_row(
        "policy_target_encoding_audit",
        [str(row["row_id"]) for row in policy_rows],
        [
            str(row["row_id"])
            for row in policy_rows
            if row.get("searched_selected_move") != row.get("reference_move")
        ],
        policy_audit["report_path"],
    )

    learned_payload = learned_policy["payload"] or {}
    learned_rows = []
    focus_row = learned_payload.get("focus_row") or {}
    preservation_row = learned_payload.get("preservation_row") or {}
    if focus_row:
        learned_rows.append((focus_row.get("row_id"), focus_row))
    if preservation_row:
        learned_rows.append((preservation_row.get("row_id"), preservation_row))
    add_row(
        "learned_policy_vs_root_corrected_prior_capture",
        [row_id for row_id, _row in learned_rows if isinstance(row_id, str)],
        [
            row_id
            for row_id, row in learned_rows
            if isinstance(row_id, str)
            and ((row.get("interventions") or {}).get("original_prior") or {}).get(
                "searched_selected_move"
            )
            != ((row.get("interventions") or {}).get("original_prior") or {}).get(
                "reference_move"
            )
        ],
        learned_policy["report_path"],
    )

    arbitration_rows = (arbitration["payload"] or {}).get("rows") or {}
    add_row(
        "search_policy_arbitration",
        sorted(arbitration_rows),
        [
            row_id
            for row_id, row in arbitration_rows.items()
            if ((row.get("search_view") or {}).get("searched_selected_move"))
            != row.get("reference_move")
        ],
        arbitration["report_path"],
    )

    hard_rows = (
        ((hard_validation["payload"] or {}).get("systems") or {}).get("current") or {}
    ).get("rows") or []
    add_row(
        "hard_state_validation",
        [str(row["id"]) for row in hard_rows],
        [
            str(row["id"])
            for row in hard_rows
            if row.get("reference_move") is not None
            and row.get("selected_move") != row.get("reference_move")
        ],
        hard_validation["report_path"],
    )

    return rows


def build_inventory(
    *,
    suite: list[Any],
    effective_reference_by_id: dict[str, dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
    capture_summary_rows: list[dict[str, Any]],
    hard_validation: dict[str, Any],
    current_path: Path,
) -> list[dict[str, Any]]:
    integrity_index = {row["row_id"]: row for row in integrity_rows}
    capture_majority = {}
    for row in capture_summary_rows:
        current = capture_majority.get(row["row_id"])
        if current is None or int(row["budget"]) >= int(current["budget"]):
            capture_majority[row["row_id"]] = row
    hard_rows = (
        ((hard_validation["payload"] or {}).get("systems") or {}).get("current") or {}
    ).get("rows") or []
    hard_rows_by_id = {str(row["id"]): row for row in hard_rows}
    evaluator = ArtifactEvaluator(current_path)
    inventory = []
    for position in suite:
        reference_row = effective_reference_by_id.get(position.id)
        integrity_row = integrity_index[position.id]
        probe = evaluate_artifact_position(
            artifact_path=current_path,
            evaluator=evaluator,
            state=dict(position.state),
            simulations=384,
            seed=17,
            c_puct=1.25,
            search_options=dict(SEARCH_OPTIONS),
            ablation_mode="full",
        )
        selected_move = probe.get("selected_move")
        child_stats = list(probe.get("child_stats") or [])
        total_visits = max(1, sum(int(child.get("visits", 0)) for child in child_stats))
        reference_visit_share = None
        if reference_row and reference_row.get("reference_move") is not None:
            for child in child_stats:
                if int(child["move"]) == int(reference_row["reference_move"]):
                    reference_visit_share = round(
                        int(child.get("visits", 0)) / total_visits, 4
                    )
                    break
        hard_row = hard_rows_by_id.get(position.id)
        corrected_failure = bool(
            hard_row
            and hard_row.get("reference_move") is not None
            and hard_row.get("selected_move") != hard_row.get("reference_move")
        )
        if integrity_row["decision"] == "reference_integrity_error":
            failure_status = "reference_integrity_error"
            severity = "high"
            recommended_use = "exclude_until_rebuilt"
        elif position.id == "capture_available-008" and any(
            row["row_id"] == position.id and row["budget"] < 2400 and not row["stable"]
            for row in capture_summary_rows
        ):
            failure_status = "unstable_reference"
            severity = "low"
            recommended_use = "exclude_until_rebuilt"
        elif corrected_failure:
            failure_status = "fail_corrected_reference"
            severity = "high" if position.bucket == "capture_available" else "medium"
            recommended_use = "candidate_failure_family"
        else:
            failure_status = "pass_corrected_reference"
            severity = "none"
            recommended_use = "keep_validation_gate"
        inventory.append(
            {
                "row_id": position.id,
                "family": position.bucket,
                "corrected_reference_move": None
                if reference_row is None
                else reference_row.get("reference_move"),
                "current_artifact_selected_move": selected_move,
                "current_artifact_reference_visit_share": reference_visit_share,
                "ClassicMCTS_majority_move": capture_majority.get(position.id, {}).get(
                    "majority_move"
                ),
                "PUCT_selected_move": selected_move,
                "failure_status": failure_status,
                "severity": severity,
                "recommended_use": recommended_use,
                "notes": integrity_row["notes"],
            }
        )
    return inventory


def classify_run(
    *,
    integrity_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> tuple[str, str]:
    integrity_errors = [
        row for row in integrity_rows if row["decision"] == "reference_integrity_error"
    ]
    if len(integrity_errors) >= max(5, len(integrity_rows) // 4):
        return (
            "reference_integrity_broken_broadly",
            "rebuild the entire forensic reference artifact from the suite and freeze state-hash validation in CI.",
        )
    true_failures = [
        row for row in inventory if row["failure_status"] == "fail_corrected_reference"
    ]
    unstable = [
        row for row in inventory if row["failure_status"] == "unstable_reference"
    ]
    became_pass = sum(
        len(row["became_pass_after_reference_fix"]) for row in diagnostic_rows
    )
    corrected_failures = sum(len(row["corrected_failures"]) for row in diagnostic_rows)
    if unstable and len(unstable) >= max(2, len(inventory) // 8):
        return (
            "reference_suite_too_noisy",
            "exclude unstable rows from hard pass/fail gates and replace them with stable alternatives.",
        )
    if became_pass > corrected_failures and not true_failures:
        return (
            "reference_artifact_poisoning_resolved",
            "rebuild failure-family datasets from corrected failure inventory before any training.",
        )
    if true_failures and len(true_failures) <= 5:
        return (
            "corrected_failure_family_identified",
            "design a focused experiment only around those remaining rows.",
        )
    if true_failures:
        return (
            "genuine_model_search_gap",
            "run a new targeted hard-state mining/replay experiment, but only from corrected references.",
        )
    return (
        "reference_artifact_poisoning_resolved",
        "rebuild failure-family datasets from corrected failure inventory before any training.",
    )


def render_markdown(
    *,
    summary: dict[str, Any],
    integrity_rows: list[dict[str, Any]],
    capture_summary_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> str:
    lines = [
        "# AlphaZero-lite Forensic Reference Rebaseline Results",
        "",
        "## Context",
        "",
        "- PR #31 corrected the tracked capture reference labels for 002, 003, and 007.",
        "- This sweep rebuilt the effective tracked reference artifact to current-suite coverage before rerunning local diagnostics.",
        f"- Effective tracked reference artifact: `{summary['effective_reference_artifact']}`.",
        "",
        "## Why PR #31 Invalidated Earlier Local-Failure Conclusions",
        "",
        "- Earlier local-failure conclusions for `capture_available-002` were anchored to the wrong reference move.",
        "- Several local diagnostics also still defaulted to the stale train-only reference artifact instead of the tracked forensic artifact.",
        "- This run rebaselined those diagnostics against the corrected tracked default and separated current-suite integrity issues from true model/search failures.",
        "",
        "## Reference Artifact Integrity Check",
        "",
        "| row_id | suite_state_hash | reference_state_hash | state_match | corrected_reference_move | legal | reference_unstable | decision | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in integrity_rows:
        lines.append(
            "| {row_id} | `{suite_state_hash}` | `{reference_state_hash}` | {state_match} | {corrected_reference_move} | {legal} | {reference_unstable} | {decision} | {notes} |".format(
                row_id=row["row_id"],
                suite_state_hash=row["suite_state_hash"],
                reference_state_hash=row["reference_state_hash"],
                state_match=str(bool(row["state_match"])).lower(),
                corrected_reference_move=row["corrected_reference_move"],
                legal=str(bool(row["legal"])).lower(),
                reference_unstable=str(bool(row["reference_unstable"])).lower()
                if row["reference_unstable"] is not None
                else "-",
                decision=row["decision"],
                notes=row["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Corrected Capture-Row Teacher Confirmation",
            "",
            "| row_id | budget | seeds | corrected_reference_move | observed_moves | majority_move | majority_fraction | selected_matches_corrected_reference | stable | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in capture_summary_rows:
        lines.append(
            "| {row_id} | {budget} | `{seeds}` | {corrected_reference_move} | `{observed_moves}` | {majority_move} | {majority_fraction} | {selected_matches_corrected_reference} | {stable} | {notes} |".format(
                row_id=row["row_id"],
                budget=row["budget"],
                seeds=json.dumps(row["seeds"]),
                corrected_reference_move=row["corrected_reference_move"],
                observed_moves=json.dumps(row["observed_moves"]),
                majority_move=row["majority_move"],
                majority_fraction=row["majority_fraction"],
                selected_matches_corrected_reference=str(
                    bool(row["selected_matches_corrected_reference"])
                ).lower(),
                stable=str(bool(row["stable"])).lower(),
                notes=row["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Re-run Local Diagnostics Under Corrected References",
            "",
            "| diagnostic | rows_evaluated | old_failures | corrected_failures | became_pass_after_reference_fix | still_fails_corrected_reference | unstable_or_excluded | report_path |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            "| {diagnostic} | {rows_evaluated} | `{old_failures}` | `{corrected_failures}` | `{became_pass_after_reference_fix}` | `{still_fails_corrected_reference}` | `{unstable_or_excluded}` | `{report_path}` |".format(
                diagnostic=row["diagnostic"],
                rows_evaluated=row["rows_evaluated"],
                old_failures=json.dumps(row["old_failures"]),
                corrected_failures=json.dumps(row["corrected_failures"]),
                became_pass_after_reference_fix=json.dumps(
                    row["became_pass_after_reference_fix"]
                ),
                still_fails_corrected_reference=json.dumps(
                    row["still_fails_corrected_reference"]
                ),
                unstable_or_excluded=json.dumps(row["unstable_or_excluded"]),
                report_path=row["report_path"],
            )
        )
    lines.extend(
        [
            "",
            "## Corrected Failure Inventory",
            "",
            "| row_id | family | corrected_reference_move | selected_move_current | failure_status | severity | recommended_use | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in inventory:
        lines.append(
            "| {row_id} | {family} | {corrected_reference_move} | {current_artifact_selected_move} | {failure_status} | {severity} | {recommended_use} | {notes} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Classification: `{summary['classification']}`.",
            f"- Corrected failures remaining: `{summary['corrected_failure_count']}`.",
            f"- Reference integrity errors remaining: `{summary['reference_integrity_error_count']}`.",
            f"- False failures cleared by the reference fix: `{summary['became_pass_count']}`.",
            "",
            "## Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    budgets = parse_csv_ints(args.budgets)
    seeds = parse_csv_ints(args.seeds)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    suite = load_suite(args.suite_path)
    suite_positions_by_id = {position.id: position for position in suite}
    legacy_reference_path = (
        root
        / "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
    )
    effective_reference_path = (
        out_root / "incumbent_forensic_references_v1_rebased.json"
    )
    corrected_failure_inventory_path = out_root / "corrected_failure_inventory.json"
    summary_path = out_root / "forensic_reference_rebaseline_summary.json"
    doc_path = root / "docs/alphazero-lite-forensic-reference-rebaseline-results.md"

    effective_reference_payload = rebuild_effective_reference_artifact(
        suite_path=args.suite_path,
        tracked_reference_path=args.reference_artifact,
        legacy_reference_path=legacy_reference_path,
        out_path=effective_reference_path,
    )
    effective_reference_by_id = row_map_from_reference(effective_reference_payload)
    legacy_reference_by_id = row_map_from_reference(load_json(legacy_reference_path))
    integrity_rows = reference_integrity_rows(
        suite=suite,
        reference_payload=effective_reference_payload,
    )

    capture_seed_rows, capture_summary_rows = capture_confirmation_rows(
        suite_positions_by_id=suite_positions_by_id,
        legacy_reference_by_id=legacy_reference_by_id,
        effective_reference_by_id=effective_reference_by_id,
        current_path=args.current_path,
        budgets=budgets,
        seeds=seeds,
    )
    write_json(
        out_root / "capture_teacher_confirmation_seed_rows.json", capture_seed_rows
    )
    write_json(
        out_root / "capture_teacher_confirmation_summary.json", capture_summary_rows
    )

    policy_audit = run_policy_target_audit(
        root=root, out_root=out_root, dry_run=args.dry_run
    )
    learned_policy = run_learned_policy_capture(
        root=root,
        out_root=out_root,
        current_path=args.current_path,
        reference_path=effective_reference_path,
        dry_run=args.dry_run,
    )
    arbitration = run_search_policy_arbitration(
        root=root,
        out_root=out_root,
        current_path=args.current_path,
        reference_path=effective_reference_path,
        dry_run=args.dry_run,
    )
    hard_validation = run_hard_state_validation(
        root=root,
        out_root=out_root,
        current_path=args.current_path,
        reference_path=effective_reference_path,
        dry_run=args.dry_run,
    )

    diagnostic_rows = summarize_diagnostics(
        policy_audit=policy_audit,
        learned_policy=learned_policy,
        arbitration=arbitration,
        hard_validation=hard_validation,
        integrity_rows=integrity_rows,
    )
    inventory = build_inventory(
        suite=suite,
        effective_reference_by_id=effective_reference_by_id,
        integrity_rows=integrity_rows,
        capture_summary_rows=capture_summary_rows,
        hard_validation=hard_validation,
        current_path=args.current_path,
    )
    write_json(corrected_failure_inventory_path, inventory)

    classification, recommended_next_action = classify_run(
        integrity_rows=integrity_rows,
        diagnostic_rows=diagnostic_rows,
        inventory=inventory,
    )
    summary = {
        "schema": "azlite_forensic_reference_rebaseline_v1",
        "summary_path": str(summary_path),
        "doc_path": str(doc_path),
        "effective_reference_artifact": str(effective_reference_path),
        "corrected_failure_inventory_path": str(corrected_failure_inventory_path),
        "classification": classification,
        "recommended_next_action": recommended_next_action,
        "corrected_failure_count": sum(
            1
            for row in inventory
            if row["failure_status"] == "fail_corrected_reference"
        ),
        "reference_integrity_error_count": sum(
            1
            for row in inventory
            if row["failure_status"] == "reference_integrity_error"
        ),
        "became_pass_count": sum(
            len(row["became_pass_after_reference_fix"]) for row in diagnostic_rows
        ),
        "dry_run": bool(args.dry_run),
    }
    write_json(summary_path, summary)
    doc_path.write_text(
        render_markdown(
            summary=summary,
            integrity_rows=integrity_rows,
            capture_summary_rows=capture_summary_rows,
            diagnostic_rows=diagnostic_rows,
            inventory=inventory,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "doc_path": str(doc_path),
                "effective_reference_artifact": str(effective_reference_path),
                "classification": classification,
                "recommended_next_action": recommended_next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
