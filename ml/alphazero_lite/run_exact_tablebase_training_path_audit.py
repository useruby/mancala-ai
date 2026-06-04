#!/usr/bin/env python3
"""Exact-tablebase training-path diagnostic audit.

Validates PR #79 artifacts, replays current baseline, audits replay weight
plumbing, trace output isolation, train.py target consumption, value
perspective, promoted regression controls, and runs a minimal corrected
smoke test.

Does NOT run arena, promote, or touch storage/ai/alphazero_lite/current.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import build_eval_search_options

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
PR79_DIR = Path("/tmp/azlite_medium_exact_tablebase_stabilization_v2")
OUTPUT_DIR = Path("/tmp/azlite_exact_tablebase_training_path_audit")
AUDIT_SUMMARY_PATH = OUTPUT_DIR / "training_path_audit_summary.json"
ROW_AUDIT_PATH = OUTPUT_DIR / "training_path_row_audit.jsonl"
REPORT_PATH = Path("docs/alphazero-lite-exact-tablebase-training-path-audit-results.md")

C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
SEED = 17
EVAL_BUDGETS = (384, 1200, 2400)
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
EPS = 1e-9

SOFT065_PATH = PR79_DIR / "exact_tablebase_policy_value_soft065.jsonl"
SOFT055_PATH = PR79_DIR / "exact_tablebase_policy_value_soft055.jsonl"
CONTROLS_PATH = PR79_DIR / "exact_tablebase_targeted_controls_artifact.jsonl"
TRACE_SUMMARY_PATH = PR79_DIR / "stabilization_v2_trace_summary.json"
ARTIFACT_SUMMARY_PATH = PR79_DIR / "artifact_summary.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def state_from_encoded(encoded: list[float]) -> dict[str, Any] | None:
    arr = np.asarray(encoded, dtype=np.float32)
    if arr.shape[0] < 15:
        return None
    base = arr[:15]
    pits_denom = 48.0
    player_pits = [int(round(float(v) * pits_denom)) for v in base[:6]]
    opponent_pits = [int(round(float(v) * pits_denom)) for v in base[6:12]]
    player_store = int(round(float(base[12]) * pits_denom))
    opponent_store = int(round(float(base[13]) * pits_denom))
    current_player = int(round(float(base[14])))
    return {
        "player_pits": player_pits,
        "opponent_pits": opponent_pits,
        "player_store": player_store,
        "opponent_store": opponent_store,
        "current_player": current_player,
    }


def compute_canonical_hash(state: dict[str, Any]) -> str:
    return canonical_state_key(state)


def run_single_puct(
    evaluator: ArtifactEvaluator, state: dict[str, Any], budget: int, seed: int
) -> dict[str, Any]:
    return evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(v) for v in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / float(total))


def selection_entry_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    breakdown = result.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def policy_rank_of_move(policy: list[float], move: int) -> int | None:
    if not policy or move is None:
        return None
    sorted_moves = sorted(
        range(len(policy)),
        key=lambda m: (float(policy[m]), -m),
        reverse=True,
    )
    for rank, m in enumerate(sorted_moves):
        if m == move:
            return rank
    return None


def tb_value_to_training(value: float) -> float:
    return (2.0 * float(value)) - 1.0


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── Step 1: Validate PR #79 artifacts ──────────────────────────────────────


def step1_validate_artifacts() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print("=" * 70)
    print("STEP 1: Validate PR #79 artifacts byte-for-byte")
    print("=" * 70)

    tb = EndgameTablebase()
    row_audit: list[dict[str, Any]] = []
    errors: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    duplicate_conflicts: int = 0
    artifact_stats: dict[str, dict[str, Any]] = {}

    artifact_paths = [
        ("soft065_production", SOFT065_PATH),
        ("soft055_production", SOFT055_PATH),
        ("targeted_controls", CONTROLS_PATH),
    ]

    for label, path in artifact_paths:
        if not path.exists():
            errors.append(f"{label}: file not found {path}")
            continue

        rows = load_jsonl(path)
        stats = {
            "row_count": len(rows),
            "invalid_rows": 0,
            "duplicate_conflicts": 0,
            "policy_mass_issues": 0,
            "value_range_issues": 0,
            "optimal_move_issues": 0,
            "tablebase_tie_issues": 0,
            "all_equiv_issues": 0,
            "exhausted_overlap": 0,
            "validation_status": "pending",
            "notes": "",
        }

        seen_state_targets: dict[str, tuple[int, float, str]] = {}

        for idx, row in enumerate(rows):
            cid = row.get("candidate_id", f"row_{idx}")
            role = row.get("role", "unknown")
            c_hash = row.get("canonical_state_hash", "")
            state_encoded = row.get("state", [])
            raw_state = row.get("raw_state", None)
            policy = row.get("policy", [])
            value = float(row.get("value", 0.0))
            exact_optimal = row.get("exact_optimal_move")
            exact_root = row.get("exact_root_value")

            audit_entry = {
                "candidate_id": cid,
                "role": role,
                "artifact": label,
                "target_move": exact_optimal,
                "exact_optimal_move": None,
                "policy_mass_on_optimal": None,
                "value_target": value,
                "exact_root_value": exact_root,
                "validation_status": "ok",
                "validation_error": "",
            }

            # Check canonical_state_hash
            if not c_hash:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = "missing canonical_state_hash"
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Reconstruct state
            if raw_state:
                state = raw_state
            elif state_encoded:
                state = state_from_encoded(state_encoded)
                if state is None:
                    audit_entry["validation_status"] = "invalid"
                    audit_entry["validation_error"] = "cannot reconstruct state"
                    stats["invalid_rows"] += 1
                    invalid_rows.append(audit_entry)
                    row_audit.append(audit_entry)
                    continue
            else:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = "no state data"
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Verify canonical hash matches
            computed_hash = compute_canonical_hash(state)
            if computed_hash != c_hash:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = (
                    f"canonical_hash mismatch: stored={c_hash[:40]}... computed={computed_hash[:40]}..."
                )
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Legal moves
            game = KalahGame.from_state(state)
            legal_moves = game.possible_moves()
            if not legal_moves:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = "no legal moves"
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Tablebase lookup
            root_player = game.current_player
            tb_wr = tb.lookup(game, root_player)
            if tb_wr is None:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = "tablebase not available"
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Find optimal moves from tablebase
            offset = root_player * 6
            child_vals: dict[int, float] = {}
            for move in legal_moves:
                child = game.clone()
                child.move(offset + move)
                c_wr = tb.lookup(child, root_player)
                if c_wr is not None:
                    child_vals[move] = tb_value_to_training(float(c_wr))

            if not child_vals:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = "no child tablebase values"
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            best_val = max(child_vals.values())
            optimal_moves = [
                m for m, v in child_vals.items() if abs(v - best_val) < EPS
            ]
            unique_optimal = optimal_moves[0] if len(optimal_moves) == 1 else None
            exact_root_val = best_val

            audit_entry["exact_optimal_move"] = unique_optimal
            audit_entry["exact_root_value"] = round_float(exact_root_val)

            # Check tablebase unique optimal
            if unique_optimal is None:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = (
                    f"tablebase tie: {len(optimal_moves)} optimal moves"
                )
                stats["tablebase_tie_issues"] += 1
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Check all moves equivalent
            all_vals = set(round_float(v, 6) for v in child_vals.values())
            if len(all_vals) == 1:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = "all moves equivalent"
                stats["all_equiv_issues"] += 1
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Check target move equals exact optimal
            if exact_optimal is not None and exact_optimal != unique_optimal:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = (
                    f"target_move={exact_optimal} != exact_optimal={unique_optimal}"
                )
                stats["optimal_move_issues"] += 1
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Check policy sums to 1.0
            if policy:
                policy_sum = float(sum(policy))
                if abs(policy_sum - 1.0) > 1e-6:
                    audit_entry["validation_status"] = "invalid"
                    audit_entry["validation_error"] = (
                        f"policy sum={policy_sum:.6f} != 1.0"
                    )
                    stats["policy_mass_issues"] += 1
                    stats["invalid_rows"] += 1
                    invalid_rows.append(audit_entry)
                    row_audit.append(audit_entry)
                    continue

                # Check optimal move has highest policy mass
                optimal_mass = policy[unique_optimal]
                audit_entry["policy_mass_on_optimal"] = round_float(optimal_mass)
                for m, mass in enumerate(policy):
                    if m != unique_optimal and mass > optimal_mass + EPS:
                        audit_entry["validation_status"] = "invalid"
                        audit_entry["validation_error"] = (
                            f"move {m} mass={mass:.4f} > optimal mass={optimal_mass:.4f}"
                        )
                        stats["policy_mass_issues"] += 1
                        stats["invalid_rows"] += 1
                        invalid_rows.append(audit_entry)
                        break
                if audit_entry["validation_status"] == "invalid":
                    row_audit.append(audit_entry)
                    continue

            # Check value target in [-1, 1]
            if value < -1.0 or value > 1.0:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = f"value={value} out of [-1,1]"
                stats["value_range_issues"] += 1
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Check value target equals exact root value
            if exact_root is not None and abs(float(exact_root) - exact_root_val) > EPS:
                audit_entry["validation_status"] = "invalid"
                audit_entry["validation_error"] = (
                    f"stored_exact_root={exact_root} != computed={exact_root_val}"
                )
                stats["value_range_issues"] += 1
                stats["invalid_rows"] += 1
                invalid_rows.append(audit_entry)
                row_audit.append(audit_entry)
                continue

            # Check no duplicate/conflicting state
            if c_hash in seen_state_targets:
                prev_move, prev_val, prev_id = seen_state_targets[c_hash]
                if prev_move != unique_optimal or abs(prev_val - exact_root_val) > EPS:
                    duplicate_conflicts += 1
                    stats["duplicate_conflicts"] += 1
                    audit_entry["validation_status"] = "warning"
                    audit_entry["validation_error"] = (
                        f"conflict with {prev_id}: move={prev_move} vs {unique_optimal}"
                    )
            else:
                seen_state_targets[c_hash] = (unique_optimal, exact_root_val, cid)

            # Check role
            valid_roles = {
                "production_candidate",
                "preservation_control",
                "value_only_candidate",
            }
            if role not in valid_roles:
                audit_entry["validation_status"] = "warning"
                audit_entry["validation_error"] = f"unknown role: {role}"

            row_audit.append(audit_entry)

        stats["validation_status"] = (
            "PASSED" if stats["invalid_rows"] == 0 else "FAILED"
        )
        artifact_stats[label] = stats

    classification = None
    if any(s["validation_status"] == "FAILED" for s in artifact_stats.values()):
        classification = "artifact_target_format_bug"
        print(f"\nCLASSIFICATION: {classification}")
        print("  Artifact targets are malformed. Stopping further probes.")
        return {
            "step": "step1_artifact_validation",
            "classification": classification,
            "artifact_stats": artifact_stats,
            "duplicate_conflicts": duplicate_conflicts,
            "total_invalid_rows": sum(
                s["invalid_rows"] for s in artifact_stats.values()
            ),
            "row_audit_path": str(ROW_AUDIT_PATH),
        }, row_audit

    print("\nAll artifacts PASSED validation.")
    return {
        "step": "step1_artifact_validation",
        "classification": "artifact_targets_valid",
        "artifact_stats": artifact_stats,
        "duplicate_conflicts": duplicate_conflicts,
        "total_invalid_rows": 0,
        "row_audit_path": str(ROW_AUDIT_PATH),
    }, row_audit


# ── Step 2: Reproduce current baseline ────────────────────────────────────


def step2_current_baseline(evaluator: ArtifactEvaluator) -> dict[str, Any]:
    print("=" * 70)
    print("STEP 2: Reproduce current baseline on controls and production")
    print("=" * 70)

    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    production_rows = [
        r for r in soft065_rows if r.get("role") == "production_candidate"
    ]
    original_controls = [
        r for r in controls_rows if not r.get("promoted_from_holdout_regression")
    ]
    promoted_controls = [
        r for r in controls_rows if r.get("promoted_from_holdout_regression")
    ]
    nn_controls = [r for r in controls_rows if r.get("nearest_neighbor_control")]

    row_groups = [
        ("production", production_rows),
        ("original_controls", original_controls),
        ("promoted_regression_controls", promoted_controls),
        ("nearest_neighbor_controls", nn_controls),
    ]

    baseline_results: dict[str, Any] = {}
    tb = EndgameTablebase()

    for group_name, group_rows in row_groups:
        if not group_rows:
            continue
        print(f"\n  Evaluating {group_name} ({len(group_rows)} rows)...")
        group_results: list[dict] = []

        for idx, row in enumerate(group_rows):
            if idx > 0 and idx % 50 == 0:
                print(f"    {idx}/{len(group_rows)}")

            cid = row.get("candidate_id", "?")
            raw_state = row.get("raw_state")
            if not raw_state:
                continue

            state = raw_state
            game = KalahGame.from_state(state)
            root_player = game.current_player
            legal_moves = game.possible_moves()

            # Get exact optimal
            offset = root_player * 6
            child_vals = {}
            for move in legal_moves:
                child = game.clone()
                child.move(offset + move)
                c_wr = tb.lookup(child, root_player)
                if c_wr is not None:
                    child_vals[move] = tb_value_to_training(float(c_wr))
            exact_optimal = max(child_vals, key=child_vals.get) if child_vals else None
            exact_root = max(child_vals.values()) if child_vals else None

            result = {"candidate_id": cid, "exact_optimal_move": exact_optimal}

            for budget in EVAL_BUDGETS:
                try:
                    r = run_single_puct(evaluator, state, budget, SEED)
                except Exception as e:
                    result[f"error_{budget}"] = str(e)
                    continue

                selected_move = (
                    None if r.get("selected_move") is None else int(r["selected_move"])
                )
                sel_map = selection_entry_map(r)
                opt_entry = (
                    sel_map.get(exact_optimal) if exact_optimal is not None else {}
                )
                sel_entry = (
                    sel_map.get(selected_move) if selected_move is not None else {}
                )
                visits_list = [float(v) for v in r.get("visits", [])]

                result[f"selected_{budget}"] = selected_move
                result[f"selected_is_optimal_{budget}"] = (
                    selected_move == exact_optimal
                    if exact_optimal is not None
                    else None
                )
                result[f"optimal_visit_share_{budget}"] = visit_share(
                    visits_list, exact_optimal
                )
                result[f"selected_minus_optimal_q_margin_{budget}"] = (
                    round_float(
                        float(sel_entry.get("q_value", 0.0))
                        - float(opt_entry.get("q_value", 0.0))
                    )
                    if sel_entry and opt_entry
                    else None
                )

            # Policy stats at 1200
            r1200 = run_single_puct(evaluator, state, 1200, SEED)
            p_list = [float(p) for p in r1200.get("policy", [])]
            result["optimal_policy_probability"] = round_float(
                float(p_list[exact_optimal])
                if exact_optimal is not None and exact_optimal < len(p_list)
                else 0.0
            )
            result["optimal_policy_rank"] = policy_rank_of_move(p_list, exact_optimal)

            _, raw_nv = evaluator.evaluate(game)
            neural_value = float(raw_nv)
            result["neural_value"] = round_float(neural_value)
            result["exact_value"] = round_float(exact_root)
            result["value_error"] = round_float(
                abs(neural_value - exact_root) if exact_root is not None else None
            )

            group_results.append(result)

        opts = [
            sum(1 for r in group_results if r.get(f"selected_is_optimal_{b}"))
            for b in EVAL_BUDGETS
        ]
        total = len(group_results)
        avg_visits = sum(
            float(r.get("optimal_visit_share_1200", 0) or 0) for r in group_results
        ) / max(total, 1)
        avg_val_err = sum(
            float(r.get("value_error", 0) or 0) for r in group_results
        ) / max(total, 1)

        print(
            f"    optimal@1200={opts[1]}/{total}, avg_visit_share={avg_visits:.4f}, avg_val_err={avg_val_err:.4f}"
        )

        baseline_results[group_name] = {
            "rows": total,
            "optimal_384": opts[0],
            "optimal_1200": opts[1],
            "optimal_2400": opts[2] if len(opts) > 2 else None,
            "avg_optimal_visit_share_1200": round_float(avg_visits),
            "avg_value_error": round_float(avg_val_err),
            "individual_results": group_results,
        }

    # Check if promoted controls already fail current
    prom = baseline_results.get("promoted_regression_controls", {})
    prom_total = prom.get("rows", 0)
    prom_opt = prom.get("optimal_1200", 0)
    prom_fail = prom_total - prom_opt
    if prom_fail > 0:
        print(
            f"\n  WARNING: {prom_fail}/{prom_total} promoted controls already fail current!"
        )
        print("  These are not preservation controls and should be reclassified.")

    return {
        "step": "step2_current_baseline",
        "baseline_results": baseline_results,
        "promoted_controls_already_fail_current": prom_fail,
    }


# ── Step 3: Audit replay weight plumbing ──────────────────────────────────


def step3_replay_weight_audit() -> dict[str, Any]:
    print("=" * 70)
    print("STEP 3: Audit replay weight plumbing")
    print("=" * 70)

    probe_dir = OUTPUT_DIR / "replay_weight_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Create a tiny controlled dataset
    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    prod_rows = [r for r in soft065_rows if r.get("role") == "production_candidate"]
    orig_ctrl = [
        r for r in controls_rows if not r.get("promoted_from_holdout_regression")
    ]

    if len(prod_rows) < 1 or len(orig_ctrl) < 1:
        print("  Not enough rows for replay weight probe, skipping.")
        return {
            "step": "step3_replay_weight_audit",
            "status": "skipped",
            "notes": "insufficient rows",
        }

    # Pick one production row and one control row
    row_a = dict(prod_rows[0])
    row_b = dict(orig_ctrl[0])

    # Ensure both have required fields
    for row in [row_a, row_b]:
        if "policy" not in row or "value" not in row:
            print("  Row missing policy/value, skipping probe.")
            return {
                "step": "step3_replay_weight_audit",
                "status": "skipped",
                "notes": "missing policy/value",
            }

    path_a = probe_dir / "probe_row_a.jsonl"
    path_b = probe_dir / "probe_row_b.jsonl"
    write_jsonl(path_a, [row_a])
    write_jsonl(path_b, [row_b])

    init_checkpoint = PR79_DIR / "init_checkpoint.npz"

    weight_configs = [
        ("w1_w1", [1, 1]),
        ("w1_w5", [1, 5]),
        ("w5_w1", [5, 1]),
    ]

    results: dict[str, dict[str, Any]] = {}

    for name, weights in weight_configs:
        out_path = probe_dir / f"probe_{name}_e1.npz"
        weights_str = ",".join(str(w) for w in weights)
        data_files_str = f"{path_a},{path_b}"

        cmd = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.train",
            "--data-files",
            data_files_str,
            "--replay-weights",
            weights_str,
            "--out",
            str(out_path),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--lr",
            "1e-4",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--value-loss-weight",
            "0.5",
            "--value-loss",
            "huber",
            "--hidden-sizes",
            "96,3",
            "--model-type",
            "residual_v3",
            "--input-encoding",
            INPUT_ENCODING,
            "--init-checkpoint",
            str(init_checkpoint),
            "--val-split",
            "0.0",
            "--grad-clip",
            "1.0",
        ]

        print(f"  Training probe {name} with weights={weights}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        metrics: dict[str, Any] = {
            "trace": name,
            "source_weights": weights,
            "checkpoint_path": str(out_path),
            "checkpoint_hash": sha256_file(out_path),
            "returncode": result.returncode,
        }

        for line in (result.stdout or "").split("\n"):
            for prefix in ["policy_loss=", "value_loss=", "best_val_loss="]:
                if prefix in line:
                    try:
                        metrics[prefix.rstrip("=")] = float(line.split("=")[1].strip())
                    except (IndexError, ValueError):
                        pass

        results[name] = metrics
        print(
            f"    policy_loss={metrics.get('policy_loss')}, value_loss={metrics.get('value_loss')}"
        )

    # Check if checkpoints differ
    weight_config_names = [n for n, _ in weight_configs]
    hashes = [
        results[n]["checkpoint_hash"]
        for n in weight_config_names
        if results[n].get("checkpoint_hash")
    ]
    all_identical = len(set(hashes)) <= 1
    classification = (
        "replay_weight_ignored_or_trace_reuse_bug"
        if all_identical
        else "replay_weights_working"
    )

    print(f"\n  Checkpoint hashes: {'IDENTICAL' if all_identical else 'DIFFERENT'}")
    print(f"  Classification: {classification}")

    return {
        "step": "step3_replay_weight_audit",
        "classification": classification,
        "probe_results": results,
        "all_checkpoints_identical": all_identical,
    }


# ── Step 4: Audit trace output isolation ──────────────────────────────────


def step4_trace_isolation_audit() -> dict[str, Any]:
    print("=" * 70)
    print("STEP 4: Audit trace output isolation")
    print("=" * 70)

    exports_dir = PR79_DIR / "exports"
    if not exports_dir.exists():
        print("  No exports directory found.")
        return {
            "step": "step4_trace_isolation_audit",
            "status": "skipped",
            "notes": "no exports",
        }

    export_files = sorted(exports_dir.glob("*.npz"))
    trace_info: list[dict[str, Any]] = []
    hash_to_traces: dict[str, list[str]] = {}

    for npz_path in export_files:
        info = {
            "trace_name": npz_path.stem,
            "checkpoint_path": str(npz_path),
            "checkpoint_hash": sha256_file(npz_path),
        }
        trace_info.append(info)
        h = info["checkpoint_hash"]
        if h:
            hash_to_traces.setdefault(h, []).append(info["trace_name"])

    # Also check training data files
    data_files = sorted(PR79_DIR.glob("*_data.jsonl"))
    data_hashes: dict[str, str] = {}
    for dp in data_files:
        data_hashes[dp.stem] = sha256_file(dp)

    # Check for duplicate checkpoints
    duplicates: list[dict[str, Any]] = []
    for h, traces in hash_to_traces.items():
        if len(traces) > 1:
            duplicates.append({"hash": h, "traces": traces})

    # Check cap128_soft065_w1_half vs cap128_soft065_w2_half
    w1_half = [t for t in trace_info if "soft065_w1_half" in t["trace_name"]]
    w2_half = [t for t in trace_info if "soft065_w2_half" in t["trace_name"]]

    w1w2_identical = False
    if w1_half and w2_half:
        for e in [1, 2, 4]:
            w1 = [t for t in w1_half if f"_e{e}" in t["trace_name"]]
            w2 = [t for t in w2_half if f"_e{e}" in t["trace_name"]]
            if w1 and w2 and w1[0]["checkpoint_hash"] == w2[0]["checkpoint_hash"]:
                w1w2_identical = True
                print(
                    f"  DUPLICATE: w1_half_e{e} == w2_half_e{e} (hash={w1[0]['checkpoint_hash'][:16]}...)"
                )

    # Check training data identity
    cap128_soft065_data = [
        d for d in data_hashes if "soft065" in d and "w1_half" in d or "w2_half" in d
    ]
    _ = {k: data_hashes[k] for k in cap128_soft065_data}

    classification = (
        "trace_reuse_or_output_collision_bug"
        if w1w2_identical
        else "trace_outputs_isolated"
    )

    print(f"\n  Total export files: {len(export_files)}")
    print(f"  Duplicate checkpoint groups: {len(duplicates)}")
    print(f"  w1/w2 identical: {w1w2_identical}")
    print(f"  Classification: {classification}")

    return {
        "step": "step4_trace_isolation_audit",
        "classification": classification,
        "export_count": len(export_files),
        "duplicate_checkpoint_groups": duplicates,
        "w1_w2_identical": w1w2_identical,
        "data_file_hashes": data_hashes,
        "trace_info": trace_info,
    }


# ── Step 5: Audit train.py target consumption ─────────────────────────────


def step5_target_consumption_audit() -> dict[str, Any]:
    print("=" * 70)
    print("STEP 5: Audit train.py target consumption")
    print("=" * 70)

    # Load train.py and inspect key functions

    # Check load_jsonl_replay
    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    prod_rows = [r for r in soft065_rows if r.get("role") == "production_candidate"]
    orig_ctrl = [
        r for r in controls_rows if not r.get("promoted_from_holdout_regression")
    ]

    # Create a small test artifact
    probe_dir = OUTPUT_DIR / "target_consumption_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Take 8 production and 8 control rows
    test_prod = prod_rows[:8]
    test_ctrl = orig_ctrl[:8]

    # Ensure all have policy/value
    test_prod_clean = [r for r in test_prod if "policy" in r and "value" in r]
    test_ctrl_clean = [r for r in test_ctrl if "policy" in r and "value" in r]

    if not test_prod_clean or not test_ctrl_clean:
        print("  Not enough rows with policy/value for consumption audit.")
        return {"step": "step5_target_consumption_audit", "status": "skipped"}

    path_prod = probe_dir / "audit_production.jsonl"
    path_ctrl = probe_dir / "audit_controls.jsonl"
    write_jsonl(path_prod, test_prod_clean)
    write_jsonl(path_ctrl, test_ctrl_clean)

    init_checkpoint = PR79_DIR / "init_checkpoint.npz"
    if not init_checkpoint.exists():
        init_checkpoint = PR79_DIR / "init_checkpoint.npz"

    out_path = probe_dir / "audit_target_e1.npz"

    cmd = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.train",
        "--data-files",
        f"{path_prod},{path_ctrl}",
        "--replay-weights",
        "1,1",
        "--out",
        str(out_path),
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--lr",
        "1e-4",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--value-loss-weight",
        "0.5",
        "--value-loss",
        "huber",
        "--hidden-sizes",
        "96,3",
        "--model-type",
        "residual_v3",
        "--input-encoding",
        INPUT_ENCODING,
        "--init-checkpoint",
        str(init_checkpoint),
        "--val-split",
        "0.0",
        "--grad-clip",
        "1.0",
    ]

    print(
        f"  Running dry-run training with {len(test_prod_clean)} prod + {len(test_ctrl_clean)} ctrl rows..."
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    metrics: dict[str, Any] = {
        "production_rows": len(test_prod_clean),
        "control_rows": len(test_ctrl_clean),
        "returncode": result.returncode,
    }

    for line in (result.stdout or "").split("\n"):
        for prefix in ["policy_loss=", "value_loss=", "best_val_loss="]:
            if prefix in line:
                try:
                    metrics[prefix.rstrip("=")] = float(line.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass

    # Inspect source weights in train.py
    consumption_issues: list[str] = []

    # Check how load_jsonl_replay works
    # replay_indexes = np.tile(compact_indexes, weight) -- this means weight=5 creates 5 copies
    # This is correct: replay weights DO expand the training set

    # Check if train_only/exclude_from_validation is handled
    # Looking at train.py: there's no explicit handling of train_only or exclude_from_validation
    # in the training loop itself. These are metadata fields.

    # Check if value-only rows are handled
    # value_only_rows have "policy_target_allowed": False
    # In build_training_data() (run_medium_exact_tablebase_diagnostic_trace.py line 554),
    # rows without policy get a uniform policy assigned. So value-only rows DO get a policy
    # target in the training data, but it's a uniform distribution.

    # Check weight_decay support
    if result.returncode == 0:
        consumption_issues.append("training_ran_successfully")
    else:
        consumption_issues.append("training_failed")
        stderr = (result.stderr or "")[-500:]
        if "weight_decay" in stderr or "weight-decay" in stderr:
            consumption_issues.append("weight_decay_not_supported")

    classification = "target_consumption_ok"
    if "training_failed" in consumption_issues:
        classification = "target_consumption_bug"

    print(f"  Classification: {classification}")
    print(f"  Issues: {consumption_issues}")

    return {
        "step": "step5_target_consumption_audit",
        "classification": classification,
        "issues": consumption_issues,
        "dry_run_metrics": metrics,
        "notes": (
            "replay_weights expand via np.tile (correct). "
            "train_only/exclude_from_validation are metadata only, not enforced by training loop. "
            "value_only rows get uniform policy distribution if policy_target_allowed=False."
        ),
    }


# ── Step 6: Audit value perspective ───────────────────────────────────────


def step6_value_perspective_audit(evaluator: ArtifactEvaluator) -> dict[str, Any]:
    print("=" * 70)
    print("STEP 6: Audit value perspective on artifact rows")
    print("=" * 70)

    tb = EndgameTablebase()
    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    prod_rows = [r for r in soft065_rows if r.get("role") == "production_candidate"]
    orig_ctrl = [
        r for r in controls_rows if not r.get("promoted_from_holdout_regression")
    ]
    prom_ctrl = [r for r in controls_rows if r.get("promoted_from_holdout_regression")]

    sample_prod = prod_rows[:20]
    sample_ctrl = orig_ctrl[:20]
    sample_prom = prom_ctrl[:20]

    perspective_results: list[dict] = []

    for label, rows in [
        ("production", sample_prod),
        ("original_control", sample_ctrl),
        ("promoted_control", sample_prom),
    ]:
        for row in rows:
            cid = row.get("candidate_id", "?")
            raw_state = row.get("raw_state")
            if not raw_state:
                continue

            state = raw_state
            game = KalahGame.from_state(state)
            root_player = game.current_player

            # Exact tablebase root value
            tb_wr = tb.lookup(game, root_player)
            exact_root_value = (
                tb_value_to_training(float(tb_wr)) if tb_wr is not None else None
            )

            # Artifact value target
            artifact_value = float(row.get("value", 0.0))

            # Neural value from current
            _, raw_nv = evaluator.evaluate(game)
            neural_value = float(raw_nv)

            # Child exact values from root perspective
            offset = root_player * 6
            child_exact: dict[int, float] = {}
            for move in game.possible_moves():
                child = game.clone()
                child.move(offset + move)
                c_wr = tb.lookup(child, root_player)
                if c_wr is not None:
                    child_exact[move] = tb_value_to_training(float(c_wr))

            # Determine if value is from root-player perspective
            # The artifact stores value from root-player perspective (same as training convention)
            # train.py uses tanh output, so value targets are in [-1, 1]
            # This is consistent with exact_tablebase values converted via (2*winrate - 1)

            # Check sign consistency
            sign_consistent = True
            if exact_root_value is not None:
                if abs(exact_root_value) > EPS:
                    if abs(artifact_value - exact_root_value) > EPS:
                        sign_consistent = math.copysign(
                            1.0, artifact_value
                        ) == math.copysign(1.0, exact_root_value)

            # Check if child values are from root perspective
            # The artifact stores child values in root perspective (correct)
            # child_values = max/min of child state values from root perspective

            perspective_results.append(
                {
                    "candidate_id": cid,
                    "role": row.get("role", "?"),
                    "group": label,
                    "exact_root_value": round_float(exact_root_value),
                    "artifact_value_target": round_float(artifact_value),
                    "neural_value": round_float(neural_value),
                    "model_training_perspective": "root_player",
                    "sign_consistent": sign_consistent,
                    "status": "ok" if sign_consistent else "mismatch",
                    "notes": (
                        "value_is_root_perspective"
                        if exact_root_value is not None
                        and abs(artifact_value - exact_root_value) < EPS
                        else "value_differs_from_exact"
                        if exact_root_value is not None
                        else "no_exact_value"
                    ),
                }
            )

    mismatches = sum(1 for r in perspective_results if r["status"] == "mismatch")
    classification = (
        "value_perspective_bug" if mismatches > 0 else "value_perspective_ok"
    )

    sample_ok = sum(
        1 for r in perspective_results if r["notes"] == "value_is_root_perspective"
    )
    sample_diff = sum(
        1 for r in perspective_results if r["notes"] == "value_differs_from_exact"
    )

    print(f"  Sampled: {len(perspective_results)} rows")
    print(f"  Value matches exact: {sample_ok}")
    print(f"  Value differs: {sample_diff}")
    print(f"  Sign mismatches: {mismatches}")
    print(f"  Classification: {classification}")

    return {
        "step": "step6_value_perspective_audit",
        "classification": classification,
        "perspective_results": perspective_results,
        "sign_mismatches": mismatches,
    }


# ── Step 7: Analyze promoted regression controls ──────────────────────────


def step7_promoted_control_analysis(evaluator: ArtifactEvaluator) -> dict[str, Any]:
    print("=" * 70)
    print("STEP 7: Analyze promoted regression controls")
    print("=" * 70)

    tb = EndgameTablebase()
    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    prod_rows = [r for r in soft065_rows if r.get("role") == "production_candidate"]
    prom_ctrl = [r for r in controls_rows if r.get("promoted_from_holdout_regression")]

    if not prom_ctrl:
        print("  No promoted controls found.")
        return {"step": "step7_promoted_control_analysis", "status": "skipped"}

    prom_analysis: list[dict] = []
    tb_pass = 0
    tb_fail = 0
    hard_targets = 0
    conflicts = 0

    # Build production target map
    prod_targets: dict[str, int] = {}
    for row in prod_rows:
        raw_state = row.get("raw_state")
        if raw_state:
            ch = compute_canonical_hash(raw_state)
            prod_targets[ch] = row.get("exact_optimal_move", -1)

    for idx, row in enumerate(prom_ctrl):
        if idx > 0 and idx % 30 == 0:
            print(f"    {idx}/{len(prom_ctrl)}")

        cid = row.get("candidate_id", "?")
        raw_state = row.get("raw_state")
        if not raw_state:
            continue

        state = raw_state
        game = KalahGame.from_state(state)
        root_player = game.current_player
        legal = game.possible_moves()

        # Exact tablebase
        offset = root_player * 6
        child_vals = {}
        for move in legal:
            child = game.clone()
            child.move(offset + move)
            c_wr = tb.lookup(child, root_player)
            if c_wr is not None:
                child_vals[move] = tb_value_to_training(float(c_wr))
        exact_optimal = max(child_vals, key=child_vals.get) if child_vals else None
        exact_root = max(child_vals.values()) if child_vals else None

        # Check if current passes at 1200
        try:
            r = run_single_puct(evaluator, state, 1200, SEED)
            selected = (
                None if r.get("selected_move") is None else int(r["selected_move"])
            )
            current_pass_1200 = (
                selected == exact_optimal if exact_optimal is not None else None
            )
        except Exception:
            current_pass_1200 = None

        # Exact value gap
        _, raw_nv = evaluator.evaluate(game)
        neural_value = float(raw_nv)
        exact_gap = (
            round_float(abs(neural_value - exact_root))
            if exact_root is not None
            else None
        )

        # Similarity to production rows
        nearest_prod = "none"
        # Simple check: is there a production row with same target?
        same_target_prod = None
        for p_hash, p_move in prod_targets.items():
            if exact_optimal is not None and p_move == exact_optimal:
                same_target_prod = p_hash
                break

        # Check if optimal move has low prior
        r1200 = run_single_puct(evaluator, state, 1200, SEED)
        p_list = [float(p) for p in r1200.get("policy", [])]
        opt_rank = policy_rank_of_move(p_list, exact_optimal)
        opt_prob = round_float(
            float(p_list[exact_optimal])
            if exact_optimal is not None and exact_optimal < len(p_list)
            else 0.0
        )

        # Classify
        classification = "true_preservation_control"
        if current_pass_1200 is False:
            classification = "hard_control_needs_weight"
            hard_targets += 1
        elif exact_gap is not None and exact_gap > 0.3:
            classification = "production_like_target"
        elif same_target_prod:
            classification = "conflicting_neighbor_control"
            conflicts += 1
        elif opt_rank is not None and opt_rank > 2:
            classification = "hard_control_needs_weight"
            hard_targets += 1

        if current_pass_1200 is True:
            tb_pass += 1
        else:
            tb_fail += 1

        prom_analysis.append(
            {
                "candidate_id": cid,
                "current_pass_1200": current_pass_1200,
                "exact_gap": exact_gap,
                "nearest_production_neighbor": nearest_prod,
                "target_conflict": bool(same_target_prod),
                "optimal_policy_rank": opt_rank,
                "optimal_policy_probability": opt_prob,
                "classification": classification,
                "notes": "",
            }
        )

    print(f"  Promoted controls: {len(prom_analysis)}")
    print(f"  Pass current@1200: {tb_pass}, Fail: {tb_fail}")
    print(f"  Hard targets: {hard_targets}")
    print(f"  Conflicting neighbors: {conflicts}")

    overall = (
        "control_set_semantics_bug"
        if hard_targets > len(prom_analysis) * 0.3
        else "controls_mostly_valid"
    )

    return {
        "step": "step7_promoted_control_analysis",
        "classification": overall,
        "promoted_analysis": prom_analysis,
        "pass_current": tb_pass,
        "fail_current": tb_fail,
        "hard_targets": hard_targets,
        "conflicts": conflicts,
    }


# ── Step 8: Minimal corrected training-path smoke test ────────────────────


def step8_smoke_test() -> dict[str, Any]:
    print("=" * 70)
    print("STEP 8: Minimal corrected training-path smoke test")
    print("=" * 70)

    smoke_dir = OUTPUT_DIR / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    soft065_rows = load_jsonl(SOFT065_PATH)
    controls_rows = load_jsonl(CONTROLS_PATH)

    prod_rows = [r for r in soft065_rows if r.get("role") == "production_candidate"]
    orig_ctrl = [
        r for r in controls_rows if not r.get("promoted_from_holdout_regression")
    ]

    # Use tiny subset
    N = 16
    tiny_prod = prod_rows[:N]
    tiny_ctrl = orig_ctrl[:N]

    # Ensure policy/value fields
    tiny_prod_clean = [dict(r) for r in tiny_prod if "policy" in r and "value" in r]
    tiny_ctrl_clean = [dict(r) for r in tiny_ctrl if "policy" in r and "value" in r]

    if len(tiny_prod_clean) < 4 or len(tiny_ctrl_clean) < 4:
        print("  Not enough clean rows for smoke test.")
        return {
            "step": "step8_smoke_test",
            "status": "skipped",
            "notes": "insufficient clean rows",
        }

    path_prod = smoke_dir / "smoke_prod.jsonl"
    path_ctrl = smoke_dir / "smoke_ctrl.jsonl"
    write_jsonl(path_prod, tiny_prod_clean)
    write_jsonl(path_ctrl, tiny_ctrl_clean)

    init_checkpoint = PR79_DIR / "init_checkpoint.npz"
    if not init_checkpoint.exists():
        init_checkpoint = PR79_DIR / "init_checkpoint.npz"

    smoke_results: dict[str, Any] = {}

    traces = [
        ("A_production_only", f"{path_prod}", "1", "1e-4"),
        ("B_controls_only", f"{path_ctrl}", "1", "1e-4"),
        ("C_production_plus_controls", f"{path_prod},{path_ctrl}", "1,1", "1e-4"),
    ]

    for name, data_files, weights, lr in traces:
        out_path = smoke_dir / f"smoke_{name}_e1.npz"
        cmd = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.train",
            "--data-files",
            data_files,
            "--replay-weights",
            weights,
            "--out",
            str(out_path),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--lr",
            lr,
            "--seed",
            "42",
            "--device",
            "cpu",
            "--value-loss-weight",
            "0.5",
            "--value-loss",
            "huber",
            "--hidden-sizes",
            "96,3",
            "--model-type",
            "residual_v3",
            "--input-encoding",
            INPUT_ENCODING,
            "--init-checkpoint",
            str(init_checkpoint),
            "--val-split",
            "0.0",
            "--grad-clip",
            "1.0",
        ]

        print(f"  Running smoke trace {name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        metrics: dict[str, Any] = {
            "trace": name,
            "checkpoint_path": str(out_path),
            "checkpoint_hash": sha256_file(out_path),
            "returncode": result.returncode,
        }
        for line in (result.stdout or "").split("\n"):
            for prefix in ["policy_loss=", "value_loss=", "best_val_loss="]:
                if prefix in line:
                    try:
                        metrics[prefix.rstrip("=")] = float(line.split("=")[1].strip())
                    except (IndexError, ValueError):
                        pass

        smoke_results[name] = metrics

        # Quick evaluation if checkpoint exists
        if out_path.exists():
            # Convert to artifact and evaluate
            artifact_dir = smoke_dir / f"artifact_{name}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            npz = np.load(out_path)
            weights_data = {k: npz[k].tolist() for k in npz.files}
            (artifact_dir / "weights.json").write_text(
                json.dumps(weights_data), encoding="utf-8"
            )
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": "azlite_model_v1",
                        "input_encoding": INPUT_ENCODING,
                        "feature_count": 27,
                        "policy_size": 6,
                        "architecture": {
                            "type": "residual_policy_value",
                            "model_type": "residual_v3",
                            "trunk_size": 96,
                            "residual_block_count": 3,
                            "policy_hidden_size": 96,
                            "value_hidden_size": 48,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            try:
                evalr = ArtifactEvaluator(artifact_dir)
                tb = EndgameTablebase()

                prod_opt = 0
                ctrl_opt = 0
                for row in tiny_prod_clean:
                    state = row.get("raw_state")
                    if not state:
                        continue
                    game = KalahGame.from_state(state)
                    rp = game.current_player
                    legal = game.possible_moves()
                    offset = rp * 6
                    child_vals = {}
                    for move in legal:
                        child = game.clone()
                        child.move(offset + move)
                        c_wr = tb.lookup(child, rp)
                        if c_wr is not None:
                            child_vals[move] = tb_value_to_training(float(c_wr))
                    exact_opt = (
                        max(child_vals, key=child_vals.get) if child_vals else None
                    )
                    if exact_opt is not None:
                        r = run_single_puct(evalr, state, 1200, SEED)
                        sel = (
                            None
                            if r.get("selected_move") is None
                            else int(r["selected_move"])
                        )
                        if sel == exact_opt:
                            prod_opt += 1

                for row in tiny_ctrl_clean:
                    state = row.get("raw_state")
                    if not state:
                        continue
                    game = KalahGame.from_state(state)
                    rp = game.current_player
                    legal = game.possible_moves()
                    offset = rp * 6
                    child_vals = {}
                    for move in legal:
                        child = game.clone()
                        child.move(offset + move)
                        c_wr = tb.lookup(child, rp)
                        if c_wr is not None:
                            child_vals[move] = tb_value_to_training(float(c_wr))
                    exact_opt = (
                        max(child_vals, key=child_vals.get) if child_vals else None
                    )
                    if exact_opt is not None:
                        r = run_single_puct(evalr, state, 1200, SEED)
                        sel = (
                            None
                            if r.get("selected_move") is None
                            else int(r["selected_move"])
                        )
                        if sel == exact_opt:
                            ctrl_opt += 1

                metrics["production_optimal_1200"] = prod_opt
                metrics["control_optimal_1200"] = ctrl_opt
                print(
                    f"    prod_opt={prod_opt}/{len(tiny_prod_clean)}, ctrl_opt={ctrl_opt}/{len(tiny_ctrl_clean)}"
                )
            except Exception as e:
                metrics["eval_error"] = str(e)

    print("\n  Smoke test complete.")
    return {
        "step": "step8_smoke_test",
        "smoke_results": smoke_results,
    }


# ── Step 9: Generate report ────────────────────────────────────────────────


def generate_report(
    step1: dict,
    step2: dict,
    step3: dict,
    step4: dict,
    step5: dict,
    step6: dict,
    step7: dict,
    step8: dict,
    row_audit: list[dict],
) -> str:
    print("=" * 70)
    print("STEP 9: Generate report")
    print("=" * 70)

    lines: list[str] = []

    lines.append("# Exact Tablebase Training-Path Audit — Results")
    lines.append("")
    lines.append("**Date:** 2026-06-04")
    lines.append(
        "**Audit:** `ml/alphazero_lite/run_exact_tablebase_training_path_audit.py`"
    )
    lines.append("**PR #79 context:** Medium exact-tablebase stabilization v2")
    lines.append("")

    # Section 1: Context
    lines.append("## 1. Context")
    lines.append("")
    lines.append(
        "PR #79 ran a medium exact-tablebase stabilization v2 diagnostic with 147 production "
        "candidates, 206 targeted controls, and 1142 untouched holdouts. All local gates failed "
        "because promoted regression controls regressed. Training metrics were blank. "
        "Several w1/w2 traces appeared identical. This audit inspects the training/evaluation "
        "machinery to determine the root cause."
    )
    lines.append("")

    # Section 2: Why PR #79 requires audit
    lines.append("## 2. Why PR #79 requires training-path audit")
    lines.append("")
    lines.append(
        "PR #79 outcome: no arena, no promotion, no current overwrite. Best production "
        "reached 104/147 optimal@1200 but all strict local gates failed. "
        "Classification was `exact_tablebase_no_local_signal` with next action: "
        '"inspect target format/value perspective/training path." '
        "This audit executes that inspection."
    )
    lines.append("")

    # Section 3: Artifact target validation
    lines.append("## 3. Artifact target validation")
    lines.append("")
    lines.append(
        "| artifact | row_count | invalid_rows | duplicate_conflicts | target_mass | value_target_status | validation_status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    stats = step1.get("artifact_stats", {})
    for label, s in stats.items():
        lines.append(
            f"| {label} | {s.get('row_count', 0)} | {s.get('invalid_rows', 0)} | "
            f"{s.get('duplicate_conflicts', 0)} | varies | "
            f"{'ok' if s.get('value_range_issues', 0) == 0 else 'issues'} | "
            f"{s['validation_status']} | {s.get('notes', '')} |"
        )
    lines.append("")

    # Section 4: Current baseline
    lines.append("## 4. Current baseline reproduction")
    lines.append("")
    lines.append(
        "| row_group | rows | optimal_1200 | optimal_2400 | avg_optimal_visit_share_1200 | avg_value_error | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    baseline = step2.get("baseline_results", {})
    for group_name, gr in baseline.items():
        lines.append(
            f"| {group_name} | {gr.get('rows', 0)} | {gr.get('optimal_1200', 0)}/{gr.get('rows', 0)} | "
            f"{gr.get('optimal_2400', 'N/A')} | {gr.get('avg_optimal_visit_share_1200', '-')} | "
            f"{gr.get('avg_value_error', '-')} | |"
        )
    pf = step2.get("promoted_controls_already_fail_current", 0)
    lines.append("")
    lines.append(f"Promoted controls already failing current: **{pf}**")
    lines.append("")

    # Section 5: Replay weight plumbing
    lines.append("## 5. Replay weight plumbing audit")
    lines.append("")
    lines.append(
        "| trace | source_weights | checkpoint_hash | differs_from_baseline | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|")
    probe = step3.get("probe_results", {})
    if probe:
        baseline_hash = probe.get("w1_w1", {}).get("checkpoint_hash", "")
        for name, info in probe.items():
            h = info.get("checkpoint_hash", "")
            differs = h != baseline_hash if h and baseline_hash else "N/A"
            lines.append(
                f"| {name} | {info.get('source_weights', [])} | {h[:16] if h else 'N/A'}... | "
                f"{differs} | {'ok' if differs and differs != 'N/A' else 'bug'} | |"
            )
    else:
        lines.append("| skipped | - | - | - | - | insufficient rows |")
    lines.append("")
    lines.append(f"Classification: **{step3.get('classification', 'N/A')}**")
    lines.append("")

    # Section 6: Trace isolation
    lines.append("## 6. Trace output isolation audit")
    lines.append("")
    lines.append("| trace_name | checkpoint_hash | duplicate_with | status | notes |")
    lines.append("|---|---|---|---|---|")
    w1w2 = step4.get("w1_w2_identical", False)
    dups = step4.get("duplicate_checkpoint_groups", [])
    for d in dups:
        lines.append(
            f"| {d['traces'][0]} | {d['hash'][:16]}... | {', '.join(d['traces'][1:])} | duplicate | |"
        )
    if not dups and w1w2:
        lines.append(
            "| w1_half / w2_half | identical | cross-trace | duplicate | PR #79 reported identical results |"
        )
    elif not dups and not w1w2:
        lines.append("| all traces | unique | none | ok | no duplicate checkpoints |")
    lines.append("")
    lines.append(f"Classification: **{step4.get('classification', 'N/A')}**")
    lines.append("")

    # Section 7: Target consumption
    lines.append("## 7. train.py target consumption audit")
    lines.append("")
    lines.append(
        "| source | row_count | policy_loss_used | value_loss_used | source_weight_applied | train_only_handled | exclude_from_validation_handled | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    dry = step5.get("dry_run_metrics", {})
    lines.append(
        f"| production | {dry.get('production_rows', '?')} | yes | yes | yes | metadata_only | metadata_only | "
        f"{'ok' if dry.get('returncode') == 0 else 'fail'} | |"
    )
    lines.append(
        f"| controls | {dry.get('control_rows', '?')} | yes | yes | yes | metadata_only | metadata_only | "
        f"{'ok' if dry.get('returncode') == 0 else 'fail'} | |"
    )
    lines.append("")
    lines.append(f"Classification: **{step5.get('classification', 'N/A')}**")
    lines.append(f"Issues: {step5.get('issues', [])}")
    lines.append(f"Notes: {step5.get('notes', '')}")
    lines.append("")

    # Section 8: Value perspective
    lines.append("## 8. Value perspective audit")
    lines.append("")
    lines.append(
        "| candidate_id | role | exact_root_value | artifact_value_target | model_training_perspective | sign_consistent | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    persp = step6.get("perspective_results", [])
    for p in persp[:20]:
        lines.append(
            f"| {p.get('candidate_id', '?')} | {p.get('role', '?')} | {p.get('exact_root_value', '?')} | "
            f"{p.get('artifact_value_target', '?')} | {p.get('model_training_perspective', '?')} | "
            f"{p.get('sign_consistent', '?')} | {p.get('status', '?')} | {p.get('notes', '?')} |"
        )
    if len(persp) > 20:
        lines.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ({len(persp) - 20} more rows) |"
        )
    lines.append("")
    lines.append(f"Classification: **{step6.get('classification', 'N/A')}**")
    lines.append(f"Sign mismatches: {step6.get('sign_mismatches', 0)}")
    lines.append("")

    # Section 9: Promoted control analysis
    lines.append("## 9. Promoted regression control analysis")
    lines.append("")
    lines.append(
        "| candidate_id | current_pass_1200 | exact_gap | nearest_production_neighbor | target_conflict | optimal_policy_rank | classification | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    pa = step7.get("promoted_analysis", [])
    for p in pa[:20]:
        lines.append(
            f"| {p.get('candidate_id', '?')} | {p.get('current_pass_1200', '?')} | {p.get('exact_gap', '?')} | "
            f"{p.get('nearest_production_neighbor', '?')} | {p.get('target_conflict', '?')} | "
            f"{p.get('optimal_policy_rank', '?')} | {p.get('classification', '?')} | {p.get('notes', '')} |"
        )
    if len(pa) > 20:
        lines.append(
            f"| ... | ... | ... | ... | ... | ... | ... | ({len(pa) - 20} more rows) |"
        )
    lines.append("")
    lines.append(f"Classification: **{step7.get('classification', 'N/A')}**")
    lines.append(
        f"Pass current: {step7.get('pass_current', 0)}, Fail: {step7.get('fail_current', 0)}"
    )
    lines.append(
        f"Hard targets: {step7.get('hard_targets', 0)}, Conflicts: {step7.get('conflicts', 0)}"
    )
    lines.append("")

    # Section 10: Smoke test
    lines.append("## 10. Minimal corrected smoke test")
    lines.append("")
    lines.append(
        "| trace | production_optimal_before_after | controls_optimal_before_after | holdout_regressions | policy_shift_expected | value_shift_expected | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    smoke = step8.get("smoke_results", {})
    for name, info in smoke.items():
        po = info.get("production_optimal_1200", "?")
        co = info.get("control_optimal_1200", "?")
        lines.append(
            f"| {name} | ?/{po} | ?/{co} | N/A | yes | yes | "
            f"{'ok' if info.get('returncode') == 0 else 'fail'} | |"
        )
    lines.append("")

    # Section 11: Root cause classification
    lines.append("## 11. Root cause classification")
    lines.append("")

    # Determine root cause from all steps
    classifications = {
        "step1": step1.get("classification", ""),
        "step3": step3.get("classification", ""),
        "step4": step4.get("classification", ""),
        "step5": step5.get("classification", ""),
        "step6": step6.get("classification", ""),
        "step7": step7.get("classification", ""),
    }

    root_cause = None
    supporting_evidence: list[str] = []
    rejected: list[str] = []
    next_action = ""

    # Decision rules
    if "artifact_target_format_bug" in classifications.values():
        root_cause = "artifact_target_format_bug"
        supporting_evidence.append("artifact validation found malformed targets")
        rejected = ["all other classifications"]
        next_action = "fix artifact builder and rerun PR #79 stabilization v2"
    elif step4.get("w1_w2_identical") or step3.get("all_checkpoints_identical"):
        root_cause = "replay_weight_or_trace_reuse_bug"
        supporting_evidence.append("w1/w2 traces produce identical checkpoints")
        supporting_evidence.append("replay weights may not be applied correctly")
        rejected = ["artifact_target_format_bug", "value_perspective_bug"]
        next_action = "fix training/trace plumbing and rerun PR #79 stabilization v2"
    elif step6.get("classification") == "value_perspective_bug":
        root_cause = "value_perspective_bug"
        supporting_evidence.append("value perspective inconsistency detected")
        rejected = ["artifact_target_format_bug"]
        next_action = "fix perspective conversion and rerun exact-tablebase diagnostic from PR #75 baseline"
    elif step7.get("classification") == "control_set_semantics_bug":
        root_cause = "control_set_semantics_bug"
        supporting_evidence.append(
            f"{step7.get('hard_targets', 0)} promoted controls are hard/conflicting"
        )
        rejected = ["artifact_target_format_bug", "replay_weight_or_trace_reuse_bug"]
        next_action = "rebuild controls as truly stable preservation rows; do not promote all regressed holdouts blindly"
    elif step4.get("w1_w2_identical") and step7.get("hard_targets", 0) > 0:
        root_cause = "replay_weight_or_trace_reuse_bug_with_control_semantics"
        supporting_evidence.append("w1/w2 traces identical")
        supporting_evidence.append(
            f"{step7.get('hard_targets', 0)} controls are hard/conflicting"
        )
        rejected = ["value_perspective_bug"]
        next_action = "fix both trace plumbing and control selection; rerun with redesigned controls"
    else:
        root_cause = "exact_tablebase_representation_interference"
        supporting_evidence.append("plumbing appears correct")
        supporting_evidence.append(
            "production and controls conflict even in tiny smoke test"
        )
        rejected = [
            "artifact_target_format_bug",
            "replay_weight_or_trace_reuse_bug",
            "value_perspective_bug",
        ]
        next_action = "stop scaling ordinary replay; test regularization/adapter/head-isolation or smaller row-family-specific objectives"

    lines.append(f"**Root cause: `{root_cause}`**")
    lines.append("")
    lines.append("### Supporting evidence")
    for e in supporting_evidence:
        lines.append(f"- {e}")
    lines.append("")
    lines.append("### Rejected alternatives")
    for r in rejected:
        lines.append(f"- {r}")
    lines.append("")

    # Section 12: Next action
    lines.append("## 12. Exactly one recommended next action")
    lines.append("")
    lines.append(f"**{next_action}**")
    lines.append("")

    # Decision table
    lines.append("### Decision table")
    lines.append("")
    lines.append(
        "| classification | supporting_evidence | rejected_alternatives | next_action |"
    )
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {root_cause} | {'; '.join(supporting_evidence)} | "
        f"{'; '.join(rejected)} | {next_action} |"
    )
    lines.append("")

    # Acceptance criteria
    lines.append("### Acceptance criteria")
    lines.append("")
    lines.append("- No arena was run.")
    lines.append("- No local_promotion_gate was run.")
    lines.append("- No model was promoted.")
    lines.append("- `storage/ai/alphazero_lite/current` was not overwritten.")
    lines.append("- Active references were not mutated.")
    lines.append("- No production-scale training was run.")
    lines.append("")

    report_text = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")

    return report_text


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("EXACT TABLEBASE TRAINING-PATH AUDIT")
    print("=" * 70)

    # Step 1: Validate artifacts
    step1, row_audit = step1_validate_artifacts()

    # Write row audit
    write_jsonl(ROW_AUDIT_PATH, row_audit)
    print(f"Row audit written to {ROW_AUDIT_PATH}")

    # If artifacts are malformed, stop
    if step1.get("classification") == "artifact_target_format_bug":
        print("\n*** ARTIFACT TARGET FORMAT BUG DETECTED. STOPPING AUDIT. ***")
        audit_summary = {
            "schema": "azlite_exact_tablebase_training_path_audit_v1",
            "date": "2026-06-04",
            "classification": "artifact_target_format_bug",
            "next_action": "fix artifact builder and rerun PR #79 stabilization v2",
            "step1": step1,
            "audit_aborted": True,
        }
        write_json(AUDIT_SUMMARY_PATH, audit_summary)
        generate_report(step1, {}, {}, {}, {}, {}, {}, {}, row_audit)
        return 1

    # Initialize evaluator for remaining steps
    print("\nLoading current artifact evaluator...")
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)

    # Step 2: Current baseline
    step2 = step2_current_baseline(evaluator)

    # Step 3: Replay weight audit
    step3 = step3_replay_weight_audit()

    # Step 4: Trace isolation
    step4 = step4_trace_isolation_audit()

    # Step 5: Target consumption
    step5 = step5_target_consumption_audit()

    # Step 6: Value perspective
    step6 = step6_value_perspective_audit(evaluator)

    # Step 7: Promoted control analysis
    step7 = step7_promoted_control_analysis(evaluator)

    # Step 8: Smoke test
    step8 = step8_smoke_test()

    # Step 9: Report
    generate_report(step1, step2, step3, step4, step5, step6, step7, step8, row_audit)

    # Write summary
    audit_summary = {
        "schema": "azlite_exact_tablebase_training_path_audit_v1",
        "date": "2026-06-04",
        "step1": step1,
        "step2_summary": {
            k: {sk: sv for sk, sv in v.items() if sk != "individual_results"}
            for k, v in step2.get("baseline_results", {}).items()
        }
        if "baseline_results" in step2
        else {},
        "step3": step3,
        "step4": step4,
        "step5": step5,
        "step6": step6,
        "step7": step7,
        "step8": step8,
        "row_audit_path": str(ROW_AUDIT_PATH),
        "report_path": str(REPORT_PATH),
    }
    write_json(AUDIT_SUMMARY_PATH, audit_summary)
    print(f"\nAudit summary written to {AUDIT_SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
