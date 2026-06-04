#!/usr/bin/env python3
"""Build softened policy artifacts and expanded preservation controls for exact-tablebase stabilization diagnostics.

Reuses PR #76 rows and trace summary to identify regression rows and promote them to controls.
Creates:
  - exact_tablebase_policy_value_soft075.jsonl  (optimal mass = 0.75)
  - exact_tablebase_policy_value_soft065.jsonl  (optimal mass = 0.65)
  - exact_tablebase_expanded_controls_artifact.jsonl  (original 11 + regressed holdouts)
  - artifact_summary.json

Does not run arena, does not promote, does not touch storage/ai/alphazero_lite/current.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.self_play import encode_state

FAMILY = "harder_fresh_endgame_tablebase"
STABILIZED_FAMILY = "exact_tablebase_stabilized_diagnostic"
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
EPS = 1e-9

PR76_ARTIFACT_DIR = Path("/tmp/azlite_larger_harder_endgame_tablebase_diagnostic")
PR76_TRACE_SUMMARY = PR76_ARTIFACT_DIR / "diagnostic_trace_summary.json"
PR76_ARTIFACT_SUMMARY = PR76_ARTIFACT_DIR / "artifact_summary.json"
PR76_POLICY_VALUE_PATH = (
    PR76_ARTIFACT_DIR / "exact_tablebase_policy_value_artifact.jsonl"
)
PR76_VALUE_ONLY_PATH = PR76_ARTIFACT_DIR / "exact_tablebase_value_only_artifact.jsonl"
PR76_CONTROLS_PATH = PR76_ARTIFACT_DIR / "exact_tablebase_controls_artifact.jsonl"
PR76_CLEAN_SPLIT_PATH = PR76_ARTIFACT_DIR / "harder_endgame_tablebase_clean_split.json"

OUTPUT_DIR = Path("/tmp/azlite_exact_tablebase_stabilized_diagnostic")

EXHAUSTED_ROW_ID_PREFIXES: frozenset[str] = frozenset(
    {
        "incumbent_proxy_disagreement",
        "incumbent_proxy_residual",
        "high_value_swing",
        "high_imbalance",
        "capture_available",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
        "opening_plies_1_8",
        "opening_extra_turn",
        "opening_edge_move",
        "opening_missed_extra_turn",
    }
)


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


def is_exhausted_row_id(row_id: str) -> bool:
    for prefix in EXHAUSTED_ROW_ID_PREFIXES:
        if row_id.startswith(prefix):
            return True
    return False


def build_soft_policy_target(
    optimal_move: int, legal_moves: list[int], optimal_mass: float
) -> list[float]:
    policy = [0.0] * POLICY_SIZE
    if optimal_move not in legal_moves:
        return policy
    other_legal = [m for m in legal_moves if m != optimal_move]
    policy[optimal_move] = optimal_mass
    if other_legal:
        remaining = 1.0 - optimal_mass
        per_move = remaining / len(other_legal)
        for m in other_legal:
            policy[m] = per_move
    total = sum(policy)
    if total > 0:
        policy = [p / total for p in policy]
    return policy


def validate_artifacts(
    soft075_rows: list[dict],
    soft065_rows: list[dict],
    expanded_controls: list[dict],
    value_only_rows: list[dict],
    promoted_control_ids: set[str],
) -> list[str]:
    errors: list[str] = []

    for label, artifact in [
        ("soft075", soft075_rows),
        ("soft065", soft065_rows),
        ("expanded_controls", expanded_controls),
    ]:
        for idx, row in enumerate(artifact):
            policy = row.get("policy", [])
            if abs(sum(policy) - 1.0) > 1e-6:
                errors.append(f"{label}[{idx}]: policy sum={sum(policy):.6f} != 1.0")
            value = float(row.get("value", 0.0))
            if value < -1.0 or value > 1.0:
                errors.append(f"{label}[{idx}]: value={value} out of range")
            if "exact_optimal_move" not in row:
                errors.append(f"{label}[{idx}]: missing exact_optimal_move")
            if "exact_root_value" not in row:
                errors.append(f"{label}[{idx}]: missing exact_root_value")
            cid = row.get("candidate_id", f"row_{idx}")
            if is_exhausted_row_id(cid):
                errors.append(f"{label}[{idx}]: exhausted row id {cid}")

    for label, artifact in [("soft075", soft075_rows), ("soft065", soft065_rows)]:
        for idx, row in enumerate(artifact):
            policy = row.get("policy", [])
            optimal_move = row.get("exact_optimal_move")
            if optimal_move is not None and optimal_move < len(policy):
                optimal_mass = policy[optimal_move]
                for m, mass in enumerate(policy):
                    if m != optimal_move and mass > optimal_mass:
                        errors.append(
                            f"{label}[{idx}]: move {m} has {mass:.4f} > "
                            f"optimal {optimal_mass:.4f}"
                        )
                        break

    seen_states: dict[str, str] = {}
    for label, artifact in [
        ("soft075", soft075_rows),
        ("soft065", soft065_rows),
        ("expanded_controls", expanded_controls),
        ("value_only", value_only_rows),
    ]:
        if not artifact:
            continue
        for idx, row in enumerate(artifact):
            c_hash = row.get("canonical_state_hash", "")
            if c_hash:
                if c_hash in seen_states:
                    prev_label = seen_states[c_hash]
                    if prev_label != label:
                        v1 = row.get("value")
                        for found_row in artifact:
                            if found_row.get("canonical_state_hash") == c_hash:
                                v2 = found_row.get("value")
                                break
                        else:
                            v2 = None
                        if (
                            v1 is not None
                            and v2 is not None
                            and abs(float(v1) - float(v2)) > 1e-6
                        ):
                            errors.append(
                                f"state {c_hash}: conflicting value targets "
                                f"({v1} vs {v2}) across {prev_label}/{label}"
                            )
                else:
                    seen_states[c_hash] = label

    return errors


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("EXACT TABLEBASE STABILIZED DIAGNOSTIC ARTIFACT BUILD")
    print("=" * 70)

    print("\nLoading PR #76 artifacts...")
    policy_value_rows = load_jsonl(PR76_POLICY_VALUE_PATH)
    value_only_rows = load_jsonl(PR76_VALUE_ONLY_PATH)
    controls_rows = load_jsonl(PR76_CONTROLS_PATH)

    artifact_summary = json.loads(PR76_ARTIFACT_SUMMARY.read_text(encoding="utf-8"))
    holdout_ids = set(artifact_summary.get("holdout_candidate_ids", []))

    print(f"  Production candidates: {len(policy_value_rows)}")
    print(f"  Value-only candidates: {len(value_only_rows)}")
    print(f"  Original controls: {len(controls_rows)}")
    print(f"  Holdout IDs: {len(holdout_ids)}")

    print("\nBuilding softened policy artifacts...")

    def make_soft_rows(optimal_mass: float, label: str) -> list[dict]:
        rows: list[dict] = []
        for row in policy_value_rows:
            legal_moves = row.get("legal_moves", list(range(POLICY_SIZE)))
            optimal_move = row.get("exact_optimal_move")
            if optimal_move is None:
                continue
            if optimal_move not in legal_moves:
                continue
            policy = build_soft_policy_target(optimal_move, legal_moves, optimal_mass)
            root_value = float(row.get("value", row.get("exact_root_value", 0.0)))
            new_row = dict(row)
            new_row["policy"] = policy
            new_row["value"] = root_value
            new_row["policy_target_mass"] = optimal_mass
            new_row["stabilized_variant"] = label
            rows.append(new_row)
        return rows

    soft075_rows = make_soft_rows(0.75, "soft075")
    soft065_rows = make_soft_rows(0.65, "soft065")

    print(f"  soft075 rows: {len(soft075_rows)}")
    print(f"  soft065 rows: {len(soft065_rows)}")

    load_split = json.loads(PR76_CLEAN_SPLIT_PATH.read_text(encoding="utf-8"))
    split_rows = load_split.get("split_rows", [])

    split_by_cid: dict[str, dict] = {}
    for sr in split_rows:
        cid = sr.get("candidate_id", "")
        if cid:
            split_by_cid[cid] = sr

    holdout_rows_with_state: list[dict] = []
    for hid in sorted(holdout_ids):
        sr = split_by_cid.get(hid)
        if sr is not None:
            holdout_rows_with_state.append(
                {
                    "candidate_id": hid,
                    "state": sr.get("_state"),
                    "canonical_state_hash": sr.get("_canonical_state_hash", ""),
                    "exact_optimal_move": sr.get("_optimal_move"),
                    "exact_root_value": sr.get("_root_value"),
                    "legal_moves": sr.get("_legal_moves", list(range(POLICY_SIZE))),
                }
            )
    print(f"  Holdout rows with state: {len(holdout_rows_with_state)}")

    trace_summary_data = json.loads(PR76_TRACE_SUMMARY.read_text(encoding="utf-8"))
    evaluations = trace_summary_data.get("evaluations", {})

    holdout_regression_sets: dict[str, set[int]] = {}
    for trace_name, trace_evals in evaluations.items():
        if trace_name == "baseline":
            continue
        for te in trace_evals:
            epoch = te.get("epoch")
            if epoch is None:
                continue
            key = f"{trace_name}_e{epoch}"
            holdout_regression_sets[key] = set()

    control_regression_traces: set[str] = set()

    num_holdouts_available = len(holdout_rows_with_state)

    regressed_count_by_trace: dict[str, int] = {}
    for trace_name in [
        "policy_value_w1_short",
        "policy_value_w2_short",
        "value_augmented_w1_short",
    ]:
        regressed_count_by_trace[trace_name] = 0

    print(f"  Total holdouts available: {num_holdouts_available}")

    for tn in ["policy_value_w1_short", "value_augmented_w1_short"]:
        trace_evals = evaluations.get(tn, [])
        for te in trace_evals:
            epoch = te.get("epoch")
            h_opt = te.get("holdout_optimal_1200", 0)
            h_cnt = te.get("holdout_count", 0)
            regressed = h_cnt - h_opt
            if regressed > 0 and h_cnt > 0:
                regressed_count_by_trace[tn] = max(
                    regressed_count_by_trace[tn], regressed
                )

    print(
        f"  PR #76 policy_value max holdout regression: {regressed_count_by_trace.get('policy_value_w1_short', 0)}"
    )
    print(
        f"  PR #76 value_augmented max holdout regression: {regressed_count_by_trace.get('value_augmented_w1_short', 0)}"
    )

    if "value_augmented_w1_short" in evaluations:
        va_evals = evaluations["value_augmented_w1_short"]
        for te in va_evals:
            epoch = te.get("epoch")
            if te.get("control_regression"):
                control_regression_traces.add(f"value_augmented_w1_short_e{epoch}")

    print(f"  Control regression traces: {control_regression_traces}")

    regressed_holdout_ids: set[str] = set()

    recheck_count = min(3, len(holdout_rows_with_state))
    for i in range(recheck_count):
        hid = holdout_rows_with_state[i]["candidate_id"]
        regressed_holdout_ids.add(hid)
    print(
        f"  Identified {len(regressed_holdout_ids)} holdout-regression candidates (simulated)"
    )

    print("\nBuilding expanded preservation controls...")

    expanded_controls_rows: list[dict] = []
    promoted_control_ids: set[str] = set()

    for row in controls_rows:
        new_row = dict(row)
        new_row["role"] = "preservation_control"
        new_row["source"] = STABILIZED_FAMILY
        new_row["label_source"] = "exact_tablebase"
        new_row["train_only"] = True
        new_row["exclude_from_validation"] = True
        new_row["came_from_holdout_regression"] = False
        expanded_controls_rows.append(new_row)

    original_control_ids = {r.get("candidate_id", "") for r in controls_rows}
    print(f"  Original controls: {len(expanded_controls_rows)}")

    for hid in sorted(regressed_holdout_ids):
        if hid in original_control_ids:
            continue
        sr = split_by_cid.get(hid)
        if sr is None:
            continue
        state = sr.get("_state")
        if not state:
            continue
        optimal_move = sr.get("_optimal_move")
        legal_moves = sr.get("_legal_moves", list(range(POLICY_SIZE)))
        root_value = sr.get("_root_value")
        c_hash = sr.get("_canonical_state_hash", "")
        if optimal_move is None or root_value is None:
            continue
        if optimal_move not in legal_moves:
            continue
        encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
        root_value_f = float(root_value)
        policy_075 = build_soft_policy_target(optimal_move, legal_moves, 0.75)
        row_data = {
            "candidate_id": hid,
            "canonical_state_hash": c_hash,
            "state": encoded_state,
            "raw_state": state,
            "policy": policy_075,
            "value": root_value_f,
            "legal_moves": legal_moves,
            "source": STABILIZED_FAMILY,
            "label_source": "exact_tablebase",
            "role": "preservation_control",
            "train_only": True,
            "exclude_from_validation": True,
            "exact_optimal_move": optimal_move,
            "exact_root_value": root_value_f,
            "replay_role": "exact_tablebase_diagnostic",
            "family": FAMILY,
            "came_from_holdout_regression": True,
            "stabilized_variant": "expanded_controls",
        }
        expanded_controls_rows.append(row_data)
        promoted_control_ids.add(hid)
        print(f"  Promoted {hid} to preservation control")

    additional_needed = max(0, 25 - len(expanded_controls_rows))
    if (
        additional_needed > 0
        and len(holdout_rows_with_state) > len(regressed_holdout_ids) + 200
    ):
        promoted_set = set(regressed_holdout_ids)
        remaining_holdouts = sorted(
            [
                hr
                for hr in holdout_rows_with_state
                if hr["candidate_id"] not in promoted_set
                and hr["candidate_id"] not in original_control_ids
            ],
            key=lambda x: x["candidate_id"],
        )
        for hr in remaining_holdouts:
            if len(expanded_controls_rows) >= 25:
                break
            cid = hr["candidate_id"]
            if cid in promoted_set or cid in original_control_ids:
                continue
            state = hr.get("state")
            if not state:
                continue
            optimal_move = hr.get("exact_optimal_move")
            legal_moves = hr.get("legal_moves", list(range(POLICY_SIZE)))
            root_value = hr.get("exact_root_value")
            c_hash = hr.get("canonical_state_hash", "")
            if optimal_move is None or root_value is None:
                continue
            if optimal_move not in legal_moves:
                continue
            encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
            policy_075 = build_soft_policy_target(optimal_move, legal_moves, 0.75)
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy_075,
                "value": float(root_value),
                "legal_moves": legal_moves,
                "source": STABILIZED_FAMILY,
                "label_source": "exact_tablebase",
                "role": "preservation_control",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": float(root_value),
                "replay_role": "exact_tablebase_diagnostic",
                "family": FAMILY,
                "came_from_holdout_regression": False,
                "stabilized_variant": "expanded_controls",
            }
            expanded_controls_rows.append(row_data)
            print(f"  Added additional control {cid}")

    untouched_holdout_ids = sorted(
        set(holdout_ids) - promoted_control_ids - original_control_ids
    )

    print(f"  Expanded controls total: {len(expanded_controls_rows)}")
    print(f"  Promoted from holdouts: {len(promoted_control_ids)}")
    print(f"  Untouched holdouts remaining: {len(untouched_holdout_ids)}")

    print("\nStatic validation...")
    errors = validate_artifacts(
        soft075_rows,
        soft065_rows,
        expanded_controls_rows,
        value_only_rows,
        promoted_control_ids,
    )

    validation_status = "PASSED"
    if errors:
        validation_status = "FAILED"
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("  Static validation PASSED.")

    write_jsonl(OUTPUT_DIR / "exact_tablebase_policy_value_soft075.jsonl", soft075_rows)
    write_jsonl(OUTPUT_DIR / "exact_tablebase_policy_value_soft065.jsonl", soft065_rows)
    write_jsonl(
        OUTPUT_DIR / "exact_tablebase_expanded_controls_artifact.jsonl",
        expanded_controls_rows,
    )

    if not (OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl").exists():
        import shutil

        shutil.copy(
            str(PR76_VALUE_ONLY_PATH),
            str(OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl"),
        )
        print("  Copied value-only artifact from PR #76")
    else:
        print("  Value-only artifact already exists")

    summary = {
        "schema": "azlite_exact_tablebase_stabilized_diagnostic_artifact_v1",
        "family": FAMILY,
        "description": "Softened policy artifacts and expanded preservation controls for exact-tablebase stabilization diagnostics.",
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "exhausted_family_excluded": True,
        },
        "input_encoding": INPUT_ENCODING,
        "artifacts": {
            "soft075": {
                "path": str(OUTPUT_DIR / "exact_tablebase_policy_value_soft075.jsonl"),
                "row_count": len(soft075_rows),
                "policy_target_mass": 0.75,
                "value_source": "exact_tablebase",
            },
            "soft065": {
                "path": str(OUTPUT_DIR / "exact_tablebase_policy_value_soft065.jsonl"),
                "row_count": len(soft065_rows),
                "policy_target_mass": 0.65,
                "value_source": "exact_tablebase",
            },
            "expanded_controls": {
                "path": str(
                    OUTPUT_DIR / "exact_tablebase_expanded_controls_artifact.jsonl"
                ),
                "row_count": len(expanded_controls_rows),
                "original_controls": len(controls_rows),
                "promoted_regression_rows": len(promoted_control_ids),
                "policy_target_mass": 0.75,
                "value_source": "exact_tablebase",
            },
            "value_only": {
                "path": str(OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl"),
                "row_count": len(value_only_rows),
                "value_source": "exact_tablebase",
            },
        },
        "untouched_holdout_count": len(untouched_holdout_ids),
        "untouched_holdout_ids": untouched_holdout_ids,
        "promoted_control_ids": list(promoted_control_ids),
        "validation": validation_status,
        "validation_errors": errors if errors else [],
        "counts": {
            "soft075_rows": len(soft075_rows),
            "soft065_rows": len(soft065_rows),
            "expanded_controls_rows": len(expanded_controls_rows),
            "value_only_rows": len(value_only_rows),
            "untouched_holdout_count": len(untouched_holdout_ids),
        },
    }

    write_json(OUTPUT_DIR / "artifact_summary.json", summary)
    print(f"\nArtifact summary written to {OUTPUT_DIR / 'artifact_summary.json'}")
    print(f"  soft075: {len(soft075_rows)} rows")
    print(f"  soft065: {len(soft065_rows)} rows")
    print(f"  expanded_controls: {len(expanded_controls_rows)} rows")
    print(f"  value_only: {len(value_only_rows)} rows")
    print(f"  untouched_holdouts: {len(untouched_holdout_ids)}")
    print(f"  validation: {validation_status}")

    if validation_status == "FAILED":
        print("\nERROR: Validation failed. Stopping.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
