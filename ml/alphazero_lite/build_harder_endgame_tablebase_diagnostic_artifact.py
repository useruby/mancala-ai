#!/usr/bin/env python3
"""Build tiny train-only exact-tablebase diagnostic artifacts from PR #74 split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.self_play import encode_state


FAMILY = "harder_fresh_endgame_tablebase"
INPUT_ENCODING = "kalah_v3"
POLICY_SIZE = 6
OPTIMAL_POLICY_MASS = 0.85
EPS = 1e-9

CLEAN_SPLIT_PATH = Path(
    "/tmp/azlite_harder_endgame_tablebase_local_diagnostics/"
    "harder_endgame_tablebase_clean_split.json"
)
ROW_DIAGNOSTICS_PATH = Path(
    "/tmp/azlite_harder_endgame_tablebase_local_diagnostics/"
    "harder_endgame_tablebase_local_row_diagnostics.jsonl"
)
OUTPUT_DIR = Path("/tmp/azlite_harder_endgame_tablebase_diagnostic_artifact")

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


def is_exhausted_row_id(row_id: str) -> bool:
    for prefix in EXHAUSTED_ROW_ID_PREFIXES:
        if row_id.startswith(prefix):
            return True
    return False


def build_policy_target(optimal_move: int, legal_moves: list[int]) -> list[float]:
    policy = [0.0] * POLICY_SIZE
    if optimal_move not in legal_moves:
        return policy
    other_legal = [m for m in legal_moves if m != optimal_move]
    policy[optimal_move] = OPTIMAL_POLICY_MASS
    if other_legal:
        remaining = 1.0 - OPTIMAL_POLICY_MASS
        per_move = remaining / len(other_legal)
        for m in other_legal:
            policy[m] = per_move
    total = sum(policy)
    if total > 0:
        policy = [p / total for p in policy]
    return policy


def validate_artifacts(
    policy_artifact: list[dict],
    value_artifact: list[dict],
    controls_artifact: list[dict],
    summary: dict,
) -> list[str]:
    errors: list[str] = []

    for label, artifact in [
        ("policy_value", policy_artifact),
        ("controls", controls_artifact),
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
            canonical = row.get("raw_state")
            if canonical:
                cid = row.get("candidate_id", f"row_{idx}")
                if is_exhausted_row_id(cid):
                    errors.append(f"{label}[{idx}]: exhausted row id {cid}")

    for idx, row in enumerate(policy_artifact):
        policy = row.get("policy", [])
        optimal_move = row.get("exact_optimal_move")
        if optimal_move is not None and optimal_move < len(policy):
            optimal_mass = policy[optimal_move]
            for m, mass in enumerate(policy):
                if m != optimal_move and mass > optimal_mass:
                    errors.append(
                        f"policy_value[{idx}]: move {m} has {mass:.4f} > "
                        f"optimal {optimal_mass:.4f}"
                    )
                    break

    seen_states: dict[str, str] = {}
    for label, artifact in [
        ("policy_value", policy_artifact),
        ("value_only", value_artifact),
        ("controls", controls_artifact),
    ]:
        for idx, row in enumerate(artifact):
            c_hash = row.get("canonical_state_hash", "")
            if c_hash:
                if c_hash in seen_states:
                    prev_label = seen_states[c_hash]
                    if prev_label != label:
                        row1 = next(
                            (
                                r
                                for r in artifact
                                if r.get("canonical_state_hash") == c_hash
                            ),
                            {},
                        )
                        v1 = row1.get("value")
                        v2 = row.get("value")
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

    train_hashes = set()
    for label, artifact in [
        ("policy_value", policy_artifact),
        ("value_only", value_artifact),
        ("controls", controls_artifact),
    ]:
        for row in artifact:
            c_hash = row.get("canonical_state_hash", "")
            if c_hash:
                train_hashes.add(c_hash)

    return errors


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean split...")
    clean_split = json.loads(CLEAN_SPLIT_PATH.read_text(encoding="utf-8"))
    split_rows = clean_split.get("split_rows", [])
    counts = clean_split.get("counts", {})

    expected_counts = {
        "production_candidate_later": 12,
        "value_only_candidate": 20,
        "preservation_control": 3,
        "holdout_candidate": 56,
    }
    for bucket, expected in expected_counts.items():
        actual = counts.get(bucket, 0)
        if actual != expected:
            print(f"  WARNING: {bucket}: expected {expected}, got {actual}")
        else:
            print(f"  {bucket}: {actual} (ok)")

    bucket_map: dict[str, list[dict]] = {
        "production_candidate_later": [],
        "value_only_candidate": [],
        "preservation_control": [],
        "holdout_candidate": [],
    }
    for sr in split_rows:
        b = sr.get("bucket", "")
        if b in bucket_map:
            bucket_map[b].append(sr)

    print("\nLoading row diagnostics...")
    diag_rows = load_jsonl(ROW_DIAGNOSTICS_PATH)
    diag_by_cid: dict[str, dict] = {}
    for dr in diag_rows:
        cid = str(dr.get("candidate_id", ""))
        diag_by_cid[cid] = dr

    print("Building artifact rows...")

    policy_value_rows: list[dict] = []
    value_only_rows: list[dict] = []
    controls_rows: list[dict] = []
    holdout_candidate_ids: list[str] = []
    validation_errors: list[str] = []

    for sr in split_rows:
        cid = sr.get("candidate_id", "")
        bucket = sr.get("bucket", "")
        mechanism = sr.get("mechanism", "")

        if cid not in diag_by_cid:
            validation_errors.append(f"Missing diag row: {cid}")
            continue

        dr = diag_by_cid[cid]
        state = dr.get("state")
        if not state:
            validation_errors.append(f"Missing state: {cid}")
            continue

        exact_tb = dr.get("exact_tablebase", {})
        root_value_wr = exact_tb.get("root_value")
        optimal_move = exact_tb.get("optimal_move")
        best_minus_second = exact_tb.get("best_minus_second_best")

        legal_moves = dr.get("legal_moves", [])
        encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
        c_hash = str(dr.get("canonical_state_hash", ""))

        if root_value_wr is None or optimal_move is None:
            validation_errors.append(f"{cid}: missing exact tablebase data")
            continue

        if optimal_move not in legal_moves:
            validation_errors.append(f"{cid}: optimal move not legal")
            continue

        root_value = float(root_value_wr)

        if bucket == "holdout_candidate":
            holdout_candidate_ids.append(cid)
            continue

        if bucket == "production_candidate_later":
            policy = build_policy_target(optimal_move, legal_moves)
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy,
                "value": root_value,
                "legal_moves": legal_moves,
                "source": FAMILY,
                "label_source": "exact_tablebase",
                "role": "production_candidate_later",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": root_value,
                "best_minus_second_best": best_minus_second,
                "row_mechanism": mechanism,
                "replay_role": "exact_tablebase_diagnostic",
                "bucket": "harder_fresh_endgame_tablebase",
                "family": FAMILY,
            }
            policy_value_rows.append(row_data)

        elif bucket == "value_only_candidate":
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "value": root_value,
                "legal_moves": legal_moves,
                "source": FAMILY,
                "label_source": "exact_tablebase",
                "role": "value_only_candidate",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": root_value,
                "best_minus_second_best": best_minus_second,
                "replay_role": "exact_tablebase_diagnostic_value_only",
                "policy_target_allowed": False,
                "reason": "tablebase exact value label, not policy target",
                "bucket": "harder_fresh_endgame_tablebase",
                "family": FAMILY,
            }
            value_only_rows.append(row_data)

        elif bucket == "preservation_control":
            policy = build_policy_target(optimal_move, legal_moves)
            row_data = {
                "candidate_id": cid,
                "canonical_state_hash": c_hash,
                "state": encoded_state,
                "raw_state": state,
                "policy": policy,
                "value": root_value,
                "legal_moves": legal_moves,
                "source": FAMILY,
                "label_source": "exact_tablebase",
                "role": "preservation_control",
                "train_only": True,
                "exclude_from_validation": True,
                "exact_optimal_move": optimal_move,
                "exact_root_value": root_value,
                "best_minus_second_best": best_minus_second,
                "replay_role": "exact_tablebase_diagnostic",
                "bucket": "harder_fresh_endgame_tablebase",
                "family": FAMILY,
            }
            controls_rows.append(row_data)

    print("\nArtifact rows:")
    print(f"  policy_value: {len(policy_value_rows)}")
    print(f"  value_only: {len(value_only_rows)}")
    print(f"  controls: {len(controls_rows)}")
    print(f"  holdout: {len(holdout_candidate_ids)}")

    if len(policy_value_rows) < 5:
        print("\nERROR: Fewer than 5 production candidates remain valid.")
        print("Classify: exact_tablebase_artifact_not_enough_signal")
        return 1

    summary = {
        "schema": "azlite_harder_endgame_tablebase_diagnostic_artifact_v1",
        "family": FAMILY,
        "description": (
            "Tiny train-only exact-tablebase diagnostic artifact "
            "built from PR #74 clean split."
        ),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "exhausted_family_excluded": True,
        },
        "input_encoding": INPUT_ENCODING,
        "artifacts": {
            "policy_value": {
                "path": str(OUTPUT_DIR / "exact_tablebase_policy_value_artifact.jsonl"),
                "row_count": len(policy_value_rows),
                "roles": ["production_candidate_later"],
                "target_types": ["policy", "value"],
                "policy_target_mass": OPTIMAL_POLICY_MASS,
                "value_source": "exact_tablebase",
            },
            "value_only": {
                "path": str(OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl"),
                "row_count": len(value_only_rows),
                "roles": ["value_only_candidate"],
                "target_types": ["value"],
                "policy_target_allowed": False,
                "value_source": "exact_tablebase",
            },
            "controls": {
                "path": str(OUTPUT_DIR / "exact_tablebase_controls_artifact.jsonl"),
                "row_count": len(controls_rows),
                "roles": ["preservation_control"],
                "target_types": ["policy", "value"],
                "policy_target_mass": OPTIMAL_POLICY_MASS,
                "value_source": "exact_tablebase",
            },
        },
        "holdout_candidate_ids": holdout_candidate_ids,
        "counts": {
            "policy_value_rows": len(policy_value_rows),
            "value_only_rows": len(value_only_rows),
            "controls_rows": len(controls_rows),
            "holdout_count": len(holdout_candidate_ids),
        },
    }

    write_jsonl(
        OUTPUT_DIR / "exact_tablebase_policy_value_artifact.jsonl", policy_value_rows
    )
    write_jsonl(
        OUTPUT_DIR / "exact_tablebase_value_only_artifact.jsonl", value_only_rows
    )
    write_jsonl(OUTPUT_DIR / "exact_tablebase_controls_artifact.jsonl", controls_rows)

    errors = validate_artifacts(
        policy_value_rows, value_only_rows, controls_rows, summary
    )
    if errors:
        print(f"\nValidation ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        print("Artifact validation FAILED.")
        return 1
    else:
        print("\nStatic validation PASSED.")

    write_json(OUTPUT_DIR / "artifact_summary.json", summary)
    print(f"\nArtifact summary written to {OUTPUT_DIR / 'artifact_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
