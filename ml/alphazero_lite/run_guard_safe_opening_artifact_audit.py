#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_tracked_opening_capture_policy_artifact import (
    derive_policy,
    encode_raw_state,
)
from ml.alphazero_lite.capture_002_003_rule_collision_diagnostic import (
    simulate_move_rule_features,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
    row_map_from_reference,
    write_json,
)


DEFAULT_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_FALLBACK_REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_CURRENT_PATH = Path("storage/ai/alphazero_lite/current")
DEFAULT_SOURCE_ARTIFACT = Path(
    "/tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact.jsonl"
)
DEFAULT_OPENING_SUBFAMILY_DIAGNOSTIC = Path(
    "/tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_guard_safe_opening_replay")
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-guard-safe-opening-replay-artifact-audit-results.md"
)
GUARD_ROW_IDS = (
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
PRIMARY_GUARD_ROW_IDS = ("capture_available-002", "capture_available-003")
LEAVE_ONE_OUT_SUBFAMILIES = (
    "opening_extra_turn_overbias",
    "opening_edge_move_5_preference",
    "opening_missed_extra_turn_continuation",
)
SCHEMA = "azlite_guard_safe_opening_replay_audit_v1"
RULE_COLLISION_EXTRA_TURN_GUARD_ROLE = "rule_collision_extra_turn_reference_guard"
RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE = "rule_collision_no_extra_turn_reference_guard"
POLICY_BUCKET_KEYS = (
    "extra_turn_capture",
    "extra_turn_no_capture",
    "no_extra_turn_capture",
    "no_extra_turn_no_capture",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--current-path", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--source-artifact", type=Path, default=DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument(
        "--opening-subfamily-diagnostic",
        type=Path,
        default=DEFAULT_OPENING_SUBFAMILY_DIAGNOSTIC,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def entropy(policy: list[float], legal_moves: list[int]) -> float:
    total = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            total -= probability * math.log(probability, 2)
    return round(total, 4)


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int:
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def side_to_move_pits(state: dict[str, Any]) -> list[int]:
    if int(state["current_player"]) == 0:
        return list(state["player_pits"])
    return list(state["opponent_pits"])


def opponent_pits(state: dict[str, Any]) -> list[int]:
    if int(state["current_player"]) == 0:
        return list(state["opponent_pits"])
    return list(state["player_pits"])


def ply_bucket(state: dict[str, Any]) -> str:
    total_store = int(state["player_store"]) + int(state["opponent_store"])
    if total_store <= 1:
        return "store_0_1"
    if total_store <= 3:
        return "store_2_3"
    return "store_4_plus"


def mask_for_legal_moves(legal_moves: list[int]) -> list[int]:
    return [1 if move in legal_moves else 0 for move in range(6)]


def move_bucket_key(*, gives_extra_turn: bool, produces_capture: bool) -> str:
    if gives_extra_turn and produces_capture:
        return "extra_turn_capture"
    if gives_extra_turn:
        return "extra_turn_no_capture"
    if produces_capture:
        return "no_extra_turn_capture"
    return "no_extra_turn_no_capture"


def move_consequences(*, state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    original_player = game.current_player
    original_store = game.captured_seeds[original_player]
    original_pits = list(game.pits)
    absolute_index = game.pit_index(move)
    seeds = game.pits[absolute_index]
    if move not in game.possible_moves() or seeds <= 0:
        raise ValueError(f"move {move} must be legal for provided state")

    track = [
        *[("player", pit_index) for pit_index in range(6)],
        ("store", None),
        *[("opponent", pit_index) for pit_index in reversed(range(6))],
    ]
    position = move
    last_side = None
    last_index = None
    game.pits[absolute_index] = 0

    for _ in range(seeds):
        position = (position + 1) % len(track)
        last_side, last_index = track[position]
        if last_side == "player":
            game.pits[last_index] += 1
        elif last_side == "opponent":
            game.pits[6 + last_index] += 1
        else:
            game.captured_seeds[original_player] += 1

    gives_extra_turn = last_side == "store"
    capture_count = 0
    if not gives_extra_turn:
        if last_index is None:
            raise ValueError("legal non-store move must land in a pit")
        landing_absolute = last_index if last_side == "player" else 6 + int(last_index)
        if last_side == "player" and game.pits[landing_absolute] == 1:
            opposite_index = game.opposite_pit_index(landing_absolute)
            opposite_seeds = game.pits[opposite_index]
            if opposite_seeds > 0:
                capture_count = int(opposite_seeds)
        game._capture(landing_absolute)
        game.current_player = game.opposite_player()

    if game.over():
        game._after_game_over()

    immediate_store_delta = int(game.captured_seeds[original_player] - original_store)
    landing_seed = 1 if capture_count > 0 else 0
    post_state = game.to_state()
    feature_view = simulate_move_rule_features(state=state, move=move)
    produced_capture = capture_count > 0
    seed_pass_delta = max(0, immediate_store_delta - capture_count - landing_seed)
    return {
        "move": int(move),
        "gives_extra_turn": bool(gives_extra_turn),
        "produces_capture": bool(produced_capture),
        "capture_count": int(capture_count),
        "store_pass_gain": int(seed_pass_delta),
        "immediate_store_delta": int(immediate_store_delta),
        "side_to_move_after": int(post_state["current_player"]),
        "landing_pit": feature_view.get("landing_pit"),
        "lands_in_store": bool(feature_view.get("lands_in_store", False)),
        "post_move_state": post_state,
        "pit_delta": [
            int(after - before)
            for before, after in zip(original_pits, game.pits, strict=False)
        ],
    }


def deterministic_features(
    *,
    state: dict[str, Any],
    legal_moves: list[int],
    move_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    extra_turn_mask = [
        1 if move in legal_moves and move_map[move]["gives_extra_turn"] else 0
        for move in range(6)
    ]
    capture_mask = [
        1 if move in legal_moves and move_map[move]["produces_capture"] else 0
        for move in range(6)
    ]
    current_vector = side_to_move_pits(state)
    return {
        "ply_bucket": ply_bucket(state),
        "current_player": int(state["current_player"]),
        "legal_move_mask": mask_for_legal_moves(legal_moves),
        "extra_turn_move_mask": extra_turn_mask,
        "capture_move_mask": capture_mask,
        "seed_count_pattern": {
            "current": list(current_vector),
            "opponent": list(opponent_pits(state)),
        },
        "current_player_pit_vector": list(current_vector),
    }


def pit_distance(left: list[int], right: list[int]) -> int:
    return int(sum(abs(int(a) - int(b)) for a, b in zip(left, right, strict=False)))


def similarity_to_guard(
    *,
    row_features: dict[str, Any],
    guard_features: dict[str, Any],
) -> dict[str, Any]:
    row_vector = list(row_features["current_player_pit_vector"])
    guard_vector = list(guard_features["current_player_pit_vector"])
    vector_distance = pit_distance(row_vector, guard_vector)
    mask_matches = {
        "legal_move_mask": row_features["legal_move_mask"]
        == guard_features["legal_move_mask"],
        "extra_turn_move_mask": row_features["extra_turn_move_mask"]
        == guard_features["extra_turn_move_mask"],
        "capture_move_mask": row_features["capture_move_mask"]
        == guard_features["capture_move_mask"],
    }
    exact_mask_match_count = sum(1 for matched in mask_matches.values() if matched)
    same_bucket = row_features["ply_bucket"] == guard_features["ply_bucket"]
    same_player = row_features["current_player"] == guard_features["current_player"]
    seed_distance_score = max(0.0, 1.0 - (vector_distance / 12.0))
    score = (
        (1.0 if same_bucket else 0.0)
        + (1.0 if same_player else 0.0)
        + float(exact_mask_match_count)
        + seed_distance_score
    )
    high_similarity = (
        same_player
        and same_bucket
        and mask_matches["legal_move_mask"]
        and mask_matches["extra_turn_move_mask"]
        and mask_matches["capture_move_mask"]
        and vector_distance <= 4
    )
    return {
        "score": round(score, 4),
        "high_similarity": bool(high_similarity),
        "vector_distance": int(vector_distance),
        "mask_matches": mask_matches,
    }


def source_family_for_row(
    row_id: str, bucket: str | None, subfamily: str | None
) -> str:
    if subfamily:
        return subfamily
    if row_id.startswith("capture_available-"):
        return "corrected_capture_guard"
    if bucket:
        return bucket
    return "unknown"


def build_supplemental_guard_row(reference_row: dict[str, Any]) -> dict[str, Any]:
    raw_state = dict(reference_row["state"])
    reference_move = int(reference_row["reference_move"])
    reference_features = simulate_move_rule_features(
        state=raw_state, move=reference_move
    )
    replay_role = (
        RULE_COLLISION_EXTRA_TURN_GUARD_ROLE
        if reference_features["extra_turn_available"]
        else RULE_COLLISION_NO_EXTRA_TURN_GUARD_ROLE
    )
    child_stats = []
    for child in list(reference_row.get("child_stats") or []):
        child_stats.append(
            {
                "move": int(child["move"]),
                "visits": int(child.get("visits", 0)),
                "win_rate": float(child.get("win_rate", 0.0)),
            }
        )
    return {
        "canonical_state": str(reference_row["canonical_state"]),
        "state": encode_raw_state(raw_state=raw_state, input_encoding="kalah_v3"),
        "raw_state": raw_state,
        "side_to_move": int(raw_state["current_player"]),
        "legal_moves": [int(child["move"]) for child in child_stats],
        "policy": derive_policy(child_stats),
        "value": float(reference_row["teacher_value"]),
        "bucket": "capture_available",
        "bucket_group": "tactical",
        "input_encoding": "kalah_v3",
        "policy_target_mode": "sharpened",
        "value_target_mode": "sharpened",
        "selection_reasons": [
            "guard_safe_opening_replay_audit",
            "corrected_guard_preservation",
        ],
        "source_artifacts": [],
        "source_runs": [
            {
                "kind": "guard_safe_opening_guard",
                "id": str(reference_row["id"]),
                "reference_move": reference_move,
            }
        ],
        "priority_score": 25.0,
        "teacher_policy_simulations": 1200,
        "teacher_value_simulations": 1800,
        "teacher_seed": 2040,
        "teacher_policy_seed": 2040,
        "teacher_value_seed": 2040,
        "teacher_selected_move": reference_move,
        "teacher_child_stats": child_stats,
        "replay_role": replay_role,
        "reference_move": reference_move,
        "reference_move_extra_turn_available": bool(
            reference_features["extra_turn_available"]
        ),
    }


def reference_row_for_id(
    *,
    row_id: str,
    reference_rows: dict[str, dict[str, Any]],
    fallback_reference_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_row = reference_rows.get(row_id)
    if reference_row is not None:
        return reference_row
    fallback_row = fallback_reference_rows.get(row_id)
    if fallback_row is not None:
        return fallback_row
    raise KeyError(row_id)


def normalize_source_row(
    *,
    row: dict[str, Any],
    reference_rows: dict[str, dict[str, Any]],
    subfamily_by_row_id: dict[str, str],
) -> dict[str, Any]:
    source_runs = list(row.get("source_runs") or [])
    source_row = source_runs[0] if source_runs else {}
    row_id = str(source_row.get("id") or row.get("id"))
    state = dict(row["raw_state"])
    legal_moves = [int(move) for move in list(row.get("legal_moves") or [])]
    policy = [float(value) for value in list(row.get("policy") or [])]
    corrected_reference_move_value = (reference_rows.get(row_id) or {}).get(
        "reference_move"
    )
    if corrected_reference_move_value is None:
        corrected_reference_move_value = row.get(
            "teacher_selected_move", row.get("reference_move")
        )
    if corrected_reference_move_value is None:
        raise ValueError(f"row {row_id} is missing corrected reference move")
    corrected_reference_move = int(corrected_reference_move_value)
    move_map = {move: move_consequences(state=state, move=move) for move in legal_moves}
    top_move = top_policy_move(policy, legal_moves)
    features = deterministic_features(
        state=state, legal_moves=legal_moves, move_map=move_map
    )
    subfamily = subfamily_by_row_id.get(row_id)
    bucket = row.get("bucket") if isinstance(row.get("bucket"), str) else None
    source_family = source_family_for_row(row_id, bucket, subfamily)
    target_consequence = move_map[top_move]
    return {
        "row_id": row_id,
        "canonical_state": str(row.get("canonical_state", "")),
        "source_family": source_family,
        "subfamily": subfamily,
        "bucket": bucket,
        "row": row,
        "raw_state": state,
        "legal_moves": legal_moves,
        "policy": policy,
        "corrected_reference_move": corrected_reference_move,
        "policy_target_top_move": top_move,
        "policy_target_entropy": entropy(policy, legal_moves),
        "per_move_consequences": {str(move): move_map[move] for move in legal_moves},
        "target_move_gives_extra_turn": bool(target_consequence["gives_extra_turn"]),
        "target_move_is_capture": bool(target_consequence["produces_capture"]),
        "has_no_extra_turn_capture_alternative": any(
            move_map[move]["produces_capture"]
            and not move_map[move]["gives_extra_turn"]
            for move in legal_moves
        ),
        "deterministic_features": features,
        "is_explicit_guard_row": row_id in GUARD_ROW_IDS,
    }


def assign_conflicts(audits: list[dict[str, Any]]) -> None:
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        by_canonical[audit["canonical_state"]].append(audit)

    feature_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        features = audit["deterministic_features"]
        group_key = (
            features["current_player"],
            features["ply_bucket"],
            tuple(features["legal_move_mask"]),
            tuple(features["extra_turn_move_mask"]),
            tuple(features["capture_move_mask"]),
        )
        feature_groups[group_key].append(audit)

    guard_audits = [audit for audit in audits if audit["row_id"] in GUARD_ROW_IDS]
    no_extra_turn_guard_audits = [
        audit
        for audit in guard_audits
        if not audit["per_move_consequences"][str(audit["corrected_reference_move"])][
            "gives_extra_turn"
        ]
    ]

    for audit in audits:
        similarity_rows = []
        for guard in guard_audits:
            similarity = similarity_to_guard(
                row_features=audit["deterministic_features"],
                guard_features=guard["deterministic_features"],
            )
            similarity_rows.append(
                {
                    "guard_row_id": guard["row_id"],
                    "score": similarity["score"],
                    "high_similarity": similarity["high_similarity"],
                    "vector_distance": similarity["vector_distance"],
                    "mask_matches": similarity["mask_matches"],
                }
            )
        audit["guard_similarity"] = similarity_rows
        high_similarity_to_different_target = [
            row
            for row in similarity_rows
            if row["high_similarity"]
            and audit["policy_target_top_move"]
            != next(
                guard["corrected_reference_move"]
                for guard in guard_audits
                if guard["row_id"] == row["guard_row_id"]
            )
        ]
        audit["conflict_flags"] = {
            "target_extra_turn_over_corrected_no_extra_turn_guard": any(
                audit["policy_target_top_move"] != guard["corrected_reference_move"]
                and audit["target_move_gives_extra_turn"]
                and any(
                    similarity_row["guard_row_id"] == guard["row_id"]
                    and similarity_row["high_similarity"]
                    for similarity_row in similarity_rows
                )
                for guard in no_extra_turn_guard_audits
            ),
            "target_move5_edge_bias": audit["policy_target_top_move"] == 5
            and audit["corrected_reference_move"] != 5,
            "contradicts_corrected_reference_artifact": audit["policy_target_top_move"]
            != audit["corrected_reference_move"],
            "near_duplicate_with_different_target": False,
            "high_similarity_to_guard_with_different_target": bool(
                high_similarity_to_different_target
            ),
        }
        audit["high_similarity_guard_row_ids"] = [
            row["guard_row_id"] for row in similarity_rows if row["high_similarity"]
        ]

    for duplicates in by_canonical.values():
        top_moves = {audit["policy_target_top_move"] for audit in duplicates}
        if len(top_moves) > 1:
            for audit in duplicates:
                audit["conflict_flags"]["near_duplicate_with_different_target"] = True

    for duplicates in feature_groups.values():
        top_moves = {audit["policy_target_top_move"] for audit in duplicates}
        if len(top_moves) > 1:
            for audit in duplicates:
                audit.setdefault("near_duplicate_group_row_ids", []).extend(
                    other["row_id"]
                    for other in duplicates
                    if other["row_id"] != audit["row_id"]
                )


def conflict_flag_count(audit: dict[str, Any]) -> int:
    return sum(1 for value in audit["conflict_flags"].values() if value)


def policy_bucket_rates(audits: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter()
    total = 0
    for audit in audits:
        top_move = audit["policy_target_top_move"]
        consequence = audit["per_move_consequences"][str(top_move)]
        counts[
            move_bucket_key(
                gives_extra_turn=bool(consequence["gives_extra_turn"]),
                produces_capture=bool(consequence["produces_capture"]),
            )
        ] += 1
        total += 1
    rates = {}
    for key in POLICY_BUCKET_KEYS:
        rates[key] = 0.0 if total == 0 else round(counts[key] / total, 4)
    return rates


def guard_validation_for_artifact(
    audits: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    audits_by_id = {audit["row_id"]: audit for audit in audits}
    if any(row_id not in audits_by_id for row_id in GUARD_ROW_IDS):
        rows = []
        for row_id in GUARD_ROW_IDS:
            audit = audits_by_id.get(row_id)
            rows.append(
                {
                    "row_id": row_id,
                    "corrected_reference_move": None
                    if audit is None
                    else audit["corrected_reference_move"],
                    "guard_row_present": audit is not None,
                    "conflicting_duplicate_count": 0,
                    "near_duplicate_conflict_count": 0,
                    "classification": "invalid_missing_guard",
                    "notes": "missing_guard_row" if audit is None else "present",
                }
            )
        return "invalid_missing_guard", rows

    rows = []
    overall = "guard_safe"
    for row_id in GUARD_ROW_IDS:
        audit = audits_by_id[row_id]
        duplicates = [
            other
            for other in audits
            if other["canonical_state"] == audit["canonical_state"]
            and other["row_id"] != row_id
            and other["policy_target_top_move"] != audit["corrected_reference_move"]
        ]
        near_conflicts = [
            other
            for other in audits
            if other["row_id"] != row_id
            and row_id in other.get("high_similarity_guard_row_ids", [])
            and other["policy_target_top_move"] != audit["corrected_reference_move"]
        ]
        classification = "guard_safe"
        notes = []
        if duplicates:
            classification = "invalid_conflicting_duplicate"
            notes.append("conflicting_duplicate_guard_state")
        elif near_conflicts:
            classification = "guard_risky"
            notes.append("near_duplicate_conflict")
        if classification == "invalid_conflicting_duplicate":
            overall = "invalid_conflicting_duplicate"
        elif classification == "guard_risky" and overall == "guard_safe":
            overall = "guard_risky"
        rows.append(
            {
                "row_id": row_id,
                "corrected_reference_move": audit["corrected_reference_move"],
                "guard_row_present": True,
                "conflicting_duplicate_count": len(duplicates),
                "near_duplicate_conflict_count": len(near_conflicts),
                "classification": classification,
                "notes": ",".join(notes) if notes else "ok",
            }
        )
    return overall, rows


def summarize_row_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows = []
    for source_family in sorted({audit["source_family"] for audit in audits}):
        family_rows = [
            audit for audit in audits if audit["source_family"] == source_family
        ]
        high_similarity_to_002 = sum(
            1
            for audit in family_rows
            if "capture_available-002" in audit["high_similarity_guard_row_ids"]
        )
        high_similarity_to_003 = sum(
            1
            for audit in family_rows
            if "capture_available-003" in audit["high_similarity_guard_row_ids"]
        )
        summary_rows.append(
            {
                "source_family": source_family,
                "rows": len(family_rows),
                "conflict_rows": sum(
                    1 for audit in family_rows if conflict_flag_count(audit) > 0
                ),
                "target_extra_turn_rate": round(
                    sum(
                        1
                        for audit in family_rows
                        if audit["target_move_gives_extra_turn"]
                    )
                    / len(family_rows),
                    4,
                ),
                "no_extra_turn_capture_available_rate": round(
                    sum(
                        1
                        for audit in family_rows
                        if audit["has_no_extra_turn_capture_alternative"]
                    )
                    / len(family_rows),
                    4,
                ),
                "high_similarity_to_002_count": high_similarity_to_002,
                "high_similarity_to_003_count": high_similarity_to_003,
                "notes": "guard_rows"
                if source_family == "corrected_capture_guard"
                else "subfamily_conflicts"
                if any(conflict_flag_count(audit) > 0 for audit in family_rows)
                else "clean",
            }
        )
    return {
        "schema": SCHEMA,
        "row_count": len(audits),
        "summary_rows": summary_rows,
        "conflict_flag_totals": {
            key: sum(1 for audit in audits if audit["conflict_flags"][key])
            for key in audits[0]["conflict_flags"]
        }
        if audits
        else {},
    }


def variant_note(name: str, classification: str, opening_row_count: int) -> str:
    if classification == "invalid_missing_guard":
        return "missing corrected guard rows"
    if classification == "invalid_conflicting_duplicate":
        return "duplicate canonical guard conflict remains"
    if classification == "guard_risky":
        return "near-duplicate guard conflicts remain"
    if opening_row_count <= 0:
        return "diagnostic-only guard artifact"
    return "guard-safe static artifact"


def included_families(audits: list[dict[str, Any]]) -> list[str]:
    return sorted({audit["source_family"] for audit in audits})


def select_artifact(artifact_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        summary
        for summary in artifact_summaries
        if summary["classification"] == "guard_safe"
        and summary["opening_row_count"] >= 5
        and summary["artifact_name"] != "guard_safe_controls_only"
    ]
    if not eligible:
        return None
    preferred = sorted(
        eligible,
        key=lambda row: (
            row["row_count"],
            0 if row["artifact_name"] == "guard_safe_no_extra_turn_overbias" else 1,
            row["artifact_name"],
        ),
    )
    return preferred[0]


def artifact_rows_from_audits(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(json.dumps(audit["row"])) for audit in audits]


def build_variants(
    *,
    audits: list[dict[str, Any]],
    output_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    guard_audits = [audit for audit in audits if audit["is_explicit_guard_row"]]
    opening_audits = [audit for audit in audits if not audit["is_explicit_guard_row"]]
    by_subfamily: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audit in opening_audits:
        if audit["subfamily"]:
            by_subfamily[str(audit["subfamily"])].append(audit)

    variants: dict[str, list[dict[str, Any]]] = {}
    variants["guard_safe_strict"] = sorted(
        [audit for audit in opening_audits if conflict_flag_count(audit) == 0]
        + guard_audits,
        key=lambda audit: audit["row_id"],
    )
    variants["guard_safe_no_extra_turn_overbias"] = sorted(
        [
            audit
            for audit in opening_audits
            if not (
                audit["subfamily"] == "opening_extra_turn_overbias"
                and (
                    audit["conflict_flags"][
                        "target_extra_turn_over_corrected_no_extra_turn_guard"
                    ]
                    or audit["conflict_flags"][
                        "high_similarity_to_guard_with_different_target"
                    ]
                )
            )
        ]
        + guard_audits,
        key=lambda audit: audit["row_id"],
    )
    variants["guard_safe_controls_only"] = sorted(
        [
            audit
            for audit in opening_audits
            if conflict_flag_count(audit) == 0
            and any(
                row_id in GUARD_ROW_IDS
                for row_id in audit["high_similarity_guard_row_ids"]
            )
            and audit["policy_target_top_move"] == audit["corrected_reference_move"]
        ]
        + guard_audits,
        key=lambda audit: audit["row_id"],
    )
    for subfamily in LEAVE_ONE_OUT_SUBFAMILIES:
        if subfamily not in by_subfamily:
            continue
        variants[f"family_leave_one_out_without_{subfamily}"] = sorted(
            [audit for audit in opening_audits if audit["subfamily"] != subfamily]
            + guard_audits,
            key=lambda audit: audit["row_id"],
        )

    artifact_summaries = []
    validation_rows_by_artifact = {}
    audits_by_artifact = {}
    for artifact_name, artifact_audits in variants.items():
        artifact_path = output_root / f"{artifact_name}.jsonl"
        artifact_summary_path = output_root / f"{artifact_name}_summary.json"
        write_jsonl(artifact_path, artifact_rows_from_audits(artifact_audits))
        classification, validation_rows = guard_validation_for_artifact(artifact_audits)
        opening_row_count = sum(
            1 for audit in artifact_audits if not audit["is_explicit_guard_row"]
        )
        excluded_rows = [
            audit
            for audit in audits
            if audit["row_id"] not in {item["row_id"] for item in artifact_audits}
        ]
        summary = {
            "artifact_name": artifact_name,
            "path": str(artifact_path),
            "row_count": len(artifact_audits),
            "opening_row_count": opening_row_count,
            "excluded_count": len(audits) - len(artifact_audits),
            "included_families": included_families(artifact_audits),
            "excluded_families": included_families(excluded_rows),
            "guard_rows_present": all(
                row["guard_row_present"] for row in validation_rows
            ),
            "conflict_flag_count": sum(
                conflict_flag_count(audit) for audit in artifact_audits
            ),
            "conflict_flag_counts": {
                key: sum(1 for audit in artifact_audits if audit["conflict_flags"][key])
                for key in artifact_audits[0]["conflict_flags"]
            }
            if artifact_audits
            else {},
            "policy_target_bucket_rates": policy_bucket_rates(artifact_audits),
            "classification": classification,
            "notes": variant_note(artifact_name, classification, opening_row_count),
        }
        write_json(artifact_summary_path, summary)
        artifact_summaries.append(summary)
        validation_rows_by_artifact[artifact_name] = validation_rows
        audits_by_artifact[artifact_name] = artifact_audits
    return artifact_summaries, validation_rows_by_artifact, audits_by_artifact


def maybe_run_probe(
    *,
    selected_artifact: dict[str, Any] | None,
    current_path: Path,
    reference_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if selected_artifact is None:
        return {
            "status": "skipped",
            "reason": "no_guard_safe_artifact_selected",
            "rows": [],
        }
    # The checked-in training path initializes from .npz, while the live artifact ships as weights.json.
    return {
        "status": "skipped",
        "reason": "tiny_probe_requires_weight_json_to_train_init_bridge",
        "rows": [],
        "selected_artifact": selected_artifact["artifact_name"],
        "current_path": str(current_path),
        "guard_row_ids": list(GUARD_ROW_IDS),
    }


def render_report(
    *,
    source_artifact_path: Path,
    source_row_count: int,
    row_audit_summary: dict[str, Any],
    artifact_summaries: list[dict[str, Any]],
    validation_rows_by_artifact: dict[str, list[dict[str, Any]]],
    probe_summary: dict[str, Any],
    selected_artifact: dict[str, Any] | None,
    overall_decision: dict[str, str],
) -> str:
    lines = [
        "# AlphaZero-lite Guard-Safe Opening Replay Artifact Audit Results",
        "",
        "## 1. Context",
        "",
        "- PR #33 classified corrected opening replay as `replay_induced_guard_regression`.",
        "- This audit avoids another broad replay sweep and instead isolates static artifact rows that conflict with corrected guard behavior.",
        "- Corrected tracked references remained the default source of truth throughout the audit.",
        "",
        "## 2. Why PR #33 paused replay",
        "",
        "- Current already passes corrected `capture_available-002/003` at baseline under the corrected tracked references.",
        "- Corrected hard-state replay stayed locally acceptable on the 1200-budget guard rows.",
        "- The replay regression was isolated to the corrected opening replay branch, especially `opening_extra_turn_overbias_corrected_w1/w2`.",
        "- The right next step was therefore artifact surgery, not more replay weight search, training, arena, or promotion.",
        "",
        "## 3. Source artifact summary",
        "",
        f"- Source artifact: `{source_artifact_path}`.",
        f"- Source row count audited after adding missing corrected guards 006/007/008: `{source_row_count}`.",
        "- The source mix contains opening replay rows plus explicit corrected guard rows.",
        "",
        "## 4. Row-level conflict audit",
        "",
        "| source_family | rows | conflict_rows | target_extra_turn_rate | no_extra_turn_capture_available_rate | high_similarity_to_002_count | high_similarity_to_003_count | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in row_audit_summary["summary_rows"]:
        lines.append(
            f"| {row['source_family']} | {row['rows']} | {row['conflict_rows']} | {row['target_extra_turn_rate']:.4f} | {row['no_extra_turn_capture_available_rate']:.4f} | {row['high_similarity_to_002_count']} | {row['high_similarity_to_003_count']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Filtered artifact variants",
            "",
            "| artifact_name | path | row_count | excluded_count | included_families | excluded_families | guard_rows_present | conflict_flag_count | classification | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in artifact_summaries:
        lines.append(
            f"| {row['artifact_name']} | `{row['path']}` | {row['row_count']} | {row['excluded_count']} | `{json.dumps(row['included_families'])}` | `{json.dumps(row['excluded_families'])}` | {str(bool(row['guard_rows_present'])).lower()} | {row['conflict_flag_count']} | `{row['classification']}` | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Offline guard-safety validation",
            "",
            "| artifact_name | row_id | corrected_reference_move | guard_row_present | conflicting_duplicate_count | near_duplicate_conflict_count | classification | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for artifact_name in sorted(validation_rows_by_artifact):
        for row in validation_rows_by_artifact[artifact_name]:
            lines.append(
                f"| {artifact_name} | {row['row_id']} | {row['corrected_reference_move']} | {str(bool(row['guard_row_present'])).lower()} | {row['conflicting_duplicate_count']} | {row['near_duplicate_conflict_count']} | `{row['classification']}` | {row['notes']} |"
            )
    lines.extend(
        [
            "",
            "## 7. Optional overfit probe",
            "",
            f"- Status: `{probe_summary['status']}`.",
            f"- Notes: `{probe_summary['reason']}`.",
            "- The probe was intentionally not replaced with production training, arena, or export logic.",
            "",
            "## 8. Selected artifact for next experiment",
            "",
        ]
    )
    if selected_artifact is None:
        lines.append(
            "- No opening artifact met the guard-safe plus minimum-coverage bar for a later training lane."
        )
    else:
        lines.extend(
            [
                f"- Selected artifact: `{selected_artifact['artifact_name']}`.",
                f"- Path: `{selected_artifact['path']}`.",
                f"- Why: smallest guard-safe artifact that still retains meaningful opening coverage (`{selected_artifact['opening_row_count']}` opening rows) while preserving corrected 002/003/006/007/008 guards.",
            ]
        )
    lines.extend(
        [
            "",
            "## 9. Exactly one recommended next action",
            "",
            f"Classification: `{overall_decision['classification']}`.",
            "",
            f"Recommendation: **{overall_decision['recommended_next_action']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def classify_overall(
    *,
    row_audits: list[dict[str, Any]],
    artifact_summaries: list[dict[str, Any]],
    selected_artifact: dict[str, Any] | None,
) -> dict[str, str]:
    opening_conflict_rows = [
        audit
        for audit in row_audits
        if not audit["is_explicit_guard_row"] and conflict_flag_count(audit) > 0
    ]
    subfamily_conflicts = Counter()
    for audit in row_audits:
        if audit["subfamily"] and conflict_flag_count(audit) > 0:
            subfamily_conflicts[str(audit["subfamily"])] += 1
    risky_opening_artifacts = [
        row
        for row in artifact_summaries
        if row["artifact_name"] != "guard_safe_controls_only"
        and row["opening_row_count"] > 0
        and row["classification"] != "guard_safe"
    ]
    if any(
        row["classification"] == "invalid_conflicting_duplicate"
        for row in artifact_summaries
    ):
        return {
            "classification": "canonical_target_conflict",
            "recommended_next_action": "fix artifact canonicalization/target generation before any training.",
        }
    if not opening_conflict_rows:
        selected_name = (
            None if selected_artifact is None else selected_artifact["artifact_name"]
        )
        return {
            "classification": "optimization_interaction_not_static_artifact_conflict",
            "recommended_next_action": "run a tiny overfit probe or low-epoch training trace"
            + (" on `" + selected_name + "`" if selected_name else "")
            + " to identify when corrected 002/003/006/007/008 guard policy drifts.",
        }
    if selected_artifact is not None:
        return {
            "classification": "guard_safe_artifact_ready",
            "recommended_next_action": f"run one small controlled training lane with `{selected_artifact['artifact_name']}` at weight 1 only, with a pre-arena corrected 002/003/006/007/008 guard kill gate.",
        }
    if subfamily_conflicts:
        dominant_subfamily, dominant_count = subfamily_conflicts.most_common(1)[0]
        if dominant_count >= max(4, sum(subfamily_conflicts.values()) // 2):
            return {
                "classification": "subfamily_poisoning_identified",
                "recommended_next_action": f"build a training artifact excluding `{dominant_subfamily}` and keep it diagnostic-only until a separate guard-safe lane clears corrected 002/003/006/007/008.",
            }
    if risky_opening_artifacts:
        return {
            "classification": "opening_replay_inherently_conflicted",
            "recommended_next_action": "abandon opening replay and move to broader corrected hard-state mining from non-opening failures.",
        }
    return {
        "classification": "optimization_interaction_not_static_artifact_conflict",
        "recommended_next_action": "run a tiny overfit probe or low-epoch training trace to identify when corrected guard policy drifts.",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    reference_artifact = resolve_path(root, args.reference_artifact)
    fallback_reference_artifact = resolve_path(root, args.fallback_reference_artifact)
    current_path = resolve_path(root, args.current_path)
    source_artifact = resolve_path(root, args.source_artifact)
    opening_subfamily_diagnostic = resolve_path(root, args.opening_subfamily_diagnostic)
    output_root = resolve_path(root, args.output_root)
    report_path = resolve_path(root, args.report_path)

    reference_rows = row_map_from_reference(load_json(reference_artifact))
    fallback_reference_rows = row_map_from_reference(
        load_json(fallback_reference_artifact)
    )
    source_rows = read_jsonl(source_artifact)
    subfamily_payload = load_json(opening_subfamily_diagnostic)
    subfamily_by_row_id = {
        str(row["row_id"]): str(row["subfamily"])
        for row in list(subfamily_payload.get("rows") or [])
        if isinstance(row, dict)
        and isinstance(row.get("row_id"), str)
        and isinstance(row.get("subfamily"), str)
    }

    source_rows = [
        row
        for row in source_rows
        if str((row.get("source_runs") or [{}])[0].get("id")) not in GUARD_ROW_IDS
    ]
    for row_id in GUARD_ROW_IDS:
        source_rows.append(
            build_supplemental_guard_row(
                reference_row=reference_row_for_id(
                    row_id=row_id,
                    reference_rows=reference_rows,
                    fallback_reference_rows=fallback_reference_rows,
                )
            )
        )

    audits = [
        normalize_source_row(
            row=row,
            reference_rows=reference_rows,
            subfamily_by_row_id=subfamily_by_row_id,
        )
        for row in source_rows
    ]
    assign_conflicts(audits)
    for audit in audits:
        audit["conflict_flag_count"] = conflict_flag_count(audit)

    output_root.mkdir(parents=True, exist_ok=True)
    row_audit_path = output_root / "row_audit.jsonl"
    row_audit_summary_path = output_root / "row_audit_summary.json"
    row_audit_summary = summarize_row_audits(audits)
    write_jsonl(
        row_audit_path,
        [
            {
                key: value
                for key, value in audit.items()
                if key not in {"row", "policy", "raw_state"}
            }
            for audit in audits
        ],
    )
    write_json(row_audit_summary_path, row_audit_summary)

    artifact_summaries, validation_rows_by_artifact, _audits_by_artifact = (
        build_variants(
            audits=audits,
            output_root=output_root,
        )
    )
    artifact_summary_path = output_root / "artifact_variant_summary.json"
    guard_validation_path = output_root / "guard_validation_summary.json"
    write_json(artifact_summary_path, {"artifacts": artifact_summaries})
    write_json(guard_validation_path, validation_rows_by_artifact)

    selected_artifact = select_artifact(artifact_summaries)
    probe_summary = maybe_run_probe(
        selected_artifact=selected_artifact,
        current_path=current_path,
        reference_rows=reference_rows,
    )
    write_json(output_root / "probe_summary.json", probe_summary)

    overall_decision = classify_overall(
        row_audits=audits,
        artifact_summaries=artifact_summaries,
        selected_artifact=selected_artifact,
    )
    summary_payload = {
        "schema": SCHEMA,
        "reference_artifact": str(reference_artifact),
        "fallback_reference_artifact": str(fallback_reference_artifact),
        "current_path": str(current_path),
        "source_artifact": str(source_artifact),
        "row_audit_path": str(row_audit_path),
        "row_audit_summary_path": str(row_audit_summary_path),
        "artifact_variant_summary_path": str(artifact_summary_path),
        "guard_validation_summary_path": str(guard_validation_path),
        "probe_summary_path": str(output_root / "probe_summary.json"),
        "selected_artifact": None
        if selected_artifact is None
        else selected_artifact["artifact_name"],
        "classification": overall_decision["classification"],
        "recommended_next_action": overall_decision["recommended_next_action"],
    }
    write_json(output_root / "summary.json", summary_payload)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            source_artifact_path=source_artifact,
            source_row_count=len(audits),
            row_audit_summary=row_audit_summary,
            artifact_summaries=artifact_summaries,
            validation_rows_by_artifact=validation_rows_by_artifact,
            probe_summary=probe_summary,
            selected_artifact=selected_artifact,
            overall_decision=overall_decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_path": str(output_root / "summary.json"),
                "report_path": str(report_path),
                "classification": overall_decision["classification"],
                "selected_artifact": None
                if selected_artifact is None
                else selected_artifact["artifact_name"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
