#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, PITS_PER_PLAYER
from ml.alphazero_lite.self_play import encode_state


DEFAULT_POLICY_TARGET_MODE = "default"
DEFAULT_VALUE_TARGET_MODE = "default"
SUPPORTED_TEACHER_MODES = frozenset({"classic_mcts"})


def _validate_state_pits(state: dict[str, Any], *, field_name: str, source: str) -> list[int]:
    pits = state.get(field_name)
    if (
        not isinstance(pits, list)
        or len(pits) != PITS_PER_PLAYER
        or any(not isinstance(pit, int) or isinstance(pit, bool) for pit in pits)
    ):
        raise ValueError(f"{source} state {field_name} must be a list of {PITS_PER_PLAYER} integers")
    return list(pits)


def _validate_state_store(state: dict[str, Any], *, field_name: str, source: str) -> int:
    value = state.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{source} state {field_name} must be an integer")
    return int(value)


def _validate_current_player(state: dict[str, Any], *, source: str) -> int:
    current_player = state.get("current_player")
    if not isinstance(current_player, int) or isinstance(current_player, bool):
        raise ValueError(f"{source} state current_player must be an integer 0 or 1")
    if current_player not in (0, 1):
        raise ValueError(f"{source} state current_player must be 0 or 1")
    return int(current_player)


def _validate_legal_moves(legal_moves: list[Any], *, source: str) -> list[int]:
    if not legal_moves:
        raise ValueError(f"{source} legal_moves must not be empty")

    validated_legal_moves: list[int] = []
    seen_moves: set[int] = set()
    for index, move in enumerate(legal_moves):
        if not isinstance(move, int) or isinstance(move, bool):
            raise ValueError(f"{source} legal_moves[{index}] must be an integer")
        if not 0 <= move < PITS_PER_PLAYER:
            raise ValueError(f"{source} legal_moves[{index}] must be between 0 and {PITS_PER_PLAYER - 1}")
        if move in seen_moves:
            raise ValueError(f"{source} legal_moves must not contain duplicates")
        seen_moves.add(move)
        validated_legal_moves.append(move)
    return validated_legal_moves


def validate_hard_state_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    required_fields = ("state", "legal_moves", "priority_score", "selection_reasons")
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"{source} missing required field '{field_name}'")

    state = row["state"]
    if not isinstance(state, dict):
        raise ValueError(f"{source} state must be an object")
    validated_state = {
        "player_pits": _validate_state_pits(state, field_name="player_pits", source=source),
        "opponent_pits": _validate_state_pits(state, field_name="opponent_pits", source=source),
        "player_store": _validate_state_store(state, field_name="player_store", source=source),
        "opponent_store": _validate_state_store(state, field_name="opponent_store", source=source),
        "current_player": _validate_current_player(state, source=source),
    }

    legal_moves = row["legal_moves"]
    if not isinstance(legal_moves, list):
        raise ValueError(f"{source} legal_moves must be a list")
    validated_legal_moves = _validate_legal_moves(legal_moves, source=source)
    canonical_legal_moves = KalahGame.from_state(validated_state).possible_moves()
    if validated_legal_moves != canonical_legal_moves:
        raise ValueError(
            f"{source} legal_moves must match state-derived legal moves {canonical_legal_moves}"
        )

    selection_reasons = row["selection_reasons"]
    if not isinstance(selection_reasons, list):
        raise ValueError(f"{source} selection_reasons must be a list")
    validated_selection_reasons: list[str] = []
    for index, selection_reason in enumerate(selection_reasons):
        if not isinstance(selection_reason, str) or not selection_reason:
            raise ValueError(f"{source} selection_reasons[{index}] must be a non-empty string")
        validated_selection_reasons.append(selection_reason)

    source_artifacts = row.get("source_artifacts")
    if source_artifacts is not None and not isinstance(source_artifacts, list):
        raise ValueError(f"{source} source_artifacts must be a list")
    validated_source_artifacts: list[str] | None = None
    if source_artifacts is not None:
        validated_source_artifacts = []
        for index, artifact in enumerate(source_artifacts):
            if not isinstance(artifact, str):
                raise ValueError(f"{source} source_artifacts[{index}] must be a string")
            validated_source_artifacts.append(artifact)

    source_artifact = row.get("source_artifact")
    if source_artifact is not None and not isinstance(source_artifact, str):
        raise ValueError(f"{source} source_artifact must be a string")

    try:
        priority_score = float(row["priority_score"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} priority_score must be numeric") from error
    if not math.isfinite(priority_score):
        raise ValueError(f"{source} priority_score must be finite")

    validated_row = {
        "state": validated_state,
        "legal_moves": validated_legal_moves,
        "priority_score": priority_score,
        "selection_reasons": validated_selection_reasons,
    }
    if validated_source_artifacts is not None:
        validated_row["source_artifacts"] = validated_source_artifacts
    if source_artifact is not None:
        validated_row["source_artifact"] = source_artifact
    return validated_row


def load_hard_state_rows(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{dataset_path}:{line_number} hard-state row must be an object")
            validated_row = validate_hard_state_row(payload, source=f"{dataset_path}:{line_number}")
            source_artifacts = list(validated_row.get("source_artifacts", []))
            prior_source_artifact = validated_row.get("source_artifact")
            if isinstance(prior_source_artifact, str) and prior_source_artifact:
                source_artifacts.append(prior_source_artifact)
            if source_artifacts:
                validated_row["source_artifacts"] = list(dict.fromkeys(source_artifacts))
            validated_row["source_artifact"] = str(dataset_path)
            rows.append(validated_row)
    if not rows:
        raise ValueError(f"{dataset_path} does not contain any hard-state rows")
    return rows


def select_top_ranked_rows(rows: list[dict[str, Any]], *, top_n: int | str) -> list[dict[str, Any]]:
    if isinstance(top_n, bool):
        raise ValueError("top_n must be an integer")

    if isinstance(top_n, int):
        normalized_top_n = top_n
    elif isinstance(top_n, str):
        try:
            normalized_top_n = int(top_n)
        except ValueError as error:
            raise ValueError("top_n must be an integer") from error
    else:
        raise ValueError("top_n must be an integer")

    if normalized_top_n <= 0:
        raise ValueError("top_n must be >= 1")
    return [dict(row) for row in rows[:normalized_top_n]]


def derive_policy_from_child_visits(child_stats: list[dict[str, Any]]) -> list[float]:
    policy = [0.0] * PITS_PER_PLAYER
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        raise ValueError("teacher search must produce at least one child visit")

    for child in child_stats:
        move = int(child["move"])
        policy[move] = float(child["visits"]) / float(total_visits)
    return policy


def derive_value_from_selected_move_win_rate(summary: dict[str, Any]) -> float:
    selected_move = summary.get("selected_move")
    for child in summary.get("child_stats", []):
        if int(child["move"]) == int(selected_move):
            if int(child.get("visits", 0)) == 0:
                return 0.0
            return (2.0 * float(child["win_rate"])) - 1.0
    raise ValueError(f"selected_move {selected_move} missing from child_stats")


def derive_teacher_targets_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    child_stats = summary["child_stats"]
    if len(child_stats) == 1 and int(child_stats[0].get("visits", 0)) == 0:
        move = int(child_stats[0]["move"])
        policy = [0.0] * PITS_PER_PLAYER
        policy[move] = 1.0
        return {
            "policy": policy,
            "value": derive_value_from_selected_move_win_rate(summary),
        }

    return {
        "policy": derive_policy_from_child_visits(child_stats),
        "value": derive_value_from_selected_move_win_rate(summary),
    }


def run_teacher_label(
    state: dict[str, Any],
    *,
    teacher_budget: int,
    teacher_mode: str,
    seed: int,
) -> dict[str, Any]:
    if teacher_mode not in SUPPORTED_TEACHER_MODES:
        raise ValueError(f"unsupported teacher_mode: {teacher_mode}")
    teacher_budget = _require_positive_budget(teacher_budget, field_name="teacher_budget")

    search = ClassicMCTS(
        KalahGame.from_state(state),
        simulations=teacher_budget,
        seed=int(seed),
    )
    summary = search.root_summary()
    return derive_teacher_targets_from_summary(summary)


def build_budget_pair_id(*, canonical_state: str, source_artifact: str, source_rank: int) -> str:
    return f"{canonical_state}|{source_artifact}|{int(source_rank)}"


def derive_teacher_seed(*, base_seed: int, canonical_state: str, source_artifact: str, teacher_budget: int) -> int:
    stable_seed_input = f"{canonical_state}|{source_artifact}|{int(teacher_budget)}"
    digest = hashlib.sha256(stable_seed_input.encode("utf-8")).digest()
    stable_offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(base_seed) + stable_offset


def resolve_teacher_seed_identity(row: dict[str, Any], *, canonical_state: str) -> str:
    source_artifacts = row.get("source_artifacts")
    if isinstance(source_artifacts, list):
        canonical_source_artifacts = sorted({artifact for artifact in source_artifacts if isinstance(artifact, str) and artifact})
        if canonical_source_artifacts:
            return "|".join(canonical_source_artifacts)

    return canonical_state


def resolve_source_artifact(row: dict[str, Any], *, canonical_state: str, source_rank: int) -> str:
    source_artifact = row.get("source_artifact")
    if isinstance(source_artifact, str) and source_artifact:
        return source_artifact

    source_artifacts = row.get("source_artifacts")
    if isinstance(source_artifacts, list):
        for artifact in source_artifacts:
            if isinstance(artifact, str) and artifact:
                return artifact

    fallback_identifier = row.get("canonical_state")
    if not isinstance(fallback_identifier, str) or not fallback_identifier:
        fallback_identifier = canonical_state

    return f"direct_input:{fallback_identifier}:rank{int(source_rank)}"


def build_dual_budget_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_budget: int,
    stronger_budget: int,
    teacher_mode: str,
    input_encoding: str,
    seed: int,
) -> list[dict[str, Any]]:
    canonical_budget = _require_positive_budget(canonical_budget, field_name="canonical_budget")
    stronger_budget = _require_positive_budget(stronger_budget, field_name="stronger_budget")
    labeled_rows: list[dict[str, Any]] = []

    for source_rank, row in enumerate(rows, start=1):
        validated_row = validate_hard_state_row(row, source=f"rows[{source_rank - 1}]")
        state = validated_row["state"]
        canonical_state = canonical_state_key(state)
        provenance_row = dict(row)
        provenance_row.update(validated_row)
        source_artifact = resolve_source_artifact(
            provenance_row,
            canonical_state=canonical_state,
            source_rank=source_rank,
        )
        teacher_seed_identity = resolve_teacher_seed_identity(
            provenance_row,
            canonical_state=canonical_state,
        )
        pair_id = build_budget_pair_id(
            canonical_state=canonical_state,
            source_artifact=source_artifact,
            source_rank=source_rank,
        )

        for teacher_profile, teacher_budget in (("canonical", canonical_budget), ("stronger", stronger_budget)):
            label = run_teacher_label(
                state,
                teacher_budget=teacher_budget,
                teacher_mode=teacher_mode,
                seed=derive_teacher_seed(
                    base_seed=seed,
                    canonical_state=canonical_state,
                    source_artifact=teacher_seed_identity,
                    teacher_budget=teacher_budget,
                ),
            )
            labeled_rows.append(
                {
                    "state": encode_state(state, input_encoding=input_encoding),
                    "policy": label["policy"],
                    "value": label["value"],
                    "policy_target_mode": DEFAULT_POLICY_TARGET_MODE,
                    "value_target_mode": DEFAULT_VALUE_TARGET_MODE,
                    "canonical_state": canonical_state,
                    "teacher_budget": int(teacher_budget),
                    "teacher_profile": teacher_profile,
                    "teacher_mode": teacher_mode,
                    "legal_moves": list(validated_row["legal_moves"]),
                    "source": f"hard_state_subset:{teacher_profile}:rank{source_rank}",
                    "source_artifact": source_artifact,
                    "source_rank": source_rank,
                    "source_priority_score": validated_row["priority_score"],
                    "selection_reasons": list(validated_row["selection_reasons"]),
                    "budget_pair_id": pair_id,
                }
            )

    return labeled_rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_labeled_rows(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{dataset_path}:{line_number} labeled row must be an object")
            rows.append(validate_labeled_row(payload, source=f"{dataset_path}:{line_number}"))
    if not rows:
        raise ValueError(f"{dataset_path} does not contain any labeled rows")
    return rows


def _top_move(policy: list[float]) -> int:
    return max(range(len(policy)), key=lambda move: (float(policy[move]), -move))


def _policy_divergence(left: list[float], right: list[float]) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)) / 2.0


def _require_labeled_row_string(row: dict[str, Any], *, field_name: str, source: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} {field_name} must be a non-empty string")
    return value


def _require_labeled_row_int(row: dict[str, Any], *, field_name: str, source: str) -> int:
    value = row.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{source} {field_name} must be an integer")
    return int(value)


def _require_labeled_row_numeric(row: dict[str, Any], *, field_name: str, source: str) -> float:
    try:
        value = float(row[field_name])
    except KeyError as error:
        raise ValueError(f"{source} missing required field '{field_name}'") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} {field_name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{source} {field_name} must be finite")
    return value


def _require_labeled_row_policy(row: dict[str, Any], *, source: str) -> list[float]:
    policy = row.get("policy")
    if not isinstance(policy, list) or len(policy) != PITS_PER_PLAYER:
        raise ValueError(f"{source} policy must be a list of {PITS_PER_PLAYER} numeric values")

    validated_policy: list[float] = []
    for index, entry in enumerate(policy):
        if isinstance(entry, bool):
            raise ValueError(f"{source} policy[{index}] must be numeric")
        try:
            value = float(entry)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} policy[{index}] must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"{source} policy[{index}] must be finite")
        if value < 0.0:
            raise ValueError(f"{source} policy[{index}] must be >= 0")
        validated_policy.append(value)
    if not math.isclose(sum(validated_policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{source} policy must sum to 1.0")
    return validated_policy


def validate_labeled_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    required_fields = (
        "budget_pair_id",
        "canonical_state",
        "teacher_profile",
        "teacher_budget",
        "source_rank",
        "policy",
        "value",
    )
    for field_name in required_fields:
        if field_name not in row:
            raise ValueError(f"{source} missing required field '{field_name}'")

    validated_row = dict(row)
    validated_row["budget_pair_id"] = _require_labeled_row_string(row, field_name="budget_pair_id", source=source)
    validated_row["canonical_state"] = _require_labeled_row_string(row, field_name="canonical_state", source=source)
    validated_row["teacher_profile"] = _require_labeled_row_string(row, field_name="teacher_profile", source=source)
    validated_row["teacher_budget"] = _require_labeled_row_int(row, field_name="teacher_budget", source=source)
    validated_row["source_rank"] = _require_labeled_row_int(row, field_name="source_rank", source=source)
    validated_row["policy"] = _require_labeled_row_policy(row, source=source)
    validated_row["value"] = _require_labeled_row_numeric(row, field_name="value", source=source)
    if not -1.0 <= validated_row["value"] <= 1.0:
        raise ValueError(f"{source} value must be between -1.0 and 1.0")
    return validated_row


def _require_positive_budget(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be >= 1")
    return int(value)


def pair_budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        pair_id = str(row["budget_pair_id"])
        teacher_profile = str(row["teacher_profile"])
        profiles = grouped.setdefault(pair_id, {})
        profiles.setdefault(teacher_profile, []).append(dict(row))

    pairs: list[dict[str, Any]] = []
    for pair_id, profiles in grouped.items():
        canonical_rows = profiles.get("canonical", [])
        stronger_rows = profiles.get("stronger", [])
        if len(canonical_rows) != 1 or len(stronger_rows) != 1 or set(profiles) != {"canonical", "stronger"}:
            raise ValueError(f"{pair_id} must contain exactly one canonical row and one stronger row")
        pairs.append(
            {
                "budget_pair_id": pair_id,
                "canonical": canonical_rows[0],
                "stronger": stronger_rows[0],
            }
        )
    return pairs


def build_comparison_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = pair_budget_rows(rows)
    disagreements: list[dict[str, Any]] = []
    top1_disagreements = 0
    total_policy_divergence = 0.0
    total_value_delta = 0.0

    for pair in pairs:
        canonical = pair["canonical"]
        stronger = pair["stronger"]
        canonical_top_move = _top_move(canonical["policy"])
        stronger_top_move = _top_move(stronger["policy"])
        policy_divergence = _policy_divergence(canonical["policy"], stronger["policy"])
        value_delta = abs(float(canonical["value"]) - float(stronger["value"]))
        if canonical_top_move != stronger_top_move:
            top1_disagreements += 1
        total_policy_divergence += policy_divergence
        total_value_delta += value_delta
        disagreements.append(
            {
                "budget_pair_id": pair["budget_pair_id"],
                "canonical_state": canonical["canonical_state"],
                "source_rank": canonical["source_rank"],
                "canonical_top_move": canonical_top_move,
                "stronger_top_move": stronger_top_move,
                "policy_divergence": round(policy_divergence, 6),
                "value_delta": round(value_delta, 6),
            }
        )

    disagreements.sort(key=lambda row: (-float(row["policy_divergence"]), row["canonical_state"]))
    pair_count = len(pairs)
    return {
        "pair_count": pair_count,
        "top1_disagreement_rate": 0.0 if pair_count == 0 else round(top1_disagreements / pair_count, 6),
        "average_policy_divergence": 0.0 if pair_count == 0 else round(total_policy_divergence / pair_count, 6),
        "maximum_policy_divergence": 0.0
        if pair_count == 0
        else round(max((row["policy_divergence"] for row in disagreements), default=0.0), 6),
        "average_absolute_value_delta": 0.0 if pair_count == 0 else round(total_value_delta / pair_count, 6),
        "largest_disagreements": disagreements[:10],
    }


def _validated_hard_suite_buckets(arena_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hard_suite_buckets = arena_report.get("hard_suite_buckets")
    if not isinstance(hard_suite_buckets, dict):
        raise ValueError("arena report must include hard_suite_buckets")

    validated_buckets: dict[str, dict[str, Any]] = {}
    for bucket_name, bucket in hard_suite_buckets.items():
        if not isinstance(bucket_name, str) or not bucket_name:
            raise ValueError("arena hard_suite_buckets keys must be non-empty strings")
        if not isinstance(bucket, dict):
            raise ValueError(
                f"arena hard_suite_buckets[{bucket_name!r}] must include integer games and numeric-or-null score"
            )

        games = bucket.get("games")
        score = bucket.get("score")
        if not isinstance(games, int) or isinstance(games, bool):
            raise ValueError(
                f"arena hard_suite_buckets[{bucket_name!r}] must include integer games and numeric-or-null score"
            )
        if games < 0:
            raise ValueError(
                f"arena hard_suite_buckets[{bucket_name!r}] must include integer games and numeric-or-null score"
            )
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"arena hard_suite_buckets[{bucket_name!r}] must include integer games and numeric-or-null score"
                )
            score = float(score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"arena hard_suite_buckets[{bucket_name!r}] must include integer games and numeric-or-null score"
                )

        validated_buckets[bucket_name] = {
            "games": int(games),
            "score": score,
        }

    required_bucket_names = {"opening", "midgame", "late"}
    if set(validated_buckets) != required_bucket_names:
        raise ValueError("arena hard_suite_buckets must include opening, midgame, and late buckets")
    return validated_buckets


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"arena {field_name} must be a non-negative integer")
    return int(value)


def _require_probability(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite numeric value between 0.0 and 1.0")
    normalized_value = float(value)
    if not math.isfinite(normalized_value) or not 0.0 <= normalized_value <= 1.0:
        raise ValueError(f"{field_name} must be a finite numeric value between 0.0 and 1.0")
    return normalized_value


def build_issue264_report(
    *,
    experiment: dict[str, Any],
    label_report: dict[str, Any],
    arena_report: dict[str, Any],
    baseline_checkpoint: str,
    challenger_checkpoint: str,
    challenger_artifact_dir: str,
    min_score: float,
) -> dict[str, Any]:
    min_score = _require_probability(min_score, field_name="min_score")
    arena_score = _require_probability(arena_report.get("score"), field_name="arena score")

    raw_promotion_decision = arena_report.get("promotion_decision")
    if not isinstance(raw_promotion_decision, dict):
        raise ValueError("arena promotion_decision must be an object")
    promotion_decision = dict(raw_promotion_decision)
    promotion_passed = promotion_decision.get("passed")
    if not isinstance(promotion_passed, bool):
        raise ValueError("arena promotion_decision.passed must be a boolean")

    hard_suite_buckets = _validated_hard_suite_buckets(arena_report)

    games_played = _require_non_negative_int(arena_report.get("games_played"), field_name="games_played")
    if games_played == 0:
        raise ValueError("arena games_played must be >= 1")
    wins = _require_non_negative_int(arena_report.get("wins"), field_name="wins")
    losses = _require_non_negative_int(arena_report.get("losses"), field_name="losses")
    draws = _require_non_negative_int(arena_report.get("draws"), field_name="draws")
    derived_games_played = wins + losses + draws
    if derived_games_played != games_played:
        raise ValueError("arena games_played must match wins/losses/draws total")
    derived_score = 0.0 if games_played == 0 else ((wins + (0.5 * draws)) / games_played)
    if not math.isclose(arena_score, derived_score, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("arena score must match wins/losses/draws-derived score")
    hard_suite_games = sum(bucket["games"] for bucket in hard_suite_buckets.values())
    if hard_suite_games != games_played:
        raise ValueError("arena hard_suite_buckets games must sum to games_played")

    score_clears_threshold = arena_score >= min_score
    if promotion_passed != score_clears_threshold:
        raise ValueError("arena promotion_decision.passed must match the score threshold outcome")

    arena_passed = score_clears_threshold

    if arena_passed:
        recommendation = "promote_to_standard_step"
        rationale = "arena score cleared threshold and hard-suite coverage was validated"
    else:
        recommendation = "remain_selective"
        rationale = "arena score did not clear the required threshold"

    return {
        "schema": "issue264_hard_suite_impact_v1",
        "experiment": dict(experiment),
        "label_report": dict(label_report),
        "training": {
            "baseline_checkpoint": baseline_checkpoint,
            "challenger_checkpoint": challenger_checkpoint,
            "challenger_artifact_dir": challenger_artifact_dir,
        },
        "arena": {
            "games_played": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score": arena_score,
            "promotion_decision": promotion_decision,
            "hard_suite_buckets": hard_suite_buckets,
        },
        "recommendation": {
            "recommendation": recommendation,
            "rationale": rationale,
        },
    }
