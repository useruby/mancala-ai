#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.hard_state_teacher_labeling import run_teacher_label
from ml.alphazero_lite.kalah_rules import (
    KalahGame,
    PITS_PER_PLAYER,
    move_consequence_table,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
    row_map_from_reference,
)
from ml.alphazero_lite.search_interaction_diagnostic import probe_artifact_position
from ml.alphazero_lite.self_play import build_eval_search_options, encode_state


OUT_ROOT = Path("/tmp/azlite_policy_target_encoding_audit")
SUMMARY_PATH = OUT_ROOT / "policy_target_consistency_summary.json"
FULL_RESULTS_PATH = OUT_ROOT / "policy_target_encoding_audit_results.json"
CURRENT_ARTIFACT = Path("/home/alex/Mancala/ai/storage/ai/alphazero_lite/current")
GUARDED_W2_ARTIFACT = Path(
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
REFERENCE_ARTIFACT = Path(
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
FORENSIC_SUITE = Path(
    "/home/alex/Mancala/ai/ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
)
AUDIT_ROW_IDS = [
    "capture_available-002",
    "capture_available-003",
    "capture_available-005",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
]
DATASET_PATHS = [
    Path(
        "/home/alex/Mancala/ai/ml/alphazero_lite/tactical_opening_capture_family_replay.jsonl"
    ),
    Path("/home/alex/Mancala/ai/ml/alphazero_lite/tactical_balanced_replay.jsonl"),
    Path(
        "/home/alex/Mancala/ai/ml/alphazero_lite/tactical_balanced_replay_source.jsonl"
    ),
    Path("/home/alex/Mancala/ai/ml/alphazero_lite/tactical_capture_protection.jsonl"),
    Path("/home/alex/Mancala/ai/ml/alphazero_lite/human_games_combined.jsonl"),
]
SEARCH_SIMULATIONS = 384
SEARCH_SEED = 17
SEARCH_C_PUCT = 1.25
TEACHER_BUDGETS = [384, 1200, 2400]
PROBE_SEED = 7
PROBE_EPOCHS = 400
PROBE_LR = 0.05
PROBE_SPLIT_FRACTION = 0.3


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def canonical_raw_state(row: dict) -> dict | None:
    if isinstance(row.get("raw_state"), dict):
        return dict(row["raw_state"])
    if isinstance(row.get("state"), dict):
        return dict(row["state"])
    canonical_state = row.get("canonical_state")
    if isinstance(canonical_state, str) and canonical_state:
        try:
            parsed = json.loads(canonical_state)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def normalized_bucket_name(consequence: dict) -> str:
    if consequence["gives_extra_turn"] and consequence["produces_capture"]:
        return "extra_turn_capture"
    if consequence["gives_extra_turn"]:
        return "extra_turn_no_capture"
    if consequence["produces_capture"]:
        return "no_extra_turn_capture"
    return "no_extra_turn_no_capture"


def infer_top_move_from_policy(
    policy: list[float], legal_moves: list[int]
) -> int | None:
    if not legal_moves:
        return None
    return max(legal_moves, key=lambda move: (float(policy[move]), -move))


def teacher_policy_from_child_stats(child_stats: list[dict]) -> list[float]:
    policy = [0.0] * PITS_PER_PLAYER
    total = sum(int(child.get("visits", 0)) for child in child_stats)
    if total <= 0:
        return policy
    for child in child_stats:
        move = int(child["move"])
        policy[move] = float(child["visits"]) / float(total)
    return policy


def teacher_top_move_from_row(row: dict) -> int | None:
    if isinstance(row.get("teacher_selected_move"), int):
        return int(row["teacher_selected_move"])
    if isinstance(row.get("reference_move"), int):
        return int(row["reference_move"])
    child_stats = row.get("teacher_child_stats") or row.get("child_stats") or []
    if not isinstance(child_stats, list) or not child_stats:
        return None
    policy = teacher_policy_from_child_stats(child_stats)
    legal_moves = [int(child["move"]) for child in child_stats]
    return infer_top_move_from_policy(policy, legal_moves)


def bucket_mass(policy: list[float], consequence_table: list[dict]) -> dict[str, float]:
    totals = {
        "extra_turn_capture": 0.0,
        "extra_turn_no_capture": 0.0,
        "no_extra_turn_capture": 0.0,
        "no_extra_turn_no_capture": 0.0,
    }
    for consequence in consequence_table:
        move = int(consequence["move_index"])
        if not consequence["legal"]:
            continue
        totals[normalized_bucket_name(consequence)] += float(policy[move])
    return {key: round(value, 6) for key, value in totals.items()}


def flatten_consequence_table(consequence_table: list[dict]) -> list[float]:
    flat: list[float] = []
    for consequence in consequence_table:
        flat.extend(
            [
                float(int(consequence["legal"])),
                float(consequence["seed_count"]),
                float(int(consequence["gives_extra_turn"])),
                float(int(consequence["produces_capture"])),
                float(consequence["capture_count"]),
                float(int(consequence["lands_on_own_empty_pit"])),
                float(consequence["opposite_pit_seeds"]),
                float(consequence["store_delta_immediate"]),
                float(consequence["opponent_store_delta_immediate"]),
                float(
                    -1
                    if consequence["resulting_side_to_move"] is None
                    else consequence["resulting_side_to_move"]
                ),
                float(int(consequence["game_over_after_move"])),
                float(consequence["immediate_score_delta"]),
                float(consequence["pit_index"]),
            ]
        )
    return flat


def action_feature_vector(
    state_features: list[float], consequence: dict
) -> list[float]:
    return [
        *state_features,
        float(consequence["move_index"]),
        float(consequence["seed_count"]),
        float(int(consequence["gives_extra_turn"])),
        float(int(consequence["produces_capture"])),
        float(consequence["capture_count"]),
        float(int(consequence["lands_on_own_empty_pit"])),
        float(consequence["opposite_pit_seeds"]),
        float(consequence["store_delta_immediate"]),
        float(consequence["opponent_store_delta_immediate"]),
        float(consequence["immediate_score_delta"]),
        float(int(consequence["game_over_after_move"])),
    ]


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def fit_multiclass_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    class_count: int,
    seed: int,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_count = x_train.shape[1]
    weights = rng.normal(0.0, 0.01, size=(feature_count, class_count)).astype(
        np.float64
    )
    bias = np.zeros(class_count, dtype=np.float64)
    targets = np.eye(class_count, dtype=np.float64)[y_train]
    sample_count = max(1, x_train.shape[0])
    for _ in range(epochs):
        logits = x_train @ weights + bias
        probs = np.vstack([stable_softmax(row) for row in logits])
        error = probs - targets
        grad_w = (x_train.T @ error) / sample_count
        grad_b = np.mean(error, axis=0)
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def fit_binary_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x_train.shape[1]).astype(np.float64)
    bias = 0.0
    sample_count = max(1, x_train.shape[0])
    y_train = y_train.astype(np.float64)
    for _ in range(epochs):
        logits = x_train @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = probs - y_train
        grad_w = (x_train.T @ error) / sample_count
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
    return weights, bias


def multiclass_metrics(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, float, list[np.ndarray]]:
    if x_dev.shape[0] == 0:
        return 0.0, 0.0, []
    logits = x_dev @ weights + bias
    probs = [stable_softmax(row) for row in logits]
    predictions = np.array([int(np.argmax(prob)) for prob in probs], dtype=np.int64)
    accuracy = float(np.mean(predictions == y_dev))
    cross_entropy = float(
        np.mean(
            [
                -math.log(max(float(prob[y]), 1e-12))
                for prob, y in zip(probs, y_dev, strict=True)
            ]
        )
    )
    return accuracy, cross_entropy, probs


def binary_state_ranking_metrics(
    examples: list[dict],
    weights: np.ndarray,
    bias: float,
) -> dict[str, int | None]:
    by_row_id: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for example in examples:
        feature_vector = np.asarray(example["features"], dtype=np.float64)
        score = float(feature_vector @ weights + bias)
        by_row_id[str(example["row_id"])].append(
            (int(example["move"]), score, int(example["label"]))
        )
    ranks: dict[str, int | None] = {}
    for row_id, entries in by_row_id.items():
        ranked = sorted(entries, key=lambda item: (-item[1], item[0]))
        rank = None
        for index, (_move, _score, label) in enumerate(ranked, start=1):
            if label == 1:
                rank = index
                break
        ranks[row_id] = rank
    return ranks


def split_by_canonical_state(
    items: list[dict], *, seed: int
) -> tuple[list[dict], list[dict]]:
    canonical_states = sorted({str(item["canonical_state"]) for item in items})
    rng = random.Random(seed)
    rng.shuffle(canonical_states)
    if len(canonical_states) <= 1:
        return items, []
    dev_count = max(1, int(round(len(canonical_states) * PROBE_SPLIT_FRACTION)))
    dev_count = min(dev_count, len(canonical_states) - 1)
    dev_states = set(canonical_states[:dev_count])
    train = [item for item in items if str(item["canonical_state"]) not in dev_states]
    dev = [item for item in items if str(item["canonical_state"]) in dev_states]
    return train, dev


def top_move_for_reference_row(reference_row: dict) -> int:
    return int(reference_row["reference_move"])


def load_reference_rows() -> dict[str, dict]:
    payload = load_json(REFERENCE_ARTIFACT)
    return row_map_from_reference(payload)


def load_fixture_rows() -> dict[str, dict]:
    rows = json.loads(FORENSIC_SUITE.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows}


def evaluate_policy_only(
    evaluator: ArtifactEvaluator, state: dict
) -> tuple[list[float], float]:
    policy, value = evaluator.evaluate(KalahGame.from_state(state))
    return [float(x) for x in policy.tolist()], float(value)


def known_row_audit(
    *,
    reference_rows: dict[str, dict],
    fixture_rows: dict[str, dict],
    current_evaluator: ArtifactEvaluator,
    guarded_evaluator: ArtifactEvaluator | None,
) -> list[dict]:
    current_search_options = dict(build_eval_search_options())
    rows: list[dict] = []
    for row_id in AUDIT_ROW_IDS:
        reference_row = reference_rows[row_id]
        raw_state = dict(reference_row["state"])
        state_features = encode_state(raw_state, input_encoding="kalah_v3")
        consequences = move_consequence_table(raw_state)
        legal_moves = [
            int(child["move"]) for child in list(reference_row["child_stats"])
        ]
        current_policy, current_value = evaluate_policy_only(
            current_evaluator, raw_state
        )
        current_probe = probe_artifact_position(
            artifact_path=str(CURRENT_ARTIFACT),
            state=raw_state,
            simulations=SEARCH_SIMULATIONS,
            seed=SEARCH_SEED,
            c_puct=SEARCH_C_PUCT,
            evaluator=current_evaluator,
            search_options=current_search_options,
        )
        guarded_policy = None
        guarded_value = None
        guarded_probe = None
        if guarded_evaluator is not None:
            guarded_policy, guarded_value = evaluate_policy_only(
                guarded_evaluator, raw_state
            )
            guarded_probe = probe_artifact_position(
                artifact_path=str(GUARDED_W2_ARTIFACT),
                state=raw_state,
                simulations=SEARCH_SIMULATIONS,
                seed=SEARCH_SEED,
                c_puct=SEARCH_C_PUCT,
                evaluator=guarded_evaluator,
                search_options=current_search_options,
            )
        reference_move = int(reference_row["reference_move"])
        searched_selected_move = int(current_probe["selected_move"])
        wrong_extra_turn_move = None
        wrong_extra_turn_prob = None
        for move in legal_moves:
            if move == reference_move:
                continue
            if not consequences[move]["gives_extra_turn"]:
                continue
            probability = float(current_policy[move])
            if wrong_extra_turn_prob is None or probability > wrong_extra_turn_prob:
                wrong_extra_turn_move = move
                wrong_extra_turn_prob = probability
        if row_id == "capture_available-002":
            wrong_extra_turn_move = 2
        reference_consequence = consequences[reference_move]
        wrong_consequence = (
            consequences[wrong_extra_turn_move]
            if wrong_extra_turn_move is not None
            else None
        )
        aggregate_gap = {
            "state_has_any_extra_turn": any(
                item["gives_extra_turn"] for item in consequences if item["legal"]
            ),
            "state_has_any_capture": any(
                item["produces_capture"] for item in consequences if item["legal"]
            ),
            "encoding_only_global_not_per_action": True,
        }
        diagnosis_parts = []
        if wrong_consequence is not None:
            if (
                reference_consequence["gives_extra_turn"]
                != wrong_consequence["gives_extra_turn"]
                and reference_consequence["produces_capture"]
                != wrong_consequence["produces_capture"]
            ):
                diagnosis_parts.append("extra_turn_vs_capture_tradeoff")
            if (
                reference_consequence["immediate_score_delta"]
                != wrong_consequence["immediate_score_delta"]
            ):
                diagnosis_parts.append("immediate_score_differs")
        if row_id == "capture_available-002":
            diagnosis_parts.append("known_policy_mismatch")
        rows.append(
            {
                "row_id": row_id,
                "reference_move": reference_move,
                "learned_top_move_current": infer_top_move_from_policy(
                    current_policy, legal_moves
                ),
                "learned_top_move_guarded_w2": None
                if guarded_policy is None
                else infer_top_move_from_policy(guarded_policy, legal_moves),
                "searched_selected_move": searched_selected_move,
                "wrong_extra_turn_move": wrong_extra_turn_move,
                "reference_policy_current": round(
                    float(current_policy[reference_move]), 4
                ),
                "wrong_policy_current": None
                if wrong_extra_turn_move is None
                else round(float(current_policy[wrong_extra_turn_move]), 4),
                "reference_policy_guarded_w2": None
                if guarded_policy is None
                else round(float(guarded_policy[reference_move]), 4),
                "wrong_policy_guarded_w2": None
                if guarded_policy is None or wrong_extra_turn_move is None
                else round(float(guarded_policy[wrong_extra_turn_move]), 4),
                "reference_gives_extra_turn": bool(
                    reference_consequence["gives_extra_turn"]
                ),
                "reference_produces_capture": bool(
                    reference_consequence["produces_capture"]
                ),
                "wrong_gives_extra_turn": None
                if wrong_consequence is None
                else bool(wrong_consequence["gives_extra_turn"]),
                "wrong_produces_capture": None
                if wrong_consequence is None
                else bool(wrong_consequence["produces_capture"]),
                "immediate_score_delta_reference": int(
                    reference_consequence["immediate_score_delta"]
                ),
                "immediate_score_delta_wrong": None
                if wrong_consequence is None
                else int(wrong_consequence["immediate_score_delta"]),
                "diagnosis": ",".join(diagnosis_parts) if diagnosis_parts else "none",
                "policy_probability_per_legal_move_current": {
                    str(move): round(float(current_policy[move]), 4)
                    for move in legal_moves
                },
                "policy_probability_per_legal_move_guarded_w2": None
                if guarded_policy is None
                else {
                    str(move): round(float(guarded_policy[move]), 4)
                    for move in legal_moves
                },
                "per_move_consequence_table": consequences,
                "kalah_v3_features": state_features,
                "aggregate_feature_gap": aggregate_gap,
                "teacher_value_reference": reference_row.get("teacher_value"),
                "raw_state": raw_state,
                "searched_policy_current": [float(x) for x in current_probe["policy"]],
                "searched_policy_guarded_w2": None
                if guarded_probe is None
                else [float(x) for x in guarded_probe["policy"]],
                "raw_value_current": round(current_value, 4),
                "raw_value_guarded_w2": None
                if guarded_value is None
                else round(guarded_value, 4),
                "fixture_state": fixture_rows.get(row_id),
            }
        )
    return rows


def target_policy_for_row(row: dict) -> tuple[list[float] | None, int | None]:
    policy = row.get("policy")
    if isinstance(policy, list) and len(policy) == PITS_PER_PLAYER:
        target = [float(value) for value in policy]
        top_move = infer_top_move_from_policy(
            target, [index for index, value in enumerate(target) if value > 0.0]
        )
        return target, top_move
    top_move = teacher_top_move_from_row(row)
    if top_move is None:
        return None, None
    policy = [0.0] * PITS_PER_PLAYER
    policy[top_move] = 1.0
    return policy, top_move


def dataset_policy_target_consistency(reference_rows: dict[str, dict]) -> dict:
    per_source_rows: list[dict] = []
    canonical_targets: dict[str, set[int]] = defaultdict(set)
    canonical_sources: dict[str, set[str]] = defaultdict(set)
    overall_conflicts: list[dict] = []
    all_row_summaries: list[dict] = []
    for source_path in DATASET_PATHS:
        if not source_path.exists():
            continue
        rows = read_jsonl(source_path)
        extra_turn_top_move_rate_num = 0
        no_extra_turn_capture_top_rate_num = 0
        rows_with_conflict = 0
        canonical_counts: dict[str, int] = defaultdict(int)
        per_source_canonical_targets: dict[str, set[int]] = defaultdict(set)
        scanned_row_count = 0
        for row in rows:
            raw_state = canonical_raw_state(row)
            if raw_state is None:
                continue
            canonical = canonical_state_key(raw_state)
            consequence_table = move_consequence_table(raw_state)
            policy, top_move = target_policy_for_row(row)
            if policy is None or top_move is None:
                continue
            scanned_row_count += 1
            canonical_counts[canonical] += 1
            canonical_targets[canonical].add(int(top_move))
            canonical_sources[canonical].add(str(source_path))
            per_source_canonical_targets[canonical].add(int(top_move))
            top_consequence = consequence_table[top_move]
            if top_consequence["gives_extra_turn"]:
                extra_turn_top_move_rate_num += 1
            if (
                top_consequence["produces_capture"]
                and not top_consequence["gives_extra_turn"]
            ):
                no_extra_turn_capture_top_rate_num += 1
            best_extra_turn_mass = 0.0
            best_no_extra_turn_capture_mass = 0.0
            for consequence in consequence_table:
                move = int(consequence["move_index"])
                if not consequence["legal"]:
                    continue
                mass = float(policy[move])
                if consequence["gives_extra_turn"]:
                    best_extra_turn_mass = max(best_extra_turn_mass, mass)
                if (
                    consequence["produces_capture"]
                    and not consequence["gives_extra_turn"]
                ):
                    best_no_extra_turn_capture_mass = max(
                        best_no_extra_turn_capture_mass, mass
                    )
            if best_extra_turn_mass >= 0.6 and best_no_extra_turn_capture_mass >= 0.2:
                rows_with_conflict += 1
            all_row_summaries.append(
                {
                    "source": str(source_path),
                    "canonical_state": canonical,
                    "target_top_move": int(top_move),
                    "target_bucket": normalized_bucket_name(top_consequence),
                    "bucket_mass": bucket_mass(policy, consequence_table),
                }
            )
        duplicate_states = sum(max(0, count - 1) for count in canonical_counts.values())
        conflicting_policy_targets = sum(
            1 for targets in per_source_canonical_targets.values() if len(targets) > 1
        )
        per_source_rows.append(
            {
                "source": str(source_path),
                "rows": scanned_row_count,
                "duplicate_canonical_states": max(0, duplicate_states),
                "conflicting_policy_targets": conflicting_policy_targets,
                "extra_turn_top_move_rate": 0.0
                if scanned_row_count == 0
                else round(extra_turn_top_move_rate_num / scanned_row_count, 4),
                "no_extra_turn_capture_top_move_rate": 0.0
                if scanned_row_count == 0
                else round(no_extra_turn_capture_top_rate_num / scanned_row_count, 4),
                "rows_with_extra_turn_over_no_extra_turn_capture_conflict": rows_with_conflict,
                "notes": "teacher_selected_move/teacher_child_stats/policy scanned when present",
            }
        )
    reference_scanned = 0
    reference_conflicts = 0
    reference_extra_turn_top_rate_num = 0
    reference_no_extra_turn_capture_top_rate_num = 0
    reference_rows_with_conflict = 0
    for row_id, row in reference_rows.items():
        raw_state = dict(row["state"])
        canonical = canonical_state_key(raw_state)
        consequence_table = move_consequence_table(raw_state)
        top_move = int(row["reference_move"])
        policy = teacher_policy_from_child_stats(list(row.get("child_stats") or []))
        reference_scanned += 1
        canonical_targets[canonical].add(top_move)
        canonical_sources[canonical].add(str(REFERENCE_ARTIFACT))
        if len(canonical_targets[canonical]) > 1:
            reference_conflicts += 1
        top_consequence = consequence_table[top_move]
        if top_consequence["gives_extra_turn"]:
            reference_extra_turn_top_rate_num += 1
        if (
            top_consequence["produces_capture"]
            and not top_consequence["gives_extra_turn"]
        ):
            reference_no_extra_turn_capture_top_rate_num += 1
        best_extra_turn_mass = 0.0
        best_no_extra_turn_capture_mass = 0.0
        for consequence in consequence_table:
            move = int(consequence["move_index"])
            if not consequence["legal"]:
                continue
            mass = float(policy[move])
            if consequence["gives_extra_turn"]:
                best_extra_turn_mass = max(best_extra_turn_mass, mass)
            if consequence["produces_capture"] and not consequence["gives_extra_turn"]:
                best_no_extra_turn_capture_mass = max(
                    best_no_extra_turn_capture_mass, mass
                )
        if best_extra_turn_mass >= 0.6 and best_no_extra_turn_capture_mass >= 0.2:
            reference_rows_with_conflict += 1
        all_row_summaries.append(
            {
                "source": str(REFERENCE_ARTIFACT),
                "canonical_state": canonical,
                "target_top_move": top_move,
                "target_bucket": normalized_bucket_name(top_consequence),
                "bucket_mass": bucket_mass(policy, consequence_table),
            }
        )
    per_source_rows.append(
        {
            "source": str(REFERENCE_ARTIFACT),
            "rows": reference_scanned,
            "duplicate_canonical_states": 0,
            "conflicting_policy_targets": reference_conflicts,
            "extra_turn_top_move_rate": 0.0
            if reference_scanned == 0
            else round(reference_extra_turn_top_rate_num / reference_scanned, 4),
            "no_extra_turn_capture_top_move_rate": 0.0
            if reference_scanned == 0
            else round(
                reference_no_extra_turn_capture_top_rate_num / reference_scanned, 4
            ),
            "rows_with_extra_turn_over_no_extra_turn_capture_conflict": reference_rows_with_conflict,
            "notes": "shared forensic reference artifact",
        }
    )
    for canonical_state, targets in canonical_targets.items():
        if len(targets) > 1:
            overall_conflicts.append(
                {
                    "canonical_state": canonical_state,
                    "conflicting_top_moves": sorted(int(move) for move in targets),
                    "sources": sorted(canonical_sources[canonical_state]),
                }
            )
    summary = {
        "schema": "azlite_policy_target_consistency_summary_v1",
        "sources": per_source_rows,
        "duplicate_canonical_states_total": sum(
            int(row["duplicate_canonical_states"]) for row in per_source_rows
        ),
        "conflicting_policy_targets_total": sum(
            int(row["conflicting_policy_targets"]) for row in per_source_rows
        ),
        "cross_source_conflicting_canonical_states": overall_conflicts,
        "row_summaries": all_row_summaries,
        "reference_rows_scanned": sorted(reference_rows),
    }
    return summary


def build_probe_examples(
    known_rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    state_examples: list[dict] = []
    state_plus_move_examples: list[dict] = []
    action_examples: list[dict] = []
    for row in known_rows:
        row_id = str(row["row_id"])
        canonical_state = canonical_state_key(dict(row["raw_state"]))
        state_features = list(row["kalah_v3_features"])
        consequence_table = list(row["per_move_consequence_table"])
        reference_move = int(row["reference_move"])
        legal_moves = [
            int(item["move_index"]) for item in consequence_table if item["legal"]
        ]
        state_examples.append(
            {
                "row_id": row_id,
                "canonical_state": canonical_state,
                "features": state_features,
                "label": reference_move,
                "legal_moves": legal_moves,
            }
        )
        state_plus_move_examples.append(
            {
                "row_id": row_id,
                "canonical_state": canonical_state,
                "features": [
                    *state_features,
                    *flatten_consequence_table(consequence_table),
                ],
                "label": reference_move,
                "legal_moves": legal_moves,
            }
        )
        for consequence in consequence_table:
            if not consequence["legal"]:
                continue
            move = int(consequence["move_index"])
            action_examples.append(
                {
                    "row_id": row_id,
                    "canonical_state": canonical_state,
                    "move": move,
                    "features": action_feature_vector(state_features, consequence),
                    "label": 1 if move == reference_move else 0,
                }
            )
    return state_examples, state_plus_move_examples, action_examples


def evaluate_probe_feature_set(
    *,
    name: str,
    multiclass_examples: list[dict] | None,
    action_examples: list[dict] | None,
) -> dict:
    if multiclass_examples is not None:
        train_examples, dev_examples = split_by_canonical_state(
            multiclass_examples, seed=PROBE_SEED
        )
        x_train = np.asarray(
            [item["features"] for item in train_examples], dtype=np.float64
        )
        y_train = np.asarray(
            [int(item["label"]) for item in train_examples], dtype=np.int64
        )
        x_dev = np.asarray(
            [item["features"] for item in dev_examples], dtype=np.float64
        )
        y_dev = np.asarray(
            [int(item["label"]) for item in dev_examples], dtype=np.int64
        )
        weights, bias = fit_multiclass_logreg(
            x_train, y_train, class_count=PITS_PER_PLAYER, seed=PROBE_SEED
        )
        accuracy, cross_entropy, _dev_probs = multiclass_metrics(
            x_dev, y_dev, weights, bias
        )
        capture_ranks: dict[str, int | None] = {
            "capture_available-002": None,
            "capture_available-003": None,
        }
        all_probs = {}
        for example in multiclass_examples:
            feature_vector = np.asarray(example["features"], dtype=np.float64)
            all_probs[str(example["row_id"])] = stable_softmax(
                feature_vector @ weights + bias
            )
        for row_id in capture_ranks:
            if row_id in all_probs:
                ranked = sorted(
                    range(PITS_PER_PLAYER),
                    key=lambda move: (-float(all_probs[row_id][move]), move),
                )
                capture_ranks[row_id] = (
                    ranked.index(
                        next(
                            item["label"]
                            for item in multiclass_examples
                            if item["row_id"] == row_id
                        )
                    )
                    + 1
                )
        return {
            "feature_set": name,
            "train_rows": len(train_examples),
            "dev_rows": len(dev_examples),
            "top1_accuracy": round(accuracy, 4),
            "capture_002_reference_rank": capture_ranks["capture_available-002"],
            "capture_003_reference_rank": capture_ranks["capture_available-003"],
            "average_cross_entropy": round(cross_entropy, 4),
            "notes": "multiclass logistic regression on canonical-state split",
        }
    assert action_examples is not None
    train_examples, dev_examples = split_by_canonical_state(
        action_examples, seed=PROBE_SEED
    )
    x_train = np.asarray(
        [item["features"] for item in train_examples], dtype=np.float64
    )
    y_train = np.asarray(
        [int(item["label"]) for item in train_examples], dtype=np.int64
    )
    x_dev = np.asarray([item["features"] for item in dev_examples], dtype=np.float64)
    y_dev = np.asarray([int(item["label"]) for item in dev_examples], dtype=np.int64)
    weights, bias = fit_binary_logreg(x_train, y_train, seed=PROBE_SEED)
    logits = x_dev @ weights + bias if x_dev.shape[0] else np.zeros(0, dtype=np.float64)
    probs = (
        1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        if x_dev.shape[0]
        else np.zeros(0, dtype=np.float64)
    )
    predictions = (
        (probs >= 0.5).astype(np.int64)
        if x_dev.shape[0]
        else np.zeros(0, dtype=np.int64)
    )
    accuracy = float(np.mean(predictions == y_dev)) if x_dev.shape[0] else 0.0
    cross_entropy = (
        float(
            np.mean(
                [
                    -(
                        label * math.log(max(float(prob), 1e-12))
                        + (1 - label) * math.log(max(float(1.0 - prob), 1e-12))
                    )
                    for prob, label in zip(probs, y_dev, strict=True)
                ]
            )
        )
        if x_dev.shape[0]
        else 0.0
    )
    ranks = binary_state_ranking_metrics(action_examples, weights, bias)
    row_ids = sorted({str(item["row_id"]) for item in action_examples})
    top1_hits = 0
    for row_id in row_ids:
        if ranks.get(row_id) == 1:
            top1_hits += 1
    return {
        "feature_set": name,
        "train_rows": len(train_examples),
        "dev_rows": len(dev_examples),
        "top1_accuracy": round(top1_hits / max(1, len(row_ids)), 4),
        "capture_002_reference_rank": ranks.get("capture_available-002"),
        "capture_003_reference_rank": ranks.get("capture_available-003"),
        "average_cross_entropy": round(cross_entropy, 4),
        "notes": "binary action-ranking logistic regression on legal actions",
    }


def teacher_budget_audit(reference_row: dict) -> dict:
    state = dict(reference_row["state"])
    budgets = []
    for budget in TEACHER_BUDGETS:
        label = run_teacher_label(
            state,
            teacher_budget=budget,
            teacher_mode="classic_mcts",
            seed=42,
        )
        policy = [float(value) for value in label["policy"]]
        top_move = infer_top_move_from_policy(
            policy, [move for move, value in enumerate(policy) if value > 0.0]
        )
        move2 = policy[2] if len(policy) > 2 else 0.0
        move4 = policy[4] if len(policy) > 4 else 0.0
        budgets.append(
            {
                "teacher_budget": budget,
                "top_move": top_move,
                "policy_move_2": round(move2, 4),
                "policy_move_4": round(move4, 4),
                "move_4_minus_move_2": round(move4 - move2, 4),
                "value": round(float(label["value"]), 4),
            }
        )
    search_rows = []
    for budget in TEACHER_BUDGETS:
        search = ClassicMCTS(KalahGame.from_state(state), simulations=budget, seed=42)
        summary = search.root_summary()
        child_by_move = {int(child["move"]): child for child in summary["child_stats"]}
        move2 = child_by_move.get(2)
        move4 = child_by_move.get(4)
        q2 = (
            None
            if move2 is None or int(move2["visits"]) <= 0
            else (2.0 * float(move2["win_rate"])) - 1.0
        )
        q4 = (
            None
            if move4 is None or int(move4["visits"]) <= 0
            else (2.0 * float(move4["win_rate"])) - 1.0
        )
        search_rows.append(
            {
                "teacher_budget": budget,
                "selected_move": summary["selected_move"],
                "move_2_visits": None if move2 is None else int(move2["visits"]),
                "move_4_visits": None if move4 is None else int(move4["visits"]),
                "move_2_q": None if q2 is None else round(q2, 4),
                "move_4_q": None if q4 is None else round(q4, 4),
                "move_4_minus_move_2_q": None
                if q2 is None or q4 is None
                else round(q4 - q2, 4),
            }
        )
    stable_reference = all(row["top_move"] == 4 for row in budgets)
    return {
        "budgets": budgets,
        "search_rows": search_rows,
        "move_4_consistently_preferred": stable_reference,
    }


def classify_next_action(
    dataset_summary: dict, probe_rows: list[dict], teacher_audit: dict
) -> tuple[str, str]:
    if (
        dataset_summary["cross_source_conflicting_canonical_states"]
        or dataset_summary["conflicting_policy_targets_total"] > 0
    ):
        return (
            "target_inconsistency",
            "fix target generation / canonicalization before any new training",
        )
    if not teacher_audit["move_4_consistently_preferred"]:
        return (
            "teacher_label_uncertain",
            "rebuild references with stronger teacher budget and rerun local diagnostics",
        )
    probe_by_name = {row["feature_set"]: row for row in probe_rows}
    state_only = probe_by_name.get("state_only_kalah_v3")
    action_augmented = probe_by_name.get("action_scoring_state_plus_consequences")
    if (
        state_only is not None
        and action_augmented is not None
        and (
            (state_only["capture_002_reference_rank"] or 99)
            > (action_augmented["capture_002_reference_rank"] or 99)
            or (state_only["capture_003_reference_rank"] or 99)
            > (action_augmented["capture_003_reference_rank"] or 99)
        )
    ):
        return (
            "encoding_gap_confirmed",
            "implement a kalah_v4 encoding with explicit per-move consequence features and run a small controlled training lane",
        )
    if (
        state_only is not None
        and action_augmented is not None
        and state_only["top1_accuracy"] >= action_augmented["top1_accuracy"]
    ):
        return (
            "optimization_or_capacity_gap",
            "run a small architecture/capacity or policy-head experiment, not more replay",
        )
    return (
        "unresolved_search_policy_interaction",
        "return to search/value trace audit, not training",
    )


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reference_rows = load_reference_rows()
    fixture_rows = load_fixture_rows()
    current_evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)
    guarded_evaluator = (
        ArtifactEvaluator(GUARDED_W2_ARTIFACT) if GUARDED_W2_ARTIFACT.exists() else None
    )
    known_rows = known_row_audit(
        reference_rows=reference_rows,
        fixture_rows=fixture_rows,
        current_evaluator=current_evaluator,
        guarded_evaluator=guarded_evaluator,
    )
    dataset_summary = dataset_policy_target_consistency(reference_rows)
    SUMMARY_PATH.write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_examples, state_plus_move_examples, action_examples = build_probe_examples(
        known_rows
    )
    probe_rows = [
        evaluate_probe_feature_set(
            name="state_only_kalah_v3",
            multiclass_examples=state_examples,
            action_examples=None,
        ),
        evaluate_probe_feature_set(
            name="state_plus_flattened_consequences",
            multiclass_examples=state_plus_move_examples,
            action_examples=None,
        ),
        evaluate_probe_feature_set(
            name="action_scoring_state_plus_consequences",
            multiclass_examples=None,
            action_examples=action_examples,
        ),
    ]
    teacher_audit = teacher_budget_audit(reference_rows["capture_available-002"])
    classification, next_action = classify_next_action(
        dataset_summary, probe_rows, teacher_audit
    )
    payload = {
        "schema": "azlite_policy_target_encoding_audit_v1",
        "known_row_audit": known_rows,
        "dataset_policy_target_consistency": dataset_summary,
        "probe_results": probe_rows,
        "teacher_target_audit_002": teacher_audit,
        "classification": classification,
        "recommended_next_action": next_action,
        "guardrails": {
            "trained_production_model": False,
            "ran_arena": False,
            "promoted_model": False,
        },
    }
    FULL_RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary_path": str(SUMMARY_PATH),
                "results_path": str(FULL_RESULTS_PATH),
                "classification": classification,
                "recommended_next_action": next_action,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
