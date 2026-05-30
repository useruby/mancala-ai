#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    validated_diagnostic_state,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.root_prior_transforms import (
    apply_root_prior_transform,
    move_feature_annotations_for,
)
from ml.alphazero_lite.run_capture_002_root_prior_transform_followup import (
    ROW_IDS as LOCAL_GUARD_ROW_IDS,
)
from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    build_probe_row,
    load_json,
    repo_root,
    row_map_from_reference,
    write_json,
)


SCHEMA = "azlite_root_prior_transform_activation_conditioned_eval_v1"
DEFAULT_OUT_ROOT = "/tmp/azlite_root_prior_activation_eval"
DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"
DEFAULT_GUARDED_W2_PATH = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_TRANSFORM = "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5"
DEFAULT_RANDOM_PREFIX_COUNT = 800
DEFAULT_RANDOM_PREFIX_MIN_PLIES = 2
DEFAULT_RANDOM_PREFIX_MAX_PLIES = 8
DEFAULT_SELF_PLAY_SAMPLE_LIMIT = 12000
DEFAULT_ACTIVATED_LIMIT = 96
DEFAULT_CHANGED_STATE_LIMIT = 24
DEFAULT_CONTINUATIONS = 16
DEFAULT_OUTCOME_BUDGET = 384
DEFAULT_SCAN_MIN_NON_GUARD_ACTIVATIONS = 3
DEFAULT_SCAN_MIN_NON_GUARD_RATE = 0.001
DEFAULT_SEED = 17
C_PUCT = 1.25
PAIRED_SEARCH_BUDGETS = (128, 384, 1200)
SEARCH_SETTINGS = {
    "fpu_mode": "parent_q",
    "reuse_subtree": True,
    "normalize_values": True,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
}
CURRENT_SELF_PLAY_CANDIDATES = (
    "/tmp/azlite_exp_v3_selfplay_sims384_versions/exp-v3-selfplay-sims384-iter1/self_play.jsonl",
    "/tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1/self_play.jsonl",
    "/tmp/azlite_v3_clone_extend_versions/aggressive-v3-clone-extend-iter1/self_play.jsonl",
)
NON_DEPLOYMENT_GATE_SOURCES = frozenset({"fixtures", "guard_rows"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--guarded-w2-path", default=DEFAULT_GUARDED_W2_PATH)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--transform-name", default=DEFAULT_TRANSFORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--random-prefix-count", type=int, default=DEFAULT_RANDOM_PREFIX_COUNT
    )
    parser.add_argument(
        "--random-prefix-min-plies", type=int, default=DEFAULT_RANDOM_PREFIX_MIN_PLIES
    )
    parser.add_argument(
        "--random-prefix-max-plies", type=int, default=DEFAULT_RANDOM_PREFIX_MAX_PLIES
    )
    parser.add_argument(
        "--self-play-sample-limit", type=int, default=DEFAULT_SELF_PLAY_SAMPLE_LIMIT
    )
    parser.add_argument("--activated-limit", type=int, default=DEFAULT_ACTIVATED_LIMIT)
    parser.add_argument(
        "--changed-state-limit", type=int, default=DEFAULT_CHANGED_STATE_LIMIT
    )
    parser.add_argument("--continuations", type=int, default=DEFAULT_CONTINUATIONS)
    parser.add_argument("--outcome-budget", type=int, default=DEFAULT_OUTCOME_BUDGET)
    parser.add_argument(
        "--scan-min-non-guard-activations",
        type=int,
        default=DEFAULT_SCAN_MIN_NON_GUARD_ACTIVATIONS,
    )
    parser.add_argument(
        "--scan-min-non-guard-rate",
        type=float,
        default=DEFAULT_SCAN_MIN_NON_GUARD_RATE,
    )
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def artifact_label(path: Path) -> str:
    return "current" if path.name == "current" else path.name


def load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def decode_v3_state(features: list[Any]) -> dict[str, Any]:
    if not isinstance(features, list) or len(features) < 15:
        raise ValueError("encoded state must be a feature list with at least 15 values")
    return {
        "player_pits": [
            max(0, int(round(float(features[idx]) * 48.0))) for idx in range(6)
        ],
        "opponent_pits": [
            max(0, int(round(float(features[6 + idx]) * 48.0))) for idx in range(6)
        ],
        "player_store": max(0, int(round(float(features[12]) * 48.0))),
        "opponent_store": max(0, int(round(float(features[13]) * 48.0))),
        "current_player": int(round(float(features[14]))),
    }


def canonical_state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def side_to_move_label(state: dict[str, Any]) -> str:
    return f"player_{int(state.get('current_player', 0))}"


def initial_state() -> dict[str, Any]:
    return {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }


def state_after_random_prefix(*, seed: int, plies: int) -> dict[str, Any] | None:
    rng = random.Random(seed)
    game = KalahGame.from_state(initial_state())
    applied = 0
    for _ in range(max(0, plies)):
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        move = rng.choice(legal_moves)
        if not game.move(game.pit_index(move)):
            break
        applied += 1
        if game.over():
            break
    if applied <= 0:
        return None
    return game.to_state()


def collect_states_from_finished_games(
    games_path: Path, *, limit: int
) -> list[dict[str, Any]]:
    payload = json.loads(games_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    states: list[dict[str, Any]] = []
    for game_index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        moves = row.get("move_history")
        if not isinstance(moves, list) or not moves:
            continue
        game = KalahGame.from_state(initial_state())
        for ply, move_row in enumerate(moves, start=1):
            if len(states) >= limit:
                return states
            if not isinstance(move_row, dict):
                break
            raw_pit = move_row.get("pit")
            try:
                move = int(raw_pit)
            except (TypeError, ValueError):
                break
            legal_moves = game.possible_moves()
            if move not in legal_moves:
                break
            state = game.to_state()
            states.append(
                {
                    "state": state,
                    "ply": ply - 1,
                    "source": f"finished_games_2026_04_06:game_{game_index:03d}",
                    "source_group": "finished_games_2026_04_06",
                }
            )
            if not game.move(game.pit_index(move)):
                break
    return states


def load_current_self_play_source(current_path: Path) -> Path | None:
    version = None
    metadata_path = current_path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        version = metadata.get("version")
    for candidate in CURRENT_SELF_PLAY_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        meta_path = path.parent / "metadata.json"
        if not meta_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if version is not None and metadata.get("version") == version:
            return path
    return None


def telemetry_for_state(
    *,
    state: dict[str, Any],
    transform_name: str,
    model_prior: list[float] | np.ndarray | None = None,
) -> dict[str, Any] | None:
    game = KalahGame.from_state(state)
    legal_moves = game.possible_moves()
    if not legal_moves:
        return None
    annotations = move_feature_annotations_for(state=state, legal_moves=legal_moves)
    prior = (
        np.asarray(model_prior, dtype=np.float32)
        if model_prior is not None
        else np.full(6, 1.0 / 6.0, dtype=np.float32)
    )
    transformed, telemetry = apply_root_prior_transform(
        state=state,
        legal_moves=legal_moves,
        original_root_prior=prior,
        move_feature_annotations=annotations,
        transform_name=transform_name,
    )
    return {
        "legal_moves": [int(move) for move in legal_moves],
        "move_annotations": {
            str(move): dict(annotations[move]) for move in legal_moves
        },
        "telemetry": telemetry,
        "transformed_prior": [float(value) for value in transformed.tolist()],
    }


def fixture_state_entries(
    reference_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row_id, row in reference_rows.items():
        if not row_id.startswith("capture_available-"):
            continue
        if row_id in LOCAL_GUARD_ROW_IDS:
            continue
        state = validated_diagnostic_state(row=build_probe_row(row))
        entries.append(
            {
                "state": state,
                "ply": int(row_id.split("-")[-1]),
                "source": f"fixture:{row_id}",
                "source_group": "fixtures",
                "fixture_row_id": row_id,
            }
        )
    return entries


def guard_state_entries(
    reference_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row_id in LOCAL_GUARD_ROW_IDS:
        row = reference_rows[row_id]
        state = validated_diagnostic_state(row=build_probe_row(row))
        entries.append(
            {
                "state": state,
                "ply": int(row_id.split("-")[-1]),
                "source": f"guard_row:{row_id}",
                "source_group": "guard_rows",
                "fixture_row_id": row_id,
                "is_guard_row": True,
            }
        )
    return entries


def summarize_scan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_group"]), str(row["artifact"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (source_group, artifact), items in sorted(grouped.items()):
        activation_count = sum(1 for item in items if bool(item.get("activated")))
        plies = [int(item["ply"]) for item in items]
        legal_move_counts = Counter(int(item["legal_move_count"]) for item in items)
        side_counts = Counter(str(item["side_to_move"]) for item in items)
        notes: list[str] = []
        if source_group == "guard_rows":
            notes.append("known_guard_rows")
        if activation_count <= 0:
            notes.append("no_activation")
        summary_rows.append(
            {
                "source": source_group,
                "artifact": artifact,
                "states_scanned": len(items),
                "activation_count": activation_count,
                "activation_rate": round(activation_count / len(items), 4)
                if items
                else None,
                "median_ply": statistics.median(plies) if plies else None,
                "ply_distribution": dict(sorted(Counter(plies).items())),
                "legal_move_count_distribution": dict(
                    sorted(legal_move_counts.items())
                ),
                "side_to_move_distribution": dict(sorted(side_counts.items())),
                "notes": ",".join(notes) if notes else "ok",
            }
        )
    return summary_rows


def scan_source_entries(
    *,
    entries: list[dict[str, Any]],
    artifact: str,
    transform_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        state = entry["state"]
        telemetry_bundle = telemetry_for_state(
            state=state, transform_name=transform_name
        )
        if telemetry_bundle is None:
            continue
        telemetry = telemetry_bundle["telemetry"]
        rows.append(
            {
                "artifact": artifact,
                "source": entry["source"],
                "source_group": entry["source_group"],
                "state": state,
                "canonical_state": canonical_state_key(state),
                "ply": int(entry.get("ply", 0)),
                "side_to_move": side_to_move_label(state),
                "legal_move_count": len(telemetry_bundle["legal_moves"]),
                "legal_moves": telemetry_bundle["legal_moves"],
                "activated": bool(telemetry.get("activated")),
                "transform_telemetry": telemetry,
                "fixture_row_id": entry.get("fixture_row_id"),
                "is_guard_row": bool(entry.get("is_guard_row", False)),
            }
        )
    return rows


def load_self_play_entries(
    *,
    path: Path,
    source_group: str,
    source_prefix: str,
    limit: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(load_jsonl(path, limit=limit)):
        state = decode_v3_state(row["state"])
        entries.append(
            {
                "state": state,
                "ply": int(row.get("move_index", 0)),
                "source": f"{source_prefix}:{index:05d}",
                "source_group": source_group,
            }
        )
    return entries


def split_label(index: int) -> str:
    remainder = index % 10
    if remainder < 6:
        return "train"
    if remainder < 8:
        return "dev"
    return "holdout"


def child_metric_by_move(summary: dict[str, Any], key: str) -> dict[str, float] | None:
    child_stats = summary.get("child_stats")
    if not isinstance(child_stats, list):
        return None
    result: dict[str, float] = {}
    for row in child_stats:
        if not isinstance(row, dict) or row.get("move") is None:
            continue
        result[str(int(row["move"]))] = round(float(row.get(key, 0.0)), 4)
    return result


def policy_distribution(summary: dict[str, Any]) -> dict[str, float] | None:
    legal_moves = summary.get("legal_moves")
    policy = summary.get("policy")
    if not isinstance(legal_moves, list) or not isinstance(policy, list):
        return None
    return {str(int(move)): round(float(policy[int(move)]), 4) for move in legal_moves}


def selected_is_reference(reference_move: int | None, move: int | None) -> bool | None:
    if reference_move is None or move is None:
        return None
    return int(reference_move) == int(move)


def build_activated_state_rows(
    *,
    activated_scan_rows: list[dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
    evaluators: dict[str, ArtifactEvaluator],
    transform_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, scan_row in enumerate(activated_scan_rows):
        state = scan_row["state"]
        source = str(scan_row["source"])
        artifact = str(scan_row["artifact"])
        evaluator = evaluators[artifact]
        game = KalahGame.from_state(state)
        base_prior, _value = evaluator.evaluate(game)
        telemetry_bundle = telemetry_for_state(
            state=state,
            transform_name=transform_name,
            model_prior=base_prior,
        )
        assert telemetry_bundle is not None
        fixture_row_id = scan_row.get("fixture_row_id")
        reference_move = None
        if isinstance(fixture_row_id, str) and fixture_row_id in reference_rows:
            reference_move = int(reference_rows[fixture_row_id]["reference_move"])
        rows.append(
            {
                "state": state,
                "canonical_state": scan_row["canonical_state"],
                "source": source,
                "source_group": scan_row["source_group"],
                "artifact": artifact,
                "ply": int(scan_row["ply"]),
                "side_to_move": scan_row["side_to_move"],
                "legal_moves": telemetry_bundle["legal_moves"],
                "transform_telemetry": telemetry_bundle["telemetry"],
                "original_prior_by_move": {
                    str(move): round(float(base_prior[move]), 4)
                    for move in telemetry_bundle["legal_moves"]
                },
                "transformed_prior_by_move": {
                    str(move): round(
                        float(telemetry_bundle["transformed_prior"][move]), 4
                    )
                    for move in telemetry_bundle["legal_moves"]
                },
                "fixture_row_id": fixture_row_id,
                "is_known_fixture": fixture_row_id is not None,
                "is_guard_row": bool(scan_row.get("is_guard_row", False)),
                "reference_move": reference_move,
                "split": split_label(index),
            }
        )
    return rows


def paired_search_for_state(
    *,
    artifact_path: Path,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    simulations: int,
    seed: int,
    transform_name: str,
) -> dict[str, Any]:
    base_summary = evaluate_artifact_position(
        artifact_path=artifact_path,
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=C_PUCT,
        search_options=dict(SEARCH_SETTINGS),
        ablation_mode="full",
        root_prior_transform=None,
    )
    transformed_summary = evaluate_artifact_position(
        artifact_path=artifact_path,
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=C_PUCT,
        search_options=dict(SEARCH_SETTINGS),
        ablation_mode="full",
        root_prior_transform=transform_name,
    )
    base_selected = base_summary.get("selected_move")
    transformed_selected = transformed_summary.get("selected_move")
    changed = base_selected != transformed_selected
    transformed_telemetry = (
        (transformed_summary.get("root_prior_telemetry") or {})
        if isinstance(transformed_summary, dict)
        else {}
    )
    legal_moves = list(
        base_summary.get("legal_moves") or transformed_summary.get("legal_moves") or []
    )
    base_move_features = move_feature_annotations_for(
        state=state, legal_moves=legal_moves
    )
    changed_to_capture = None
    changed_from_extra_turn = None
    if changed and transformed_selected is not None:
        changed_to_capture = bool(
            base_move_features[int(transformed_selected)]["produces_capture"]
        )
    if changed and base_selected is not None:
        changed_from_extra_turn = bool(
            base_move_features[int(base_selected)]["gives_extra_turn"]
        )
    return {
        "selected_move_no_transform": base_selected,
        "selected_move_transform": transformed_selected,
        "selected_move_changed": changed,
        "changed_to_capture": changed_to_capture,
        "changed_from_extra_turn": changed_from_extra_turn,
        "visit_share_by_move_before": policy_distribution(base_summary),
        "visit_share_by_move_after": policy_distribution(transformed_summary),
        "q_by_move_before": child_metric_by_move(base_summary, "q_value"),
        "q_by_move_after": child_metric_by_move(transformed_summary, "q_value"),
        "prior_mass_shift": (
            transformed_telemetry.get("mass_shift")
            if isinstance(transformed_telemetry, dict)
            else None
        ),
        "transform_activation_telemetry": transformed_telemetry,
    }


def select_search_move(
    *,
    artifact_path: Path,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    simulations: int,
    seed: int,
    transform_name: str | None,
) -> int | None:
    summary = evaluate_artifact_position(
        artifact_path=artifact_path,
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=C_PUCT,
        search_options=dict(SEARCH_SETTINGS),
        ablation_mode="full",
        root_prior_transform=transform_name,
    )
    move = summary.get("selected_move")
    return None if move is None else int(move)


def rollout_outcome_for_side(state: dict[str, Any], *, side_to_move: int) -> float:
    game = KalahGame.from_state(state)
    if not game.over():
        return 0.5
    if game.winner is None:
        return 0.5
    return 1.0 if int(game.winner) == int(side_to_move) else 0.0


def continue_game_after_move(
    *,
    artifact_path: Path,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    first_move: int,
    transform_name: str | None,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    starting_player = int(game.current_player)
    legal_moves = game.possible_moves()
    if first_move not in legal_moves:
        raise ValueError("first_move must be legal")
    capture_events = 0
    extra_turn_events = 0
    move_count = 0

    def apply_move_and_track(move: int) -> None:
        nonlocal capture_events, extra_turn_events, move_count
        before = game.to_state()
        annotations = move_feature_annotations_for(
            state=before, legal_moves=game.possible_moves()
        )
        if annotations[int(move)]["produces_capture"]:
            capture_events += 1
        acting_player = int(game.current_player)
        if not game.move(game.pit_index(move)):
            raise ValueError("illegal move during rollout")
        move_count += 1
        if game.current_player == acting_player and not game.over():
            extra_turn_events += 1

    apply_move_and_track(first_move)
    while not game.over() and move_count < 200:
        move = select_search_move(
            artifact_path=artifact_path,
            evaluator=evaluator,
            state=game.to_state(),
            simulations=simulations,
            seed=seed + move_count,
            transform_name=transform_name,
        )
        if move is None:
            break
        apply_move_and_track(move)
    final_state = game.to_state()
    return {
        "outcome": rollout_outcome_for_side(final_state, side_to_move=starting_player),
        "capture_count": capture_events,
        "extra_turn_chain_length": extra_turn_events,
        "game_length": move_count,
    }


def mean_confidence_interval(
    values: list[float],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    std = statistics.stdev(values)
    margin = 1.96 * (std / math.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def gate_scan_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in summary_rows
        if row["source"] not in NON_DEPLOYMENT_GATE_SOURCES
        and row["states_scanned"] > 0
    ]


def classify_summary(summary: dict[str, Any]) -> tuple[str, str]:
    gate_rows = gate_scan_rows(summary["activation_scan"])
    gate_activations = sum(int(row["activation_count"]) for row in gate_rows)
    gate_scanned = sum(int(row["states_scanned"]) for row in gate_rows)
    gate_rate = 0.0 if gate_scanned <= 0 else gate_activations / gate_scanned
    min_activations = int(
        summary.get(
            "scan_min_non_guard_activations", DEFAULT_SCAN_MIN_NON_GUARD_ACTIVATIONS
        )
    )
    min_rate = float(
        summary.get("scan_min_non_guard_rate", DEFAULT_SCAN_MIN_NON_GUARD_RATE)
    )
    if gate_activations < min_activations or gate_rate < min_rate:
        return (
            "transform_too_rare_for_deployment",
            "abandon this transform as a deployment feature; move to input encoding / policy-target audit.",
        )

    paired_rows = summary.get("paired_search_rows") or []
    changed_count = sum(
        1 for row in paired_rows if bool(row.get("selected_move_changed"))
    )
    if paired_rows and changed_count <= max(1, len(paired_rows) // 20):
        return (
            "transform_low_leverage",
            "abandon deployment; keep only as diagnostic.",
        )

    outcome_rows = summary.get("outcome_rows") or []
    if outcome_rows:
        means = [
            float(row["mean_outcome_delta"])
            for row in outcome_rows
            if row.get("mean_outcome_delta") is not None
        ]
        if means and max(means) <= 0.0:
            return (
                "local_policy_fix_not_value_improvement",
                "audit teacher labels/value backup for the affected context.",
            )
        current_means = [
            float(row["mean_outcome_delta"])
            for row in outcome_rows
            if row.get("artifact") == "current"
            and row.get("mean_outcome_delta") is not None
        ]
        guarded_means = [
            float(row["mean_outcome_delta"])
            for row in outcome_rows
            if row.get("artifact") == "guarded-w2"
            and row.get("mean_outcome_delta") is not None
        ]
        if current_means and guarded_means:
            current_best = max(current_means)
            guarded_best = max(guarded_means)
            if abs(current_best - guarded_best) <= 0.03:
                return (
                    "search_patch_not_model_improvement",
                    "evaluate as optional search mode on current only; do not promote guarded-w2 weights.",
                )
            if guarded_best > current_best + 0.03:
                return (
                    "candidate_transform_interaction",
                    "run a larger equal-budget targeted micro-arena for guarded-w2 with transform.",
                )
        if means and max(means) > 0.0:
            return (
                "targeted_search_patch_promising",
                "run a larger activated-start-position micro-arena, not general promotion.",
            )

    return (
        "transform_low_leverage",
        "abandon deployment; keep only as diagnostic.",
    )


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Root-Prior Transform Activation-Conditioned Eval Results",
        "",
        "## Context",
        "",
        "This experiment tests whether the narrow root-prior transform activates in realistic or semi-realistic positions and whether activation changes downstream search or outcome.",
        "",
        f"- transform: `{summary['transform_name']}`",
        f"- current artifact: `{summary['current_path_display']}`",
        f"- guarded-w2 artifact: `{summary['guarded_w2_path_display']}`",
        f"- classification: `{summary['classification']}`",
        f"- deployment-gate activation count: `{summary['deployment_gate_activation_count']}` / `{summary['deployment_gate_states_scanned']}`",
        f"- deployment-gate activation rate: `{summary['deployment_gate_activation_rate']}`",
        "",
        "## Activation-rate Scan",
        "",
        "| source | artifact | states_scanned | activation_count | activation_rate | median_ply | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["activation_scan"]:
        lines.append(
            "| {source} | {artifact} | {states_scanned} | {activation_count} | {activation_rate} | {median_ply} | {notes} |".format(
                source=row["source"],
                artifact=row["artifact"],
                states_scanned=row["states_scanned"],
                activation_count=row["activation_count"],
                activation_rate=row["activation_rate"],
                median_ply=row["median_ply"],
                notes=row["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Activated-state Dataset Summary",
            "",
            f"- activated states written: `{summary['activated_state_count']}`",
            f"- activated states path: `{summary['activated_states_path']}`",
            f"- non-guard activated states: `{summary['non_guard_activated_state_count']}`",
            f"- current self-play source status: `{summary['current_self_play_status']}`",
            "",
            "## Paired Search Results",
            "",
            "| artifact | simulations | activated_states | changed_selection_count | changed_selection_rate | reference_improvement_count | reference_regression_count | average_prior_mass_shift | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    paired_rows = summary.get("paired_search_summary", [])
    for row in paired_rows:
        lines.append(
            "| {artifact} | {simulations} | {activated_states} | {changed_selection_count} | {changed_selection_rate} | {reference_improvement_count} | {reference_regression_count} | {average_prior_mass_shift} | {notes} |".format(
                **row
            )
        )
    if not paired_rows:
        lines.append(
            "No paired search run because the activation-rate gate stopped the experiment after Step 1."
        )
    lines.extend(
        [
            "",
            "## Outcome Rollout Results",
            "",
            "| artifact | simulations | changed_states | continuations_per_state | mean_outcome_delta | ci_low | ci_high | positive_count | negative_count | neutral_count | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    outcome_rows = summary.get("outcome_rows", [])
    for row in outcome_rows:
        lines.append(
            "| {artifact} | {simulations} | {changed_states} | {continuations_per_state} | {mean_outcome_delta} | {ci_low} | {ci_high} | {positive_count} | {negative_count} | {neutral_count} | {notes} |".format(
                **row
            )
        )
    if not outcome_rows:
        lines.append(
            "No outcome rollout run because the activation-rate gate stopped the experiment after Step 1."
        )
    lines.extend(
        [
            "",
            "## Optional Activated-state Micro-arena",
            "",
            "Not run because the activation-rate gate did not justify continuing to paired search or rollout.",
            "",
            "## Interpretation",
            "",
            f"- classification: `{summary['classification']}`",
            f"- interpretation: {summary['interpretation']}",
            f"- gate basis: only `{summary['deployment_gate_activation_count']}` activations across `{summary['deployment_gate_states_scanned']}` realistic or semi-realistic scanned states, excluding the hand-built fixture and guard sources",
            "",
            "## Recommended Next Action",
            "",
            f"Recommendation: **{summary['recommended_next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out_root = resolve_path(root, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    current_path = resolve_path(root, args.current_path)
    guarded_w2_path = resolve_path(root, args.guarded_w2_path)
    reference_path = resolve_path(root, args.reference_artifact)
    activated_states_path = out_root / "activated_states.jsonl"
    summary_path = out_root / "activation_conditioned_eval_summary.json"
    report_path = (
        root
        / "docs/alphazero-lite-root-prior-transform-activation-conditioned-eval-results.md"
    )

    reference_rows = row_map_from_reference(load_json(reference_path))
    random_prefix_entries: list[dict[str, Any]] = []
    for index in range(max(0, int(args.random_prefix_count))):
        plies = random.Random(int(args.seed) + index).randint(
            int(args.random_prefix_min_plies), int(args.random_prefix_max_plies)
        )
        state = state_after_random_prefix(seed=int(args.seed) + index, plies=plies)
        if state is None:
            continue
        random_prefix_entries.append(
            {
                "state": state,
                "ply": plies,
                "source": f"random_prefix:{index:05d}",
                "source_group": "random_prefixes",
            }
        )

    finished_games_entries = collect_states_from_finished_games(
        root / "ml/alphazero_lite/fixtures/superhuman_strength_games_2026_04_06.json",
        limit=max(200, int(args.random_prefix_count) // 2),
    )
    fixture_entries = fixture_state_entries(reference_rows)
    guard_entries = guard_state_entries(reference_rows)
    guarded_self_play_entries = load_self_play_entries(
        path=guarded_w2_path / "self_play.jsonl",
        source_group="guarded_w2_self_play",
        source_prefix="guarded_w2_self_play",
        limit=int(args.self_play_sample_limit),
    )
    current_self_play_entries: list[dict[str, Any]] = []
    current_self_play_path = load_current_self_play_source(current_path)
    current_self_play_status = "unavailable_exact_match"
    if current_self_play_path is not None:
        current_self_play_entries = load_self_play_entries(
            path=current_self_play_path,
            source_group="current_self_play",
            source_prefix="current_self_play",
            limit=int(args.self_play_sample_limit),
        )
        current_self_play_status = str(current_self_play_path)

    scan_rows: list[dict[str, Any]] = []
    for artifact in ("current", "guarded-w2"):
        scan_rows.extend(
            scan_source_entries(
                entries=random_prefix_entries,
                artifact=artifact,
                transform_name=args.transform_name,
            )
        )
        scan_rows.extend(
            scan_source_entries(
                entries=fixture_entries,
                artifact=artifact,
                transform_name=args.transform_name,
            )
        )
        scan_rows.extend(
            scan_source_entries(
                entries=guard_entries,
                artifact=artifact,
                transform_name=args.transform_name,
            )
        )
        scan_rows.extend(
            scan_source_entries(
                entries=finished_games_entries,
                artifact=artifact,
                transform_name=args.transform_name,
            )
        )
    if current_self_play_entries:
        scan_rows.extend(
            scan_source_entries(
                entries=current_self_play_entries,
                artifact="current",
                transform_name=args.transform_name,
            )
        )
    if guarded_self_play_entries:
        scan_rows.extend(
            scan_source_entries(
                entries=guarded_self_play_entries,
                artifact="guarded-w2",
                transform_name=args.transform_name,
            )
        )

    activation_scan = summarize_scan_rows(scan_rows)
    non_guard_activated_rows = [
        row
        for row in scan_rows
        if bool(row["activated"])
        and not bool(row.get("is_guard_row"))
        and row["source_group"] != "guard_rows"
    ]
    activated_guard_rows = [
        row
        for row in scan_rows
        if bool(row["activated"]) and bool(row.get("is_guard_row"))
    ]
    non_guard_activated_rows.sort(
        key=lambda row: (
            int(row["source_group"] == "guard_rows"),
            int(row["ply"]),
            row["canonical_state"],
            row["artifact"],
        )
    )
    selected_activated_rows = (
        activated_guard_rows + non_guard_activated_rows[: int(args.activated_limit)]
    )

    evaluators = {
        "current": ArtifactEvaluator(current_path),
        "guarded-w2": ArtifactEvaluator(guarded_w2_path),
    }
    activated_state_rows = build_activated_state_rows(
        activated_scan_rows=selected_activated_rows,
        reference_rows=reference_rows,
        evaluators=evaluators,
        transform_name=args.transform_name,
    )
    write_jsonl(activated_states_path, activated_state_rows)

    paired_search_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    paired_search_summary: list[dict[str, Any]] = []
    gate_rows = gate_scan_rows(activation_scan)
    gate_activation_count = sum(int(row["activation_count"]) for row in gate_rows)
    gate_scanned = sum(int(row["states_scanned"]) for row in gate_rows)
    gate_rate = 0.0 if gate_scanned <= 0 else gate_activation_count / gate_scanned
    continue_beyond_scan = gate_activation_count >= int(
        args.scan_min_non_guard_activations
    ) and gate_rate >= float(args.scan_min_non_guard_rate)

    if continue_beyond_scan:
        for artifact, artifact_path in (
            ("current", current_path),
            ("guarded-w2", guarded_w2_path),
        ):
            evaluator = evaluators[artifact]
            artifact_rows = [
                row
                for row in activated_state_rows
                if row["artifact"] == artifact and not bool(row["is_guard_row"])
            ]
            for budget in PAIRED_SEARCH_BUDGETS:
                budget_rows: list[dict[str, Any]] = []
                for state_index, row in enumerate(artifact_rows):
                    paired = paired_search_for_state(
                        artifact_path=artifact_path,
                        evaluator=evaluator,
                        state=row["state"],
                        simulations=budget,
                        seed=int(args.seed) + (1000 * (state_index + 1)) + budget,
                        transform_name=args.transform_name,
                    )
                    paired["artifact"] = artifact
                    paired["simulations"] = budget
                    paired["state_index"] = state_index
                    paired["source"] = row["source"]
                    paired["fixture_row_id"] = row.get("fixture_row_id")
                    reference_move = row.get("reference_move")
                    paired["selected_is_reference_before"] = selected_is_reference(
                        reference_move, paired["selected_move_no_transform"]
                    )
                    paired["selected_is_reference_after"] = selected_is_reference(
                        reference_move, paired["selected_move_transform"]
                    )
                    paired_search_rows.append(paired)
                    budget_rows.append(paired)

                changed_count = sum(
                    1 for row in budget_rows if row["selected_move_changed"]
                )
                reference_improvement_count = sum(
                    1
                    for row in budget_rows
                    if row.get("selected_is_reference_before") is False
                    and row.get("selected_is_reference_after") is True
                )
                reference_regression_count = sum(
                    1
                    for row in budget_rows
                    if row.get("selected_is_reference_before") is True
                    and row.get("selected_is_reference_after") is False
                )
                shifts = [
                    float(row["prior_mass_shift"])
                    for row in budget_rows
                    if row.get("prior_mass_shift") is not None
                ]
                paired_search_summary.append(
                    {
                        "artifact": artifact,
                        "simulations": budget,
                        "activated_states": len(budget_rows),
                        "changed_selection_count": changed_count,
                        "changed_selection_rate": round(
                            changed_count / len(budget_rows), 4
                        )
                        if budget_rows
                        else None,
                        "reference_improvement_count": reference_improvement_count,
                        "reference_regression_count": reference_regression_count,
                        "average_prior_mass_shift": round(statistics.fmean(shifts), 4)
                        if shifts
                        else None,
                        "notes": "ok"
                        if budget_rows
                        else "no_non_guard_activated_states",
                    }
                )

            changed_candidates = [
                row
                for row in paired_search_rows
                if row["artifact"] == artifact
                and int(row["simulations"]) == int(args.outcome_budget)
                and bool(row["selected_move_changed"])
            ][: int(args.changed_state_limit)]
            if not changed_candidates:
                outcome_rows.append(
                    {
                        "artifact": artifact,
                        "simulations": int(args.outcome_budget),
                        "changed_states": 0,
                        "continuations_per_state": int(args.continuations),
                        "mean_outcome_delta": None,
                        "ci_low": None,
                        "ci_high": None,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "notes": "no_changed_states",
                    }
                )
                continue
            deltas: list[float] = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            for changed_index, changed_row in enumerate(changed_candidates):
                source_state = next(
                    row["state"]
                    for row in activated_state_rows
                    if row["artifact"] == artifact
                    and row["source"] == changed_row["source"]
                )
                branch_deltas: list[float] = []
                for continuation_index in range(int(args.continuations)):
                    branch_seed = (
                        int(args.seed)
                        + (changed_index * 10000)
                        + continuation_index
                        + int(args.outcome_budget)
                    )
                    no_transform_result = continue_game_after_move(
                        artifact_path=artifact_path,
                        evaluator=evaluator,
                        state=source_state,
                        first_move=int(changed_row["selected_move_no_transform"]),
                        transform_name=None,
                        simulations=int(args.outcome_budget),
                        seed=branch_seed,
                    )
                    transform_result = continue_game_after_move(
                        artifact_path=artifact_path,
                        evaluator=evaluator,
                        state=source_state,
                        first_move=int(changed_row["selected_move_transform"]),
                        transform_name=args.transform_name,
                        simulations=int(args.outcome_budget),
                        seed=branch_seed,
                    )
                    delta = float(
                        transform_result["outcome"] - no_transform_result["outcome"]
                    )
                    deltas.append(delta)
                    branch_deltas.append(delta)
                mean_delta = statistics.fmean(branch_deltas) if branch_deltas else 0.0
                if mean_delta > 0.02:
                    positive_count += 1
                elif mean_delta < -0.02:
                    negative_count += 1
                else:
                    neutral_count += 1
            mean_delta, ci_low, ci_high = mean_confidence_interval(deltas)
            outcome_rows.append(
                {
                    "artifact": artifact,
                    "simulations": int(args.outcome_budget),
                    "changed_states": len(changed_candidates),
                    "continuations_per_state": int(args.continuations),
                    "mean_outcome_delta": None
                    if mean_delta is None
                    else round(mean_delta, 4),
                    "ci_low": None if ci_low is None else round(ci_low, 4),
                    "ci_high": None if ci_high is None else round(ci_high, 4),
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                    "notes": "ok",
                }
            )

    summary = {
        "schema": SCHEMA,
        "transform_name": args.transform_name,
        "current_path": str(current_path),
        "current_path_display": display_path(root, current_path),
        "guarded_w2_path": str(guarded_w2_path),
        "guarded_w2_path_display": display_path(root, guarded_w2_path),
        "reference_artifact": str(reference_path),
        "scan_min_non_guard_activations": int(args.scan_min_non_guard_activations),
        "scan_min_non_guard_rate": float(args.scan_min_non_guard_rate),
        "current_self_play_status": current_self_play_status,
        "activation_scan": activation_scan,
        "activated_state_count": len(activated_state_rows),
        "non_guard_activated_state_count": len(non_guard_activated_rows),
        "deployment_gate_activation_count": gate_activation_count,
        "deployment_gate_states_scanned": gate_scanned,
        "deployment_gate_activation_rate": round(gate_rate, 6)
        if gate_scanned > 0
        else None,
        "activated_states_path": str(activated_states_path),
        "paired_search_rows": paired_search_rows,
        "paired_search_summary": paired_search_summary,
        "outcome_rows": outcome_rows,
        "micro_arena": {"ran": False, "reason": "not_requested_by_decision_gate"},
    }
    classification, recommendation = classify_summary(summary)
    summary["classification"] = classification
    summary["interpretation"] = classification.replace("_", " ")
    summary["recommended_next_action"] = recommendation
    write_json(summary_path, summary)
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(f"wrote summary to {summary_path}")
    print(f"wrote report to {report_path}")
    print(f"classification={classification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
