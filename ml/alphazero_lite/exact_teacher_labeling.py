"""Production exact-teacher labeling with the native hybrid MTD(f) teacher.

This module owns the pure, MCTS-free conversion from native hybrid exact
labels (player-zero final stone margins) into training-eligible rows.

Value semantics (validated against ``native_probe.c`` and PR #281):

* the native ``label`` operation returns ``exact_value`` and per-action
  ``action_values`` as player-zero final store margins (signed integers);
* the root selects ``max`` for player 0 and ``min`` for player 1;
* ``optimal_actions`` are exactly the legal actions attaining that root.

Training conventions (repository-grounded, no invented normalization):

* training ``value`` targets are root-player perspective values in
  ``[-1, 1]`` (see ``value_from_classic_mcts_root``,
  ``tb_value_to_training``, and PR #76/#77 exact artifacts). The exact
  training value is therefore the sign of the root-perspective margin:
  ``+1.0`` forced win, ``0.0`` forced draw, ``-1.0`` forced loss.
* the training ``policy`` target is uniform over the exact optimal-action
  set and zero elsewhere (illegal actions always receive zero mass, as
  required by ``train.py`` validation).
* the raw integer margins are preserved verbatim alongside the targets so
  nothing is lost; ``exact_value_normalized_48`` is stored only as an
  explicitly labeled diagnostic (margin divided by the 48-stone encoding
  divisor), never as a training target.

This module never imports classic MCTS. Failures stay explicit: the
conversion raises instead of falling back to any approximate teacher.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ml.alphazero_lite.exact_kalah_solver import ExactState
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

POLICY_SIZE = 6
PITS_DIVISOR = 48.0
TEACHER_IDENTITY = "native_hybrid_exact"
TEACHER_VERSION = "pr281_production_v1"
POLICY_TARGET_MODE = "default"
VALUE_TARGET_MODE = "default"
INPUT_ENCODING = "kalah_v3"

SOURCE_BUCKETS = ("8-16", "17-24", "25-32", "33-40")
SOURCE_BUCKET_RANGES = {
    "8-16": (8, 16),
    "17-24": (17, 24),
    "25-32": (25, 32),
    "33-40": (33, 40),
}


class ExactLabelError(ValueError):
    """Raised when a native exact label is inconsistent or unusable."""


def validate_native_label(
    *,
    action_values: dict[int, int],
    optimal_actions: list[int],
    exact_value: int,
    legal_moves: list[int],
    current_player: int,
) -> int:
    """Validate native value semantics and return the root margin.

    Raises ExactLabelError when the response violates the documented native
    contract (root extremum, optimal set identity, or legality) instead of
    silently accepting or falling back to another teacher.
    """
    if current_player not in (0, 1):
        raise ExactLabelError(f"current_player must be 0 or 1, got {current_player}")
    if not legal_moves:
        raise ExactLabelError("state exposes no legal moves")
    if not action_values:
        raise ExactLabelError("native response carries no action values")
    illegal_valued = sorted(set(action_values) - set(legal_moves))
    if illegal_valued:
        raise ExactLabelError(f"native values for illegal actions: {illegal_valued}")
    missing_valued = sorted(set(legal_moves) - set(action_values))
    if missing_valued:
        raise ExactLabelError(f"native values missing for actions: {missing_valued}")
    expected_root = (
        max(action_values.values())
        if current_player == 0
        else min(action_values.values())
    )
    if exact_value != expected_root:
        raise ExactLabelError(
            f"native exact_value {exact_value} != root extremum {expected_root}"
        )
    expected_optimal = sorted(
        action for action, value in action_values.items() if value == expected_root
    )
    if sorted(optimal_actions) != expected_optimal:
        raise ExactLabelError(
            f"native optimal_actions {sorted(optimal_actions)} != "
            f"extremum actions {expected_optimal}"
        )
    illegal_optimal = sorted(set(optimal_actions) - set(legal_moves))
    if illegal_optimal:
        raise ExactLabelError(
            f"native optimal set contains illegal actions: {illegal_optimal}"
        )
    if not optimal_actions:
        raise ExactLabelError("native optimal-action set is empty")
    return expected_root


def policy_target_from_optimal(
    *, optimal_actions: list[int], legal_moves: list[int]
) -> list[float]:
    """Return a 6-vector policy target uniform over the optimal set."""
    if not optimal_actions:
        raise ExactLabelError("cannot build a policy target from an empty optimal set")
    optimal_set = set(optimal_actions)
    if not optimal_set <= set(legal_moves):
        raise ExactLabelError(
            f"optimal actions {sorted(optimal_set)} include illegal moves "
            f"for legal set {sorted(legal_moves)}"
        )
    mass = 1.0 / len(optimal_set)
    policy = [0.0] * POLICY_SIZE
    for move in sorted(optimal_set):
        policy[move] = mass
    if abs(sum(policy) - 1.0) > 1e-9:
        raise ExactLabelError("uniform optimal policy does not sum to 1.0")
    return policy


def exact_margin_to_training_value(*, exact_margin: int, current_player: int) -> float:
    """Map a player-zero final margin to a root-perspective training value.

    Returns +1.0 (forced win), 0.0 (forced draw), or -1.0 (forced loss) from
    the perspective of the player to move, matching the repository's
    win-rate-derived exact-value convention (``2 * win_rate - 1``).
    """
    if current_player not in (0, 1):
        raise ExactLabelError(f"current_player must be 0 or 1, got {current_player}")
    sign = 0.0 if exact_margin == 0 else (1.0 if exact_margin > 0 else -1.0)
    if current_player == 1:
        sign = -sign
    return sign


def active_stone_bucket(active_stones: int) -> str:
    for bucket in SOURCE_BUCKETS:
        low, high = SOURCE_BUCKET_RANGES[bucket]
        if low <= active_stones <= high:
            return bucket
    raise ExactLabelError(f"active stone count {active_stones} outside 8-40 scope")


def phase_label(move_index: int) -> str:
    if move_index <= 8:
        return "early"
    if move_index <= 24:
        return "mid"
    return "late"


def game_state_from_row_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_pits": [int(v) for v in state["player_pits"]],
        "opponent_pits": [int(v) for v in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
        "current_player": int(state["current_player"]),
    }


def build_training_row(
    *,
    source: dict[str, Any],
    action_values: dict[int, int],
    optimal_actions: list[int],
    exact_value: int,
    label_wall_seconds: float,
    label_cpu_seconds: float,
    teacher_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Convert one validated native label into a train.py-compatible row."""
    state = game_state_from_row_state(source["state"])
    game = KalahGame.from_state(state)
    legal_moves = game.possible_moves()
    root_margin = validate_native_label(
        action_values=action_values,
        optimal_actions=optimal_actions,
        exact_value=exact_value,
        legal_moves=legal_moves,
        current_player=int(state["current_player"]),
    )
    policy = policy_target_from_optimal(
        optimal_actions=list(optimal_actions), legal_moves=legal_moves
    )
    value = exact_margin_to_training_value(
        exact_margin=root_margin, current_player=int(state["current_player"])
    )
    encoded = encode_state(state, input_encoding=INPUT_ENCODING)
    row = {
        "state": encoded,
        "policy": policy,
        "value": value,
        "policy_target_mode": POLICY_TARGET_MODE,
        "value_target_mode": VALUE_TARGET_MODE,
        "player": int(state["current_player"]),
        "move_index": int(source["move_index"]),
        "legal_moves": legal_moves,
        "canonical_state": str(source["canonical_state"]),
        "source_id": str(source["source_id"]),
        "teacher": TEACHER_IDENTITY,
        "teacher_version": TEACHER_VERSION,
        "exact_root_margin": int(root_margin),
        "exact_action_margins": {str(k): int(v) for k, v in action_values.items()},
        "exact_optimal_actions": sorted(int(a) for a in optimal_actions),
        "exact_value_training": value,
        "exact_value_normalized_48": round(root_margin / PITS_DIVISOR, 6),
        "label_wall_seconds": float(label_wall_seconds),
        "label_cpu_seconds": float(label_cpu_seconds),
    }
    row.update({f"teacher_{k}": v for k, v in teacher_provenance.items()})
    row["source_provenance"] = dict(source.get("provenance", {}))
    return row


def build_failed_row(
    *,
    source: dict[str, Any],
    status: str,
    error: str,
    teacher_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build an explicit failed/unsolved label attempt. Never a fallback."""
    if status not in ("failed", "timeout"):
        raise ExactLabelError(f"failed-row status must be failed/timeout, got {status}")
    row = {
        "source_id": str(source["source_id"]),
        "canonical_state": str(source["canonical_state"]),
        "status": status,
        "error": str(error),
        "teacher": TEACHER_IDENTITY,
        "teacher_version": TEACHER_VERSION,
        "exact": False,
    }
    row.update({f"teacher_{k}": v for k, v in teacher_provenance.items()})
    row["source_provenance"] = dict(source.get("provenance", {}))
    return row


def freeze_source_cohort(
    *,
    seed: int,
    per_bucket: int,
    exclusion_keys: dict[str, set[str]] | None = None,
    max_games: int = 400_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze a deterministic, deduplicated training-eligible source cohort.

    Random-play games from the standard opening are scanned for positions
    with 8-40 active stones. States are deduplicated by canonical key and
    stratified evenly across the four active-stone buckets. Excluded keys
    (feasibility corpus, evaluation suites, opening suites) are skipped and
    counted, never labeled.
    """
    excluded = exclusion_keys or {}
    feasibility_keys = set(excluded.get("feasibility_corpus", set()))
    forensic_keys = set(excluded.get("forensic_suite", set()))
    opening_hashes = set(excluded.get("opening_suites", set()))
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in SOURCE_BUCKETS}
    seen: set[str] = set()
    stats: dict[str, Any] = {
        "games_played": 0,
        "positions_visited": 0,
        "duplicates_skipped": 0,
        "terminal_skipped": 0,
        "out_of_scope_skipped": 0,
        "excluded_feasibility": 0,
        "excluded_forensic_suite": 0,
        "excluded_opening_suite": 0,
    }
    game_index = 0
    while any(len(rows) < per_bucket for rows in buckets.values()):
        if game_index >= max_games:
            raise ExactLabelError(
                f"could not fill all buckets after {max_games} games: "
                + str({b: len(r) for b, r in buckets.items()})
            )
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for move_index in range(200):
            if game.over():
                stats["terminal_skipped"] += 1
                break
            legal = game.possible_moves()
            if not legal:
                break
            raw = game.to_state()
            active = sum(raw["player_pits"]) + sum(raw["opponent_pits"])
            stats["positions_visited"] += 1
            key = canonical_state_key(raw)
            if key in seen:
                stats["duplicates_skipped"] += 1
            else:
                seen.add(key)
                if key in feasibility_keys:
                    stats["excluded_feasibility"] += 1
                elif key in forensic_keys:
                    stats["excluded_forensic_suite"] += 1
                elif _opening_hash(raw) in opening_hashes:
                    stats["excluded_opening_suite"] += 1
                elif not any(
                    low <= active <= high for low, high in SOURCE_BUCKET_RANGES.values()
                ):
                    stats["out_of_scope_skipped"] += 1
                else:
                    bucket = active_stone_bucket(active)
                    if len(buckets[bucket]) < per_bucket:
                        buckets[bucket].append(
                            {
                                "source_id": (
                                    f"exact-prod-frozen-{bucket}-"
                                    f"{len(buckets[bucket]):05d}"
                                ),
                                "state": game_state_from_row_state(raw),
                                "canonical_state": key,
                                "active_stones": int(active),
                                "stones_bucket": bucket,
                                "phase": phase_label(move_index),
                                "move_index": int(move_index),
                                "player": int(raw["current_player"]),
                                "legal_moves": list(legal),
                                "training_eligible": True,
                                "provenance": {
                                    "generator": "exact_teacher_label_production",
                                    "generator_version": TEACHER_VERSION,
                                    "source_seed": int(seed),
                                    "source_game_index": int(game_index),
                                    "source_ply": int(move_index),
                                },
                            }
                        )
            if not game.move(game.pit_index(rng.choice(legal))):
                break
        game_index += 1
    stats["games_played"] = game_index
    rows = [row for bucket in SOURCE_BUCKETS for row in buckets[bucket]]
    stats["frozen_rows"] = len(rows)
    stats["rows_per_bucket"] = {b: len(r) for b, r in buckets.items()}
    return rows, stats


def _opening_hash(state: dict[str, Any]) -> str:
    from ml.alphazero_lite import build_opening_suite as suites

    return suites.canonical_key(state)


def deterministic_split(
    rows: list[dict[str, Any]], *, train_split: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/holdout by sorted canonical key.

    Canonical keys sort deterministically, so train and holdout are disjoint
    by construction and stable across resumed runs.
    """
    if not 0.0 < train_split < 1.0:
        raise ExactLabelError(f"train_split must be in (0, 1), got {train_split}")
    ordered = sorted(rows, key=lambda row: str(row["canonical_state"]))
    split_index = max(1, round(len(ordered) * train_split))
    train_rows = ordered[:split_index]
    holdout_rows = ordered[split_index:]
    train_keys = {str(r["canonical_state"]) for r in train_rows}
    holdout_keys = {str(r["canonical_state"]) for r in holdout_rows}
    if train_keys & holdout_keys:
        raise ExactLabelError("train/holdout canonical-state overlap detected")
    return train_rows, holdout_rows


def derive_mcts_audit_seed(
    *, base_seed: int, canonical_state: str, simulations: int
) -> int:
    material = f"exact-teacher-mcts-audit|{canonical_state}|{int(simulations)}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**31)


def summarize_latencies(seconds: list[float]) -> dict[str, Any]:
    if not seconds:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "max": None,
        }
    ordered = sorted(seconds)
    n = len(ordered)

    def quantile(q: float) -> float:
        pos = min(n - 1, max(0, int(round(q * (n - 1)))))
        return float(ordered[pos])

    return {
        "count": n,
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "mean": float(sum(ordered) / n),
        "max": float(ordered[-1]),
    }


def compute_audit_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired MCTS-vs-exact audit metrics from aligned label pairs.

    Each pair carries ``exact_optimal_actions`` (list), ``exact_single`` (bool
    derived), ``mcts_top`` (int|None), ``mcts_value`` and ``exact_value``
    (both root-perspective, in [-1, 1]), ``mcts_top_visit_share``, and an
    optional ``bucket`` label. Exact-optimal-set membership is the primary
    policy metric because several actions can be exactly tied.
    """
    total = len(pairs)
    if total == 0:
        raise ExactLabelError("audit requires at least one pair")
    in_set = sum(1 for p in pairs if p["mcts_top"] in set(p["exact_optimal_actions"]))
    singles = [p for p in pairs if len(p["exact_optimal_actions"]) == 1]
    single_agree = sum(
        1 for p in singles if p["mcts_top"] == p["exact_optimal_actions"][0]
    )
    multis = [p for p in pairs if len(p["exact_optimal_actions"]) > 1]
    disagreements = [
        abs(float(p["mcts_value"]) - float(p["exact_value"])) for p in pairs
    ]
    correct = [p for p in pairs if p["mcts_top"] in set(p["exact_optimal_actions"])]
    incorrect = [
        p for p in pairs if p["mcts_top"] not in set(p["exact_optimal_actions"])
    ]
    by_bucket: dict[str, Any] = {}
    for bucket in sorted({str(p.get("bucket", "unknown")) for p in pairs}):
        subset = [p for p in pairs if str(p.get("bucket", "unknown")) == bucket]
        sub_in_set = sum(
            1 for p in subset if p["mcts_top"] in set(p["exact_optimal_actions"])
        )
        sub_dis = [
            abs(float(p["mcts_value"]) - float(p["exact_value"])) for p in subset
        ]
        by_bucket[bucket] = {
            "n": len(subset),
            "mcts_top_in_exact_set_rate": sub_in_set / len(subset),
            "mean_abs_value_disagreement": sum(sub_dis) / len(sub_dis),
        }
    return {
        "n": total,
        "mcts_top_in_exact_optimal_set": in_set,
        "mcts_top_in_exact_optimal_set_rate": in_set / total,
        "exact_single_optimum_n": len(singles),
        "exact_single_optimum_mcts_top1_agreement": (
            single_agree / len(singles) if singles else None
        ),
        "exact_multi_optimum_n": len(multis),
        "exact_multi_optimum_rate": len(multis) / total,
        "mean_abs_value_disagreement": sum(disagreements) / total,
        "max_abs_value_disagreement": max(disagreements),
        "mean_mcts_top_visit_share_when_correct": (
            sum(float(p["mcts_top_visit_share"]) for p in correct) / len(correct)
            if correct
            else None
        ),
        "mean_mcts_top_visit_share_when_incorrect": (
            sum(float(p["mcts_top_visit_share"]) for p in incorrect) / len(incorrect)
            if incorrect
            else None
        ),
        "by_bucket": by_bucket,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_rows(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_state_from_game_state(state: dict[str, Any]) -> ExactState:
    return ExactState.from_game_state(game_state_from_row_state(state))


def load_feasibility_corpus_keys() -> set[str]:
    from ml.alphazero_lite.run_exact_solver_feasibility_preflight import (
        DEFAULT_SEED,
        generate_feasibility_corpus,
    )

    return {
        canonical_state_key(row["state"])
        for row in generate_feasibility_corpus(DEFAULT_SEED)
    }


def load_forensic_suite_keys(
    path: str | Path = "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json",
) -> set[str]:
    from ml.alphazero_lite.forensic_suite import load_suite

    return {position.canonical_key for position in load_suite(path)}


def load_opening_suite_hashes(
    paths: list[str | Path],
) -> tuple[set[str], dict[str, int]]:
    from ml.alphazero_lite import build_opening_suite as suites

    hashes: set[str] = set()
    per_file: dict[str, int] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            per_file[str(path)] = -1
            continue
        entries = suites.load_suite_jsonl(str(path))
        file_hashes = {suites.canonical_key(entry["state"]) for entry in entries}
        hashes |= file_hashes
        per_file[str(path)] = len(file_hashes)
    return hashes, per_file
