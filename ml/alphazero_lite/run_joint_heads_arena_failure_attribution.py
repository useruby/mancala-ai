#!/usr/bin/env python3
"""No-training causal attribution for a frozen joint-head candidate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
)
from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    derive_search_seed,
    stable_hash,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    canonical_manifest_hash,
    sha256_file,
)
from ml.alphazero_lite.run_residual_v3_iteration0_target_causal_audit import (  # noqa: E402
    forced_continuation,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402

LANES = {
    "current_policy_current_value": ("current", "current"),
    "candidate_policy_current_value": ("candidate", "current"),
    "current_policy_candidate_value": ("current", "candidate"),
    "candidate_policy_candidate_value": ("candidate", "candidate"),
}
HEAD_KEYS = {
    "w_policy",
    "b_policy",
    "w_policy_hidden",
    "b_policy_hidden",
    "w_value",
    "b_value",
    "w_value_hidden",
    "b_value_hidden",
}
COMPOSITION_BUDGETS = (
    "384:256",
    "768:256",
    "768:768",
    "1200:1200",
    "1200:256",
    "256:768",
)
TRACE_BUDGETS = ("384:256",)
STAGES = ("composition_scores", "trace_384", "coverage", "forced", "classify")
_WORKER_EVALUATORS: dict[str, arena.ComposedArtifactEvaluator] | None = None
_WORKER_CURRENT: arena.ArtifactEvaluator | None = None


def canonical_hash(value: Any) -> str:
    """Hash a raw JSON representation. Never use this for encoded state vectors."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(canonical_hash(list(parts))[:16], 16) % (2**31)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a JSON artifact so interrupted stages remain resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically publish a JSONL shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def phase(game: KalahGame) -> str:
    remaining = sum(game.pits)
    return "late" if remaining <= 12 else "midgame" if remaining <= 24 else "opening"


def composition_hash(current_hash: str, candidate_hash: str, lane: str) -> str:
    return canonical_hash(
        {
            "current_weights_sha256": current_hash,
            "candidate_weights_sha256": candidate_hash,
            "composition": lane,
        }
    )


def lane_evaluators(
    current: Path, candidate: Path
) -> dict[str, arena.ComposedArtifactEvaluator]:
    incumbent, joint = (
        arena.ArtifactEvaluator(current),
        arena.ArtifactEvaluator(candidate),
    )
    return {
        name: arena.ComposedArtifactEvaluator(
            incumbent, joint, policy_source=p, value_source=v
        )
        for name, (p, v) in LANES.items()
    }


def initialize_play_worker(current: str, candidate: str) -> None:
    """Create immutable evaluators once per worker process."""
    global _WORKER_EVALUATORS, _WORKER_CURRENT
    _WORKER_CURRENT = arena.ArtifactEvaluator(Path(current))
    candidate_evaluator = arena.ArtifactEvaluator(Path(candidate))
    _WORKER_EVALUATORS = {
        name: arena.ComposedArtifactEvaluator(
            _WORKER_CURRENT,
            candidate_evaluator,
            policy_source=policy,
            value_source=value,
        )
        for name, (policy, value) in LANES.items()
    }


def forced_item_worker(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Run one persisted intervention using the worker's initialized evaluator."""
    if _WORKER_EVALUATORS is None or _WORKER_CURRENT is None:
        raise RuntimeError("forced worker was not initialized")
    row = payload["row"]
    baseline, move = int(payload["baseline_move"]), int(payload["alternative_move"])
    continuation_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    records = []
    for mode, budget, cpuct in payload["contexts"]:
        item_seed = stable_seed(
            payload["seed"],
            row["state_hash"],
            row["budget_pair"],
            payload["component"],
            mode,
            budget,
        )

        def continuation(forced_move: int) -> dict[str, Any]:
            key = (row["state_hash"], forced_move, budget, cpuct, item_seed)
            if key not in continuation_cache:
                continuation_cache[key] = forced_continuation(
                    state=row["state"],
                    forced_move=forced_move,
                    evaluator=_WORKER_CURRENT,
                    simulations=budget,
                    c_puct=cpuct,
                    search_options=options(),
                    seed=item_seed,
                )
            return continuation_cache[key]

        base, forced = continuation(baseline), continuation(move)
        records.append(
            {
                "mode": mode,
                "component": payload["component"],
                "state_hash": row["state_hash"],
                "budget_pair": row["budget_pair"],
                "challenger_player": row["challenger_player"],
                "acting_player": row["acting_player"],
                "phase": row["phase"],
                "coverage_status": "covered"
                if not row["no_exact_replay_overlap"]
                else "non_overlapping",
                "nearest_distance_quartile": row.get("nearest_distance_quartile"),
                "delta": float(forced["outcome_root"] - base["outcome_root"]),
                "continuation_simulations": budget,
                "effective_c_puct": cpuct,
            }
        )
    return payload["item_id"], records


def verify_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    current_weights, candidate_weights = (
        Path(args.current) / "weights.json",
        Path(args.candidate) / "weights.json",
    )
    if sha256_file(current_weights) != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    summary = json.loads(Path(args.pr164_summary).read_text(encoding="utf-8"))
    if sha256_file(candidate_weights) != summary["run_a"]["artifact_weights_sha256"]:
        raise RuntimeError("candidate weights hash mismatch")
    manifest = json.loads(Path(args.training_manifest).read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256_excluding_this_field") != canonical_manifest_hash(
        manifest
    ):
        raise RuntimeError("training manifest hash mismatch")
    if sha256_file(Path(args.replay)) != manifest.get("replay_sha256"):
        raise RuntimeError("replay hash mismatch")
    if sha256_file(Path(manifest["artifact_paths"]["batch_indexes"])) != manifest.get(
        "batch_plan_sha256"
    ):
        raise RuntimeError("batch-plan hash mismatch")
    current, candidate = (
        json.loads(current_weights.read_text()),
        json.loads(candidate_weights.read_text()),
    )
    if set(current) != set(candidate) or any(
        current[key] != candidate[key] for key in set(current) - HEAD_KEYS
    ):
        raise RuntimeError("candidate trunk differs from current")
    if not any(current[key] != candidate[key] for key in HEAD_KEYS):
        raise RuntimeError("candidate heads did not change")
    return summary, manifest


def options() -> dict[str, Any]:
    return build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )


def search(
    evaluator: Any, state: dict[str, Any], simulations: int, seed: int, c_puct: float
) -> dict[str, Any]:
    return arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=c_puct,
        search_options=options(),
    )


def profile_hash(budget_pair: str, c_puct: float) -> str:
    return canonical_hash(
        {
            "budget_pair": budget_pair,
            "c_puct": c_puct,
            "root_policy": "deterministic",
            "normalize_values": False,
            "tactical_root_bias": 0.0,
            "value_interpolation": False,
            "value_trust": False,
            "root_prior_transform": None,
        }
    )


def play_unit(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One opening/seat/composition work unit with private immutable evaluators."""
    evaluators = _WORKER_EVALUATORS or lane_evaluators(
        Path(payload["current"]), Path(payload["candidate"])
    )
    lane, opening, challenger_player = (
        payload["lane"],
        payload["opening"],
        payload["challenger_player"],
    )
    challenger_sims, current_sims, c_puct = (
        payload["challenger_sims"],
        payload["current_sims"],
        payload["c_puct"],
    )
    game = KalahGame.from_state(opening["state"])
    ply = int(opening.get("ply", len(opening.get("prefix_moves", []))))
    traces, trajectory = [], []
    while not game.over():
        state, legal = game.to_state(), [int(move) for move in game.possible_moves()]
        if not legal:
            break
        acting_challenger = game.current_player == challenger_player
        simulations = challenger_sims if acting_challenger else current_sims
        acting_role = "challenger" if acting_challenger else "current"
        seed, seed_context_hash = derive_search_seed(
            base_seed=payload["seed"],
            suite_sha256=payload.get(
                "suite_sha256", stable_hash([opening.get("state")])
            ),
            budget_pair=payload["budget_pair"],
            opening_index=payload["opening_index"],
            opening_state_hash=payload.get(
                "opening_state_hash", stable_hash(opening["state"])
            ),
            challenger_player=challenger_player,
            game_within_opening=challenger_player,
            ply=ply,
            canonical_current_state_hash=stable_hash(state),
            acting_role=acting_role,
            simulations=simulations,
            effective_c_puct=c_puct,
        )
        selected = search(
            evaluators[lane]
            if acting_challenger
            else evaluators["current_policy_current_value"],
            state,
            simulations,
            seed,
            c_puct,
        )
        if payload.get("trace", True):
            traces.append(
                {
                    "state": state,
                    "state_hash": canonical_hash(state),
                    "budget_pair": payload["budget_pair"],
                    "challenger_simulations": challenger_sims,
                    "incumbent_simulations": current_sims,
                    "effective_c_puct": c_puct,
                    "search_seed": seed,
                    "seed_contract": SEED_CONTRACT_VERSION,
                    "seed_context_hash": seed_context_hash,
                    "challenger_player": challenger_player,
                    "acting_player": game.current_player,
                    "composition_trajectory_source": lane,
                    "opening_identifier": opening.get("id", payload["opening_index"]),
                    "game_identifier": f"{payload['budget_pair']}:{lane}:{payload['opening_index']}:{challenger_player}",
                    "ply": ply,
                    "phase": phase(game),
                    "legal_moves": legal,
                    # The trajectory search is itself this composition's result.
                    # Other compositions are evaluated once per deduplicated context
                    # after all trajectories have completed.
                    "active_selected_move": selected["selected_move"],
                    "active_child_visits": selected["visits"],
                    "search_profile_hash": profile_hash(payload["budget_pair"], c_puct),
                }
            )
        move = int(selected["selected_move"])
        trajectory.append(move)
        game.move(game.pit_index(move))
        ply += 1
    score = (
        1.0 if game.winner == challenger_player else 0.5 if game.winner is None else 0.0
    )
    return {
        "lane": lane,
        "challenger_player": challenger_player,
        "score": score,
        "trajectory": ",".join(map(str, trajectory)),
    }, traces


def run_budget(
    *,
    current: Path,
    candidate: Path,
    openings: list[dict[str, Any]],
    label: str,
    seed: int,
    c_puct: float,
    workers: int,
    trace: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    challenger_sims, current_sims = map(int, label.split(":"))
    suite_sha256 = stable_hash(
        [opening.get("prefix_moves", opening.get("state")) for opening in openings]
    )
    units = [
        {
            "current": str(current),
            "candidate": str(candidate),
            "opening": opening,
            "opening_index": i,
            "lane": lane,
            "challenger_player": player,
            "challenger_sims": challenger_sims,
            "current_sims": current_sims,
            "budget_pair": label,
            "c_puct": c_puct,
            "seed": seed,
            "suite_sha256": suite_sha256,
            "opening_state_hash": stable_hash(opening["state"]),
            "trace": trace,
        }
        for i, opening in enumerate(openings)
        for lane in LANES
        for player in (0, 1)
    ]
    # Each process initializes immutable evaluators once; map preserves task order.
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_play_worker,
            initargs=(str(current), str(candidate)),
        ) as pool:
            completed = list(pool.map(play_unit, units))
    else:
        completed = [play_unit(unit) for unit in units]
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces: list[dict[str, Any]] = []
    for game, rows in completed:
        games[game["lane"]].append(game)
        if trace:
            traces.extend(rows)
    metrics = {}
    for lane in LANES:
        lane_games = games[lane]
        p0 = [row["score"] for row in lane_games if row["challenger_player"] == 0]
        p1 = [row["score"] for row in lane_games if row["challenger_player"] == 1]
        trajectories = Counter(row["trajectory"] for row in lane_games)
        metrics[lane] = {
            "raw_ds": statistics.fmean(p0) - statistics.fmean(p1),
            "p0_score": statistics.fmean(p0),
            "p1_score": statistics.fmean(p1),
            "unique_trajectories": len(trajectories),
            "duplicate_trajectory_count": sum(
                n for n in trajectories.values() if n > 1
            ),
            "challenger_simulations": challenger_sims,
            "incumbent_simulations": current_sims,
            "effective_c_puct": c_puct,
        }
    return metrics, traces


def raw_state_vector(state: dict[str, Any]) -> np.ndarray:
    if "pits" in state:
        return np.asarray(
            list(state["pits"]) + list(state["captured_seeds"]), dtype=np.int16
        )
    return np.asarray(
        list(state["player_pits"])
        + list(state["opponent_pits"])
        + [state["player_store"], state["opponent_store"]],
        dtype=np.int16,
    )


def replay_coverage(
    traces: list[dict[str, Any]],
    replay: Path,
    manifest: dict[str, Any],
    workers: int,
    distance_selector: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(replay)
    train, validation = (
        set(manifest["train_source_row_indexes"]),
        set(manifest["validation_source_row_indexes"]),
    )
    indexes: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        # Replay's training tensor is encoded in ``state``; only ``raw_state``
        # participates in arena identity and distance diagnostics.
        raw_state = row.get("raw_state")
        if not isinstance(raw_state, dict):
            raise RuntimeError("replay row is missing raw_state required for coverage")
        raw_hash = canonical_hash(raw_state)
        replay_game = KalahGame.from_state(raw_state)
        entry = indexes.setdefault(
            raw_hash,
            {
                "count": 0,
                "train_game_ids": set(),
                "validation_game_ids": set(),
                "vector": raw_state_vector(raw_state),
                "player": raw_state.get("current_player"),
                "phase": phase(replay_game),
                "legal_move_count": len(replay_game.possible_moves()),
            },
        )
        entry["count"] += 1
        game_id = row.get("game_index")
        if index in train:
            entry["train_game_ids"].add(game_id)
        if index in validation:
            entry["validation_game_ids"].add(game_id)
    replay_entries = list(indexes.values())
    replay_vectors = np.asarray(
        [entry["vector"] for entry in replay_entries], dtype=np.int16
    )
    replay_train = np.asarray(
        [entry["vector"] for entry in replay_entries if entry["train_game_ids"]],
        dtype=np.int16,
    )
    replay_validation = np.asarray(
        [entry["vector"] for entry in replay_entries if entry["validation_game_ids"]],
        dtype=np.int16,
    )

    unique_rows = {row["state_hash"]: row for row in traces}
    changed_hashes = {
        row["state_hash"]
        for row in unique_rows.values()
        if any(
            move != row["selected_moves"]["current_policy_current_value"]
            for lane, move in row.get("selected_moves", {}).items()
            if lane != "current_policy_current_value"
        )
    }

    def stratum(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row["acting_player"],
            row["phase"],
            row.get("opening_identifier"),
            len(row["legal_moves"]),
            row.get("budget_pair"),
        )

    changed_strata = Counter(stratum(unique_rows[key]) for key in changed_hashes)
    controls: set[str] = set()
    for key, row in sorted(unique_rows.items()):
        group = stratum(row)
        if (
            key not in changed_hashes
            and changed_strata[group] > 0
            and len(controls) < 512
        ):
            controls.add(key)
            changed_strata[group] -= 1
    # Fill unmatched control strata deterministically without exceeding the cap.
    for key in sorted(unique_rows):
        if key not in changed_hashes and len(controls) < 512:
            controls.add(key)
    distance_hashes = changed_hashes | controls

    def nearest_distances(
        vector: np.ndarray,
    ) -> tuple[int | None, int | None, int | None, dict[str, Any] | None]:
        def nearest(vectors: np.ndarray) -> int | None:
            if not len(vectors):
                return None
            best: int | None = None
            for start in range(0, len(vectors), 2048):
                distance = int(
                    np.abs(vectors[start : start + 2048] - vector).sum(axis=1).min()
                )
                best = distance if best is None else min(best, distance)
            return best

        best_index, best_distance = 0, None
        for start in range(0, len(replay_vectors), 2048):
            distances = np.abs(replay_vectors[start : start + 2048] - vector).sum(
                axis=1
            )
            index = int(np.argmin(distances))
            distance = int(distances[index])
            if best_distance is None or distance < best_distance:
                best_index, best_distance = start + index, distance
        return (
            best_distance,
            nearest(replay_train),
            nearest(replay_validation),
            replay_entries[best_index] if best_distance is not None else None,
        )

    def annotate(row: dict[str, Any]) -> dict[str, Any]:
        raw_hash = canonical_hash(row["state"])
        assert raw_hash == row["state_hash"], (
            "trace state hashes must be raw-state hashes"
        )
        entry = indexes.get(raw_hash)
        include_distances = raw_hash in distance_hashes
        vector = raw_state_vector(row["state"])
        nearest_replay, nearest_train, nearest_validation, nearest = (
            nearest_distances(vector) if include_distances else (None, None, None, None)
        )

        return {
            **row,
            "exact_training_replay_overlap": bool(entry and entry["train_game_ids"]),
            "exact_validation_replay_overlap": bool(
                entry and entry["validation_game_ids"]
            ),
            "no_exact_replay_overlap": entry is None,
            "replay_duplicate_occurrence_count": 0 if entry is None else entry["count"],
            "train_replay_game_ids": []
            if entry is None
            else sorted(entry["train_game_ids"]),
            "validation_replay_game_ids": []
            if entry is None
            else sorted(entry["validation_game_ids"]),
            "distance_evaluated": include_distances,
            "same_current_player": None
            if nearest is None
            else nearest["player"] == row["acting_player"],
            "same_phase": None if nearest is None else nearest["phase"] == row["phase"],
            "same_legal_move_count": None
            if nearest is None
            else nearest["legal_move_count"] == len(row["legal_moves"]),
            "pit_store_l1_distance": nearest_replay,
            "nearest_replay_state_distance": nearest_replay,
            "nearest_train_state_distance": nearest_train,
            "nearest_validation_state_distance": nearest_validation,
        }

    annotated_unique = {key: annotate(row) for key, row in unique_rows.items()}
    distances = sorted(
        row["nearest_replay_state_distance"]
        for row in annotated_unique.values()
        if row["nearest_replay_state_distance"] is not None
    )
    edges = np.percentile(distances, [25, 50, 75]).tolist() if distances else []
    for row in annotated_unique.values():
        distance = row["nearest_replay_state_distance"]
        row["nearest_distance_quartile"] = (
            None
            if distance is None
            else str(1 + sum(distance > edge for edge in edges))
        )
    annotated = [
        {
            **row,
            **{
                key: value
                for key, value in annotated_unique[row["state_hash"]].items()
                if key not in row
            },
        }
        for row in traces
    ]
    # Identical raw states are a protocol invariant; encoded representations are never compared here.
    assert all(canonical_hash(row["state"]) == row["state_hash"] for row in annotated)
    return {
        "arena_trace_rows": len(annotated),
        "exact_training_overlap_rate": sum(
            row["exact_training_replay_overlap"] for row in annotated
        )
        / max(1, len(annotated)),
        "exact_validation_overlap_rate": sum(
            row["exact_validation_replay_overlap"] for row in annotated
        )
        / max(1, len(annotated)),
        "no_exact_overlap_rate": sum(
            row["no_exact_replay_overlap"] for row in annotated
        )
        / max(1, len(annotated)),
        "distance_rows_evaluated": sum(row["distance_evaluated"] for row in annotated),
        "unique_traced_states": len(unique_rows),
        "changed_unique_states": len(changed_hashes),
        "matched_unchanged_controls": len(controls),
        "distance_chunk_rows": 2048,
        "nearest_distance_quartile_edges": edges,
        "hash_representation": "canonical_hash(raw_state_dict)",
        "encoded_hashes_compared": False,
    }, annotated


def aggregate(
    values: list[float],
    seed: int,
    clusters: list[str] | None = None,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Summarize effects with deterministic cluster (state) bootstrap samples."""
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[(clusters or [str(index)] * len(values))[index]].append(value)
    cluster_means = [statistics.fmean(group) for _, group in sorted(grouped.items())]
    boot = (
        [
            float(np.mean(rng.choice(cluster_means, len(cluster_means), replace=True)))
            for _ in range(bootstrap_samples)
        ]
        if cluster_means
        else [0.0]
    )
    return {
        "unique_state_hashes": len(cluster_means),
        "continuation_records": len(values),
        "mean_forced_outcome_delta": statistics.fmean(values) if values else 0.0,
        "median_forced_outcome_delta": statistics.median(values) if values else 0.0,
        "bootstrap_95_ci": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "fraction_delta_lte_negative_quarter": sum(v <= -0.25 for v in values)
        / max(1, len(values)),
        "fraction_delta_gte_positive_quarter": sum(v >= 0.25 for v in values)
        / max(1, len(values)),
    }


def forced_move_audit(
    traces: list[dict[str, Any]],
    current: Path,
    candidate: Path,
    shard_dir: Path,
    continuation_budgets: list[int],
    default_c_puct: float,
    schedule: dict[str, float],
    seed: int,
    workers: int,
    sample_size: int = 128,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Run a bounded, resumable forced audit with one atomic shard per item."""
    candidates: dict[str, list[tuple[dict[str, Any], int, int]]] = defaultdict(list)
    seen: set[tuple[str, str, str, int, int]] = set()
    for row in sorted(
        traces,
        key=lambda r: (
            r["state_hash"],
            r["budget_pair"],
            r["composition_trajectory_source"],
            r["game_identifier"],
        ),
    ):
        baseline = int(row["selected_moves"]["current_policy_current_value"])
        moves = row["selected_moves"]
        component_moves = {
            "policy": moves["candidate_policy_current_value"],
            "value": moves["current_policy_candidate_value"],
            "joint_only": moves["candidate_policy_candidate_value"],
        }
        for component, move in component_moves.items():
            move = int(move)
            if move == baseline:
                continue
            if component == "joint_only" and (
                moves["candidate_policy_current_value"] != baseline
                or moves["current_policy_candidate_value"] != baseline
            ):
                continue
            key = (row["state_hash"], row["budget_pair"], component, baseline, move)
            if key not in seen:
                seen.add(key)
                candidates[component].append((row, baseline, move))
    selected: list[dict[str, Any]] = []
    for component in ("policy", "value", "joint_only"):
        unique_states: set[str] = set()
        for row, baseline, move in sorted(
            candidates[component],
            key=lambda item: stable_seed(
                seed, component, item[0]["state_hash"], item[0]["opening_identifier"]
            ),
        ):
            if len(unique_states) >= sample_size or row["state_hash"] in unique_states:
                continue
            unique_states.add(row["state_hash"])
            challenger, incumbent = map(int, row["budget_pair"].split(":"))
            contexts = [("neutral", budget, 1.25) for budget in continuation_budgets]
            contexts.append(
                (
                    "context_matched",
                    challenger,
                    resolve_budget_cpuct(
                        schedule=schedule,
                        challenger_simulations=challenger,
                        current_simulations=incumbent,
                        default_c_puct=default_c_puct,
                    ),
                )
            )
            item_id = canonical_hash(
                [
                    "forced-v3",
                    row["state_hash"],
                    row["budget_pair"],
                    component,
                    baseline,
                    move,
                ]
            )
            selected.append(
                {
                    "item_id": item_id,
                    "row": row,
                    "component": component,
                    "baseline_move": baseline,
                    "alternative_move": move,
                    "contexts": contexts,
                    "seed": seed,
                }
            )
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, list[dict[str, Any]]] = {}
    pending = []
    for item in selected:
        path = shard_dir / f"{item['item_id']}.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("item_id") == item["item_id"]:
                    completed[item["item_id"]] = payload["records"]
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        pending.append(item)
    if pending:
        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=initialize_play_worker,
                initargs=(str(current), str(candidate)),
            ) as pool:
                futures = [pool.submit(forced_item_worker, item) for item in pending]
                for future in concurrent.futures.as_completed(futures):
                    item_id, item_records = future.result()
                    write_json(
                        shard_dir / f"{item_id}.json",
                        {"item_id": item_id, "records": item_records},
                    )
                    completed[item_id] = item_records
        else:
            initialize_play_worker(str(current), str(candidate))
            for item in pending:
                item_id, item_records = forced_item_worker(item)
                write_json(
                    shard_dir / f"{item_id}.json",
                    {"item_id": item_id, "records": item_records},
                )
                completed[item_id] = item_records
    records = [record for item in selected for record in completed[item["item_id"]]]
    result: dict[str, Any] = {
        "selected_context_moves": len(selected),
        "available_unique_states": {
            component: len({row["state_hash"] for row, _, _ in values})
            for component, values in candidates.items()
        },
        "records": records,
        "neutral": {},
        "context_matched": {},
    }
    for mode in ("neutral", "context_matched"):
        subset = [record for record in records if record["mode"] == mode]
        for component in ("policy", "value", "joint_only"):
            values = [
                record["delta"] for record in subset if record["component"] == component
            ]
            summary = aggregate(
                values,
                stable_seed(seed, mode, component),
                [
                    record["state_hash"]
                    for record in subset
                    if record["component"] == component
                ],
                bootstrap_samples,
            )
            component_records = [r for r in subset if r["component"] == component]
            summary["unique_state_context_pairs"] = len(
                {
                    (
                        r["state_hash"],
                        r["budget_pair"],
                        r["continuation_simulations"],
                        r["effective_c_puct"],
                    )
                    for r in component_records
                }
            )
            result[mode][component] = summary
        for field in (
            "challenger_player",
            "acting_player",
            "phase",
            "coverage_status",
            "nearest_distance_quartile",
        ):
            result[mode][f"by_{field}"] = {
                key: aggregate(
                    [r["delta"] for r in subset if str(r[field]) == key],
                    stable_seed(seed, mode, field, key),
                    [r["state_hash"] for r in subset if str(r[field]) == key],
                    bootstrap_samples,
                )
                for key in sorted({str(r[field]) for r in subset})
            }
    return result


def trace_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    states = {
        lane: {
            row["state_hash"]
            for row in traces
            if row["composition_trajectory_source"] == lane
        }
        for lane in LANES
    }

    shared = set.intersection(*states.values()) if states else set()
    unique = {
        lane: values
        - set.union(*(other for key, other in states.items() if key != lane))
        for lane, values in states.items()
    }
    rates = {}
    for lane, values in states.items():
        rows = [row for row in traces if row["composition_trajectory_source"] == lane]
        rates[lane] = {
            "shared_states_changed_move_rate": statistics.fmean(
                [
                    int(
                        row["selected_moves"][lane]
                        != row["selected_moves"]["current_policy_current_value"]
                    )
                    for row in rows
                    if row["state_hash"] in shared
                ]
            )
            if any(row["state_hash"] in shared for row in rows)
            else 0.0,
            "composition_unique_states_changed_move_rate": statistics.fmean(
                [
                    int(
                        row["selected_moves"][lane]
                        != row["selected_moves"]["current_policy_current_value"]
                    )
                    for row in rows
                    if row["state_hash"] in unique[lane]
                ]
            )
            if any(row["state_hash"] in unique[lane] for row in rows)
            else 0.0,
            "all_trajectory_states": len(values),
        }
    return {
        "unique_states_reached_by_composition": {
            lane: len(values) for lane, values in states.items()
        },
        "states_shared_by_all_four": len(shared),
        "states_unique_to_policy_only_trajectory": len(
            unique["candidate_policy_current_value"]
        ),
        "states_unique_to_value_only_trajectory": len(
            unique["current_policy_candidate_value"]
        ),
        "states_unique_to_joint_trajectory": len(
            unique["candidate_policy_candidate_value"]
        ),
        "changed_move_rates": rates,
    }


def distance_harm_evidence(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Apply the preregistered low/high-distance harm criterion by state."""
    grouped = {
        quartile: [r for r in records if r.get("nearest_distance_quartile") == quartile]
        for quartile in ("1", "2", "3", "4")
    }

    def state_means(rows: list[dict[str, Any]]) -> np.ndarray:
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            values[row["state_hash"]].append(row["delta"])
        return np.asarray(
            [statistics.fmean(value) for _, value in sorted(values.items())]
        )

    low, high = state_means(grouped["1"]), state_means(grouped["4"])
    difference, ci, concentrated = None, None, False
    if len(low) >= 64 and len(high) >= 64:
        rng = np.random.default_rng(stable_seed(seed, "distance-harm"))
        samples = [
            float(rng.choice(high, len(high)).mean() - rng.choice(low, len(low)).mean())
            for _ in range(1000)
        ]
        difference = float(high.mean() - low.mean())
        ci = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        concentrated = difference <= -0.10 and ci[1] < 0
    return {
        "by_quartile": {
            key: {
                "unique_state_hashes": len(state_means(rows)),
                "mean_forced_delta": float(state_means(rows).mean())
                if len(rows)
                else None,
            }
            for key, rows in grouped.items()
        },
        "high_minus_low_mean_forced_delta": difference,
        "high_minus_low_bootstrap_95_ci": ci,
        "distance_harm_concentrated": concentrated,
    }


def attribution(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table = {}
    for label in COMPOSITION_BUDGETS:
        base = metrics["current_policy_current_value"][label]["raw_ds"]
        policy = metrics["candidate_policy_current_value"][label]["raw_ds"] - base
        value = metrics["current_policy_candidate_value"][label]["raw_ds"] - base
        joint = metrics["candidate_policy_candidate_value"][label]["raw_ds"] - base
        residual = joint - policy - value
        table[label] = {
            "joint_delta": joint,
            "policy_only_delta": policy,
            "value_only_delta": value,
            "interaction_residual": residual,
            "policy_signed_fraction": policy / joint if joint else None,
            "value_signed_fraction": value / joint if joint else None,
            "interaction_signed_fraction": residual / joint if joint else None,
        }
    return table


def classify(
    metrics: dict[str, Any],
    forced: dict[str, Any],
    coverage: dict[str, Any],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    required = (
        bool(traces)
        and all(lane in metrics and "384:256" in metrics[lane] for lane in LANES)
        and all(mode in forced for mode in ("neutral", "context_matched"))
    )
    evidence: dict[str, Any] = {
        "required_stages_present": required,
        "failed_criteria": [],
    }
    if not required:
        return {
            "classification": "attribution_experiment_incomplete",
            "evidence": evidence,
        }
    base = metrics["current_policy_current_value"]["384:256"]["raw_ds"]
    policy = metrics["candidate_policy_current_value"]["384:256"]["raw_ds"] - base
    value = metrics["current_policy_candidate_value"]["384:256"]["raw_ds"] - base
    joint = metrics["candidate_policy_candidate_value"]["384:256"]["raw_ds"] - base
    values = {
        "joint_delta": joint,
        "policy_only_delta": policy,
        "value_only_delta": value,
        "interaction_residual": joint - policy - value,
    }
    evidence.update(values)
    if values["joint_delta"] >= 0:
        return {
            "classification": "no_joint_384_256_regression_reproduced",
            "evidence": evidence,
        }

    def component_harm(component: str) -> bool:
        return any(
            forced[mode].get(component, {}).get("bootstrap_95_ci", [0, 0])[1] < 0
            for mode in ("neutral", "context_matched")
        )

    def component_states(mode: str, component: str) -> int:
        summary = forced[mode].get(component, {})
        return int(
            summary.get("unique_state_hashes", summary.get("unique_changed_states", 0))
        )

    for component, delta_key in (
        ("policy", "policy_only_delta"),
        ("value", "value_only_delta"),
    ):
        enough = (
            component_states("neutral", component) >= 64
            or forced["context_matched"]
            .get(component, {})
            .get(
                "unique_state_hashes",
                forced["context_matched"]
                .get(component, {})
                .get("unique_changed_states", 0),
            )
            >= 64
        )
        if (
            values[delta_key] / values["joint_delta"] >= 0.70
            and component_harm(component)
            and enough
        ):
            return {
                "classification": f"{component}_component_primary_harm",
                "evidence": evidence,
            }
    interaction = values["interaction_residual"] / values["joint_delta"]
    if (
        abs(values["policy_only_delta"]) <= 0.03
        and abs(values["value_only_delta"]) <= 0.03
        and interaction >= 0.5
        and (
            component_states("neutral", "joint_only") >= 64
            or component_states("context_matched", "joint_only") >= 64
        )
        and component_harm("joint_only")
    ):
        return {
            "classification": "destructive_policy_value_interaction",
            "evidence": evidence,
        }
    exact = coverage.get("exact_training_overlap_rate", 0.0) + coverage.get(
        "exact_validation_overlap_rate", 0.0
    )
    covered, uncovered = (
        coverage.get("covered_harm_count", 0),
        coverage.get("uncovered_harm_count", 0),
    )
    evidence.update(
        {
            "exact_overlap_rate": exact,
            "covered_harm_count": covered,
            "uncovered_harm_count": uncovered,
        }
    )
    if (
        (exact < 0.10 or coverage.get("distance_harm_concentrated", False))
        and covered >= 64
        and uncovered >= 64
        and coverage.get("overlapping_materially_better", False)
    ):
        return {
            "classification": "replay_to_arena_distribution_shift",
            "evidence": evidence,
        }
    if (
        exact >= 0.10
        and coverage.get("covered_harmful_changes", False)
        and coverage.get("covered_forced_ci_upper", 0.0) < 0
    ):
        return {
            "classification": "replay_targets_bad_on_arena_states",
            "evidence": evidence,
        }
    evidence["failed_criteria"].append(
        "no head, interaction, or coverage mechanism met its prespecified criteria"
    )
    return {"classification": "proxy_probe_not_predictive", "evidence": evidence}


def markdown(summary: dict[str, Any]) -> str:
    return (
        "# Joint-Head Arena Failure Attribution\n\n## Input Hashes And PR #164 Reproduction\n\n```json\n"
        + json.dumps(summary["reproduction"], indent=2, sort_keys=True)
        + "\n```\n\n## Composition Semantic Checks\n\n```json\n"
        + json.dumps(summary["composition_semantic_checks"], indent=2)
        + "\n```\n\n## Medium DS And Signed Attribution\n\n```json\n"
        + json.dumps(
            {
                "medium": summary["medium_head_compositions"],
                "attribution": summary["signed_attribution"],
            },
            indent=2,
        )
        + "\n```\n\n## Trace Coverage And Replay Overlap\n\n```json\n"
        + json.dumps(
            {
                "trace": summary["trace_coverage"],
                "replay": summary["replay_to_arena_coverage"],
            },
            indent=2,
        )
        + "\n```\n\n## Neutral Forced-Move Audit\n\n```json\n"
        + json.dumps(summary["forced_move_audit"]["neutral"], indent=2)
        + "\n```\n\n## Context-Matched Forced-Move Audit\n\n```json\n"
        + json.dumps(summary["forced_move_audit"]["context_matched"], indent=2)
        + "\n```\n\n## Classification Evidence\n\n```json\n"
        + json.dumps(summary["classification"], indent=2)
        + "\n```\n\n## Next Recommended Scientific Action\n\n"
        + summary["next_recommended_scientific_action"]
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "workdir",
        "current",
        "expected-current-weights-sha256",
        "candidate",
        "pr164-summary",
        "training-manifest",
        "replay",
        "medium-suite",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--continuation-budgets", default="384,768,1200")
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages: composition_scores,trace_384,coverage,forced,classify (default: all).",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-recompute-stage", action="store_true", help="Rebuild selected stages."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Deterministically limit medium-suite openings for a bounded run.",
    )
    parser.add_argument("--forced-sample-per-component", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def parse_stages(value: str) -> set[str]:
    """Validate the explicitly requested stage subset."""
    aliases = {"composition": "composition_scores", "trace": "trace_384"}
    stages = {
        aliases.get(stage, stage)
        for stage in (STAGES if value == "all" else filter(None, value.split(",")))
    }
    unknown = stages - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {', '.join(sorted(unknown))}")
    if not stages:
        raise ValueError("at least one stage is required")
    return stages


def stage_manifest_path(workdir: Path, stage: str) -> Path:
    return workdir / "stages" / stage / "manifest.json"


def stage_is_ready(workdir: Path, stage: str, fingerprint: str) -> bool:
    path = stage_manifest_path(workdir, stage)
    if not path.exists():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return (
        manifest.get("fingerprint") == fingerprint
        and manifest.get("completion_status") == "complete"
        and all(
            Path(output).is_file() and sha256_file(Path(output)) == output_hash
            for output, output_hash in manifest.get("output_shard_hashes", {}).items()
        )
    )


def publish_stage(
    workdir: Path,
    stage: str,
    fingerprint: str,
    outputs: list[Path],
    inputs: dict[str, str] | None = None,
    configuration: dict[str, Any] | None = None,
) -> None:
    """Publish stage completion only after every atomically-written output exists."""
    if not all(path.is_file() for path in outputs):
        raise RuntimeError(f"cannot publish incomplete {stage} stage")
    run_manifest = workdir / "run_manifest.json"
    run_inputs = (
        json.loads(run_manifest.read_text(encoding="utf-8"))
        if run_manifest.exists()
        else {}
    )
    write_json(
        stage_manifest_path(workdir, stage),
        {
            "schema": "azlite_joint_heads_attribution_stage_v1",
            "stage": stage,
            "fingerprint": fingerprint,
            "current_artifact_hash": run_inputs.get("current_artifact_hash"),
            "candidate_artifact_hash": run_inputs.get("candidate_artifact_hash"),
            "replay_hash": run_inputs.get("replay_hash"),
            "training_manifest_hash": run_inputs.get("training_manifest_hash"),
            "medium_suite_hash": run_inputs.get("medium_suite_hash"),
            "runtime_profile_hash": run_inputs.get("runtime_profile_hash"),
            "input_shard_hashes": inputs or {},
            "output_shard_hashes": {str(path): sha256_file(path) for path in outputs},
            "stage_configuration": configuration or {},
            "completion_status": "complete",
        },
    )


def rejoin_trace_contexts(
    traces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate trajectory contexts without recomputing their active result."""
    cache: dict[str, dict[str, Any]] = {}
    rejoined: list[dict[str, Any]] = []
    for row in traces:
        context = {
            "state_hash": row["state_hash"],
            "budget_pair": row["budget_pair"],
            "effective_c_puct": row["effective_c_puct"],
            "challenger_player": row.get(
                "challenger_player", row.get("acting_player", 0)
            ),
            "simulations": (
                row.get("challenger_simulations", 384)
                if row.get("acting_player", 0)
                == row.get("challenger_player", row.get("acting_player", 0))
                else row.get("incumbent_simulations", 256)
            ),
            "search_seed": row.get("search_seed"),
        }
        key = canonical_hash(context)
        legacy_moves = row.get("selected_moves", {})
        legacy_visits = row.get("composition_child_visits", {})
        observation = {
            "state": row.get("state"),
            "active_lane": row.get(
                "composition_trajectory_source", "current_policy_current_value"
            ),
            "active_selected_move": row.get(
                "active_selected_move", legacy_moves.get("current_policy_current_value")
            ),
            "active_child_visits": row.get(
                "active_child_visits", legacy_visits.get("current_policy_current_value")
            ),
        }
        prior = cache.setdefault(key, {**context, **observation})
        if prior["active_lane"] == observation["active_lane"] and prior != {
            **context,
            **observation,
        }:
            raise RuntimeError(
                "non-deterministic trace observation for identical context"
            )
        rejoined.append({**row, "trace_context_key": key})
    return rejoined, {"unique_contexts": len(cache), "cache": cache}


def search_trace_context(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Evaluate all compositions once for a trace context in an initialized worker."""
    if _WORKER_EVALUATORS is None:
        raise RuntimeError("trace search worker was not initialized")
    results = {
        lane: search(
            evaluator,
            context["state"],
            int(context["simulations"]),
            int(context["search_seed"]),
            float(context["effective_c_puct"]),
        )
        for lane, evaluator in _WORKER_EVALUATORS.items()
    }
    # The active composition result was already paid for during trajectory play.
    results[context["active_lane"]] = {
        "selected_move": context["active_selected_move"],
        "visits": context["active_child_visits"],
    }
    return canonical_hash(
        {
            key: context[key]
            for key in context
            if key != "state"
            and key
            not in {"active_lane", "active_selected_move", "active_child_visits"}
        }
    ), results


def complete_trace_searches(
    traces: list[dict[str, Any]], current: Path, candidate: Path, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate missing heads once per context and deterministically rejoin occurrences."""
    occurrences, context_data = rejoin_trace_contexts(traces)
    contexts = [context_data["cache"][key] for key in sorted(context_data["cache"])]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_play_worker,
            initargs=(str(current), str(candidate)),
        ) as pool:
            searched = list(pool.map(search_trace_context, contexts))
    else:
        initialize_play_worker(str(current), str(candidate))
        searched = [search_trace_context(context) for context in contexts]
    results = dict(searched)
    joined = [
        {
            **row,
            "selected_moves": {
                lane: result["selected_move"]
                for lane, result in results[row["trace_context_key"]].items()
            },
            "composition_child_visits": {
                lane: result["visits"]
                for lane, result in results[row["trace_context_key"]].items()
            },
        }
        for row in occurrences
    ]
    report = {
        "raw_trace_occurrences": len(traces),
        "unique_search_contexts": len(contexts),
        "search_cache_hit_rate": 1.0 - len(contexts) / max(1, len(traces)),
        "searches_avoided_vs_pr166": max(
            0, len(traces) * 4 - (len(traces) + len(contexts) * 3)
        ),
    }
    return joined, contexts, report


def main() -> int:
    args = parse_args()
    if args.tactical_root_bias != 0.0:
        raise ValueError("diagnostic profile requires tactical_root_bias=0.0")
    workdir, current, candidate = (
        Path(args.workdir),
        Path(args.current),
        Path(args.candidate),
    )
    workdir.mkdir(parents=True, exist_ok=True)
    prior, manifest = verify_inputs(args)
    evaluators, openings = (
        lane_evaluators(current, candidate),
        read_jsonl(Path(args.medium_suite)),
    )
    if not openings:
        raise RuntimeError("medium suite is empty")
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError("--sample must be positive")
        openings = openings[: args.sample]
    if args.forced_sample_per_component <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("forced sample and bootstrap sizes must be positive")
    semantic = {
        "four_compositions_available": set(evaluators) == set(LANES),
        "terminal_uncomposed": all(
            e.evaluate(
                KalahGame(pits=[0] * 12, captured_seeds=[24, 24], current_player=0)
            )[1]
            == 0.0
            for e in evaluators.values()
        ),
        "deterministic_root_policy": options()["root_policy_mode"] == "deterministic",
    }
    if not all(semantic.values()):
        raise RuntimeError(f"composition semantic checks failed: {semantic}")
    schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    selected_stages = parse_stages(args.stages)
    last_stage = max(selected_stages, key=STAGES.index)
    input_fingerprint = canonical_hash(
        {
            "current": sha256_file(current / "weights.json"),
            "candidate": sha256_file(candidate / "weights.json"),
            "replay": sha256_file(Path(args.replay)),
            "openings": sha256_file(Path(args.medium_suite)),
            "seed": args.seed,
            "sample": args.sample,
            "schedule": schedule,
            "default_c_puct": args.default_c_puct,
        }
    )
    write_json(
        workdir / "run_manifest.json",
        {
            "schema": "azlite_joint_heads_attribution_run_v1",
            "current_artifact_hash": sha256_file(current / "weights.json"),
            "candidate_artifact_hash": sha256_file(candidate / "weights.json"),
            "replay_hash": sha256_file(Path(args.replay)),
            "training_manifest_hash": canonical_manifest_hash(manifest),
            "medium_suite_hash": sha256_file(Path(args.medium_suite)),
            "runtime_profile_hash": canonical_hash(
                {
                    "default_c_puct": args.default_c_puct,
                    "schedule": schedule,
                    "tactical_root_bias": args.tactical_root_bias,
                    "options": options(),
                }
            ),
            "configuration": {
                "seed": args.seed,
                "continuation_budgets": args.continuation_budgets,
                "forced_sample_per_component": args.forced_sample_per_component,
            },
        },
    )
    shards = workdir / "stages"
    composition_paths = [
        shards
        / "composition_scores"
        / f"{label.replace(':', '_')}-{lane}-{player}.json"
        for label in COMPOSITION_BUDGETS
        for lane in LANES
        for player in (0, 1)
    ]
    composition_fingerprint = canonical_hash(
        [input_fingerprint, "composition", COMPOSITION_BUDGETS]
    )
    if "composition_scores" in selected_stages and (
        args.force_recompute_stage
        or not args.resume
        or not stage_is_ready(workdir, "composition_scores", composition_fingerprint)
    ):
        for label in COMPOSITION_BUDGETS:
            label_paths = [
                shards
                / "composition_scores"
                / f"{label.replace(':', '_')}-{lane}-{player}.json"
                for lane in LANES
                for player in (0, 1)
            ]
            if (
                args.resume
                and not args.force_recompute_stage
                and all(path.is_file() for path in label_paths)
            ):
                continue
            challenger, incumbent = map(int, label.split(":"))
            result, _ = run_budget(
                current=current,
                candidate=candidate,
                openings=openings,
                label=label,
                seed=args.seed,
                c_puct=resolve_budget_cpuct(
                    schedule=schedule,
                    challenger_simulations=challenger,
                    current_simulations=incumbent,
                    default_c_puct=args.default_c_puct,
                ),
                workers=args.workers,
                trace=False,
            )
            for lane in LANES:
                for player in (0, 1):
                    path = (
                        shards
                        / "composition_scores"
                        / f"{label.replace(':', '_')}-{lane}-{player}.json"
                    )
                    write_json(
                        path,
                        {
                            "label": label,
                            "lane": lane,
                            "challenger_player": player,
                            "metrics": result[lane],
                        },
                    )
        publish_stage(
            workdir, "composition_scores", composition_fingerprint, composition_paths
        )
    if not stage_is_ready(workdir, "composition_scores", composition_fingerprint):
        raise RuntimeError(
            "matching composition shards are required; run --stages composition_scores first"
        )
    if last_stage == "composition_scores":
        print(json.dumps({"completed_stage": "composition_scores"}))
        return 0
    metrics = {lane: {} for lane in LANES}
    for path in composition_paths:
        shard = json.loads(path.read_text(encoding="utf-8"))
        metrics[shard["lane"]][shard["label"]] = shard["metrics"]
    for label in COMPOSITION_BUDGETS:
        baseline = metrics["current_policy_current_value"][label]["raw_ds"]
        for lane in LANES:
            metrics[lane][label]["composition_minus_current_ds"] = (
                metrics[lane][label]["raw_ds"] - baseline
            )
    trace_path, trace_cache_path = (
        shards / "trace_384" / "trace_384_joined.jsonl",
        shards / "trace_384" / "trace_384_unique_states.jsonl",
    )
    trace_fingerprint = canonical_hash([input_fingerprint, "trace", TRACE_BUDGETS])
    if "trace_384" in selected_stages and (
        args.force_recompute_stage
        or not args.resume
        or not stage_is_ready(workdir, "trace_384", trace_fingerprint)
    ):
        label = TRACE_BUDGETS[0]
        challenger, incumbent = map(int, label.split(":"))
        _, traces = run_budget(
            current=current,
            candidate=candidate,
            openings=openings,
            label=label,
            seed=args.seed,
            c_puct=resolve_budget_cpuct(
                schedule=schedule,
                challenger_simulations=challenger,
                current_simulations=incumbent,
                default_c_puct=args.default_c_puct,
            ),
            workers=args.workers,
            trace=True,
        )
        trajectory_path = shards / "trace_384" / "trace_384_trajectory_rows.jsonl"
        write_jsonl(trajectory_path, traces)
        traces, contexts, cache = complete_trace_searches(
            traces, current, candidate, args.workers
        )
        write_jsonl(trace_path, traces)
        write_jsonl(trace_cache_path, contexts)
        searches_path = shards / "trace_384" / "trace_384_composition_searches.jsonl"
        write_jsonl(searches_path, contexts)
        write_json(shards / "trace_384" / "cache_report.json", cache)
        publish_stage(
            workdir,
            "trace_384",
            trace_fingerprint,
            [
                trajectory_path,
                trace_path,
                trace_cache_path,
                searches_path,
                shards / "trace_384" / "cache_report.json",
            ],
        )
    if not stage_is_ready(workdir, "trace_384", trace_fingerprint):
        raise RuntimeError(
            "matching trace shard is required; run --stages trace_384 first"
        )
    traces = read_jsonl(trace_path)
    if last_stage == "trace_384":
        print(json.dumps({"completed_stage": "trace_384", "trace_rows": len(traces)}))
        return 0
    expected = (
        prior.get("medium", {}).get("384:256", {}).get("candidate_minus_current_ds")
    )
    reproduced = (
        expected is None
        or abs(
            metrics["candidate_policy_candidate_value"]["384:256"][
                "composition_minus_current_ds"
            ]
            - float(expected)
        )
        <= 1e-9
    )
    current_hash, candidate_hash = (
        sha256_file(current / "weights.json"),
        sha256_file(candidate / "weights.json"),
    )
    reproduction = {
        "current_weights_sha256": current_hash,
        "candidate_weights_sha256": candidate_hash,
        "replay_sha256": sha256_file(Path(args.replay)),
        "training_manifest_sha256": canonical_manifest_hash(manifest),
        "batch_plan_sha256": manifest["batch_plan_sha256"],
        "medium_384_256_reproduced": reproduced,
    }
    coverage_path, covered_trace_path = (
        shards / "coverage.json",
        shards / "trace-with-coverage.jsonl",
    )
    coverage_fingerprint = canonical_hash(
        [input_fingerprint, "coverage-v2", trace_fingerprint]
    )
    if "coverage" in selected_stages and (
        args.force_recompute_stage
        or not args.resume
        or not stage_is_ready(workdir, "coverage", coverage_fingerprint)
    ):
        coverage, covered_traces = replay_coverage(
            traces,
            Path(args.replay),
            manifest,
            args.workers,
            lambda row: any(
                move != row["selected_moves"]["current_policy_current_value"]
                for move in row["selected_moves"].values()
            ),
        )
        write_json(coverage_path, coverage)
        write_jsonl(covered_trace_path, covered_traces)
        publish_stage(
            workdir,
            "coverage",
            coverage_fingerprint,
            [coverage_path, covered_trace_path],
        )
    if not stage_is_ready(workdir, "coverage", coverage_fingerprint):
        raise RuntimeError(
            "matching coverage artifacts are required; run --stages coverage first"
        )
    coverage, traces = (
        json.loads(coverage_path.read_text(encoding="utf-8")),
        read_jsonl(covered_trace_path),
    )
    if last_stage == "coverage":
        print(json.dumps({"completed_stage": "coverage", "trace_rows": len(traces)}))
        return 0
    forced_dir = shards / "forced"
    forced_path, forced_records_path = (
        forced_dir / "summary.json",
        forced_dir / "forced_records.jsonl",
    )
    forced_fingerprint = canonical_hash(
        [
            input_fingerprint,
            "forced-v2",
            coverage_fingerprint,
            args.forced_sample_per_component,
            args.bootstrap_samples,
        ]
    )
    if "forced" in selected_stages and (
        args.force_recompute_stage
        or not args.resume
        or not stage_is_ready(workdir, "forced", forced_fingerprint)
    ):
        forced = forced_move_audit(
            traces,
            current,
            candidate,
            forced_dir / "items",
            [int(value) for value in args.continuation_budgets.split(",")],
            args.default_c_puct,
            schedule,
            args.seed,
            args.workers,
            args.forced_sample_per_component,
            args.bootstrap_samples,
        )
        write_json(
            forced_path,
            {key: value for key, value in forced.items() if key != "records"},
        )
        write_jsonl(forced_records_path, forced["records"])
        write_json(
            forced_dir / "forced_sample_manifest.json",
            {
                "forced_sample_per_component": args.forced_sample_per_component,
                "available_unique_states": forced["available_unique_states"],
            },
        )
        write_jsonl(
            forced_dir / "forced_tasks.jsonl",
            [
                {"item_id": path.stem, "sha256": sha256_file(path)}
                for path in sorted((forced_dir / "items").glob("*.json"))
            ],
        )
        publish_stage(
            workdir,
            "forced",
            forced_fingerprint,
            [
                forced_path,
                forced_records_path,
                forced_dir / "forced_sample_manifest.json",
                forced_dir / "forced_tasks.jsonl",
            ],
        )
    if not stage_is_ready(workdir, "forced", forced_fingerprint):
        raise RuntimeError(
            "matching forced artifacts are required; run --stages forced first"
        )
    forced = json.loads(forced_path.read_text(encoding="utf-8"))
    forced["records"] = read_jsonl(forced_records_path)
    if last_stage == "forced":
        print(
            json.dumps(
                {"completed_stage": "forced", "forced_records": len(forced["records"])}
            )
        )
        return 0
    changed = [r for r in forced["records"] if r["component"] in {"policy", "value"}]
    covered = [r for r in changed if r["coverage_status"] == "covered"]
    uncovered = [r for r in changed if r["coverage_status"] == "non_overlapping"]
    distance_evidence = distance_harm_evidence(forced["records"], args.seed)
    coverage.update(
        {
            "covered_harm_count": len(covered),
            "uncovered_harm_count": len(uncovered),
            "covered_harmful_changes": any(r["delta"] < 0 for r in covered),
            "covered_forced_ci_upper": aggregate(
                [r["delta"] for r in covered], args.seed
            )["bootstrap_95_ci"][1],
            "overlapping_materially_better": statistics.fmean(
                [r["delta"] for r in covered]
            )
            > statistics.fmean([r["delta"] for r in uncovered]) + 0.03
            if covered and uncovered
            else False,
            **distance_evidence,
        }
    )
    classification = classify(metrics, forced, coverage, traces)
    action = {
        "policy_component_primary_harm": "Close full PUCT visit-distribution CE on generic replay; audit target temperature/concentration and causal move quality on arena-derived states.",
        "value_component_primary_harm": "Close terminal-outcome value learning on this replay; retain policy-target investigation only if policy composition is safe.",
        "destructive_policy_value_interaction": "Test exactly one alternating-head update (policy first, then value with updated policy frozen); do not implement it here.",
        "replay_to_arena_distribution_shift": "Generate deterministic PUCT/outcome replay seeded from diverse medium/fixed openings with held-out opening-family splits.",
        "replay_targets_bad_on_arena_states": "Reject the current replay target construction before generating more volume.",
        "proxy_probe_not_predictive": "Make a cheap medium game-strength screen mandatory immediately after each future training checkpoint.",
    }.get(
        classification["classification"],
        "Resolve incomplete or non-reproduced evidence before proposing a follow-up model.",
    )
    summary = {
        "schema": "azlite_joint_heads_arena_failure_attribution_v2",
        "classification": classification,
        "reproduction": reproduction,
        "composition_semantic_checks": semantic,
        "composition_definitions": LANES,
        "composition_profile_hashes": {
            lane: composition_hash(current_hash, candidate_hash, lane) for lane in LANES
        },
        "medium_head_compositions": metrics,
        "signed_attribution": attribution(metrics),
        "trace_coverage": trace_summary(traces),
        "replay_to_arena_coverage": coverage,
        "forced_move_audit": {k: v for k, v in forced.items() if k != "records"},
        "next_recommended_scientific_action": action,
    }
    summary_path = workdir / "summary_metrics.json"
    classify_fingerprint = canonical_hash(
        [input_fingerprint, "classify", forced_fingerprint]
    )
    if "classify" in selected_stages and (
        args.force_recompute_stage
        or not args.resume
        or not stage_is_ready(workdir, "classify", classify_fingerprint)
    ):
        write_jsonl(workdir / "arena_state_traces.jsonl", traces)
        write_jsonl(workdir / "forced_move_records.jsonl", forced["records"])
        write_json(summary_path, summary)
        write_json(
            REPO_ROOT
            / "docs/data/alphazero-lite-joint-heads-arena-failure-attribution-summary.json",
            summary,
        )
        results_path = (
            REPO_ROOT
            / "docs/alphazero-lite-joint-heads-arena-failure-attribution-results.md"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = results_path.with_suffix(results_path.suffix + ".tmp")
        temporary.write_text(markdown(summary), encoding="utf-8")
        os.replace(temporary, results_path)
        publish_stage(
            workdir,
            "classify",
            classify_fingerprint,
            [
                summary_path,
                workdir / "arena_state_traces.jsonl",
                workdir / "forced_move_records.jsonl",
            ],
        )
    print(
        json.dumps(
            {
                "classification": classification["classification"],
                "trace_rows": len(traces),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
