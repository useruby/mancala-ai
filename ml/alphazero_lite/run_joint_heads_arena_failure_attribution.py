#!/usr/bin/env python3
"""No-training causal attribution for a frozen joint-head candidate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import sys
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
TRACE_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")


def canonical_hash(value: Any) -> str:
    """Hash a raw JSON representation. Never use this for encoded state vectors."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(canonical_hash(list(parts))[:16], 16) % (2**31)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    evaluators = lane_evaluators(Path(payload["current"]), Path(payload["candidate"]))
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
        seed = stable_seed(
            payload["seed"],
            lane,
            payload["opening_index"],
            challenger_player,
            ply,
            state,
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
        all_results = (
            {
                name: search(evaluator, state, simulations, seed, c_puct)
                for name, evaluator in evaluators.items()
            }
            if payload.get("trace", True)
            else {}
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
                    "challenger_player": challenger_player,
                    "acting_player": game.current_player,
                    "composition_trajectory_source": lane,
                    "opening_identifier": opening.get("id", payload["opening_index"]),
                    "game_identifier": f"{payload['budget_pair']}:{lane}:{payload['opening_index']}:{challenger_player}",
                    "ply": ply,
                    "phase": phase(game),
                    "legal_moves": legal,
                    "selected_moves": {
                        name: result["selected_move"]
                        for name, result in all_results.items()
                    },
                    "composition_child_visits": {
                        name: result["visits"] for name, result in all_results.items()
                    },
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
            "trace": trace,
        }
        for i, opening in enumerate(openings)
        for lane in LANES
        for player in (0, 1)
    ]
    # Every process builds private evaluators; merge uses input order for determinism.
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
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
    traces: list[dict[str, Any]], replay: Path, manifest: dict[str, Any], workers: int
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

    def annotate(row: dict[str, Any]) -> dict[str, Any]:
        raw_hash = canonical_hash(row["state"])
        assert raw_hash == row["state_hash"], (
            "trace state hashes must be raw-state hashes"
        )
        entry = indexes.get(raw_hash)
        vector = raw_state_vector(row["state"])
        distances = np.abs(replay_vectors - vector).sum(axis=1)
        nearest = replay_entries[int(np.argmin(distances))]

        def distance(vectors: np.ndarray) -> int | None:
            return (
                int(np.abs(vectors - vector).sum(axis=1).min())
                if len(vectors)
                else None
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
            "same_current_player": nearest["player"] == row["acting_player"],
            "same_phase": nearest["phase"] == row["phase"],
            "same_legal_move_count": nearest["legal_move_count"]
            == len(row["legal_moves"]),
            "pit_store_l1_distance": int(distances.min()),
            "nearest_replay_state_distance": distance(replay_vectors),
            "nearest_train_state_distance": distance(replay_train),
            "nearest_validation_state_distance": distance(replay_validation),
        }

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            annotated = list(pool.map(annotate, traces))
    else:
        annotated = [annotate(row) for row in traces]
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
        "hash_representation": "canonical_hash(raw_state_dict)",
        "encoded_hashes_compared": False,
    }, annotated


def aggregate(values: list[float], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    boot = (
        [
            float(np.mean(rng.choice(values, len(values), replace=True)))
            for _ in range(1000)
        ]
        if values
        else [0.0]
    )
    return {
        "unique_changed_states": len(values),
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
    continuation_budgets: list[int],
    default_c_puct: float,
    schedule: dict[str, float],
    seed: int,
    workers: int,
) -> dict[str, Any]:
    selected = []
    seen_context_moves: set[tuple[str, str, str]] = set()
    for row in sorted(
        traces,
        key=lambda r: (
            r["state_hash"],
            r["budget_pair"],
            r["composition_trajectory_source"],
            r["game_identifier"],
        ),
    ):
        baseline = row["selected_moves"]["current_policy_current_value"]
        for lane, move in sorted(row["selected_moves"].items()):
            if (
                move != baseline
                and (key := (row["state_hash"], row["budget_pair"], lane))
                not in seen_context_moves
            ):
                seen_context_moves.add(key)
                selected.append((row, lane, baseline, move))
    # Cap contexts per state only for aggregate weighting, not trace retention.
    capped, counts = [], Counter()
    for item in selected:
        if counts[item[0]["state_hash"]] < 4:
            capped.append(item)
            counts[item[0]["state_hash"]] += 1

    def task(item: tuple[dict[str, Any], str, int, int]) -> list[dict[str, Any]]:
        row, lane, baseline, move = item
        evaluator = arena.ArtifactEvaluator(current)
        component = (
            "policy"
            if lane == "candidate_policy_current_value"
            else "value"
            if lane == "current_policy_candidate_value"
            else "joint"
        )
        contexts = [("neutral", budget, 1.25) for budget in continuation_budgets]
        challenger, incumbent = map(int, row["budget_pair"].split(":"))
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
        records = []
        for mode, budget, cpuct in contexts:
            item_seed = stable_seed(
                seed, row["state_hash"], row["budget_pair"], lane, mode, budget
            )
            base = forced_continuation(
                state=row["state"],
                forced_move=baseline,
                evaluator=evaluator,
                simulations=budget,
                c_puct=cpuct,
                search_options=options(),
                seed=item_seed,
            )
            forced = forced_continuation(
                state=row["state"],
                forced_move=move,
                evaluator=evaluator,
                simulations=budget,
                c_puct=cpuct,
                search_options=options(),
                seed=item_seed,
            )
            records.append(
                {
                    "mode": mode,
                    "component": component,
                    "state_hash": row["state_hash"],
                    "budget_pair": row["budget_pair"],
                    "challenger_player": row["challenger_player"],
                    "phase": row["phase"],
                    "coverage_status": "covered"
                    if not row["no_exact_replay_overlap"]
                    else "non_overlapping",
                    "delta": float(forced["outcome_root"] - base["outcome_root"]),
                    "continuation_simulations": budget,
                    "effective_c_puct": cpuct,
                }
            )
        return records

    # Each independent continuation has a private evaluator and private PUCT state.
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            records = [record for result in pool.map(task, capped) for record in result]
    else:
        records = [record for item in capped for record in task(item)]
    result: dict[str, Any] = {
        "selected_context_moves": len(capped),
        "records": records,
        "neutral": {},
        "context_matched": {},
    }
    for mode in ("neutral", "context_matched"):
        subset = [record for record in records if record["mode"] == mode]
        for component in ("policy", "value", "joint"):
            values = [
                record["delta"] for record in subset if record["component"] == component
            ]
            result[mode][component] = aggregate(
                values, stable_seed(seed, mode, component)
            )
        for field in ("budget_pair", "challenger_player", "phase", "coverage_status"):
            result[mode][f"by_{field}"] = {
                key: aggregate(
                    [r["delta"] for r in subset if str(r[field]) == key],
                    stable_seed(seed, mode, field, key),
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


def attribution(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table = {}
    for label in ("384:256", "768:768", "1200:1200", "1200:256"):
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

    for component, delta_key in (
        ("policy", "policy_only_delta"),
        ("value", "value_only_delta"),
    ):
        enough = (
            forced["neutral"].get(component, {}).get("unique_changed_states", 0) >= 64
            or forced["context_matched"]
            .get(component, {})
            .get("unique_changed_states", 0)
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
        and component_harm("joint")
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
    return parser.parse_args()


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
    labels = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
    metrics = {lane: {} for lane in LANES}
    traces = []
    for label in labels:
        challenger, incumbent = map(int, label.split(":"))
        cpuct = resolve_budget_cpuct(
            schedule=schedule,
            challenger_simulations=challenger,
            current_simulations=incumbent,
            default_c_puct=args.default_c_puct,
        )
        result, rows = run_budget(
            current=current,
            candidate=candidate,
            openings=openings,
            label=label,
            seed=args.seed,
            c_puct=cpuct,
            workers=args.workers,
            trace=label in TRACE_BUDGETS,
        )
        for lane in LANES:
            metrics[lane][label] = result[lane]
            metrics[lane][label]["composition_minus_current_ds"] = (
                result[lane]["raw_ds"]
                - result["current_policy_current_value"]["raw_ds"]
            )
        traces.extend(rows)
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
    coverage, traces = replay_coverage(
        traces, Path(args.replay), manifest, args.workers
    )
    forced = forced_move_audit(
        traces,
        current,
        [int(value) for value in args.continuation_budgets.split(",")],
        args.default_c_puct,
        schedule,
        args.seed,
        args.workers,
    )
    changed = [r for r in forced["records"] if r["component"] in {"policy", "value"}]
    covered = [r for r in changed if r["coverage_status"] == "covered"]
    uncovered = [r for r in changed if r["coverage_status"] == "non_overlapping"]
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
            "distance_harm_concentrated": False,
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
    with (workdir / "arena_state_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (workdir / "forced_move_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in forced["records"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-joint-heads-arena-failure-attribution-summary.json",
        summary,
    )
    (
        REPO_ROOT
        / "docs/alphazero-lite-joint-heads-arena-failure-attribution-results.md"
    ).write_text(markdown(summary), encoding="utf-8")
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
