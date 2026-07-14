#!/usr/bin/env python3
"""Diagnostic attribution of a frozen-trunk joint policy/value head update.

This script never trains or exports a model.  It evaluates four in-memory
compositions against the incumbent using the normal deterministic PUCT path.
The intentionally detailed JSONL trace is kept only in the requested workdir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
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
from ml.alphazero_lite.self_play import build_eval_search_options, encode_state  # noqa: E402

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
TRUNK_KEYS = frozenset()


def canonical_hash(value: Any) -> str:
    """Return a stable hash for a JSON-compatible diagnostic value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def composition_hash(current_hash: str, candidate_hash: str, lane: str) -> str:
    return canonical_hash(
        {
            "current_weights_sha256": current_hash,
            "candidate_weights_sha256": candidate_hash,
            "composition": lane,
        }
    )


def verify_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify immutable PR inputs before any arena work is performed."""
    current_weights = Path(args.current) / "weights.json"
    candidate_weights = Path(args.candidate) / "weights.json"
    if sha256_file(current_weights) != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    summary = json.loads(Path(args.pr164_summary).read_text(encoding="utf-8"))
    expected_candidate = str(summary["run_a"]["artifact_weights_sha256"])
    if sha256_file(candidate_weights) != expected_candidate:
        raise RuntimeError("candidate weights hash mismatch")
    manifest = json.loads(Path(args.training_manifest).read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256_excluding_this_field") != canonical_manifest_hash(
        manifest
    ):
        raise RuntimeError("training manifest hash mismatch")
    if sha256_file(Path(args.replay)) != manifest.get("replay_sha256"):
        raise RuntimeError("replay hash mismatch")
    batch_path = Path(manifest["artifact_paths"]["batch_indexes"])
    if sha256_file(batch_path) != manifest.get("batch_plan_sha256"):
        raise RuntimeError("batch-plan hash mismatch")
    current = json.loads(current_weights.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_weights.read_text(encoding="utf-8"))
    if set(current) != set(candidate):
        raise RuntimeError("candidate/current weight key sets differ")
    trunk_keys = sorted(set(current) - HEAD_KEYS)
    if any(current[key] != candidate[key] for key in trunk_keys):
        raise RuntimeError("candidate trunk differs from current")
    if not any(current[key] != candidate[key] for key in HEAD_KEYS):
        raise RuntimeError("candidate policy/value heads did not change")
    return summary, manifest


def lane_evaluators(
    current: Path, candidate: Path
) -> dict[str, arena.ComposedArtifactEvaluator]:
    incumbent = arena.ArtifactEvaluator(current)
    joint = arena.ArtifactEvaluator(candidate)
    return {
        name: arena.ComposedArtifactEvaluator(
            incumbent, joint, policy_source=policy, value_source=value
        )
        for name, (policy, value) in LANES.items()
    }


def raw_logits(evaluator: arena.ArtifactEvaluator, game: KalahGame) -> np.ndarray:
    """Mirror ArtifactEvaluator's residual-v3 policy forward pass before softmax."""
    x = np.asarray(
        encode_state(game.to_state(), input_encoding=evaluator.input_encoding),
        dtype=np.float32,
    )
    assert evaluator.w_input is not None and evaluator.b_input is not None
    hidden = np.maximum(0.0, (x @ evaluator.w_input) + evaluator.b_input)
    for (w1, b1), (w2, b2) in evaluator.residual_blocks:
        residual = hidden
        hidden = np.maximum(0.0, (hidden @ w1) + b1)
        hidden = np.maximum(0.0, (hidden @ w2) + b2 + residual)
    if evaluator.w_policy_hidden is not None:
        hidden = np.maximum(
            0.0, (hidden @ evaluator.w_policy_hidden) + evaluator.b_policy_hidden
        )
    assert evaluator.w_policy is not None and evaluator.b_policy is not None
    return np.asarray(
        (hidden @ evaluator.w_policy) + evaluator.b_policy, dtype=np.float32
    )


def phase(game: KalahGame) -> str:
    remaining = sum(game.pits)
    return "late" if remaining <= 12 else "midgame" if remaining <= 24 else "opening"


def search(
    evaluator: Any,
    state: dict[str, Any],
    simulations: int,
    seed: int,
    c_puct: float,
    options: dict[str, Any],
) -> dict[str, Any]:
    return arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=simulations,
        seed=seed,
        c_puct=c_puct,
        search_options=options,
    )


def _shares(visits: list[float], legal: list[int]) -> list[float]:
    total = sum(visits[move] for move in legal)
    return [0.0 if total <= 0 else visits[move] / total for move in legal]


def _kl(left: list[float], right: list[float], legal: list[int]) -> float:
    p, q = _shares(left, legal), _shares(right, legal)
    return sum(a * math.log((a + 1e-12) / (b + 1e-12)) for a, b in zip(p, q) if a > 0)


def play_lane(
    *,
    lane: str,
    evaluators: dict[str, Any],
    openings: list[dict[str, Any]],
    challenger_sims: int,
    current_sims: int,
    c_puct: float,
    seed: int,
    trace: bool,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Play both seat splits, optionally logging every decision on trajectories."""
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    games: list[dict[str, Any]] = []
    changed = 0
    roots = 0
    kls: list[float] = []
    logit_deltas: list[float] = []
    value_deltas: list[float] = []
    q_deltas: list[float] = []
    started = time.perf_counter()
    for opening_index, opening in enumerate(openings):
        for challenger_player in (0, 1):
            game = KalahGame.from_state(opening["state"])
            ply = int(opening.get("ply", len(opening.get("prefix_moves", []))))
            trajectory: list[int] = []
            while not game.over():
                state, legal = (
                    game.to_state(),
                    [int(move) for move in game.possible_moves()],
                )
                if not legal:
                    break
                acting_challenger = game.current_player == challenger_player
                selected_evaluator = (
                    evaluators[lane]
                    if acting_challenger
                    else evaluators["current_policy_current_value"]
                )
                selected_sims = challenger_sims if acting_challenger else current_sims
                result = search(
                    selected_evaluator,
                    state,
                    selected_sims,
                    seed + ply + opening_index * 1000 + challenger_player * 100,
                    c_puct,
                    options,
                )
                move = int(result["selected_move"])
                if trace:
                    all_results = {
                        name: search(
                            evaluator,
                            state,
                            selected_sims,
                            seed + ply + opening_index * 1000 + challenger_player * 100,
                            c_puct,
                            options,
                        )
                        for name, evaluator in evaluators.items()
                    }
                    incumbent, candidate = (
                        evaluators["current_policy_current_value"].current_evaluator,
                        evaluators["current_policy_current_value"].candidate_evaluator,
                    )
                    current_logits, candidate_logits = (
                        raw_logits(incumbent, game),
                        raw_logits(candidate, game),
                    )
                    current_policy, current_value = incumbent.evaluate(game)
                    candidate_policy, candidate_value = candidate.evaluate(game)
                    current_result, candidate_result = (
                        all_results["current_policy_current_value"],
                        all_results["candidate_policy_candidate_value"],
                    )
                    changed += int(
                        current_result["selected_move"]
                        != candidate_result["selected_move"]
                    )
                    roots += 1
                    kls.append(
                        _kl(candidate_result["visits"], current_result["visits"], legal)
                    )
                    logit_deltas.append(
                        float(np.mean(np.abs(candidate_logits - current_logits)))
                    )
                    value_deltas.append(abs(candidate_value - current_value))
                    current_q = {
                        row["move"]: row["q_value"]
                        for row in current_result["child_stats"]
                    }
                    candidate_q = {
                        row["move"]: row["q_value"]
                        for row in candidate_result["child_stats"]
                    }
                    q_deltas.append(
                        float(
                            np.mean(
                                [
                                    abs(current_q.get(m, 0.0) - candidate_q.get(m, 0.0))
                                    for m in legal
                                ]
                            )
                        )
                    )
                    traces.append(
                        {
                            "state": state,
                            "state_hash": canonical_hash(state),
                            "encoded_state_hash": canonical_hash(
                                encode_state(
                                    state, input_encoding=incumbent.input_encoding
                                )
                            ),
                            "opening_identifier": opening.get("id", opening_index),
                            "game_identifier": f"{lane}:{opening_index}:{challenger_player}",
                            "ply": ply,
                            "phase": phase(game),
                            "side_to_move": game.current_player,
                            "challenger_player": challenger_player,
                            "legal_moves": legal,
                            "current_policy_logits": current_logits.tolist(),
                            "candidate_policy_logits": candidate_logits.tolist(),
                            "current_policy_probabilities": current_policy.tolist(),
                            "candidate_policy_probabilities": candidate_policy.tolist(),
                            "current_value": current_value,
                            "candidate_value": candidate_value,
                            "selected_moves": {
                                name: item["selected_move"]
                                for name, item in all_results.items()
                            },
                            "composition_child_visits": {
                                name: item["visits"]
                                for name, item in all_results.items()
                            },
                            "composition_child_q_values": {
                                name: item["child_stats"]
                                for name, item in all_results.items()
                            },
                            "selected_visit_share": max(
                                _shares(result["visits"], legal), default=0.0
                            ),
                            "effective_c_puct": c_puct,
                        }
                    )
                trajectory.append(move)
                game.move(game.pit_index(move))
                ply += 1
            score = (
                1.0
                if game.winner == challenger_player
                else 0.5
                if game.winner is None
                else 0.0
            )
            games.append(
                {
                    "challenger_player": challenger_player,
                    "score": score,
                    "trajectory": ",".join(map(str, trajectory)),
                    "margin": game.captured_seeds[challenger_player]
                    - game.captured_seeds[1 - challenger_player],
                }
            )
    p0 = [item["score"] for item in games if item["challenger_player"] == 0]
    p1 = [item["score"] for item in games if item["challenger_player"] == 1]
    counts = Counter(item["trajectory"] for item in games)
    return {
        "raw_ds": statistics.fmean(p0) - statistics.fmean(p1),
        "p0_score": statistics.fmean(p0),
        "p1_score": statistics.fmean(p1),
        "unique_trajectories": len(counts),
        "duplicate_trajectory_count": sum(
            value for value in counts.values() if value > 1
        ),
        "selected_move_changed_rate_vs_current": changed / max(roots, 1),
        "root_visit_kl": statistics.fmean(kls) if kls else 0.0,
        "mean_absolute_policy_logit_delta": statistics.fmean(logit_deltas)
        if logit_deltas
        else 0.0,
        "mean_absolute_value_delta": statistics.fmean(value_deltas)
        if value_deltas
        else 0.0,
        "mean_absolute_child_q_delta": statistics.fmean(q_deltas) if q_deltas else 0.0,
        "runtime_latency_seconds": time.perf_counter() - started,
        "games": games,
    }


def semantic_checks(
    evaluators: dict[str, Any], state: dict[str, Any], seed: int
) -> dict[str, bool]:
    game = KalahGame.from_state(state)
    current = evaluators["current_policy_current_value"].current_evaluator
    candidate = evaluators["current_policy_current_value"].candidate_evaluator
    cp, cv = current.evaluate(game)
    xp, xv = candidate.evaluate(game)
    checks = {
        "current_current_exact": np.array_equal(
            cp, evaluators["current_policy_current_value"].evaluate(game)[0]
        )
        and cv == evaluators["current_policy_current_value"].evaluate(game)[1],
        "candidate_candidate_exact": np.array_equal(
            xp, evaluators["candidate_policy_candidate_value"].evaluate(game)[0]
        )
        and xv == evaluators["candidate_policy_candidate_value"].evaluate(game)[1],
        "mixed_outputs": np.array_equal(
            xp, evaluators["candidate_policy_current_value"].evaluate(game)[0]
        )
        and cv == evaluators["candidate_policy_current_value"].evaluate(game)[1]
        and np.array_equal(
            cp, evaluators["current_policy_candidate_value"].evaluate(game)[0]
        )
        and xv == evaluators["current_policy_candidate_value"].evaluate(game)[1],
        "legal_masks_unchanged": np.array_equal(
            cp == 0, evaluators["current_policy_candidate_value"].evaluate(game)[0] == 0
        ),
        "terminal_uncomposed": all(
            evaluator.evaluate(
                KalahGame(pits=[0] * 12, captured_seeds=[24, 24], current_player=0)
            )[1]
            == 0.0
            for evaluator in evaluators.values()
        ),
    }
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    first = search(
        evaluators["candidate_policy_current_value"], state, 32, seed, 1.25, options
    )
    second = search(
        evaluators["candidate_policy_current_value"], state, 32, seed, 1.25, options
    )
    checks["deterministic_search"] = (
        first["selected_move"] == second["selected_move"]
        and first["visits"] == second["visits"]
    )
    return checks


def classify(metrics: dict[str, Any], forced: dict[str, Any]) -> str:
    base = metrics["current_policy_current_value"]["384:256"]["raw_ds"]
    joint = metrics["candidate_policy_candidate_value"]["384:256"]["raw_ds"] - base
    policy = metrics["candidate_policy_current_value"]["384:256"]["raw_ds"] - base
    value = metrics["current_policy_candidate_value"]["384:256"]["raw_ds"] - base
    if joint >= 0:
        return "no_joint_384_256_regression_reproduced"
    forced_components = forced.get("by_component", {})
    policy_harm = (
        forced_components.get("policy", {}).get("bootstrap_95_ci", [0.0, 0.0])[1] < 0.0
    )
    value_harm = (
        forced_components.get("value", {}).get("bootstrap_95_ci", [0.0, 0.0])[1] < 0.0
    )
    if abs(policy / joint) >= 0.7 and policy_harm:
        return "policy_component_primary_harm"
    if abs(value / joint) >= 0.7 and value_harm:
        return "value_component_primary_harm"
    if (
        abs((joint - policy - value) / joint) >= 0.5
        and abs(policy) < 0.03
        and abs(value) < 0.03
    ):
        return "destructive_policy_value_interaction"
    return "proxy_probe_not_predictive"


def forced_move_audit(
    traces: list[dict[str, Any]],
    evaluators: dict[str, Any],
    budgets: list[int],
    seed: int,
) -> dict[str, Any]:
    """Adjudicate distinct composed moves using only incumbent continuation play."""
    ranked = sorted(
        {row["state_hash"]: row for row in traces}.values(),
        key=lambda row: (
            row["selected_moves"]["candidate_policy_candidate_value"]
            != row["selected_moves"]["current_policy_current_value"],
            abs(row["candidate_value"] - row["current_value"]),
        ),
        reverse=True,
    )[:1024]
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    by_component: dict[str, list[float]] = defaultdict(list)
    records = []
    for index, row in enumerate(ranked):
        moves = row["selected_moves"]
        current_move = moves["current_policy_current_value"]
        for label, move in moves.items():
            if move == current_move:
                continue
            deltas = []
            outcomes = {}
            for budget in budgets:
                baseline = forced_continuation(
                    state=row["state"],
                    forced_move=current_move,
                    evaluator=evaluators[
                        "current_policy_current_value"
                    ].current_evaluator,
                    simulations=budget,
                    c_puct=1.25,
                    search_options=options,
                    seed=seed + index * 100 + budget,
                )
                forced = forced_continuation(
                    state=row["state"],
                    forced_move=move,
                    evaluator=evaluators[
                        "current_policy_current_value"
                    ].current_evaluator,
                    simulations=budget,
                    c_puct=1.25,
                    search_options=options,
                    seed=seed + index * 100 + budget,
                )
                delta = float(forced["outcome_root"] - baseline["outcome_root"])
                deltas.append(delta)
                outcomes[str(budget)] = {
                    "forced_outcome": forced["outcome_root"],
                    "final_margin": forced["store_margin_root"],
                    "move_minus_current_outcome_delta": delta,
                }
            component = (
                "policy"
                if label == "candidate_policy_current_value"
                else "value"
                if label == "current_policy_candidate_value"
                else "joint"
            )
            by_component[component].append(statistics.fmean(deltas))
            records.append(
                {
                    "state_hash": row["state_hash"],
                    "head_introduced_move": component,
                    "move": move,
                    "continuations": outcomes,
                    "agreement_across_budgets": len(set(deltas)) == 1,
                    "policy_prior_change": row["candidate_policy_probabilities"][move]
                    - row["current_policy_probabilities"][move],
                    "value_change": row["candidate_value"] - row["current_value"],
                }
            )

    def aggregate(values: list[float]) -> dict[str, Any]:
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
            "changed_states": len(values),
            "mean_forced_outcome_delta": statistics.fmean(values) if values else 0.0,
            "median_forced_outcome_delta": statistics.median(values) if values else 0.0,
            "bootstrap_95_ci": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "fraction_delta_lte_negative_quarter": sum(
                value <= -0.25 for value in values
            )
            / max(len(values), 1),
            "fraction_delta_gte_positive_quarter": sum(
                value >= 0.25 for value in values
            )
            / max(len(values), 1),
        }

    return {
        "selected_unique_states": len(ranked),
        "by_component": {
            name: aggregate(values) for name, values in by_component.items()
        },
        "records": records,
    }


def replay_coverage(
    traces: list[dict[str, Any]], replay: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    rows = read_jsonl(replay)
    train = set(manifest.get("train_source_row_indexes", []))
    validation = set(manifest.get("validation_source_row_indexes", []))
    train_hashes = {canonical_hash(rows[index]["state"]) for index in train}
    validation_hashes = {canonical_hash(rows[index]["state"]) for index in validation}
    overlap = [
        row
        for row in traces
        if row["encoded_state_hash"] in train_hashes
        or row["encoded_state_hash"] in validation_hashes
    ]
    return {
        "arena_states": len(traces),
        "exact_training_overlap": sum(
            row["encoded_state_hash"] in train_hashes for row in traces
        )
        / max(len(traces), 1),
        "exact_validation_overlap": sum(
            row["encoded_state_hash"] in validation_hashes for row in traces
        )
        / max(len(traces), 1),
        "overlapping_states": len(overlap),
        "non_overlapping_states": len(traces) - len(overlap),
        "similarity_diagnostic": "exact state identity only; non-overlap similarity is intentionally discrete and unmodeled",
    }


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
    prior_summary, manifest = verify_inputs(args)
    evaluators = lane_evaluators(current, candidate)
    openings = read_jsonl(Path(args.medium_suite))
    if not openings:
        raise RuntimeError("medium suite is empty")
    checks = semantic_checks(evaluators, openings[0]["state"], args.seed)
    if not all(checks.values()):
        raise RuntimeError(f"composition semantic checks failed: {checks}")
    schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    budgets = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
    traces: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {lane: {} for lane in LANES}
    for label in budgets:
        challenger_sims, current_sims = map(int, label.split(":"))
        cpuct = resolve_budget_cpuct(
            schedule=schedule,
            challenger_simulations=challenger_sims,
            current_simulations=current_sims,
            default_c_puct=args.default_c_puct,
        )
        for lane in LANES:
            result = play_lane(
                lane=lane,
                evaluators=evaluators,
                openings=openings,
                challenger_sims=challenger_sims,
                current_sims=current_sims,
                c_puct=cpuct,
                seed=args.seed,
                trace=label in {"384:256", "768:768", "1200:1200", "1200:256"}
                and lane
                in {"current_policy_current_value", "candidate_policy_candidate_value"},
                traces=traces,
            )
            result["composition_minus_current_ds"] = (
                result["raw_ds"]
                - metrics["current_policy_current_value"].get(label, result)["raw_ds"]
            )
            result.pop("games")
            metrics[lane][label] = result
    # PR result must agree at the deterministic score level, not merely hashes.
    expected = (
        prior_summary.get("medium", {})
        .get("384:256", {})
        .get("candidate_minus_current_ds")
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
    if not reproduced:
        raise RuntimeError("PR #164 medium 384:256 result did not reproduce")
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
        "semantic_checks": checks,
        "medium_384_256_reproduced": reproduced,
    }
    write_json(workdir / "pr164_reproduction.json", reproduction)
    for row in traces:
        row["composition_profile_hashes"] = {
            lane: composition_hash(current_hash, candidate_hash, lane) for lane in LANES
        }
    with (workdir / "arena_state_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    forced = forced_move_audit(
        traces,
        evaluators,
        [int(value) for value in args.continuation_budgets.split(",")],
        args.seed,
    )
    coverage = replay_coverage(traces, Path(args.replay), manifest)
    base = metrics["current_policy_current_value"]["384:256"]["raw_ds"]
    deltas = {lane: metrics[lane]["384:256"]["raw_ds"] - base for lane in LANES}
    joint = deltas["candidate_policy_candidate_value"]
    attribution = {
        "policy_loss_fraction": abs(deltas["candidate_policy_current_value"])
        / abs(joint)
        if joint
        else 0.0,
        "value_loss_fraction": abs(deltas["current_policy_candidate_value"])
        / abs(joint)
        if joint
        else 0.0,
        "joint_interaction_residual": joint
        - deltas["candidate_policy_current_value"]
        - deltas["current_policy_candidate_value"],
    }
    summary = {
        "schema": "azlite_joint_heads_arena_failure_attribution_v1",
        "classification": classify(metrics, forced),
        "reproduction": reproduction,
        "composition_definitions": LANES,
        "composition_profile_hashes": {
            lane: composition_hash(current_hash, candidate_hash, lane) for lane in LANES
        },
        "medium_head_compositions": metrics,
        "attribution_384_256": attribution,
        "trace_rows": len(traces),
        "forced_move_audit": {
            key: value for key, value in forced.items() if key != "records"
        },
        "replay_to_arena_coverage": coverage,
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-joint-heads-arena-failure-attribution-summary.json",
        summary,
    )
    (
        REPO_ROOT
        / "docs/alphazero-lite-joint-heads-arena-failure-attribution-results.md"
    ).write_text(
        "# Joint-Head Arena Failure Attribution\n\n- classification: `"
        + summary["classification"]
        + "`\n- trace rows: `"
        + str(len(traces))
        + "`\n- 384:256 interaction residual: `"
        + str(attribution["joint_interaction_residual"])
        + "`\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"classification": summary["classification"], "trace_rows": len(traces)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
