#!/usr/bin/env python3
"""Evaluation-only exact quality audit for the seed48 AlphaZero-lite teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state  # noqa: E402
from ml.alphazero_lite.self_play import PUCT, build_eval_search_options, state_hash  # noqa: E402


SCHEMA = "azlite_seed48_target_quality_audit_v1"
BUCKETS = (("20-21", 20, 21), ("16-19", 16, 19), ("11-15", 11, 15), ("0-10", 0, 10))
METHODS = ("raw", "mcts_96", "mcts_384", "mcts_1200")
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="zero",
    reuse_subtree=False,
    normalize_values=False,
    root_policy_mode="deterministic",
    tactical_root_bias=0.0,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def active_stones(state: dict[str, Any]) -> int:
    return sum(state["player_pits"]) + sum(state["opponent_pits"])


def bucket_for(stones: int) -> str | None:
    return next(
        (name for name, lower, upper in BUCKETS if lower <= stones <= upper), None
    )


def select_top(policy: list[float], legal_moves: list[int]) -> int:
    return max(legal_moves, key=lambda move: (float(policy[move]), -move))


def normalize_legal(
    policy: list[float] | np.ndarray, legal_moves: list[int]
) -> tuple[list[float], bool]:
    result = [0.0] * 6
    total = sum(float(policy[move]) for move in legal_moves)
    if total <= 0:
        for move in legal_moves:
            result[move] = 1.0 / len(legal_moves)
        return result, True
    for move in legal_moves:
        result[move] = float(policy[move]) / total
    return result, abs(total - 1.0) > 1e-8


def exact_action_summary(
    action_raw_pit_margins: dict[int, int], *, root_player: int, stores: list[int]
) -> dict[str, Any]:
    """Convert KVTB1 player-0 pit margins to final root-player scores."""
    store_margin_p0 = int(stores[0]) - int(stores[1])
    scores = {
        move: (raw + store_margin_p0) * (1 if root_player == 0 else -1)
        for move, raw in action_raw_pit_margins.items()
    }
    best = max(scores.values())
    optimal = sorted(move for move, score in scores.items() if score == best)
    ranks = {
        move: 1 + sum(score > value for value in scores.values())
        for move, score in scores.items()
    }
    return {
        "exact_score_by_move": scores,
        "best_exact_score": best,
        "exact_optimal_moves": optimal,
        "exact_regret_by_move": {move: best - score for move, score in scores.items()},
        "exact_rank_by_move": ranks,
        "exact_wdl_by_move": {
            move: "W" if score > 0 else "D" if score == 0 else "L"
            for move, score in scores.items()
        },
    }


def quality(
    policy: list[float], exact: dict[str, Any], legal_moves: list[int]
) -> dict[str, Any]:
    normalized, renormalized = normalize_legal(policy, legal_moves)
    top = select_top(normalized, legal_moves)
    regrets = exact["exact_regret_by_move"]
    optimal = set(exact["exact_optimal_moves"])
    return {
        "policy": normalized,
        "renormalized": renormalized,
        "top_move": top,
        "top_move_optimal": top in optimal,
        "optimal_mass": sum(normalized[move] for move in optimal),
        "top_move_regret": regrets[top],
        "expected_regret": sum(
            normalized[move] * regrets[move] for move in legal_moves
        ),
    }


def paired_bootstrap(
    left: list[float], right: list[float], *, seed: int, samples: int = 10_000
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equally sized nonempty samples")
    deltas = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    means = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    return {
        "point_estimate": statistics.fmean(deltas),
        "ci95_lower": means[int(samples * 0.025)],
        "ci95_upper": means[int(samples * 0.975)],
    }


class TablebaseProbe:
    def __init__(self, binary: Path, tablebase: Path):
        self.process = subprocess.Popen(
            [str(binary), "probe", str(tablebase)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def action_values(self, state: dict[str, Any]) -> dict[int, int]:
        assert self.process.stdin is not None and self.process.stdout is not None
        request = {
            "pits": state["player_pits"] + state["opponent_pits"],
            "player": state["current_player"],
        }
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "actions" not in response:
            raise ValueError(f"tablebase probe failed: {response}")
        return {int(move): int(value) for move, value in response["actions"].items()}

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=5)


def search(
    evaluator: ArtifactEvaluator, state: dict[str, Any], simulations: int, seed: int
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    engine = PUCT(evaluator, simulations, 1.25, random.Random(seed), **SEARCH_OPTIONS)
    visits, _root = engine.run(game)
    summary = engine.root_summary()
    policy, renormalized = normalize_legal(visits, game.possible_moves())
    return {
        "visits": [int(value) for value in visits],
        "policy": policy,
        "renormalized": renormalized,
        "root_summary": summary,
    }


def sample_visit_move(
    policy: list[float], legal_moves: list[int], rng: random.Random
) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for move in legal_moves:
        cumulative += policy[move]
        if threshold <= cumulative:
            return move
    return legal_moves[-1]


def collect_corpus(
    evaluator: ArtifactEvaluator, *, seed: int, games: int, target: int
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    rng = random.Random(seed)
    for game_id in range(games):
        game = KalahGame(pits=[4] * 12, captured_seeds=[0, 0], current_player=0)
        for ply in range(200):
            if game.over():
                break
            state = game.to_state()
            stones = active_stones(state)
            bucket = bucket_for(stones)
            if bucket is not None:
                key = canonical_key(state)
                if (
                    key not in rows
                    and sum(1 for row in rows.values() if row["bucket"] == bucket)
                    < target
                ):
                    rows[key] = {
                        "canonical_state": state,
                        "state_hash": state_hash(state),
                        "side_to_move": game.current_player,
                        "active_stones": stones,
                        "bucket": bucket,
                        "legal_moves": game.possible_moves(),
                        "source_game_id": game_id,
                        "ply": ply,
                        "training_eligible": False,
                    }
            if all(
                sum(1 for row in rows.values() if row["bucket"] == name) >= target
                for name, _, _ in BUCKETS
            ):
                return sorted(
                    rows.values(), key=lambda row: (row["bucket"], row["state_hash"])
                )
            outcome = search(evaluator, state, 384, rng.randrange(2**63))
            # Search is deterministic and noise-free; sampling its normalized visit
            # target with the audit RNG supplies a reproducible corpus of standard starts.
            game.move(
                game.pit_index(
                    sample_visit_move(outcome["policy"], game.possible_moves(), rng)
                )
            )
    return sorted(rows.values(), key=lambda row: (row["bucket"], row["state_hash"]))


def method_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "states": len(items),
            "top1_optimal": statistics.fmean(row["top_move_optimal"] for row in items),
            "optimal_mass": statistics.fmean(row["optimal_mass"] for row in items),
            "top1_regret": statistics.fmean(row["top_move_regret"] for row in items),
            "expected_regret": statistics.fmean(
                row["expected_regret"] for row in items
            ),
            "median_regret": statistics.median(row["top_move_regret"] for row in items),
            "p90_regret": sorted(row["top_move_regret"] for row in items)[
                int(0.9 * (len(items) - 1))
            ],
            "catastrophic_rate": statistics.fmean(
                row["top_move_regret"] >= 4 for row in items
            ),
        }

    results = {
        method: summarize([row["methods"][method] for row in rows])
        for method in METHODS
    }
    for method in METHODS:
        for name, _, _ in BUCKETS:
            results[method].setdefault("by_bucket", {})[name] = summarize(
                [row["methods"][method] for row in rows if row["bucket"] == name]
            )
        for label, predicate in (
            ("capture_available", lambda row: row["capture_available"]),
            ("no_capture", lambda row: not row["capture_available"]),
            ("extra_turn_available", lambda row: row["extra_turn_available"]),
            ("no_extra_turn", lambda row: not row["extra_turn_available"]),
        ):
            items = [row["methods"][method] for row in rows if predicate(row)]
            results[method].setdefault("by_tactical_condition", {})[label] = (
                summarize(items) if items else None
            )
        for legal_count in sorted({len(row["legal_moves"]) for row in rows}):
            items = [
                row["methods"][method]
                for row in rows
                if len(row["legal_moves"]) == legal_count
            ]
            results[method].setdefault("by_legal_action_count", {})[
                str(legal_count)
            ] = summarize(items)
    return results


def disagreement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        raw, m384, m1200 = (
            row["methods"][name] for name in ("raw", "mcts_384", "mcts_1200")
        )
        informative = (
            (raw["top_move_optimal"] and not m1200["top_move_optimal"])
            or (not raw["top_move_optimal"] and m1200["top_move_optimal"])
            or (
                m384["top_move"] != m1200["top_move"]
                and m384["top_move_regret"] != m1200["top_move_regret"]
            )
        )
        if informative:
            selected.append(row)
    return sorted(
        selected,
        key=lambda row: (
            -abs(
                row["methods"]["mcts_1200"]["top_move_regret"]
                - row["methods"]["raw"]["top_move_regret"]
            ),
            row["state_hash"],
        ),
    )[:20]


def classify(comparisons: dict[str, Any], metrics: dict[str, Any]) -> str:
    mass = comparisons["mcts_1200_minus_raw"]["optimal_mass"]
    regret = comparisons["mcts_1200_minus_raw"]["expected_regret"]
    improves = mass["ci95_lower"] > 0 and regret["ci95_upper"] < 0
    worse = mass["ci95_upper"] < 0 or regret["ci95_lower"] > 0
    saturated = (
        comparisons["mcts_1200_minus_mcts_384"]["optimal_mass"]["ci95_lower"]
        <= 0
        <= comparisons["mcts_1200_minus_mcts_384"]["optimal_mass"]["ci95_upper"]
        and comparisons["mcts_1200_minus_mcts_384"]["expected_regret"]["ci95_lower"]
        <= 0
        <= comparisons["mcts_1200_minus_mcts_384"]["expected_regret"]["ci95_upper"]
    )
    bucket_regression = any(
        metrics["mcts_1200"]["by_bucket"][name]["optimal_mass"]
        < metrics["raw"]["by_bucket"][name]["optimal_mass"]
        or metrics["mcts_1200"]["by_bucket"][name]["expected_regret"]
        > metrics["raw"]["by_bucket"][name]["expected_regret"]
        for name, _, _ in BUCKETS
    )
    if worse:
        return "search_targets_exactly_worse"
    if improves and saturated and not bucket_regression:
        return "search_targets_budget_saturated"
    if improves and not bucket_regression:
        return "search_targets_exactly_improve_policy"
    return "search_targets_not_improving"


def _wdl(score: int) -> str:
    return "W" if score > 0 else "D" if score == 0 else "L"


def _selection_change_class(
    exact: dict[str, Any], disabled: int, threshold: int
) -> str:
    scores = {
        int(move): int(score) for move, score in exact["exact_score_by_move"].items()
    }
    left, right = scores[disabled], scores[threshold]
    if left == right:
        return "exact_tie"
    if _wdl(left) != _wdl(right):
        return "outcome_improvement" if right > left else "outcome_regression"
    return (
        "same_outcome_margin_improvement"
        if right > left
        else "same_outcome_margin_regression"
    )


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = statistics.fmean(pair[0] for pair in pairs)
    right = statistics.fmean(pair[1] for pair in pairs)
    covariance = sum((a - left) * (b - right) for a, b in pairs)
    left_variance = sum((a - left) ** 2 for a, _ in pairs)
    right_variance = sum((b - right) ** 2 for _, b in pairs)
    if left_variance <= 0 or right_variance <= 0:
        return None
    return covariance / math.sqrt(left_variance * right_variance)


def _replay_exact_leaf_audit(args: argparse.Namespace) -> int:
    frozen = json.loads(args.replay_corpus.read_text(encoding="utf-8"))
    rows = frozen["states"]
    evaluator = ArtifactEvaluator(args.artifact)
    options = build_eval_search_options(
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
    )
    tablebase = EndgameTablebase()

    def compact(result: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
        return {
            key: result.get(key)
            for key in (
                "selected_move",
                "legal_moves",
                "policy",
                "visit_policy",
                "value",
                "search_root_value",
                "child_stats",
                "visits",
                "terminal_leaf_count",
                "nonterminal_leaf_count",
                "backed_up_value_range",
                "exact_leaf_value_telemetry",
            )
        } | {"root_child_timing": timing}

    audit_rows = []
    for index, row in enumerate(rows):
        state = row["canonical_state"]
        exact = {
            **row["exact"],
            "exact_score_by_move": {
                int(move): int(score)
                for move, score in row["exact"]["exact_score_by_move"].items()
            },
            "exact_regret_by_move": {
                int(move): int(score)
                for move, score in row["exact"]["exact_regret_by_move"].items()
            },
            "exact_wdl_by_move": {
                int(move): value
                for move, value in row["exact"]["exact_wdl_by_move"].items()
            },
            "exact_optimal_moves": [
                int(move) for move in row["exact"]["exact_optimal_moves"]
            ],
        }
        outputs = {}
        for name, threshold in (("disabled", None), ("threshold10", 10)):
            timing: dict[str, Any] = {}
            result = evaluate_artifact_position(
                evaluator=evaluator,
                state=state,
                simulations=384,
                seed=args.seed + index,
                c_puct=1.25,
                search_options=options,
                endgame_tablebase=tablebase,
                exact_solve_stone_threshold=threshold,
                exact_solve_fail_closed=True,
                root_child_telemetry=timing,
                root_snapshot_checkpoints={32, 64, 128, 256, 384},
            )
            visit_policy, _ = normalize_legal(result["visits"], row["legal_moves"])
            result["visit_policy"] = visit_policy
            outputs[name] = compact(result, timing)
        disabled, threshold10 = outputs["disabled"], outputs["threshold10"]
        change = None
        if disabled["selected_move"] != threshold10["selected_move"]:
            change = _selection_change_class(
                exact, disabled["selected_move"], threshold10["selected_move"]
            )
        audit_rows.append(
            {
                "state_hash": row["state_hash"],
                "bucket": row["bucket"],
                "legal_moves": row["legal_moves"],
                "exact": exact,
                "disabled": disabled,
                "threshold10": threshold10,
                "selection_change_class": change,
            }
        )

    def summary(name: str) -> dict[str, float]:
        qualities = [
            quality(row[name]["visit_policy"], row["exact"], row["legal_moves"])
            for row in audit_rows
        ]
        wdl_optimal = [
            select_top(row[name]["visit_policy"], row["legal_moves"])
            in {
                int(move)
                for move, score in row["exact"]["exact_score_by_move"].items()
                if _wdl(int(score))
                == _wdl(max(map(int, row["exact"]["exact_score_by_move"].values())))
            }
            for row in audit_rows
        ]
        return {
            "optimal_mass": statistics.fmean(x["optimal_mass"] for x in qualities),
            "expected_regret": statistics.fmean(
                x["expected_regret"] for x in qualities
            ),
            "top_move_optimal": statistics.fmean(
                x["top_move_optimal"] for x in qualities
            ),
            "top_move_regret": statistics.fmean(
                x["top_move_regret"] for x in qualities
            ),
            "wdl_optimal_top1": statistics.fmean(wdl_optimal),
        }

    aggregate = {name: summary(name) for name in ("disabled", "threshold10")}
    aggregate["delta"] = {
        key: aggregate["threshold10"][key] - aggregate["disabled"][key]
        for key in aggregate["disabled"]
    }
    counts = {
        name: sum(row["selection_change_class"] == name for row in audit_rows)
        for name in (
            "outcome_improvement",
            "outcome_regression",
            "same_outcome_margin_improvement",
            "same_outcome_margin_regression",
            "exact_tie",
        )
    }
    boundary_totals: dict[str, dict[str, float | int]] = {}
    for row in audit_rows:
        telemetry = row["threshold10"].get("exact_leaf_value_telemetry") or {}
        for stones, bucket in (telemetry.get("boundary_value_buckets") or {}).items():
            total = boundary_totals.setdefault(stones, {key: 0 for key in bucket})
            for key, value in bucket.items():
                total[key] = float(total[key]) + float(value)
    boundary_summary = {}
    for stones, bucket in boundary_totals.items():
        count, exact_count = int(bucket["count"]), int(bucket["exact_count"])
        network_mean = float(bucket["network_value_sum"]) / count
        boundary_summary[stones] = {
            "count": count,
            "network_value_mean": network_mean,
            "network_value_sd": math.sqrt(
                max(
                    0.0,
                    float(bucket["network_value_square_sum"]) / count - network_mean**2,
                )
            ),
            "exact_count": exact_count,
            "exact_value_mean": None
            if not exact_count
            else float(bucket["exact_value_sum"]) / exact_count,
            "exact_minus_network_mean": None
            if not exact_count
            else float(bucket["delta_sum"]) / exact_count,
            "absolute_delta_mean": None
            if not exact_count
            else float(bucket["absolute_delta_sum"]) / exact_count,
            "sign_disagreement_rate": None
            if not exact_count
            else int(bucket["sign_disagreement_count"]) / exact_count,
        }
    root_q_margin_correlation = {}
    for name in ("disabled", "threshold10"):
        pairs = []
        for row in audit_rows:
            margins = row["exact"]["exact_score_by_move"]
            pairs.extend(
                (float(child["q_value"]), float(margins[int(child["move"])]))
                for child in row[name]["child_stats"]
                if int(child["move"]) in margins
            )
        root_q_margin_correlation[name] = {
            "pairs": len(pairs),
            "pearson": _pearson(pairs),
        }
    ranked = []
    for row in audit_rows:
        disabled_quality = quality(
            row["disabled"]["visit_policy"], row["exact"], row["legal_moves"]
        )
        threshold_quality = quality(
            row["threshold10"]["visit_policy"], row["exact"], row["legal_moves"]
        )
        ranked.append(
            {
                "state_hash": row["state_hash"],
                "visit_weighted_expected_regret_delta": (
                    threshold_quality["expected_regret"]
                    - disabled_quality["expected_regret"]
                ),
                "selection_change_class": row["selection_change_class"],
            }
        )
    report = {
        "schema": SCHEMA,
        "diagnostic": "puct_exact_leaf_propagation",
        "training_eligible": False,
        "aggregate": aggregate,
        "changed_selection_counts": counts,
        "states": audit_rows,
        "boundary_value_summary": boundary_summary,
        "root_q_vs_exact_action_margin": root_q_margin_correlation,
        "largest_harmful_states": sorted(
            ranked,
            key=lambda row: (
                -row["visit_weighted_expected_regret_delta"],
                row["state_hash"],
            ),
        )[:20],
        "largest_improving_states": sorted(
            ranked,
            key=lambda row: (
                row["visit_weighted_expected_regret_delta"],
                row["state_hash"],
            ),
        )[:20],
    }
    args.replay_out.parent.mkdir(parents=True, exist_ok=True)
    args.replay_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "states": len(audit_rows),
                "aggregate": aggregate,
                "out": str(args.replay_out),
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--tablebase", type=Path, default=ROOT / ".tmp/kalah_v1_21.kvtb"
    )
    parser.add_argument(
        "--tablebase-binary",
        type=Path,
        default=ROOT / ".tmp/kalah_v1_tablebase_build/kalah_v1_tablebase",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-seed48-target-quality",
    )
    parser.add_argument("--seed", type=int, default=340)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--target-per-bucket", type=int, default=50)
    parser.add_argument("--replay-corpus", type=Path)
    parser.add_argument("--replay-out", type=Path)
    args = parser.parse_args()
    if args.replay_corpus is not None:
        if args.artifact is None or args.replay_out is None:
            raise ValueError("--replay-corpus requires --artifact and --replay-out")
        return _replay_exact_leaf_audit(args)
    if args.games > 128:
        raise ValueError("corpus collection is capped at 128 games")
    metadata = json.loads((args.artifact / "metadata.json").read_text(encoding="utf-8"))
    weights_sha = sha256_file(args.artifact / "weights.json")
    evaluator = ArtifactEvaluator(args.artifact)
    corpus = collect_corpus(
        evaluator, seed=args.seed, games=args.games, target=args.target_per_bucket
    )
    for row in corpus:
        row["incumbent_weights_sha256"] = weights_sha
        row["corpus_generation"] = {
            "seed": args.seed,
            "standard_start": True,
            "trajectory_simulations": 384,
            "c_puct": 1.25,
            "search_options": SEARCH_OPTIONS,
        }
    counts = {
        name: sum(row["bucket"] == name for row in corpus) for name, _, _ in BUCKETS
    }
    if len(corpus) < 160 or min(counts.values()) < 35:
        raise RuntimeError(f"insufficient exact corpus: {counts}")
    probe = TablebaseProbe(args.tablebase_binary, args.tablebase)
    try:
        for index, row in enumerate(corpus):
            state = row["canonical_state"]
            raw_actions = probe.action_values(state)
            if set(raw_actions) != set(row["legal_moves"]):
                raise ValueError("tablebase did not return every legal action")
            exact = exact_action_summary(
                raw_actions,
                root_player=row["side_to_move"],
                stores=[state["player_store"], state["opponent_store"]],
            )
            row["exact"] = {"raw_pit_margin_player0_by_move": raw_actions, **exact}
            prior, _value = evaluator.evaluate(KalahGame.from_state(state))
            row["methods"] = {"raw": quality(prior.tolist(), exact, row["legal_moves"])}
            for budget in (96, 384, 1200):
                outcome = search(evaluator, state, budget, args.seed + index)
                row["methods"][f"mcts_{budget}"] = {
                    **quality(outcome["policy"], exact, row["legal_moves"]),
                    "visit_counts": outcome["visits"],
                    "root_q_by_move": {
                        str(item["move"]): item["q_value"]
                        for item in outcome["root_summary"]["child_stats"]
                    },
                }
            row["capture_available"] = any(
                move_consequence_for_state(state, move)["produces_capture"]
                for move in row["legal_moves"]
            )
            row["extra_turn_available"] = any(
                move_consequence_for_state(state, move)["gives_extra_turn"]
                for move in row["legal_moves"]
            )
    finally:
        probe.close()
    metrics = method_metrics(corpus)
    comparisons = {}
    for label, left, right in (
        ("mcts_96_minus_raw", "mcts_96", "raw"),
        ("mcts_384_minus_raw", "mcts_384", "raw"),
        ("mcts_1200_minus_raw", "mcts_1200", "raw"),
        ("mcts_1200_minus_mcts_384", "mcts_1200", "mcts_384"),
    ):
        comparisons[label] = {
            metric: paired_bootstrap(
                [row["methods"][left][metric] for row in corpus],
                [row["methods"][right][metric] for row in corpus],
                seed=args.seed,
            )
            for metric in ("optimal_mass", "top_move_regret", "expected_regret")
        }
    report = {
        "schema": SCHEMA,
        "classification": classify(comparisons, metrics),
        "scope": {
            "training_eligible": False,
            "training_or_artifact_mutation": False,
            "dirichlet_noise": False,
        },
        "incumbent": {
            "version": metadata["version"],
            "weights_json_sha256": weights_sha,
        },
        "corpus_generation": {
            "seed": args.seed,
            "games_cap": args.games,
            "trajectory_search": {
                "simulations": 384,
                "c_puct": 1.25,
                "search_options": SEARCH_OPTIONS,
            },
            "trajectory_action_selection": "seeded sampling from noise-free normalized 384-visit policies",
            "bucket_counts": counts,
        },
        "tablebase": {
            "path": str(args.tablebase),
            "sha256": sha256_file(args.tablebase),
            "binary_sha256": sha256_file(args.tablebase_binary),
            "tier": 21,
            "perspective": "KVTB1 values are final player-0 pit margins; add root stores then negate for player 1.",
        },
        "historical_seed48_provenance": {
            "plan": "ml/alphazero_lite/configs/fresh_uniform1200_confirmation.json",
            "base_recipe": "ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root.json",
            "runner": "ml/alphazero_lite/run_fresh_uniform1200_confirmation.py",
        },
        "methods": metrics,
        "multi_optimal_state_frequency": statistics.fmean(
            len(row["exact"]["exact_optimal_moves"]) > 1 for row in corpus
        ),
        "paired_comparisons": comparisons,
        "states": corpus,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out_dir / "corpus.json"
    result_path = args.out_dir / "results.json"
    corpus_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "training_eligible": False,
                "states": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"methods", "exact"}
                    }
                    for row in corpus
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report["corpus_sha256"] = sha256_file(corpus_path)
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    disagreements = disagreement_rows(corpus)
    (args.out_dir / "disagreements.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "training_eligible": False, "rows": disagreements},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "corpus": str(corpus_path),
                "results": str(result_path),
                "states": len(corpus),
                "bucket_counts": counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
