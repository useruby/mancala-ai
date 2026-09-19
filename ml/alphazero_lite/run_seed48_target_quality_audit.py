#!/usr/bin/env python3
"""Evaluation-only exact quality audit for the seed48 AlphaZero-lite teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
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
    args = parser.parse_args()
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
