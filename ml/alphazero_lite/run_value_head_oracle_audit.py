#!/usr/bin/env python3
# ruff: noqa: E402
"""Matched evaluation-only audit of PR #351 policy/value search behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_native_hybrid_feasibility import (  # noqa: E402
    NativeHybridProcess,
)
from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (
    native_label_payload,
    root_training_value,
    validate_native_response,
)  # noqa: E402
from ml.alphazero_lite.self_play import PUCT, build_eval_search_options  # noqa: E402

SCHEMA = "azlite_value_head_oracle_audit_v1"
SOURCES = (401, 407, 413, 419, 443, 449)
ARMS = ("fresh_w1", "fresh_w4")
SIMULATIONS = 384
SEED = 341
BOOTSTRAP_SEED = 352
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="zero",
    reuse_subtree=False,
    normalize_values=False,
    root_policy_mode="deterministic",
    tactical_root_bias=0.0,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def active_stones(game: KalahGame) -> int:
    return sum(game.pits)


def select_top(values: list[float] | np.ndarray, legal: list[int]) -> int:
    return max(legal, key=lambda move: (float(values[move]), -move))


def normalise(values: list[float] | np.ndarray, legal: list[int]) -> list[float]:
    result = [0.0] * 6
    total = sum(float(values[move]) for move in legal)
    if total <= 0:
        for move in legal:
            result[move] = 1.0 / len(legal)
    else:
        for move in legal:
            result[move] = float(values[move]) / total
    return result


def exact_root_outcome(row: dict[str, Any]) -> int:
    score = int(row["exact"]["best_exact_score"])
    return (score > 0) - (score < 0)


def verify_exact_perspective(row: dict[str, Any]) -> None:
    """Recheck PR #340 KVTB1 player-0 margins against root-player scores."""
    state, exact = row["canonical_state"], row["exact"]
    store_margin = int(state["player_store"]) - int(state["opponent_store"])
    sign = 1 if int(state["current_player"]) == 0 else -1
    expected_scores = {
        int(move): (int(margin) + store_margin) * sign
        for move, margin in exact["raw_pit_margin_player0_by_move"].items()
    }
    actual_scores = {
        int(move): int(score) for move, score in exact["exact_score_by_move"].items()
    }
    if expected_scores != actual_scores:
        raise ValueError("value_audit_exact_tablebase_perspective_invalid")
    best = max(expected_scores.values())
    optimal = sorted(move for move, score in expected_scores.items() if score == best)
    if optimal != [int(move) for move in exact["exact_optimal_moves"]]:
        raise ValueError("value_audit_exact_optimal_ties_invalid")
    expected_wdl = {
        move: "W" if score > 0 else "D" if score == 0 else "L"
        for move, score in expected_scores.items()
    }
    if expected_wdl != {
        int(move): value for move, value in exact["exact_wdl_by_move"].items()
    }:
        raise ValueError("value_audit_exact_wdl_perspective_invalid")


class ZeroNonterminalValueEvaluator:
    """Preserve the delegate policy object while neutralising only nonterminals."""

    def __init__(self, delegate: ArtifactEvaluator) -> None:
        self.delegate = delegate

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy, value = self.delegate.evaluate(game)
        return policy, float(value) if game.over() else 0.0


class ExactWdlLeafValueEvaluator:
    """Preserve neural priors while returning exact current-player WDL values."""

    def __init__(self, delegate: ArtifactEvaluator, oracle: "NativeWdlOracle") -> None:
        self.delegate = delegate
        self.oracle = oracle

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy, value = self.delegate.evaluate(game)
        return policy, float(value) if game.over() else self.oracle.value(game)


class NativeWdlOracle:
    """Strict tier-21 exact oracle with no neural fallback path."""

    def __init__(self, probe: Path, tablebase: Path) -> None:
        self.process = NativeHybridProcess(probe, tablebase)
        self.cache: dict[str, float] = {}
        self.covered: set[str] = set()
        self.outside_coverage: list[dict[str, Any]] = []

    def close(self) -> None:
        self.process.close()

    def value(self, game: KalahGame) -> float:
        state = game.to_state()
        key = json.dumps(state, sort_keys=True, separators=(",", ":"))
        if key in self.cache:
            return self.cache[key]
        stones = active_stones(game)
        if stones > 21:
            self.outside_coverage.append({"state": state, "active_stones": stones})
            raise ValueError("value_audit_oracle_leaf_outside_coverage")
        result = self.process.request(native_label_payload(state), 30.0)
        values = {
            int(move): int(score) for move, score in result["action_values"].items()
        }
        validate_native_response(
            values,
            [int(move) for move in result["optimal_actions"]],
            int(result["exact_value"]),
            game.possible_moves(),
            game.current_player,
        )
        value = root_training_value(int(result["exact_value"]), game.current_player)
        self.cache[key] = value
        self.covered.add(key)
        return value


def recovered_models(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    cells = {
        (int(cell["self_play_source"]["seed"]), cell["arm"]): cell
        for arm in ARMS
        for cell in provenance[arm]
    }
    models = []
    for source in SOURCES:
        pair = []
        for arm in ARMS:
            cell = cells.get((source, arm))
            if cell is None:
                break
            checkpoint = Path(cell["artifact"]) / "checkpoint.npz"
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != cell["checkpoint_sha256"]
            ):
                break
            pair.append(
                {
                    "source": source,
                    "arm": arm,
                    "artifact": cell["artifact"],
                    "checkpoint_sha256": cell["checkpoint_sha256"],
                    "weights_sha256": cell["weights_sha256"],
                    "pr351_mcts_384": cell["exact"]["mcts_384"],
                }
            )
        if len(pair) == 2:
            models.extend(pair)
    return models


def rank(values: dict[int, float]) -> dict[int, float]:
    return {
        move: 1.0 + sum(other > value for other in values.values())
        for move, value in values.items()
    }


def spearman(left: dict[int, float], right: dict[int, float]) -> float | None:
    if len(left) < 2 or len(set(left.values())) < 2 or len(set(right.values())) < 2:
        return None
    a, b = rank(left), rank(right)
    av, bv = np.array(list(a.values())), np.array([b[key] for key in a])
    return float(np.corrcoef(av, bv)[0, 1])


def value_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([row["network_value"] for row in rows], dtype=float)
    targets = np.array([row["exact_outcome"] for row in rows], dtype=float)
    wrong = values * targets < 0
    draws = targets == 0
    bins = []
    for lower, upper in zip(np.arange(-1.0, 1.0, 0.2), np.arange(-0.8, 1.2, 0.2)):
        mask = (values >= lower) & (
            (values < upper) if upper < 1 else (values <= upper)
        )
        if np.any(mask):
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "states": int(mask.sum()),
                    "mean_prediction": float(values[mask].mean()),
                    "mean_exact_outcome": float(targets[mask].mean()),
                }
            )
    return {
        "mae": float(np.abs(values - targets).mean()),
        "mse": float(((values - targets) ** 2).mean()),
        "sign_accuracy": float(np.mean(np.sign(values) == targets)),
        "draw_absolute_error": None
        if not np.any(draws)
        else float(np.abs(values[draws]).mean()),
        "mean_absolute_prediction_magnitude": float(np.abs(values).mean()),
        "mean_absolute_value_when_sign_wrong": None
        if not np.any(wrong)
        else float(np.abs(values[wrong]).mean()),
        "predicted_value_bins": bins,
    }


def search_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "exact_optimal_top1_rate": statistics.fmean(row["top_optimal"] for row in rows),
        "exact_optimal_visit_mass": statistics.fmean(
            row["optimal_mass"] for row in rows
        ),
        "selected_move_exact_regret": statistics.fmean(
            row["selected_regret"] for row in rows
        ),
        "visit_weighted_expected_exact_regret": statistics.fmean(
            row["expected_regret"] for row in rows
        ),
        "catastrophic_regret_rate": statistics.fmean(
            row["selected_regret"] >= 4 for row in rows
        ),
    }


def baseline_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        abs(actual[key] - expected[historical]) <= 1e-12
        for key, historical in (
            ("exact_optimal_visit_mass", "optimal_mass"),
            ("visit_weighted_expected_exact_regret", "expected_regret"),
        )
    )


def run_search(evaluator: Any, row: dict[str, Any], seed: int) -> dict[str, Any]:
    game = KalahGame.from_state(row["canonical_state"])
    engine = PUCT(evaluator, SIMULATIONS, 1.25, random.Random(seed), **SEARCH_OPTIONS)
    visits, _ = engine.run(game, dirichlet_alpha=None)
    summary = engine.root_summary()
    legal, exact = row["legal_moves"], row["exact"]
    policy = normalise(visits, legal)
    selected = int(summary["selected_move"])
    optimal = {int(move) for move in exact["exact_optimal_moves"]}
    regrets = {
        int(move): int(value) for move, value in exact["exact_regret_by_move"].items()
    }
    return {
        "selected_move": selected,
        "visit_counts": [int(value) for value in visits],
        "visit_policy": policy,
        "child_q": {
            str(item["move"]): item["q_value"] for item in summary["child_stats"]
        },
        "top_optimal": selected in optimal,
        "optimal_mass": sum(policy[move] for move in optimal),
        "selected_regret": regrets[selected],
        "expected_regret": sum(policy[move] * regrets[move] for move in legal),
        "telemetry": {
            key: summary[key]
            for key in (
                "terminal_leaf_count",
                "nonterminal_leaf_count",
                "backed_up_value_range",
                "root_q_value",
                "selection_breakdown",
            )
        },
    }


def audit_model(
    model: dict[str, Any], corpus: list[dict[str, Any]], oracle: NativeWdlOracle
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(Path(model["artifact"]))
    calibration_rows, ranking_rows, searches = (
        [],
        [],
        {"network_value": [], "zero_nonterminal_value": [], "exact_wdl_leaf_value": []},
    )
    for index, row in enumerate(corpus):
        game, legal = KalahGame.from_state(row["canonical_state"]), row["legal_moves"]
        priors, root_value = evaluator.evaluate(game)
        exact_scores = {
            int(move): float(score)
            for move, score in row["exact"]["exact_score_by_move"].items()
        }
        child_values = {}
        for move in legal:
            child = game.clone()
            child.move(child.pit_index(move))
            _child_policy, value = evaluator.evaluate(child)
            child_values[move] = (
                value if child.current_player == game.current_player else -value
            )
        value_top = select_top(
            [child_values.get(move, -float("inf")) for move in range(6)], legal
        )
        policy_top = select_top(priors, legal)
        pairs = [
            (a, b)
            for a in legal
            for b in legal
            if a < b and exact_scores[a] != exact_scores[b]
        ]
        calibration_rows.append(
            {
                "state_hash": row["state_hash"],
                "bucket": row["bucket"],
                "network_value": float(root_value),
                "exact_outcome": exact_root_outcome(row),
            }
        )
        ranking_rows.append(
            {
                "state_hash": row["state_hash"],
                "value_top_move": value_top,
                "value_top_optimal": value_top in row["exact"]["exact_optimal_moves"],
                "value_top_regret": row["exact"]["exact_regret_by_move"][
                    str(value_top)
                ],
                "pairwise_accuracy": None
                if not pairs
                else statistics.fmean(
                    (child_values[a] - child_values[b])
                    * (exact_scores[a] - exact_scores[b])
                    > 0
                    for a, b in pairs
                ),
                "spearman": spearman(child_values, exact_scores),
                "agrees_with_raw_policy": value_top == policy_top,
                "priors": [float(value) for value in priors],
                "child_values_root_perspective": child_values,
            }
        )
        for condition, wrapped in (
            ("network_value", evaluator),
            ("zero_nonterminal_value", ZeroNonterminalValueEvaluator(evaluator)),
            ("exact_wdl_leaf_value", ExactWdlLeafValueEvaluator(evaluator, oracle)),
        ):
            searches[condition].append(run_search(wrapped, row, SEED + index))
    if oracle.outside_coverage:
        raise ValueError("value_audit_oracle_leaf_outside_coverage")
    baseline = search_metrics(searches["network_value"])
    expected = model["pr351_mcts_384"]
    if not baseline_matches(baseline, expected):
        raise ValueError("value_audit_baseline_reproduction_failed")
    by_bucket = {
        bucket: value_calibration(
            [item for item in calibration_rows if item["bucket"] == bucket]
        )
        for bucket in sorted({item["bucket"] for item in calibration_rows})
    }
    return {
        **model,
        "value_calibration": value_calibration(calibration_rows),
        "value_calibration_by_bucket": by_bucket,
        "one_ply_value_ranking": {
            "top_move_optimal_rate": statistics.fmean(
                item["value_top_optimal"] for item in ranking_rows
            ),
            "mean_regret": statistics.fmean(
                item["value_top_regret"] for item in ranking_rows
            ),
            "pairwise_ranking_accuracy": statistics.fmean(
                item["pairwise_accuracy"]
                for item in ranking_rows
                if item["pairwise_accuracy"] is not None
            ),
            "mean_spearman": statistics.fmean(
                item["spearman"]
                for item in ranking_rows
                if item["spearman"] is not None
            ),
            "agrees_with_raw_policy_rate": statistics.fmean(
                item["agrees_with_raw_policy"] for item in ranking_rows
            ),
        },
        "search": {
            condition: {"metrics": search_metrics(items), "states": items}
            for condition, items in searches.items()
        },
        "states": [
            {
                "state_hash": row["state_hash"],
                "raw_policy": ranking["priors"],
                "child_network_values_root_perspective": ranking[
                    "child_values_root_perspective"
                ],
                "value_ranking": ranking,
                "calibration": calibration,
            }
            for row, ranking, calibration in zip(corpus, ranking_rows, calibration_rows)
        ],
    }


def paired_bootstrap(w1: list[float], w4: list[float]) -> dict[str, Any]:
    deltas = [right - left for left, right in zip(w1, w4, strict=True)]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(10_000)
    )
    return {
        "point_estimate": statistics.fmean(deltas),
        "bootstrap_95": {
            "seed": BOOTSTRAP_SEED,
            "samples": 10_000,
            "lower": samples[249],
            "upper": samples[9749],
        },
        "by_source": {str(source): delta for source, delta in zip(SOURCES, deltas)},
    }


def classification(models: dict[tuple[int, str], dict[str, Any]]) -> str:
    w1, w4 = (
        [models[source, arm] for source in SOURCES for arm in ("fresh_w1", "fresh_w4")],
        [],
    )
    w1, w4 = w1[::2], w1[1::2]

    def mean(items: list[float]) -> float:
        return statistics.fmean(items)

    normal_worse = mean(
        [
            item["search"]["network_value"]["metrics"]["exact_optimal_visit_mass"]
            for item in w4
        ]
    ) < mean(
        [
            item["search"]["network_value"]["metrics"]["exact_optimal_visit_mass"]
            for item in w1
        ]
    )
    calibration_worse = mean([item["value_calibration"]["mae"] for item in w4]) > mean(
        [item["value_calibration"]["mae"] for item in w1]
    )
    repaired = mean(
        [
            item["search"]["zero_nonterminal_value"]["metrics"][
                "exact_optimal_visit_mass"
            ]
            for item in w4
        ]
    ) >= mean(
        [
            item["search"]["zero_nonterminal_value"]["metrics"][
                "exact_optimal_visit_mass"
            ]
            for item in w1
        ]
    ) or mean(
        [
            item["search"]["exact_wdl_leaf_value"]["metrics"][
                "exact_optimal_visit_mass"
            ]
            for item in w4
        ]
    ) >= mean(
        [
            item["search"]["exact_wdl_leaf_value"]["metrics"][
                "exact_optimal_visit_mass"
            ]
            for item in w1
        ]
    )
    oracle_better = all(
        item["search"]["exact_wdl_leaf_value"]["metrics"]["exact_optimal_visit_mass"]
        > item["search"]["network_value"]["metrics"]["exact_optimal_visit_mass"]
        for item in (*w1, *w4)
    )
    zero_better = all(
        item["search"]["zero_nonterminal_value"]["metrics"]["exact_optimal_visit_mass"]
        > item["search"]["network_value"]["metrics"]["exact_optimal_visit_mass"]
        for item in (*w1, *w4)
    )
    if calibration_worse and normal_worse and repaired and oracle_better:
        return "value_head_degradation_explains_mcts_regression"
    if zero_better and oracle_better:
        return "network_value_is_actively_harmful"
    if oracle_better:
        return "network_value_is_imperfect_but_helpful"
    return "policy_search_interaction_primary"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-fresh-replay-weight-ablation/results.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-seed48-target-quality/results.json",
    )
    parser.add_argument(
        "--tablebase", type=Path, default=ROOT / ".tmp/kalah_v1_21.kvtb"
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=ROOT / ".tmp/native-mtdf/girving-kalah/native_probe",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-value-head-oracle-audit",
    )
    args = parser.parse_args(argv)
    provenance, corpus_report = (
        json.loads(args.provenance.read_text()),
        json.loads(args.corpus.read_text()),
    )
    models = recovered_models(provenance)
    if len(models) < 10:
        write_json(
            args.out_dir / "results.json",
            {
                "schema": SCHEMA,
                "classification": "value_audit_model_provenance_unavailable",
            },
        )
        return 0
    corpus = corpus_report["states"]
    if len(corpus) != 200 or corpus_report["scope"]["training_eligible"] is not False:
        raise ValueError("value_audit_exact_corpus_invalid")
    for row in corpus:
        verify_exact_perspective(row)
    search_profile = {
        "simulations": SIMULATIONS,
        "c_puct": 1.25,
        "seed": SEED,
        "options": SEARCH_OPTIONS,
    }
    search_profile["sha256"] = hashlib.sha256(
        json.dumps(search_profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = {
        "schema": SCHEMA,
        "scope": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "canonical_gate": False,
        },
        "models": models,
        "corpus_sha256": corpus_report["corpus_sha256"],
        "tablebase": {
            "path": str(args.tablebase),
            "sha256": sha256_file(args.tablebase),
            "probe_sha256": sha256_file(args.probe),
            "tier": 21,
            "perspective": "KVTB1 final player-0 margin plus stores, converted to current-player WDL",
        },
        "search_profile": search_profile,
    }
    write_json(args.out_dir / "plan.json", plan)
    oracle = NativeWdlOracle(args.probe, args.tablebase)
    try:
        audited = [audit_model(model, corpus, oracle) for model in models]
    finally:
        oracle.close()
    indexed = {(item["source"], item["arm"]): item for item in audited}
    paired = {}
    for name, getter in {
        "root_value_mae": lambda item: item["value_calibration"]["mae"],
        "one_ply_value_ranking_regret": lambda item: item["one_ply_value_ranking"][
            "mean_regret"
        ],
        **{
            f"mcts_{condition}_{metric}": lambda item, condition=condition, metric=metric: (
                item["search"][condition]["metrics"][metric]
            )
            for condition in (
                "network_value",
                "zero_nonterminal_value",
                "exact_wdl_leaf_value",
            )
            for metric in (
                "exact_optimal_visit_mass",
                "visit_weighted_expected_exact_regret",
            )
        },
    }.items():
        paired[name] = paired_bootstrap(
            [getter(indexed[source, "fresh_w1"]) for source in SOURCES],
            [getter(indexed[source, "fresh_w4"]) for source in SOURCES],
        )
    result = {
        "schema": SCHEMA,
        "plan": plan,
        "models": audited,
        "oracle_coverage": {
            "unique_nonterminal_leaves": len(oracle.covered),
            "outside_coverage": oracle.outside_coverage,
        },
        "paired_bootstrap": paired,
        "classification": classification(indexed),
        "canonical_gate_run": False,
        "promotion_performed": False,
        "candidate_selection_performed": False,
    }
    write_json(args.out_dir / "results.json", result)
    # Preserve complete per-state evidence only for the most informative 25 rows.
    by_hash = {row["state_hash"]: row for row in corpus}
    scored = []
    for source in SOURCES:
        a, b = indexed[source, "fresh_w1"], indexed[source, "fresh_w4"]
        for index, state in enumerate(a["states"]):
            a0, b0 = (
                a["search"]["network_value"]["states"][index],
                b["search"]["network_value"]["states"][index],
            )
            fixes = sum(
                a["search"][condition]["states"][index]["selected_regret"]
                < a0["selected_regret"]
                for condition in ("zero_nonterminal_value", "exact_wdl_leaf_value")
            )
            if (
                fixes
                or (state["value_ranking"]["value_top_regret"] > 0)
                or (
                    state["value_ranking"]["value_top_optimal"]
                    and a0["selected_regret"] > 0
                )
                or (b0["selected_regret"] > a0["selected_regret"])
            ):
                scored.append(
                    (
                        -(
                            fixes * 100
                            + state["value_ranking"]["value_top_regret"]
                            + b0["selected_regret"]
                            - a0["selected_regret"]
                        ),
                        source,
                        index,
                    )
                )
    disagreements = []
    for _score, source, index in sorted(scored)[:25]:
        disagreements.append(
            {
                "source": source,
                "exact_state": by_hash[
                    indexed[source, "fresh_w1"]["states"][index]["state_hash"]
                ],
                "fresh_w1": {
                    "state": indexed[source, "fresh_w1"]["states"][index],
                    "search": {
                        condition: indexed[source, "fresh_w1"]["search"][condition][
                            "states"
                        ][index]
                        for condition in (
                            "network_value",
                            "zero_nonterminal_value",
                            "exact_wdl_leaf_value",
                        )
                    },
                },
                "fresh_w4": {
                    "state": indexed[source, "fresh_w4"]["states"][index],
                    "search": {
                        condition: indexed[source, "fresh_w4"]["search"][condition][
                            "states"
                        ][index]
                        for condition in (
                            "network_value",
                            "zero_nonterminal_value",
                            "exact_wdl_leaf_value",
                        )
                    },
                },
            }
        )
    write_json(
        args.out_dir / "disagreements.json", {"schema": SCHEMA, "rows": disagreements}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
