#!/usr/bin/env python3
"""Validate frozen-margin PUCT sensitivity for the immutable PR #214 artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import (  # noqa: E402
    ArtifactEvaluator,
    apply_opening_moves,
    canonical_game_state_hash,
)
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import (  # noqa: E402
    decode_kalah_v3_base_state,
    state_round_trips_kalah_v3,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.margin_sensitivity import decision_sensitivity, legal_policy  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (  # noqa: E402
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    _suite,
    state_hash as model_state_hash,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import new_model  # noqa: E402
from ml.alphazero_lite.self_play import PUCT  # noqa: E402
from ml.alphazero_lite.train import load_checkpoint_into_model  # noqa: E402

SAMPLE_SIZE = 4096
SAMPLE_SEED = 220
SIMULATIONS = 1200
C_PUCT = 1.25
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 220
HELD_OUT_HASHES = {
    "6d7a71e6007e3943a223024aa1515659887ecb275806402f645a5b6367f942fe",
    "cd6293ed266fb1db26224cb9208d2494bd9f35be6d37526c55b2a64437e98d8c",
}
_WORKER_P1: ArtifactEvaluator | None = None
_WORKER_A16: ArtifactEvaluator | None = None


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _entropy(policy: list[float]) -> float:
    values = np.asarray(policy, dtype=float)
    values = values[values > 0]
    return float(-np.sum(values * np.log(values)))


def _phase(move_index: int) -> str:
    if move_index < 5:
        return "00-04"
    if move_index < 10:
        return "05-09"
    if move_index < 20:
        return "10-19"
    if move_index < 40:
        return "20-39"
    return "40+"


def _canonical_arena_hashes() -> set[str]:
    hashes = set()
    for row in _suite()[0]:
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        apply_opening_moves(game, [int(move) for move in row["prefix_moves"]])
        hashes.add(canonical_game_state_hash(game))
    return hashes


def select_sample(rows: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    """Deterministically round-robin replay strata after required exclusions."""
    arena_hashes = _canonical_arena_hashes()
    states: dict[int, tuple[dict[str, Any], str]] = {}
    unique_hashes: set[str] = set()
    for index, row in enumerate(rows):
        state = decode_kalah_v3_base_state(list(row["state"]))
        state_hash = canonical_game_state_hash(KalahGame.from_state(state))
        if state_hash in arena_hashes | HELD_OUT_HASHES or state_hash in unique_hashes:
            continue
        unique_hashes.add(state_hash)
        states[index] = (row, state_hash)
    entropies = {index: _entropy(row["policy"]) for index, (row, _) in states.items()}
    cuts = np.quantile(list(entropies.values()), [0.25, 0.5, 0.75])
    groups: dict[tuple[int, str, int, int], list[tuple[str, int]]] = defaultdict(list)
    metadata: dict[int, dict[str, Any]] = {}
    for index, (row, state_hash) in states.items():
        legal = [int(move) for move in row["legal_moves"]]
        entropy_quartile = (
            int(np.searchsorted(cuts, entropies[index], side="right")) + 1
        )
        key = (
            int(row["player"]),
            _phase(int(row["move_index"])),
            len(legal),
            entropy_quartile,
        )
        groups[key].append(
            (
                _canonical_hash(
                    {"seed": SAMPLE_SEED, "index": index, "state": state_hash}
                ),
                index,
            )
        )
        metadata[index] = {
            "replay_index": index,
            "state_hash": state_hash,
            "player_to_move": int(row["player"]),
            "move_index": int(row["move_index"]),
            "move_index_bucket": _phase(int(row["move_index"])),
            "legal_move_count": len(legal),
            "stored_policy_entropy": entropies[index],
            "stored_entropy_quartile": entropy_quartile,
        }
    for values in groups.values():
        values.sort()
    offsets = {key: 0 for key in groups}
    selected: list[int] = []
    while len(selected) < SAMPLE_SIZE:
        added = False
        for key in sorted(groups):
            if offsets[key] < len(groups[key]) and len(selected) < SAMPLE_SIZE:
                selected.append(groups[key][offsets[key]][1])
                offsets[key] += 1
                added = True
        if not added:
            break
    if len(selected) != SAMPLE_SIZE or len(set(selected)) != SAMPLE_SIZE:
        raise RuntimeError("unable to select 4096 eligible unique replay states")
    return selected, [metadata[index] for index in selected]


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"pr220-margin-sensitivity:{state_hash}".encode()).hexdigest()[
            :16
        ],
        16,
    )


def _search(
    evaluator: ArtifactEvaluator, state: dict[str, Any], seed: int, simulations: int
) -> tuple[list[dict], dict]:
    trace: list[dict] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=C_PUCT,
        rng=random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        selection_trace=trace,
        trace_checkpoints={384, 1200},
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return trace, search.root_summary()


def _init_worker(p1_artifact: str, a16_artifact: str) -> None:
    """Load immutable evaluators once per independent replay-state worker."""
    global _WORKER_P1, _WORKER_A16
    _WORKER_P1 = ArtifactEvaluator(Path(p1_artifact))
    _WORKER_A16 = ArtifactEvaluator(Path(a16_artifact))


def _worker_evaluators() -> tuple[ArtifactEvaluator, ArtifactEvaluator]:
    if _WORKER_P1 is None or _WORKER_A16 is None:
        raise RuntimeError("margin sensitivity worker was not initialized")
    return _WORKER_P1, _WORKER_A16


def _snapshot(summary: dict, simulation: int) -> dict:
    return next(
        item
        for item in summary["trace_root_snapshots"]
        if item["simulation"] == simulation
    )


def _trace_prefix(trace: list[dict], simulations: int) -> list[dict]:
    """Remove end-of-run annotations before checking continuous-prefix identity."""
    return [
        {
            key: value
            for key, value in record.items()
            if key not in {"final_root_visits", "final_selected_root_move"}
        }
        for record in trace[:simulations]
    ]


def _visit_js(left: list[float], right: list[float]) -> float:
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2.0
    return float(
        0.5 * np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12)))
        + 0.5 * np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(midpoint, 1e-12)))
    )


def _visit_margin(snapshot: dict) -> float:
    ordered = sorted(snapshot["visits"], reverse=True)
    return float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])


def _root_policy_metrics(
    p1: ArtifactEvaluator, a16: ArtifactEvaluator, game: KalahGame
) -> dict[str, Any]:
    legal = game.possible_moves()
    p1_policy = legal_policy(p1.evaluate(game)[0], legal)
    a16_policy = legal_policy(a16.evaluate(game)[0], legal)
    delta = np.abs(a16_policy - p1_policy)
    return {
        "root_policy_l1": float(delta.sum()),
        "root_policy_js": _visit_js(
            p1_policy[legal].tolist(), a16_policy[legal].tolist()
        ),
        "root_max_abs_action_delta": float(delta.max()),
        "root_top1_disagreement": bool(
            max(legal, key=lambda move: (p1_policy[move], -move))
            != max(legal, key=lambda move: (a16_policy[move], -move))
        ),
    }


def sensitivity_through(
    trace: list[dict], root_state: dict[str, Any], a16: ArtifactEvaluator, limit: int
) -> dict[str, Any]:
    """Evaluate A16 priors on every P1-selected trace node, one root at a time."""
    cached_policies: dict[str, np.ndarray] = {}
    decisions: list[dict[str, Any]] = []
    for simulation in trace[:limit]:
        game = KalahGame.from_state(root_state)
        for decision in simulation["selection_path"]:
            node_hash = canonical_game_state_hash(game)
            if node_hash != decision["state_hash"]:
                raise RuntimeError("trace path reconstruction state hash mismatch")
            if node_hash not in cached_policies:
                cached_policies[node_hash] = a16.evaluate(game)[0]
            score = decision_sensitivity(
                decision, cached_policies[node_hash], c_puct=C_PUCT
            )
            decisions.append(
                {
                    **score,
                    "simulation": int(simulation["simulation_index"]),
                    "depth": int(decision["tree_depth"]),
                    "state_hash": node_hash,
                }
            )
            if not game.move(game.pit_index(int(decision["chosen_move"]))):
                raise RuntimeError("trace path contains an illegal selected move")
    flips = [item for item in decisions if item["counterfactual_flip"]]
    first = flips[0] if flips else None
    return {
        "max_pressure_ratio": float(
            max(item["max_pressure_ratio"] for item in decisions)
        ),
        "max_flip_excess": float(max(item["max_flip_excess"] for item in decisions)),
        "first_predicted_counterfactual_flip_simulation": None
        if first is None
        else first["simulation"],
        "first_predicted_counterfactual_flip_depth": None
        if first is None
        else first["depth"],
        "first_predicted_counterfactual_flip_state_hash": None
        if first is None
        else first["state_hash"],
        "first_predicted_counterfactual_flip_action_pair": None
        if first is None
        else [first["counterfactual_move"], first["selected_move"]],
        "predicted_flip_count": len(flips),
        "predicted_flip_fraction": float(len(flips) / len(decisions)),
        "minimum_parent_puct_winner_runner_up_margin": float(
            min(item["winner_runner_up_margin"] for item in decisions)
        ),
    }


def _same_statistics(left: dict, right: dict) -> bool:
    left_children = {entry["move"]: entry for entry in left["children"]}
    right_children = {entry["move"]: entry for entry in right["children"]}
    return left_children.keys() == right_children.keys() and all(
        left_children[move]["visit_count"] == right_children[move]["visit_count"]
        and abs(left_children[move]["q_value"] - right_children[move]["q_value"])
        <= 1e-12
        and abs(
            left_children[move]["q_component"] - right_children[move]["q_component"]
        )
        <= 1e-12
        for move in left_children
    )


def first_divergence(
    a16_trace: list[dict], p1_trace: list[dict]
) -> dict[str, Any] | None:
    """PR #220 comparison plus the full pre-divergence statistics invariant."""
    for actual, parent in zip(a16_trace, p1_trace, strict=True):
        for candidate_decision, parent_decision in zip(
            actual["selection_path"], parent["selection_path"], strict=True
        ):
            if candidate_decision["state_hash"] != parent_decision["state_hash"]:
                return {
                    "invariant_failure": "path state mismatch before action divergence"
                }
            if not _same_statistics(candidate_decision, parent_decision):
                return {
                    "invariant_failure": "visits or Q values differ before action divergence"
                }
            if candidate_decision["chosen_move"] != parent_decision["chosen_move"]:
                return {
                    "simulation": int(actual["simulation_index"]),
                    "depth": int(candidate_decision["tree_depth"]),
                    "state_hash": candidate_decision["state_hash"],
                    "action_pair": [
                        int(candidate_decision["chosen_move"]),
                        int(parent_decision["chosen_move"]),
                    ],
                }
        if abs(actual["backed_up_value"] - parent["backed_up_value"]) > 1e-12:
            return {"invariant_failure": "backup differs before selection divergence"}
    return None


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    return float(
        np.sum(np.cumsum(ranked) / (np.arange(len(ranked)) + 1) * ranked) / positives
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = np.empty(len(scores), dtype=float)
    order = np.argsort(scores, kind="stable")
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        indexes = np.flatnonzero(scores == value)
        ranks[indexes] = ranks[indexes].mean()
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    def ranks(values: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=float)
        for value in np.unique(values):
            indexes = np.flatnonzero(values == value)
            result[indexes] = (indexes.size - 1) / 2.0
        order = np.argsort(values, kind="stable")
        ordered = result[order].copy()
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]:
                stop += 1
            ordered[start:stop] = (start + stop - 1) / 2.0
            start = stop
        result[order] = ordered
        return result

    left_ranks, right_ranks = ranks(left), ranks(right)
    if len(left) < 2 or np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _predictor_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictors = {
        "max_pressure_ratio": "max_pressure_ratio",
        "max_flip_excess": "max_flip_excess",
        "policy_l1": "root_policy_l1",
        "max_action_delta": "root_max_abs_action_delta",
        "minimum_parent_puct_margin": "minimum_parent_puct_winner_runner_up_margin",
    }
    table: dict[str, Any] = {}
    for name, key in predictors.items():
        scores = np.asarray(
            [
                record["through_1200"][key]
                if key in record["through_1200"]
                else record[key]
                for record in records
            ]
        )
        table[name] = {}
        for budget in (384, 1200):
            labels = np.asarray(
                [record[f"root_move_difference_{budget}"] for record in records],
                dtype=bool,
            )
            visit_js = np.asarray(
                [record[f"visit_js_{budget}"] for record in records], dtype=float
            )
            order = np.argsort(scores)[::-1]
            quartiles = np.array_split(order, 4)
            table[name][str(budget)] = {
                "auroc_root_move_difference": _auroc(scores, labels),
                "auprc_root_move_difference": _average_precision(scores, labels),
                "top_10_capture": float(
                    labels[order[: max(1, len(order) // 10)]].mean()
                ),
                "top_25_capture": float(
                    labels[order[: max(1, len(order) // 4)]].mean()
                ),
                "spearman_visit_js": _spearman(scores, visit_js),
                "mean_visit_js_by_predictor_quartile": [
                    float(visit_js[indexes].mean()) for indexes in quartiles
                ],
            }
    return table


def _bootstrap_differences(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(records), size=(BOOTSTRAP_SAMPLES, len(records)))
    pressure = np.asarray(
        [record["through_1200"]["max_pressure_ratio"] for record in records]
    )
    l1 = np.asarray([record["root_policy_l1"] for record in records])
    result = {}
    for budget in (384, 1200):
        labels = np.asarray(
            [record[f"root_move_difference_{budget}"] for record in records], dtype=bool
        )
        visit_js = np.asarray([record[f"visit_js_{budget}"] for record in records])
        ap = []
        rho = []
        for draw in indexes:
            pressure_ap, l1_ap = (
                _average_precision(pressure[draw], labels[draw]),
                _average_precision(l1[draw], labels[draw]),
            )
            pressure_rho, l1_rho = (
                _spearman(pressure[draw], visit_js[draw]),
                _spearman(l1[draw], visit_js[draw]),
            )
            if pressure_ap is not None and l1_ap is not None:
                ap.append(pressure_ap - l1_ap)
            if pressure_rho is not None and l1_rho is not None:
                rho.append(pressure_rho - l1_rho)
        result[str(budget)] = {
            "auprc_pressure_minus_l1": _interval(ap),
            "spearman_pressure_minus_l1": _interval(rho),
        }
    return result


def _interval(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "lower_95": None, "upper_95": None, "samples": 0}
    array = np.asarray(values)
    return {
        "estimate": float(array.mean()),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
        "samples": len(values),
    }


def _percentile(value: float, distribution: list[float]) -> float:
    return float(100.0 * np.mean(np.asarray(distribution) <= value))


def _search_budget_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "cross_before_384": [],
        "first_cross_after_384": [],
        "never_cross_by_1200": [],
    }
    for record in records:
        first = record["through_1200"]["first_predicted_counterfactual_flip_simulation"]
        name = (
            "cross_before_384"
            if first is not None and first <= 384
            else "first_cross_after_384"
            if first is not None
            else "never_cross_by_1200"
        )
        groups[name].append(record)
    return {
        name: {
            "count": len(group),
            "root_move_difference_rate_384": float(
                np.mean([record["root_move_difference_384"] for record in group])
            )
            if group
            else None,
            "root_move_difference_rate_1200": float(
                np.mean([record["root_move_difference_1200"] for record in group])
            )
            if group
            else None,
            "mean_visit_js_384": float(
                np.mean([record["visit_js_384"] for record in group])
            )
            if group
            else None,
            "mean_visit_js_1200": float(
                np.mean([record["visit_js_1200"] for record in group])
            )
            if group
            else None,
        }
        for name, group in groups.items()
    }


def _classification(
    invariants: dict[str, bool],
    records: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    held_out: list[dict[str, Any]],
) -> tuple[str, str]:
    if not all(invariants.values()):
        return (
            "invariant_failure",
            "Repair artifact, state, seed, or trace-prefix invariants before ML work.",
        )
    divergent = [
        record for record in records if record["actual_first_divergence"] is not None
    ]
    if any(not record["first_flip_matches_actual"] for record in divergent):
        return (
            "search_margin_metric_not_reproducible",
            "Audit counterfactual state reconstruction and trace instrumentation before ML work.",
        )
    evidence = bootstrap["1200"]
    materially_better = (
        evidence["auprc_pressure_minus_l1"]["lower_95"] is not None
        and evidence["auprc_pressure_minus_l1"]["lower_95"] > 0.0
        and evidence["spearman_pressure_minus_l1"]["lower_95"] is not None
        and evidence["spearman_pressure_minus_l1"]["lower_95"] > 0.0
    )
    extreme_held_out = all(
        item["percentiles"]["max_pressure_ratio"] >= 90.0 for item in held_out
    )
    if materially_better and extreme_held_out:
        return (
            "search_margin_sensitivity_predictive",
            "Retrain the exact Gen-2 update with a margin-sensitive parent constraint, protecting high-risk replay states while leaving large-margin states freer.",
        )
    if (
        evidence["auprc_pressure_minus_l1"]["lower_95"] is not None
        and evidence["auprc_pressure_minus_l1"]["lower_95"] <= 0.0
        and evidence["auprc_pressure_minus_l1"]["upper_95"] >= 0.0
    ):
        return (
            "plain_policy_drift_equally_predictive",
            "Do not add expensive MCTS-derived risk scoring to training.",
        )
    if not materially_better:
        return (
            "margin_explains_first_flip_not_final_instability",
            "Model post-divergence Q/visit amplification rather than another policy constraint.",
        )
    return (
        "inconclusive",
        "Repeat the frozen replay audit without changing its metric or calibration sample.",
    )


def _report(summary: dict[str, Any]) -> str:
    comparison = summary["predictor_comparison"]
    rows = [
        "| Predictor | AUPRC move @384 | AUPRC move @1200 | Spearman JS @384 | Spearman JS @1200 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in comparison.items():
        rows.append(
            f"| {name} | {values['384']['auprc_root_move_difference']} | {values['1200']['auprc_root_move_difference']} | {values['384']['spearman_visit_js']} | {values['1200']['spearman_visit_js']} |"
        )
    held = [
        "| State hash | Pressure percentile | Flip-excess percentile | L1 percentile | Max-delta percentile |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["held_out_pr220"]:
        p = item["percentiles"]
        held.append(
            f"| `{item['state_hash']}` | {p['max_pressure_ratio']:.2f} | {p['max_flip_excess']:.2f} | {p['root_policy_l1']:.2f} | {p['root_max_abs_action_delta']:.2f} |"
        )
    return "\n".join(
        [
            "# PR #214 Search-Margin Sensitivity",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended follow-up:** {summary['recommended_follow_up']}",
            "",
            "## Predictor Comparison",
            "",
            *rows,
            "",
            "## Held-Out PR #220 States",
            "",
            *held,
            "",
            "## Validation",
            "",
            "```json",
            json.dumps(
                {
                    "invariants": summary["invariants"],
                    "bootstrap": summary["paired_bootstrap"],
                    "search_budget": summary["search_budget"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )


def _prefix_probe(task: tuple[dict[str, Any], str]) -> bool:
    state, state_hash = task
    p1, _a16 = _worker_evaluators()
    full_trace, full_summary = _search(p1, state, _seed(state_hash), SIMULATIONS)
    short_trace, short_summary = _search(p1, state, _seed(state_hash), 384)
    return _trace_prefix(full_trace, 384) == _trace_prefix(
        short_trace, 384
    ) and _snapshot(full_summary, 384) == _snapshot(short_summary, 384)


def _audit_root(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    state, metadata = task
    p1, a16 = _worker_evaluators()
    seed = _seed(metadata["state_hash"])
    p1_trace, p1_summary = _search(p1, state, seed, SIMULATIONS)
    through_384 = sensitivity_through(p1_trace, state, a16, 384)
    through_1200 = sensitivity_through(p1_trace, state, a16, 1200)
    a16_trace, a16_summary = _search(a16, state, seed, SIMULATIONS)
    actual = first_divergence(a16_trace, p1_trace)
    invariant_failure = actual is not None and "invariant_failure" in actual
    if invariant_failure:
        actual = None
    p384, p1200 = _snapshot(p1_summary, 384), _snapshot(p1_summary, 1200)
    a384, a1200 = _snapshot(a16_summary, 384), _snapshot(a16_summary, 1200)
    prediction = through_1200
    first_matches = (
        actual is None
        and prediction["first_predicted_counterfactual_flip_simulation"] is None
    )
    if actual is not None:
        first_matches = (
            prediction["first_predicted_counterfactual_flip_simulation"]
            == actual["simulation"]
            and prediction["first_predicted_counterfactual_flip_depth"]
            == actual["depth"]
            and prediction["first_predicted_counterfactual_flip_state_hash"]
            == actual["state_hash"]
            and prediction["first_predicted_counterfactual_flip_action_pair"]
            == actual["action_pair"]
        )
    return {
        **metadata,
        "search_seed": seed,
        **_root_policy_metrics(p1, a16, KalahGame.from_state(state)),
        "through_384": through_384,
        "through_1200": through_1200,
        "actual_first_divergence": actual,
        "first_flip_matches_actual": first_matches,
        "pre_divergence_invariant": not invariant_failure,
        "root_move_difference_384": p384["selected_move"] != a384["selected_move"],
        "root_move_difference_1200": p1200["selected_move"] != a1200["selected_move"],
        "visit_js_384": _visit_js(p384["visits"], a384["visits"]),
        "visit_js_1200": _visit_js(p1200["visits"], a1200["visits"]),
        "root_q_difference": float(
            a16_summary["root_q_value"] - p1_summary["root_q_value"]
        ),
        "visit_margin_difference": _visit_margin(a1200) - _visit_margin(p1200),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_adapter_margin_sensitivity"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-margin-sensitivity-results.md",
    )
    args = parser.parse_args()
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    a16_checkpoint = args.adapter_workdir / "artifacts/step_0016/checkpoint.npz"
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    rows = read_jsonl(replay)
    model = new_model(torch.device("cpu"))
    load_checkpoint_into_model(model, a16_checkpoint)
    indexes, manifest_rows = select_sample(rows)
    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "azlite_pr214_margin_sensitivity_manifest_v1",
        "sample_seed": SAMPLE_SEED,
        "sample_size": SAMPLE_SIZE,
        "excluded_state_hashes": sorted(HELD_OUT_HASHES | _canonical_arena_hashes()),
        "rows": manifest_rows,
    }
    manifest_path = args.out_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    invariants = {
        "pr220_parent_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "622c4aa^"], cwd=REPO_ROOT, text=True)
        .strip()
        == "aeaffb6a7b90982ed2e04f088f128de26674f3fe",
        "p1_checkpoint_hash": sha256_file(p1_checkpoint) == P1_CHECKPOINT_SHA,
        "a16_state_hash": model_state_hash(model.state_dict()) == A16_STATE_SHA,
        "replay_hash": sha256_file(replay) == REPLAY_SHA,
        "sample_state_round_trip": all(
            state_round_trips_kalah_v3(list(rows[index]["state"])) for index in indexes
        ),
    }
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    tasks = [
        (decode_kalah_v3_base_state(list(rows[index]["state"])), metadata)
        for index, metadata in zip(indexes, manifest_rows, strict=True)
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(p1_artifact), str(a16_artifact)),
    ) as executor:
        prefix_ok = all(
            executor.map(
                _prefix_probe,
                [(state, metadata["state_hash"]) for state, metadata in tasks[:8]],
            )
        )
        records = list(executor.map(_audit_root, tasks, chunksize=1))
    invariants["continuous_1200_prefix_equals_standalone_384"] = prefix_ok
    invariants["pre_divergence_trace_statistics"] = all(
        record["pre_divergence_invariant"] for record in records
    )
    comparison = _predictor_table(records)
    bootstrap = _bootstrap_differences(records)
    distributions = {
        key: [record["through_1200"][key] for record in records]
        for key in ("max_pressure_ratio", "max_flip_excess")
    }
    distributions.update(
        {
            key: [record[key] for record in records]
            for key in ("root_policy_l1", "root_max_abs_action_delta")
        }
    )
    divergence_summary = json.loads(
        (
            REPO_ROOT
            / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
        ).read_text()
    )
    held_out = []
    p1 = ArtifactEvaluator(p1_artifact)
    a16 = ArtifactEvaluator(a16_artifact)
    for row in {
        item["state_hash"]: item
        for item in divergence_summary["first_game_divergences"]
    }.values():
        state = row["state"]
        p1_trace, _p1_summary = _search(
            p1, state, _seed(row["state_hash"]), SIMULATIONS
        )
        sensitivity = sensitivity_through(p1_trace, state, a16, SIMULATIONS)
        root = _root_policy_metrics(p1, a16, KalahGame.from_state(state))
        values = {
            "max_pressure_ratio": sensitivity["max_pressure_ratio"],
            "max_flip_excess": sensitivity["max_flip_excess"],
            "root_policy_l1": root["root_policy_l1"],
            "root_max_abs_action_delta": root["root_max_abs_action_delta"],
        }
        held_out.append(
            {
                "state_hash": row["state_hash"],
                "values": values,
                "percentiles": {
                    key: _percentile(value, distributions[key])
                    for key, value in values.items()
                },
            }
        )
    budget = _search_budget_summary(records)
    classification, follow_up = _classification(
        invariants, records, bootstrap, held_out
    )
    summary = {
        "schema": "azlite_pr214_margin_sensitivity_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "arena_derived_calibration": False,
            "promotion": False,
            "c_puct": C_PUCT,
            "simulations": SIMULATIONS,
        },
        "hashes": {
            "pr220_commit": "622c4aaba14632cdee422ee16e63808ac0c087af",
            "pr220_parent": "aeaffb6a7b90982ed2e04f088f128de26674f3fe",
            "p1_checkpoint": sha256_file(p1_checkpoint),
            "a16_state": model_state_hash(model.state_dict()),
            "replay": sha256_file(replay),
            "manifest": sha256_file(manifest_path),
            "p1_artifact_weights": sha256_file(
                args.p1_workdir
                / "beta_095/snapshot_artifacts/step_0046/artifact/weights.json"
            ),
            "a16_artifact_weights": sha256_file(
                args.adapter_workdir / "artifacts/step_0016/artifact/weights.json"
            ),
        },
        "invariants": invariants,
        "sample": {
            "size": len(records),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "search_contract": {
            "simulations": [384, 1200],
            "c_puct": C_PUCT,
            "root_noise": False,
            "root_policy_mode": "deterministic",
            "seed": "sha256(pr220-margin-sensitivity:state-hash)",
        },
        "records": records,
        "predictor_comparison": comparison,
        "paired_bootstrap": bootstrap,
        "held_out_pr220": held_out,
        "search_budget": budget,
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(_report(summary), encoding="utf-8")
    print(classification)


if __name__ == "__main__":
    main()
