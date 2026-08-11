#!/usr/bin/env python3
# ruff: noqa: E402
"""Attribute move-quality calibration to raw value, PUCT Q, or visits.

Diagnostic only.  It never writes replay, trains, tunes search, or promotes a
model.  Detailed search and forced-continuation rows remain in ``workdir``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_seed_contract import stable_hash, stable_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_denoised_puct_convergence_audit import (
    CONTINUATION_BUDGETS,
    EXPECTED_CURRENT_SHA256,
    TEACHER_BUDGETS,
    _state_from_row,
    convergence_summary,
    entropy,
    phase_for_state,
    run_forced_tasks,
    run_teacher_tasks,
    sha256_file,
    stability_summary,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator, Evaluator, Node, PUCT

DEFAULT_WORKDIR = Path("/tmp/azlite_search_q_value_attribution")
DEFAULT_PR176 = Path(
    "/tmp/azlite_distribution_aligned_selfplay/pilot_standard_replay.jsonl"
)
DEFAULT_PR177 = Path(
    "/tmp/azlite_policy_target_noise_ablation/target_probe_states.jsonl"
)
DEFAULT_OPENING = Path("/tmp/azlite_opening_suite/large_eval.jsonl")
DEFAULT_ADDITIONAL = Path(
    "/tmp/azlite_denoised_puct_convergence/additional_standard_start_selfplay_states.jsonl"
)
SCHEMA = "azlite_search_q_value_attribution_audit_v1"
BOOTSTRAP_SAMPLES = 10_000


def verify_puct_q_semantics() -> None:
    """Fail closed if child-Q is no longer expressed for the root player."""

    class KnownValueEvaluator(Evaluator):
        def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
            policy = np.zeros(6, dtype=np.float32)
            for move in game.possible_moves():
                policy[move] = 1.0 / len(game.possible_moves())
            return policy, float(game.captured_seeds[game.current_player]) / 10.0

    engine = PUCT(
        KnownValueEvaluator(), simulations=1, c_puct=1.25, rng=random.Random(1)
    )
    parent = Node(KalahGame([1] + [0] * 11, [0, 0], 0), expanded=True)
    switched = Node(KalahGame([0] * 6 + [1] + [0] * 5, [0, 6], 1))
    parent.children[0] = switched
    if not np.isclose(engine._search(parent), -0.6) or not np.isclose(
        switched.q_value, -0.6
    ):
        raise RuntimeError(
            "CLASSIFY puct_q_perspective_or_backup_bug: turn-switch child Q"
        )
    parent = Node(KalahGame([1] + [0] * 11, [0, 0], 0), expanded=True)
    extra_turn = Node(KalahGame([0, 1] + [0] * 10, [6, 0], 0))
    parent.children[0] = extra_turn
    if not np.isclose(engine._search(parent), 0.6) or not np.isclose(
        extra_turn.q_value, 0.6
    ):
        raise RuntimeError(
            "CLASSIFY puct_q_perspective_or_backup_bug: extra-turn child Q"
        )
    parent = Node(KalahGame([1] + [0] * 11, [0, 0], 0), expanded=True)
    terminal = Node(KalahGame([0] * 12, [25, 23], 1, winner=0, _over=True))
    parent.children[0] = terminal
    if engine._search(parent) != 1.0 or terminal.q_value != 1.0:
        raise RuntimeError(
            "CLASSIFY puct_q_perspective_or_backup_bug: terminal child Q"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_provenance(sources: dict[str, tuple[Path, str, str]]) -> dict[str, Any]:
    """Record and reject accidental cross-domain source reuse."""
    records: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for domain, (path, parent_experiment, expected_domain) in sources.items():
        if not path.is_file():
            raise RuntimeError(f"missing source for {domain}: {path}")
        rows = read_jsonl(path)
        digest = sha256_file(path)
        if digest in hashes:
            raise RuntimeError(
                "distinct source-domain labels reference the same file hash: "
                f"{hashes[digest]} and {domain}; aliases are disallowed"
            )
        hashes[digest] = domain
        records[domain] = {
            "path": str(path),
            "sha256": digest,
            "parent_experiment": parent_experiment,
            "expected_row_count": len(rows),
            "expected_state_count": len(
                {stable_hash(_state_from_row(row)) for row in rows}
            ),
            "source_domain_label": expected_domain,
            "aliases_allowed": False,
        }
    return records


def _candidate_rows(
    rows: list[dict[str, Any]], domain: str, evaluator: CheckpointEvaluator
) -> list[dict[str, Any]]:
    result = []
    for index, source_row in enumerate(rows):
        state = _state_from_row(source_row)
        game = KalahGame.from_state(state)
        legal = [int(move) for move in game.possible_moves()]
        if not legal:
            continue
        policy, _ = evaluator.evaluate(game)
        result.append(
            {
                "state": state,
                "state_hash": stable_hash(state),
                "source_domain": domain,
                "source_index": index,
                "player": int(game.current_player),
                "phase": phase_for_state(state),
                "legal_moves": legal,
                "legal_move_count": len(legal),
                "current_policy_entropy": entropy(policy),
            }
        )
    return result


def select_probe_states(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    evaluator: CheckpointEvaluator,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take 256 unique states/domain, targeting <=60% opening overall."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for domain, rows in source_rows.items():
        # The independent opening suite is intentionally all opening positions;
        # counterweight it with more mid/late standard-start states overall.
        phase_targets = (
            {"opening": 256, "midgame": 0, "late": 0}
            if domain == "independent_opening_suite_diagnostic"
            else {"opening": 77, "midgame": 102, "late": 77}
        )
        buckets: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
        for row in _candidate_rows(rows, domain, evaluator):
            buckets[
                (
                    row["phase"],
                    row["player"],
                    row["legal_move_count"],
                    int(row["current_policy_entropy"] * 4),
                )
            ].append(row)
        rng = random.Random(stable_seed(seed, domain, "q-calibration-probe"))
        queues = {phase: deque() for phase in phase_targets}
        for key in sorted(buckets, key=str):
            values = list(buckets[key])
            rng.shuffle(values)
            queues[key[0]].append(deque(values))
        domain_selected = []
        for phase, target in phase_targets.items():
            while (
                queues[phase]
                and sum(r["phase"] == phase for r in domain_selected) < target
            ):
                bucket = queues[phase].popleft()
                while bucket and bucket[0]["state_hash"] in used:
                    bucket.popleft()
                if bucket:
                    row = bucket.popleft()
                    used.add(row["state_hash"])
                    domain_selected.append(row)
                if bucket:
                    queues[phase].append(bucket)
        remaining = deque(bucket for phase in phase_targets for bucket in queues[phase])
        while remaining and len(domain_selected) < 256:
            bucket = remaining.popleft()
            while bucket and bucket[0]["state_hash"] in used:
                bucket.popleft()
            if bucket:
                row = bucket.popleft()
                used.add(row["state_hash"])
                domain_selected.append(row)
            if bucket:
                remaining.append(bucket)
        if len(domain_selected) != 256:
            raise RuntimeError(
                f"only {len(domain_selected)} unique states available for {domain}; need 256"
            )
        selected.extend(domain_selected)
    if len(selected) != 768 or len({row["state_hash"] for row in selected}) != 768:
        raise RuntimeError("probe must contain exactly 768 unique states")
    cutoffs = np.quantile(
        [r["current_policy_entropy"] for r in selected], [0.25, 0.5, 0.75]
    ).tolist()
    for row in selected:
        row["policy_entropy_quartile"] = 1 + sum(
            row["current_policy_entropy"] > cutoff for cutoff in cutoffs
        )
    return selected, {
        "schema": "azlite_q_calibration_probe_v1",
        "selection_seed": seed,
        "state_count": 768,
        "state_hashes": [r["state_hash"] for r in selected],
        "source_domain_counts": dict(Counter(r["source_domain"] for r in selected)),
        "player_counts": dict(Counter(str(r["player"]) for r in selected)),
        "phase_counts": dict(Counter(r["phase"] for r in selected)),
        "legal_move_distribution": dict(
            Counter(str(r["legal_move_count"]) for r in selected)
        ),
        "policy_entropy_quartile_cutoffs": [float(value) for value in cutoffs],
    }


def root_perspective_post_move_value(
    evaluator: CheckpointEvaluator, state: dict[str, Any], move: int
) -> float:
    game = KalahGame.from_state(state)
    root_player = game.current_player
    if not game.move(game.pit_index(move)):
        raise RuntimeError(f"illegal candidate move {move}")
    _policy, value = evaluator.evaluate(game)
    return float(value if game.current_player == root_player else -value)


def audit_move_rows(
    teachers: list[dict[str, Any]], *, evaluator: CheckpointEvaluator
) -> list[dict[str, Any]]:
    rows = []
    for record in teachers:
        state = record["state"]
        legal = record["legal_moves"]
        policy, _ = evaluator.evaluate(KalahGame.from_state(state))
        candidate_moves = set(
            sorted(legal, key=lambda move: (-float(policy[move]), move))[:2]
        )
        for budget in (384, 768, 1200):
            teacher = record["teachers"][f"D{budget}"]
            candidate_moves.update(
                int(item["move"])
                for item in sorted(
                    teacher["child_stats"],
                    key=lambda item: (
                        -int(item["visits"]),
                        -float(item["q_value"]),
                        int(item["move"]),
                    ),
                )[:2]
            )
        # The selected move of every budget is required even when D128 is outside top-2 union.
        candidate_moves.update(
            record["teachers"][f"D{budget}"]["top_move"] for budget in TEACHER_BUDGETS
        )
        ordered = sorted(candidate_moves)
        for move in ordered:
            row = {
                key: record[key]
                for key in (
                    "state_hash",
                    "state",
                    "player",
                    "phase",
                    "source_domain",
                    "legal_move_count",
                    "current_policy_entropy",
                    "policy_entropy_quartile",
                )
            }
            row.update(
                {
                    "move": int(move),
                    "candidate_move_count": len(ordered),
                    "excluded_legal_move_count": len(legal) - len(ordered),
                    "policy_prior": float(policy[move]),
                    "policy_prior_rank": 1
                    + sorted(
                        legal,
                        key=lambda candidate: (-float(policy[candidate]), candidate),
                    ).index(move),
                    "raw_leaf_value_root_perspective": root_perspective_post_move_value(
                        evaluator, state, move
                    ),
                }
            )
            for budget in TEACHER_BUDGETS:
                teacher = record["teachers"][f"D{budget}"]
                child = next(
                    item for item in teacher["child_stats"] if int(item["move"]) == move
                )
                row[f"q_D{budget}"] = float(child["q_value"])
                row[f"visit_count_D{budget}"] = int(child["visits"])
                row[f"child_visit_count_D{budget}"] = int(child["visits"])
                row[f"visit_share_D{budget}"] = float(child["visits"]) / max(
                    1, sum(teacher["visits"])
                )
                row[f"visit_rank_D{budget}"] = 1 + [
                    int(item["move"])
                    for item in sorted(
                        teacher["child_stats"],
                        key=lambda item: (
                            -int(item["visits"]),
                            -float(item["q_value"]),
                            int(item["move"]),
                        ),
                    )
                ].index(move)
                row[f"root_value_D{budget}"] = float(teacher["root_value"])
                row[f"top1_top2_visit_margin_D{budget}"] = float(
                    teacher["top1_top2_visit_margin"]
                )
            rows.append(row)
    return rows


def forced_rows(
    moves: list[dict[str, Any]], *, checkpoint: Path, workers: int, seed: int
) -> list[dict[str, Any]]:
    output = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in moves:
        grouped[row["state_hash"]].append(row)
    tasks = []
    for state_hash, group in grouped.items():
        representative = group[0]
        for budget in CONTINUATION_BUDGETS:
            tasks.append(
                {
                    "state": representative["state"],
                    "state_hash": state_hash,
                    "continuation_budget": budget,
                    "experiment_seed": seed,
                    "moves": [row["move"] for row in group],
                    "comparison": "candidate_move_set",
                }
            )
    interventions = run_forced_tasks(
        tasks=tasks, checkpoint=checkpoint, workers=workers
    )
    move_rows = {(row["state_hash"], row["move"]): row for row in moves}
    for record in interventions:
        for raw_move, result in record["interventions"].items():
            row = move_rows[(record["state_hash"], int(raw_move))]
            output.append(
                {
                    **row,
                    "continuation_budget": record["continuation_budget"],
                    **result,
                    "normalized_store_margin_root": float(result["store_margin_root"])
                    / 48.0,
                }
            )
    return output


def _ranked_move(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row[signal]), -int(row["move"])))


def _state_metric(
    rows: list[dict[str, Any]], signal: str, target: str
) -> dict[str, float]:
    agreements = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            realized = float(left[target] - right[target])
            if realized:
                predicted = float(left[signal] - right[signal])
                agreements.append(float(np.sign(predicted) == np.sign(realized)))
    return {
        "concordance": float(np.mean(agreements)) if agreements else 0.5,
        "non_tied_pairs": len(agreements),
    }


def _bootstrap(values: list[float], *, seed: int) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(BOOTSTRAP_SAMPLES, len(data)), replace=True).mean(
        axis=1
    )
    return {
        "mean": float(data.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "n": len(data),
    }


def _correlation(left: list[float], right: list[float]) -> float:
    return (
        float(np.corrcoef(left, right)[0, 1])
        if len(left) > 1 and np.std(left) and np.std(right)
        else 0.0
    )


def _ordinal_ranks(values: list[float]) -> list[float]:
    ranks = np.empty(len(values), dtype=float)
    ranks[np.argsort(np.asarray(values), kind="stable")] = np.arange(len(values))
    return ranks.tolist()


def legacy_pr179_style(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Secondary only: one averaged candidate-pair observation per state/budget."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["state_hash"], row["continuation_budget"])].append(row)
    output = {}
    for continuation_budget in CONTINUATION_BUDGETS:
        output[str(continuation_budget)] = {}
        groups = [
            group
            for (_state, budget), group in grouped.items()
            if budget == continuation_budget
        ]
        for target in ("store_margin_root", "outcome_root"):
            output[str(continuation_budget)][target] = {}
            for teacher_budget in TEACHER_BUDGETS:
                observations = []
                for group in groups:
                    deltas = [
                        (
                            float(
                                left[f"q_D{teacher_budget}"]
                                - right[f"q_D{teacher_budget}"]
                            ),
                            float(left[target] - right[target]),
                        )
                        for index, left in enumerate(group)
                        for right in group[index + 1 :]
                    ]
                    if deltas:
                        observations.append(tuple(np.mean(deltas, axis=0)))
                predicted, realized = zip(*observations) if observations else ([], [])
                high_confidence = [
                    actual
                    for prediction, actual in observations
                    if abs(prediction) >= 0.25
                ]
                output[str(continuation_budget)][target][f"D{teacher_budget}"] = {
                    "n_unique_state_observations": len(observations),
                    "pearson": _correlation(list(predicted), list(realized)),
                    "spearman": _correlation(
                        _ordinal_ranks(list(predicted)), _ordinal_ranks(list(realized))
                    )
                    if observations
                    else 0.0,
                    "high_confidence_wrong_fraction": float(
                        np.mean(
                            [
                                prediction * actual < 0
                                for prediction, actual in observations
                                if abs(prediction) >= 0.25
                            ]
                        )
                    )
                    if high_confidence
                    else 0.0,
                }
    return output


def adjacent_budget_causal_summary(
    teachers: list[dict[str, Any]], forced: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    forced_by_identity = {
        (row["state_hash"], row["continuation_budget"], row["move"]): row
        for row in forced
    }
    output = {}
    for lower, higher in ((128, 384), (384, 768), (768, 1200)):
        comparison = f"D{lower}->D{higher}"
        output[comparison] = {}
        disagreements = [
            row
            for row in teachers
            if row["teachers"][f"D{lower}"]["top_move"]
            != row["teachers"][f"D{higher}"]["top_move"]
        ]
        for continuation_budget in CONTINUATION_BUDGETS:
            rows = []
            for teacher in disagreements:
                lower_row = forced_by_identity[
                    (
                        teacher["state_hash"],
                        continuation_budget,
                        teacher["teachers"][f"D{lower}"]["top_move"],
                    )
                ]
                higher_row = forced_by_identity[
                    (
                        teacher["state_hash"],
                        continuation_budget,
                        teacher["teachers"][f"D{higher}"]["top_move"],
                    )
                ]
                rows.append(
                    {
                        "outcome_delta": higher_row["outcome_root"]
                        - lower_row["outcome_root"],
                        "margin_delta": higher_row["store_margin_root"]
                        - lower_row["store_margin_root"],
                    }
                )
            margins = [float(row["margin_delta"]) for row in rows]
            outcomes = [float(row["outcome_delta"]) for row in rows]
            output[comparison][str(continuation_budget)] = {
                "unique_disagreement_states": len(rows),
                "mean_outcome_delta": float(np.mean(outcomes)) if outcomes else 0.0,
                "mean_store_margin_delta": float(np.mean(margins)) if margins else 0.0,
                "state_clustered_margin_bootstrap_95": _bootstrap(
                    margins, seed=stable_seed(seed, comparison, continuation_budget)
                ),
            }
    return output


def calibration_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    signals = (
        ["policy_prior", "raw_leaf_value_root_perspective"]
        + [f"q_D{b}" for b in TEACHER_BUDGETS]
        + [f"visit_share_D{b}" for b in TEACHER_BUDGETS]
    )
    by_budget_state: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_budget_state[(row["state_hash"], row["continuation_budget"])].append(row)
    state_rows = []
    for (state_hash, budget), group in by_budget_state.items():
        metadata = {
            key: group[0][key]
            for key in (
                "player",
                "phase",
                "source_domain",
                "legal_move_count",
                "policy_entropy_quartile",
            )
        }
        for target, label in (
            ("normalized_store_margin_root", "margin"),
            ("outcome_root", "outcome"),
        ):
            values = {
                signal: _state_metric(group, signal, target) for signal in signals
            }
            state_rows.append(
                {
                    "state_hash": state_hash,
                    "continuation_budget": budget,
                    "target": label,
                    **metadata,
                    "signals": values,
                }
            )
    result: dict[str, Any] = {
        "primary": "within-state pairwise concordance against store margin; bootstrap unit=unique state"
    }
    for target in ("margin", "outcome"):
        target_rows = [row for row in state_rows if row["target"] == target]
        result[target] = {}
        for budget in CONTINUATION_BUDGETS:
            budget_rows = [
                row for row in target_rows if row["continuation_budget"] == budget
            ]
            result[target][str(budget)] = {
                signal: _bootstrap(
                    [row["signals"][signal]["concordance"] for row in budget_rows],
                    seed=stable_seed(seed, target, budget, signal),
                )
                for signal in signals
            }
            result[target][str(budget)]["slices"] = {
                field: {
                    str(value): {
                        signal: _bootstrap(
                            [
                                r["signals"][signal]["concordance"]
                                for r in budget_rows
                                if str(r[field]) == str(value)
                            ],
                            seed=stable_seed(
                                seed, target, budget, field, value, signal
                            ),
                        )
                        for signal in signals
                    }
                    for value in sorted({r[field] for r in budget_rows}, key=str)
                }
                for field in (
                    "player",
                    "phase",
                    "source_domain",
                    "legal_move_count",
                    "policy_entropy_quartile",
                )
            }
    regrets = {}
    for budget in CONTINUATION_BUDGETS:
        groups = [
            group
            for (_state, current_budget), group in by_budget_state.items()
            if current_budget == budget
        ]
        for signal in signals:
            values = [
                max(float(row["normalized_store_margin_root"]) for row in group)
                - float(_ranked_move(group, signal)["normalized_store_margin_root"])
                for group in groups
            ]
            regrets.setdefault(str(budget), {})[signal] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "bootstrap_95": _bootstrap(
                    values, seed=stable_seed(seed, budget, signal)
                ),
            }
    result["selected_move_regret"] = regrets
    result["legacy_pr179_style_secondary"] = legacy_pr179_style(rows)
    return result


def classify(
    calibration: dict[str, Any], convergence: dict[str, Any]
) -> tuple[list[str], str]:
    margin = calibration["margin"]["1200"]
    raw, q384, q768, q1200 = (
        margin[name]["mean"]
        for name in ("raw_leaf_value_root_perspective", "q_D384", "q_D768", "q_D1200")
    )
    visit384, visit768, visit1200 = (
        margin[name]["mean"]
        for name in ("visit_share_D384", "visit_share_D768", "visit_share_D1200")
    )
    regrets = calibration["selected_move_regret"]["1200"]
    labels = []
    if raw <= 0.55 and q768 - raw < 0.05 and visit768 <= q768:
        labels.append("value_head_primary_q_bottleneck")
    if (
        q384 > raw
        and q768 - q384 >= 0.03
        and margin["q_D768"]["lower"] >= 0
        and regrets["q_D768"]["mean"] <= regrets["q_D384"]["mean"]
    ):
        labels.append("puct_q_improves_with_budget")
    if raw >= q768 + 0.05:
        labels.append("search_backup_or_q_bottleneck")
    if (
        q768 >= raw + 0.05
        and visit768 < q768
        and regrets["visit_share_D768"]["mean"] > regrets["q_D768"]["mean"]
    ):
        labels.append("visit_allocation_bottleneck")
    if (
        q768 <= 0.55
        and calibration["outcome"]["1200"]["q_D768"]["mean"] <= 0.55
        and raw <= 0.55
    ):
        labels.append("search_q_calibration_confirmed")
    if q768 > 0.55 and regrets["q_D1200"]["mean"] <= regrets["q_D384"]["mean"]:
        labels.append("q_calibration_not_primary_bottleneck")
    d768_d1200 = convergence["comparisons"]["D768->D1200"]["global"]
    if d768_d1200["top1_agreement"] >= 0.90 and q1200 >= q768 and visit1200 >= visit768:
        labels.append("stronger_teacher_signal_replicated")
    action = "No training authorization; inspect the recorded attribution evidence."
    if "value_head_primary_q_bottleneck" in labels:
        action = "Run a no-training value-target diagnostic against forced continuation and final-margin outcomes; do not increase policy-teacher budget."
    elif {"puct_q_improves_with_budget", "stronger_teacher_signal_replicated"}.issubset(
        labels
    ):
        action = "One matched D768 policy-target training experiment is justified, using identical trajectories and value targets; do not implement it in this audit."
    elif "search_backup_or_q_bottleneck" in labels:
        action = "Audit PUCT backup, value averaging, and extra-turn behavior before changing ML targets."
    elif "visit_allocation_bottleneck" in labels:
        action = "Audit prior versus Q influence and visit allocation causally before retraining."
    elif "q_calibration_not_primary_bottleneck" in labels:
        action = "Retire the broad PR #179 Q-miscalibration diagnosis and focus on high-budget target instability."
    return labels or ["inconclusive"], action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--pr176-pilot", default=str(DEFAULT_PR176))
    parser.add_argument("--pr177-probe", default=str(DEFAULT_PR177))
    parser.add_argument("--opening-suite", default=str(DEFAULT_OPENING))
    parser.add_argument("--additional-selfplay", default=str(DEFAULT_ADDITIONAL))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def report_markdown(summary: dict[str, Any]) -> str:
    calibration = summary["calibration"]
    margin = calibration["margin"]
    outcome = calibration["outcome"]

    def section(title: str, value: Any) -> list[str]:
        return [
            "",
            f"## {title}",
            "",
            "```json",
            json.dumps(value, indent=2, sort_keys=True),
            "```",
        ]

    lines = [
        "# Search-Q Value Attribution Audit",
        "",
        f"- Classification: `{', '.join(summary['classifications'])}`",
        f"- Next action: {summary['next_action']}",
        "",
    ]
    lines += section("Corrected Source Provenance", summary["source_provenance"])
    lines += [
        "",
        "## Q-Perspective Semantic Tests",
        "",
        "The runner executed turn-switch, extra-turn, and terminal root-perspective checks before collecting the audit. Any failure aborts with `CLASSIFY puct_q_perspective_or_backup_bug`.",
        "",
    ]
    lines += section(
        "Probe Manifest And Audited Move-Set Sizes",
        {
            "probe_manifest": summary["probe_manifest"],
            "move_set_sizes": summary["move_set_sizes"],
        },
    )
    lines += section(
        "Raw-Value Calibration",
        {
            budget: values["raw_leaf_value_root_perspective"]
            for budget, values in margin.items()
            if budget != "slices"
        },
    )
    lines += section(
        "Q Calibration By Search Budget",
        {
            budget: {
                name: value for name, value in values.items() if name.startswith("q_D")
            }
            for budget, values in margin.items()
            if budget != "slices"
        },
    )
    lines += section(
        "Visit-Ranking Calibration",
        {
            budget: {
                name: value
                for name, value in values.items()
                if name.startswith("visit_share_D")
            }
            for budget, values in margin.items()
            if budget != "slices"
        },
    )
    lines += section("Selected-Move Regret", calibration["selected_move_regret"])
    lines += section(
        "Continuous-Margin Versus Binary-Outcome Results",
        {
            "normalized_store_margin": margin,
            "binary_outcome": outcome,
            "legacy_pr179_style_secondary": calibration["legacy_pr179_style_secondary"],
        },
    )
    lines += section("Corrected PR #179 Convergence", summary["corrected_convergence"])
    lines += section(
        "Adjacent-Budget Causal Results", summary["adjacent_budget_causal_results"]
    )
    lines += section(
        "Player, Phase, Domain, Legal-Move, And Entropy Slices",
        {
            budget: values["slices"]
            for budget, values in margin.items()
            if budget != "slices"
        },
    )
    lines += [
        "",
        "## Exact Classification Evidence",
        "",
        "The decision used state-clustered, within-state normalized-margin concordance and selected-move regret. The full evidence is in the preceding calibration and regret sections.",
        "",
        "## Next Action",
        "",
        summary["next_action"],
        "",
        "Per-state search and forced-continuation traces remain in the workdir. No replay, training, runtime tuning, or promotion was generated.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workdir, current = Path(args.workdir), Path(args.current)
    workdir.mkdir(parents=True, exist_ok=True)
    verify_puct_q_semantics()
    weights = current / "weights.json"
    if sha256_file(weights) != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")
    sources = {
        "pr176_standard_start_pilot": (
            Path(args.pr176_pilot),
            "PR #176 distribution-aligned standard-start pilot",
            "pr176_standard_start_pilot",
        ),
        "independent_opening_suite_diagnostic": (
            Path(args.opening_suite),
            "independent opening-suite diagnostic",
            "independent_opening_suite_diagnostic",
        ),
        "additional_standard_start_selfplay": (
            Path(args.additional_selfplay),
            "diagnostic-only bounded standard-start self-play harvest",
            "additional_standard_start_selfplay",
        ),
    }
    provenance = source_provenance(sources)
    # PR #177 remains recorded only as the verified probe provenance; it is not a
    # fourth sampling domain in this corrected representative probe.
    pr177_rows = read_jsonl(Path(args.pr177_probe))
    pr177_hash = sha256_file(Path(args.pr177_probe))
    if pr177_hash in {record["sha256"] for record in provenance.values()}:
        raise RuntimeError(
            "distinct source-domain labels reference the same file hash; aliases are disallowed"
        )
    provenance["pr177_evaluation_diagnostic_reference"] = {
        "path": str(Path(args.pr177_probe)),
        "sha256": pr177_hash,
        "parent_experiment": "PR #177 policy-target probe",
        "expected_row_count": len(pr177_rows),
        "expected_state_count": len(
            {stable_hash(_state_from_row(row)) for row in pr177_rows}
        ),
        "source_domain_label": "evaluation_opening_diagnostic",
        "aliases_allowed": False,
        "used_for_sampling": False,
    }
    checkpoint = materialize_weights_json_checkpoint(
        weights_path=weights, out_path=workdir / "current.npz"
    )
    evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    probe, manifest = select_probe_states(
        {
            domain: read_jsonl(path)
            for domain, (path, _parent, _label) in sources.items()
        },
        evaluator=evaluator,
        seed=args.seed,
    )
    for row in probe:
        policy, _ = evaluator.evaluate(KalahGame.from_state(row["state"]))
        row["current_network_top_move"] = int(
            max(row["legal_moves"], key=lambda move: (float(policy[move]), -move))
        )
    manifest["source_hashes"] = {
        name: value["sha256"] for name, value in provenance.items() if "sha256" in value
    }
    manifest["current_model_hash"] = sha256_file(weights)
    write_jsonl(workdir / "q_calibration_probe_states.jsonl", probe)
    write_json(workdir / "q_calibration_probe_manifest.json", manifest)
    teacher_path = workdir / "teacher_search_records.jsonl"
    cached_teachers = read_jsonl(teacher_path) if teacher_path.is_file() else []
    teachers = (
        cached_teachers
        if [row["state_hash"] for row in cached_teachers]
        == sorted(row["state_hash"] for row in probe)
        else run_teacher_tasks(
            tasks=[{**row, "experiment_seed": args.seed} for row in probe],
            checkpoint=checkpoint,
            workers=args.workers,
        )
    )
    for row in teachers:
        if "current_network_top_move" not in row:
            policy, _ = evaluator.evaluate(KalahGame.from_state(row["state"]))
            row["current_network_top_move"] = int(
                max(row["legal_moves"], key=lambda move: (float(policy[move]), -move))
            )
    write_jsonl(teacher_path, teachers)
    move_path = workdir / "audited_move_records.jsonl"
    cached_moves = read_jsonl(move_path) if move_path.is_file() else []
    moves = (
        cached_moves
        if {row["state_hash"] for row in cached_moves}
        == {row["state_hash"] for row in teachers}
        else audit_move_rows(teachers, evaluator=evaluator)
    )
    write_jsonl(move_path, moves)
    forced_path = workdir / "forced_continuation_records.jsonl"
    cached_forced = read_jsonl(forced_path) if forced_path.is_file() else []
    forced = (
        cached_forced
        if len(cached_forced) == len(moves) * len(CONTINUATION_BUDGETS)
        and {row["state_hash"] for row in cached_forced}
        == {row["state_hash"] for row in teachers}
        else forced_rows(
            moves, checkpoint=checkpoint, workers=args.workers, seed=args.seed
        )
    )
    write_jsonl(forced_path, forced)
    calibration = calibration_summary(forced, seed=args.seed)
    convergence = convergence_summary(teachers)
    labels, next_action = classify(calibration, convergence)
    summary = {
        "schema": SCHEMA,
        "current_weights_sha256": sha256_file(weights),
        "search_configuration": {
            "teacher_budgets": list(TEACHER_BUDGETS),
            "continuation_budgets": list(CONTINUATION_BUDGETS),
            "dirichlet_epsilon": 0.0,
            "c_puct": 1.25,
            "tactical_root_bias": 0.0,
            "root_policy_mode": "deterministic",
            "normalize_values": False,
            "paired_seed_contract": "state, continuation budget, root player, experiment seed; forced move excluded",
        },
        "source_provenance": provenance,
        "probe_manifest": manifest,
        "move_set_sizes": {
            "mean": float(np.mean([r["candidate_move_count"] for r in moves])),
            "distribution": dict(
                Counter(str(r["candidate_move_count"]) for r in moves)
            ),
            "mean_excluded_legal_moves": float(
                np.mean([r["excluded_legal_move_count"] for r in moves])
            ),
        },
        "calibration": calibration,
        "corrected_convergence": convergence,
        "target_stability": stability_summary(teachers),
        "adjacent_budget_causal_results": adjacent_budget_causal_summary(
            teachers, forced, seed=args.seed
        ),
        "classification": labels[0],
        "classifications": labels,
        "next_action": next_action,
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT / "docs/data/alphazero-lite-search-q-value-attribution-summary.json",
        summary,
    )
    (
        REPO_ROOT / "docs/alphazero-lite-search-q-value-attribution-results.md"
    ).write_text(report_markdown(summary), encoding="utf-8")
    print(json.dumps({"classifications": labels, "workdir": str(workdir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
