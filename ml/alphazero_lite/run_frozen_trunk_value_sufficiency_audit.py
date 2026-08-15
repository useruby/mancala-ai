#!/usr/bin/env python3
# ruff: noqa: E402, E701, E702, E731
"""Diagnostic-only frozen-trunk value sufficiency audit.

This runner never writes an artifact and trains only ephemeral probes in its
workdir.  Evaluation-opening rows are held out as a separate diagnostic domain.
"""

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
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.evaluation_seed_contract import stable_hash, stable_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_denoised_puct_convergence_audit import (
    _state_from_row,
    entropy,
)
from ml.alphazero_lite.self_play import (
    Evaluator,
    build_eval_search_options,
    encode_state,
)

EXPECTED_CURRENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
DEFAULT_WORKDIR = Path("/tmp/azlite_frozen_trunk_value_sufficiency")
SOURCES = {
    "standard_start_selfplay": Path(
        "/tmp/azlite_distribution_aligned_selfplay/pilot_standard_replay.jsonl"
    ),
    "opening_family_diagnostic": Path("/tmp/azlite_opening_suite/large_eval.jsonl"),
    "additional_selfplay": Path(
        "/tmp/azlite_denoised_puct_convergence/additional_standard_start_selfplay_states.jsonl"
    ),
}
EVALUATION_SOURCE = Path(
    "/tmp/azlite_policy_target_noise_ablation/target_probe_states.jsonl"
)
TARGETS = {"opening": 2560, "midgame": 768, "late": 768}
RIDGE_ALPHA = 1.0
_WORKER_EVALUATOR: ArtifactEvaluator | None = None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_state_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def phase(game: KalahGame) -> str:
    remaining = sum(game.pits)
    return "late" if remaining <= 12 else "midgame" if remaining <= 24 else "opening"


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def state_rows(
    rows: list[dict[str, Any]], domain: str, evaluator: ArtifactEvaluator
) -> list[dict[str, Any]]:
    result = []
    for source_index, raw in enumerate(rows):
        state = _state_from_row(raw)
        game = KalahGame.from_state(state)
        if game.over() or not game.possible_moves():
            continue
        policy, value = evaluator.evaluate(game)
        result.append(
            {
                "state": state,
                "state_hash": stable_hash(state),
                "source_domain": domain,
                "source_game": f"{domain}:{raw.get('game_index', raw.get('opening_index', source_index))}",
                "opening_family": str(
                    raw.get("opening_family", raw.get("game_index", source_index))
                ),
                "player": int(game.current_player),
                "phase": phase(game),
                "legal_move_count": len(game.possible_moves()),
                "current_value": float(value),
                "policy_entropy": float(entropy(policy)),
            }
        )
    return result


def supplemental_standard_start_states(
    evaluator: ArtifactEvaluator, *, seed: int, count: int
) -> list[dict[str, Any]]:
    """Generate disjoint, seed-identified standard-start opening families."""
    rows: list[dict[str, Any]] = []
    for game_index in range(count):
        game = KalahGame([4] * 12, [0, 0], 0)
        # A deterministic family prefix avoids duplicating zero-noise root search.
        family_rng = random.Random(stable_seed(seed, "supplemental-family", game_index))
        opening_plies = 2 + family_rng.randrange(10)
        for ply in range(opening_plies):
            if game.over():
                break
            legal = game.possible_moves()
            move = legal[family_rng.randrange(len(legal))]
            if move is None or not game.move(game.pit_index(move)):
                break
        if game.over() or not game.possible_moves():
            continue
        policy, value = evaluator.evaluate(game)
        rows.append(
            {
                "state": game.to_state(),
                "state_hash": stable_hash(game.to_state()),
                "source_domain": "generated_opening_family_diagnostic",
                "source_game": f"generated_opening_family:{game_index}",
                "opening_family": f"generated_opening_family:{game_index}",
                "player": int(game.current_player),
                "phase": phase(game),
                "legal_move_count": len(game.possible_moves()),
                "current_value": float(value),
                "policy_entropy": float(entropy(policy)),
            }
        )
    return rows


def select_corpus(
    candidates: list[dict[str, Any]], *, seed: int, targets: dict[str, int] = TARGETS
) -> list[dict[str, Any]]:
    """Deterministically stratify, with source games as later split units."""
    seen, selected = set(), []
    rng = random.Random(seed)
    for label, target in targets.items():
        bucket = [
            r for r in candidates if r["phase"] == label and r["state_hash"] not in seen
        ]
        rng.shuffle(bucket)
        # Round robin player/legal-move combinations prevents one trajectory shape dominating.
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            groups[(row["player"], row["legal_move_count"])].append(row)
        ordered = [groups[key] for key in sorted(groups)]
        while len([r for r in selected if r["phase"] == label]) < target and any(
            ordered
        ):
            for group in ordered:
                if group and len([r for r in selected if r["phase"] == label]) < target:
                    row = group.pop()
                    if row["state_hash"] not in seen:
                        selected.append(row)
                        seen.add(row["state_hash"])
        if len([r for r in selected if r["phase"] == label]) != target:
            raise RuntimeError(
                f"insufficient unique {label} states for target {target}"
            )
    if len(selected) != 4096:
        raise RuntimeError(
            "corpus must contain exactly 4,096 unique nonterminal states"
        )
    values = np.quantile([r["current_value"] for r in selected], [0.25, 0.5, 0.75])
    entropies = np.quantile([r["policy_entropy"] for r in selected], [0.25, 0.5, 0.75])
    for row in selected:
        row["current_value_quartile"] = int(
            1 + sum(row["current_value"] > x for x in values)
        )
        row["policy_entropy_quartile"] = int(
            1 + sum(row["policy_entropy"] > x for x in entropies)
        )
    return selected


def assign_splits(rows: list[dict[str, Any]], seed: int) -> None:
    families = sorted({row["source_game"] for row in rows})
    rng = random.Random(stable_seed(seed, "source-game-split"))
    rng.shuffle(families)
    boundaries = (round(len(families) * 0.6), round(len(families) * 0.8))
    assignment = {
        family: (
            "train"
            if i < boundaries[0]
            else "validation"
            if i < boundaries[1]
            else "test"
        )
        for i, family in enumerate(families)
    }
    for row in rows:
        row["split"] = assignment[row["source_game"]]


def continuation(
    state: dict[str, Any], evaluator: Evaluator, *, budget: int, seed: int
) -> dict[str, Any]:
    game, root_player, moves = (
        KalahGame.from_state(state),
        KalahGame.from_state(state).current_player,
        [],
    )
    while not game.over():
        result = evaluate_artifact_position(
            evaluator=evaluator,
            state=game.to_state(),
            simulations=budget,
            seed=stable_seed(seed, stable_hash(game.to_state()), "continuation"),
            c_puct=1.25,
            search_options=build_eval_search_options(),
        )
        move = result["selected_move"]
        if move is None or not game.move(game.pit_index(move)):
            raise RuntimeError("continuation selected illegal move")
        moves.append(int(move))
    margin = int(
        game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    )
    return {
        "outcome": int(np.sign(margin)),
        "final_store_margin": margin,
        "normalized_final_margin": margin / 48.0,
        "trajectory_hash": stable_hash(moves),
        "game_length": len(moves),
    }


def _init_label_worker(artifact_dir: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = ArtifactEvaluator(Path(artifact_dir))


def _label_worker(task: tuple[dict[str, Any], int]) -> tuple[str, dict[str, Any]]:
    row, seed = task
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("continuation worker evaluator was not initialized")
    d768 = continuation(
        row["state"],
        _WORKER_EVALUATOR,
        budget=768,
        seed=stable_seed(seed, row["state_hash"], 768),
    )
    d1200 = continuation(
        row["state"],
        _WORKER_EVALUATOR,
        budget=1200,
        seed=stable_seed(seed, row["state_hash"], 1200),
    )
    return row["state_hash"], {
        "D768": d768,
        "D1200": d1200,
        "d768_d1200_outcome_agree": d768["outcome"] == d1200["outcome"],
        "d768_d1200_margin_difference": d768["normalized_final_margin"]
        - d1200["normalized_final_margin"],
    }


def collect_labels(
    rows: list[dict[str, Any]],
    artifact_dir: Path,
    seed: int,
    workers: int,
    checkpoint: Path,
) -> None:
    pending = [row for row in rows if "D1200" not in row or "D768" not in row]
    if not pending:
        return
    by_hash = {row["state_hash"]: row for row in rows}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_label_worker,
        initargs=(str(artifact_dir),),
    ) as executor:
        for completed, (state_hash, values) in enumerate(
            executor.map(_label_worker, ((row, seed) for row in pending), chunksize=1),
            start=1,
        ):
            by_hash[state_hash].update(values)
            if completed % 16 == 0:
                write_state_checkpoint(checkpoint, rows)
    write_state_checkpoint(checkpoint, rows)


def _signal_worker(task: tuple[dict[str, Any], int]) -> tuple[str, dict[str, float]]:
    row, seed = task
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("search-signal worker evaluator was not initialized")
    values: dict[str, float] = {}
    for budget in (128, 384, 768, 1200):
        result = evaluate_artifact_position(
            evaluator=_WORKER_EVALUATOR,
            state=row["state"],
            simulations=budget,
            seed=stable_seed(seed, row["state_hash"], "root-signal", budget),
            c_puct=1.25,
            search_options=build_eval_search_options(),
        )
        values[f"D{budget}"] = float(result.get("search_root_value", result["value"]))
    return row["state_hash"], values


def collect_search_signals(
    rows: list[dict[str, Any]],
    artifact_dir: Path,
    seed: int,
    workers: int,
    checkpoint: Path,
) -> None:
    """Record root values independently of continuation labels."""
    pending = [
        row
        for row in rows
        if set(row.get("root_search_values", {})) != {"D128", "D384", "D768", "D1200"}
    ]
    if not pending:
        return
    by_hash = {row["state_hash"]: row for row in rows}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_label_worker,
        initargs=(str(artifact_dir),),
    ) as executor:
        for completed, (state_hash, values) in enumerate(
            executor.map(_signal_worker, ((row, seed) for row in pending), chunksize=1),
            start=1,
        ):
            by_hash[state_hash]["root_search_values"] = values
            if completed % 16 == 0:
                write_state_checkpoint(checkpoint, rows)
    write_state_checkpoint(checkpoint, rows)


def ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    design = np.c_[z, np.ones(len(z))]
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[-1, -1] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y), mean, scale


def ridge_predict(
    model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray
) -> np.ndarray:
    coef, mean, scale = model
    return np.c_[(x - mean) / scale, np.ones(len(x))] @ coef


class ValueHeadProbe(nn.Module):
    def __init__(self, trunk_size: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(trunk_size, max(trunk_size // 2, 8))
        self.out = nn.Linear(max(trunk_size // 2, 8), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.out(torch.relu(self.hidden(x)))).reshape(-1)


def fit_value_head(x: np.ndarray, y: np.ndarray, seed: int) -> ValueHeadProbe:
    torch.manual_seed(seed)
    model = ValueHeadProbe(x.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=RIDGE_ALPHA / len(x)
    )
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.mean((model(tx) - ty) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)


def concordance(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return 0.5
    left, right = np.tril_indices(len(pred), k=-1)
    target_delta = target[left] - target[right]
    mask = target_delta != 0
    if not np.any(mask):
        return 0.5
    return float(
        np.mean(
            np.sign(pred[left][mask] - pred[right][mask]) == np.sign(target_delta[mask])
        )
    )


def metrics(rows: list[dict[str, Any]], prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray([r["D1200"]["normalized_final_margin"] for r in rows])
    outcome = np.asarray([r["D1200"]["outcome"] for r in rows])

    def buckets(observed: np.ndarray) -> list[dict[str, float | int]]:
        cuts = np.quantile(prediction, np.linspace(0, 1, 11))
        return [
            {
                "bucket": bucket + 1,
                "n": int(np.sum(mask)),
                "mean_prediction": float(np.mean(prediction[mask]))
                if np.any(mask)
                else 0.0,
                "mean_target": float(np.mean(observed[mask])) if np.any(mask) else 0.0,
            }
            for bucket in range(10)
            if np.any(
                mask := (
                    (prediction >= cuts[bucket]) & (prediction <= cuts[bucket + 1])
                    if bucket == 9
                    else (prediction >= cuts[bucket]) & (prediction < cuts[bucket + 1])
                )
            )
        ]

    return {
        "n": len(rows),
        "margin": {
            "mae": float(np.mean(abs(prediction - target))),
            "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "pearson": float(np.corrcoef(prediction, target)[0, 1])
            if np.std(prediction) * np.std(target)
            else 0.0,
            "spearman": float(np.corrcoef(ranks(prediction), ranks(target))[0, 1])
            if np.std(prediction) * np.std(target)
            else 0.0,
            "pairwise_concordance": concordance(prediction, target),
        },
        "outcome": {
            "mae": float(np.mean(abs(prediction - outcome))),
            "sign_accuracy": float(np.mean(np.sign(prediction) == outcome)),
            "brier_style_squared_error": float(np.mean((prediction - outcome) ** 2)),
        },
        "calibration": {
            "margin_prediction_deciles": buckets(target),
            "outcome_value_buckets": buckets(outcome),
        },
    }


def sliced_metrics(
    rows: list[dict[str, Any]], predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("player", "phase", "source_domain", "current_value_quartile"):
        output[field] = {}
        for value in sorted({str(row[field]) for row in rows}):
            indexes = np.asarray([str(row[field]) == value for row in rows])
            subset = [row for row, keep in zip(rows, indexes) if keep]
            output[field][value] = {}
            for name, values in predictions.items():
                result = metrics(subset, values[indexes])
                # Aggregate calibration is reported above; repeating ten buckets per
                # probe and slice exceeds the committed-report size guardrail.
                result.pop("calibration")
                output[field][value][name] = result
    return output


class ProbeValueEvaluator(Evaluator):
    """Diagnostic composition: immutable current policy and ephemeral probe value."""

    def __init__(self, current: ArtifactEvaluator, probe: ValueHeadProbe) -> None:
        self.current, self.probe = current, probe

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy, _value = self.current.evaluate(game)
        trunk = self.current.extract_trunk(game)
        with torch.no_grad():
            value = float(self.probe(torch.from_numpy(trunk[None, :])).item())
        return policy, value


def js(left: np.ndarray, right: np.ndarray) -> float:
    p, q = left / max(left.sum(), 1), right / max(right.sum(), 1)
    mean = (p + q) / 2
    return float(
        0.5 * np.sum(np.where(p > 0, p * np.log(p / np.maximum(mean, 1e-12)), 0))
        + 0.5 * np.sum(np.where(q > 0, q * np.log(q / np.maximum(mean, 1e-12)), 0))
    )


def causal_puct_check(
    rows: list[dict[str, Any]],
    current: ArtifactEvaluator,
    probe: ValueHeadProbe,
    seed: int,
) -> dict[str, Any]:
    """Compare policy-identical searches; continuations share seeds by state/move."""
    sampled = sorted(
        rows, key=lambda row: stable_seed(seed, row["state_hash"], "causal")
    )[:512]
    probe_evaluator = ProbeValueEvaluator(current, probe)
    results: dict[str, Any] = {}
    for budget in (384, 768, 1200):
        changed = []
        diagnostics = []
        for row in sampled:
            common = dict(
                state=row["state"],
                simulations=budget,
                c_puct=1.25,
                search_options=build_eval_search_options(),
            )
            baseline = evaluate_artifact_position(
                evaluator=current,
                seed=stable_seed(seed, row["state_hash"], budget, "search"),
                **common,
            )
            treatment = evaluate_artifact_position(
                evaluator=probe_evaluator,
                seed=stable_seed(seed, row["state_hash"], budget, "search"),
                **common,
            )
            q = lambda item: {
                int(c["move"]): float(c["q_value"]) for c in item["child_stats"]
            }
            diagnostics.append(
                {
                    "selected_changed": baseline["selected_move"]
                    != treatment["selected_move"],
                    "visit_js": js(
                        np.asarray(baseline["visits"]), np.asarray(treatment["visits"])
                    ),
                    "root_value_delta": float(
                        treatment.get("search_root_value", 0)
                        - baseline.get("search_root_value", 0)
                    ),
                    "q_ranking_changed": sorted(
                        q(baseline), key=lambda m: (-q(baseline)[m], m)
                    )
                    != sorted(q(treatment), key=lambda m: (-q(treatment)[m], m)),
                }
            )
            if baseline["selected_move"] != treatment["selected_move"]:
                changed.append(
                    (
                        row,
                        int(baseline["selected_move"]),
                        int(treatment["selected_move"]),
                    )
                )
        deltas = {"768": [], "1200": []}
        for row, current_move, probe_move in changed:
            for continuation_budget in (768, 1200):
                # The forced move is deliberately absent from the paired continuation seed.
                left = continuation_after_forced_move(
                    row["state"],
                    current,
                    current_move,
                    continuation_budget,
                    stable_seed(
                        seed, row["state_hash"], budget, continuation_budget, "paired"
                    ),
                )
                right = continuation_after_forced_move(
                    row["state"],
                    current,
                    probe_move,
                    continuation_budget,
                    stable_seed(
                        seed, row["state_hash"], budget, continuation_budget, "paired"
                    ),
                )
                deltas[str(continuation_budget)].append(
                    {
                        "margin": right["normalized_final_margin"]
                        - left["normalized_final_margin"],
                        "outcome": right["outcome"] - left["outcome"],
                    }
                )
        results[f"D{budget}"] = {
            "states": len(sampled),
            "selected_move_change_rate": float(
                np.mean([x["selected_changed"] for x in diagnostics])
            ),
            "visit_js": float(np.mean([x["visit_js"] for x in diagnostics])),
            "root_value_change": float(
                np.mean([x["root_value_delta"] for x in diagnostics])
            ),
            "q_ranking_change_rate": float(
                np.mean([x["q_ranking_changed"] for x in diagnostics])
            ),
            "changed_moves": len(changed),
            "causal": {
                key: {
                    "normalized_final_margin_delta": _bootstrap_simple(
                        [x["margin"] for x in values], seed
                    ),
                    "binary_outcome_delta": _bootstrap_simple(
                        [x["outcome"] for x in values], seed
                    ),
                }
                for key, values in deltas.items()
            },
        }
    return results


def continuation_after_forced_move(
    state: dict[str, Any], evaluator: Evaluator, move: int, budget: int, seed: int
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    root = game.current_player
    if not game.move(game.pit_index(move)):
        raise RuntimeError("forced move is illegal")
    result = continuation(game.to_state(), evaluator, budget=budget, seed=seed)
    # continuation's perspective is its starting player; reconstruct the original root margin.
    if game.current_player != root:
        result["outcome"] *= -1
        result["final_store_margin"] *= -1
        result["normalized_final_margin"] *= -1
    return result


def _bootstrap_simple(values: list[float], seed: int) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    data = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), (1000, len(data)))].mean(axis=1)
    return {
        "mean": float(data.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "n": len(data),
    }


def bootstrap_gain(
    rows: list[dict[str, Any]], left: np.ndarray, right: np.ndarray, seed: int
) -> dict[str, float]:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["source_game"]].append(i)
    families = list(groups.values())
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(1000):
        indices = np.concatenate(
            [families[i] for i in rng.integers(0, len(families), len(families))]
        )
        gains.append(
            concordance(
                right[indices],
                np.asarray(
                    [rows[i]["D1200"]["normalized_final_margin"] for i in indices]
                ),
            )
            - concordance(
                left[indices],
                np.asarray(
                    [rows[i]["D1200"]["normalized_final_margin"] for i in indices]
                ),
            )
        )
    return {
        "lower": float(np.quantile(gains, 0.025)),
        "upper": float(np.quantile(gains, 0.975)),
    }


def classify(
    current: dict[str, Any],
    affine: dict[str, Any],
    trunk: dict[str, Any],
    raw: dict[str, Any],
    ci: dict[str, float],
) -> str:
    base, probe = current["margin"], trunk["margin"]
    improvement = (base["mae"] - probe["mae"]) / max(base["mae"], 1e-9)
    gain = probe["pairwise_concordance"] - base["pairwise_concordance"]
    if (
        raw["margin"]["pairwise_concordance"] > probe["pairwise_concordance"] + 0.03
        or probe["pairwise_concordance"] < 0.55
    ):
        return "representation_bottleneck"
    if (
        improvement >= 0.15
        and gain >= 0.07
        and ci["lower"] > 0
        and probe["pairwise_concordance"] >= 0.62
    ):
        return (
            "representation_sufficient_value_head_update_justified_pending_causal_puct"
        )
    if (base["mae"] - affine["margin"]["mae"]) / max(
        base["mae"], 1e-9
    ) >= 0.10 and gain < 0.05:
        return "value_head_calibration_bottleneck"
    return "inconclusive_no_value_head_authorization"


def final_classification(preliminary: str, causal: dict[str, Any] | None) -> str:
    if (
        preliminary
        != "representation_sufficient_value_head_update_justified_pending_causal_puct"
    ):
        return preliminary
    if causal is None:
        return "inconclusive_no_value_head_authorization"
    deltas = [
        entry["causal"]["1200"]["normalized_final_margin_delta"]["mean"]
        for entry in causal.values()
    ]
    return (
        "representation_sufficient_value_head_update_justified"
        if all(value >= 0 for value in deltas)
        else "value_metrics_improve_but_search_does_not"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-labels", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.current) / "weights.json"
    if sha256(weights) != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")
    evaluator = ArtifactEvaluator(Path(args.current))
    candidates = []
    provenance = {}
    for name, path in SOURCES.items():
        if not path.is_file():
            raise RuntimeError(f"missing retained corpus: {path}")
        provenance[name] = {"path": str(path), "sha256": sha256(path)}
        candidates.extend(state_rows(read_rows(path), name, evaluator))
    retained_hashes = {row["state_hash"] for row in candidates}
    supplemental = supplemental_standard_start_states(
        evaluator, seed=args.seed, count=1024
    )
    candidates.extend(
        row for row in supplemental if row["state_hash"] not in retained_hashes
    )
    provenance["generated_opening_family_diagnostic"] = {
        "generation": "seeded disjoint standard-start opening prefixes",
        "families_requested": 1024,
    }
    state_path = workdir / "value_probe_states.jsonl"
    if state_path.is_file():
        rows = read_rows(state_path)
        if len(rows) != 4096 or len({row["state_hash"] for row in rows}) != 4096:
            raise RuntimeError("cached corpus is not the required 4,096 unique states")
    else:
        rows = select_corpus(candidates, seed=args.seed)
        assign_splits(rows, args.seed)
    for row in rows:
        game = KalahGame.from_state(row["state"])
        row["trunk"] = evaluator.extract_trunk(game).tolist()
    if not args.skip_labels:
        collect_search_signals(
            rows, Path(args.current), args.seed, args.workers, state_path
        )
        collect_labels(rows, Path(args.current), args.seed, args.workers, state_path)
    write_state_checkpoint(state_path, rows)
    manifest = {
        "schema": "azlite_frozen_trunk_value_sufficiency_v1",
        "state_count": len(rows),
        "current_weights_sha256": sha256(weights),
        "source_provenance": provenance,
        "split_unit": "source_game/opening_family",
        "split_counts": {
            s: sum(r["split"] == s for r in rows)
            for s in ("train", "validation", "test")
        },
    }
    (workdir / "value_probe_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    evaluation_domain = []
    if EVALUATION_SOURCE.is_file():
        evaluation_domain = state_rows(
            read_rows(EVALUATION_SOURCE), "evaluation_domain_diagnostic", evaluator
        )
        used = {row["state_hash"] for row in rows}
        evaluation_domain = [
            row for row in evaluation_domain if row["state_hash"] not in used
        ][:512]
        (workdir / "evaluation_domain_states.jsonl").write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in evaluation_domain
            ),
            encoding="utf-8",
        )
    if args.skip_labels:
        return 0
    train = [r for r in rows if r["split"] == "train"]
    summary = {
        "schema": manifest["schema"],
        "current_weights_sha256": sha256(weights),
        "manifest": manifest,
    }
    x = np.asarray([r["trunk"] for r in rows], np.float32)
    features = np.asarray(
        [encode_state(r["state"], input_encoding="kalah_v3") for r in rows], np.float32
    )
    y = np.asarray([r["D1200"]["normalized_final_margin"] for r in rows], np.float32)
    ti = np.asarray([r["split"] == "train" for r in rows])
    a, b = np.polyfit(np.asarray([r["current_value"] for r in train]), y[ti], 1)
    models = {
        "current_value": np.asarray([r["current_value"] for r in rows]),
        "affine_current_value": a * np.asarray([r["current_value"] for r in rows]) + b,
        "raw_feature_ridge": ridge_predict(ridge_fit(features[ti], y[ti]), features),
        "frozen_trunk_linear": ridge_predict(ridge_fit(x[ti], y[ti]), x),
    }
    head = fit_value_head(x[ti], y[ti], args.seed)
    with torch.no_grad():
        models["frozen_trunk_value_head"] = head(torch.from_numpy(x)).numpy()
    for split in ("validation", "test"):
        ix = np.asarray([r["split"] == split for r in rows])
        summary[split] = {
            name: metrics([r for r in rows if r["split"] == split], values[ix])
            for name, values in models.items()
        }
        summary[split]["slices"] = sliced_metrics(
            [r for r in rows if r["split"] == split],
            {name: values[ix] for name, values in models.items()},
        )
    test = [r for r in rows if r["split"] == "test"]
    test_ix = np.asarray([r["split"] == "test" for r in rows])
    ci = bootstrap_gain(
        test,
        models["current_value"][test_ix],
        models["frozen_trunk_value_head"][test_ix],
        args.seed,
    )
    summary["concordance_gain_bootstrap_95"] = ci
    preliminary = classify(
        summary["test"]["current_value"],
        summary["test"]["affine_current_value"],
        summary["test"]["frozen_trunk_value_head"],
        summary["test"]["raw_feature_ridge"],
        ci,
    )
    causal = (
        causal_puct_check(test, evaluator, head, args.seed)
        if preliminary.endswith("pending_causal_puct")
        else None
    )
    summary["causal_puct_check"] = causal
    summary["classification"] = final_classification(preliminary, causal)
    summary["target_stability"] = {
        "D768_D1200_outcome_agreement": float(
            np.mean([row["d768_d1200_outcome_agree"] for row in rows])
        ),
        "D768_D1200_normalized_margin_mae": float(
            np.mean([abs(row["d768_d1200_margin_difference"]) for row in rows])
        ),
    }
    summary["evaluation_domain"] = {
        "state_count": len(evaluation_domain),
        "excluded_from_probe_training": True,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    (workdir / "summary_metrics.json").write_text(payload, encoding="utf-8")
    (
        REPO_ROOT
        / "docs/data/alphazero-lite-frozen-trunk-value-sufficiency-summary.json"
    ).write_text(payload, encoding="utf-8")
    (
        REPO_ROOT / "docs/alphazero-lite-frozen-trunk-value-sufficiency-results.md"
    ).write_text(
        f"# Frozen-Trunk Value Sufficiency Audit\n\n**Classification:** `{summary['classification']}`\n\n```json\n{payload}\n```\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
