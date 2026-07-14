#!/usr/bin/env python3
"""Audit terminal-outcome replay balance and value-head-only updates.

This deliberately keeps sampling outside ``train.py``: compact replay arrays stay
unchanged while a deterministic index vector controls training mass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_seat_metrics,
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.self_play import encode_state  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    _count_parameters,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    load_jsonl,
    set_seed,
    train,
)

MODEL_TYPE = "residual_v3"
INPUT_ENCODING = "kalah_v3"
HIDDEN_SIZES = (96, 3)
BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-terminal-outcome-replay-balance-results.md"
)
SUMMARY_PATH = (
    REPO_ROOT / "docs/data/alphazero-lite-terminal-outcome-replay-balance-summary.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def outcome_sign(value: float) -> str:
    return "win" if value > 0 else "loss" if value < 0 else "draw"


def phase(row: dict[str, Any]) -> str:
    return str(row.get("phase", "unknown"))


def seat(row: dict[str, Any]) -> str:
    return f"player_{int(row.get('player', row.get('seat', 0)))}"


def swap_state(state: dict[str, Any]) -> dict[str, Any]:
    """Apply the player-label permutation (0 <-> 1) to a Kalah state."""
    return {
        "player_pits": list(state["opponent_pits"]),
        "opponent_pits": list(state["player_pits"]),
        "player_store": int(state["opponent_store"]),
        "opponent_store": int(state["player_store"]),
        "current_player": 1 - int(state["current_player"]),
    }


def swap_move(move: int) -> int:
    return int(move)


def swap_player(player: int) -> int:
    """Return the player identity under the 0 <-> 1 label permutation."""
    return 1 - int(player)


def swap_winner(winner: int | None) -> int | None:
    """Return the terminal winner identity under the label permutation."""
    return None if winner is None else 1 - int(winner)


swapped_winner = swap_winner


def outcome_for_recorded_player(winner: int | None, player: int) -> float:
    """Return a terminal outcome from the recorded absolute player's view."""
    return 0.0 if winner is None else (1.0 if int(winner) == int(player) else -1.0)


def transformed_replay_target(row: dict[str, Any]) -> float:
    """Derive a relabeled replay target without assuming a value sign flip."""
    return outcome_for_recorded_player(
        swap_winner(row.get("winner")), swap_player(int(row["player"]))
    )


def state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def apply_relative_move(state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    if not game.move(game.pit_index(move)):
        raise ValueError(f"illegal move {move}")
    return game.to_state()


def game_deciles(rows: list[dict[str, Any]]) -> dict[int, int]:
    lengths = sorted({int(row.get("game_length", 0)) for row in rows})
    if not lengths:
        return {}
    return {
        length: min(
            9, int(np.searchsorted(lengths, length, side="right") * 10 / len(lengths))
        )
        for length in lengths
    }


def correlation(values: list[float], targets: list[float]) -> float:
    if len(values) < 2 or np.std(values) == 0 or np.std(targets) == 0:
        return 0.0
    return float(np.corrcoef(values, targets)[0, 1])


def spearman(values: list[float], targets: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_ranks = np.argsort(np.argsort(np.asarray(values, dtype=float))).astype(float)
    target_ranks = np.argsort(np.argsort(np.asarray(targets, dtype=float))).astype(
        float
    )
    return correlation(value_ranks.tolist(), target_ranks.tolist())


def replay_weighting_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[int(row["game_index"])].append(row)
    deciles = game_deciles(rows)
    records = []
    for index, game_rows in sorted(by_game.items()):
        first = game_rows[0]
        records.append(
            {
                "game_index": index,
                "trajectory_hash": first.get("trajectory_hash"),
                "game_length": int(first.get("game_length", len(game_rows))),
                "winner": first.get("winner"),
                "final_margin": int(first.get("final_margin", 0))
                if int(first.get("player", 0)) == 0
                else -int(first.get("final_margin", 0)),
                "replay_rows": len(game_rows),
                "player_0_rows": sum(seat(r) == "player_0" for r in game_rows),
                "player_1_rows": sum(seat(r) == "player_1" for r in game_rows),
                "phase_counts": dict(Counter(phase(r) for r in game_rows)),
                "simulation_lane": str(first.get("simulations", "unknown")),
                "outcome_counts": dict(
                    Counter(outcome_sign(float(r["value"])) for r in game_rows)
                ),
            }
        )

    def summarize(key_fn):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(key_fn(row))].append(row)
        return {
            key: {
                "games": len({int(r["game_index"]) for r in values}),
                "replay_rows": len(values),
                "gradient_fraction": len(values) / max(len(rows), 1),
                "average_rows_per_game": len(values)
                / max(len({int(r["game_index"]) for r in values}), 1),
            }
            for key, values in sorted(buckets.items())
        }

    lengths = [float(r["game_length"]) for r in records]
    winners = [0.0 if r["winner"] is None else float(r["winner"]) for r in records]
    margins = [float(r["final_margin"]) for r in records]
    per_player_outcome = [
        (float(row["game_length"]), float(row["value"])) for row in rows
    ]
    strata = {
        "winner": summarize(lambda r: r.get("winner", "draw")),
        "player": summarize(seat),
        "outcome_sign": summarize(lambda r: outcome_sign(float(r["value"]))),
        "game_length_decile": summarize(
            lambda r: deciles[int(r.get("game_length", 0))]
        ),
        "phase": summarize(phase),
        "simulation_lane": summarize(lambda r: r.get("simulations", "unknown")),
        "winner_x_player": summarize(lambda r: f"{r.get('winner', 'draw')}:{seat(r)}"),
        "player_x_phase": summarize(lambda r: f"{seat(r)}:{phase(r)}"),
        "outcome_x_game_length_decile": summarize(
            lambda r: (
                f"{outcome_sign(float(r['value']))}:{deciles[int(r.get('game_length', 0))]}"
            )
        ),
    }
    return {
        "schema": "azlite_terminal_outcome_replay_weighting_audit_v1",
        "games": records,
        "strata": strata,
        "effective_game_weight": {
            "minimum": min((r["replay_rows"] for r in records), default=0),
            "maximum": max((r["replay_rows"] for r in records), default=0),
            "mean": statistics.fmean(r["replay_rows"] for r in records)
            if records
            else 0.0,
        },
        "correlations": {
            "game_length_vs_winner": correlation(lengths, winners),
            "game_length_vs_final_margin": correlation(lengths, margins),
            "game_length_vs_player_specific_outcome": correlation(
                *map(list, zip(*per_player_outcome))
            )
            if per_player_outcome
            else 0.0,
        },
    }


def deterministic_replay_indexes(
    rows: list[dict[str, Any]], mode: str, *, seed: int, epochs: int
) -> np.ndarray:
    """Return exactly one source-row index per intended sample, deterministically."""
    by_game: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_game[int(row["game_index"])].append(index)
    rng = random.Random(seed)
    indexes: list[int] = []
    draws = len(rows) * epochs
    games = sorted(by_game)
    if mode == "row_uniform":
        indexes = list(range(len(rows))) * epochs
        rng.shuffle(indexes)
        return np.asarray(indexes, dtype=np.int64)
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    if mode == "seat_outcome_balanced":
        for game, game_rows in by_game.items():
            # A game can have both seats; stratum is based on sampled row, below.
            for row_index in game_rows:
                strata[
                    (
                        seat(rows[row_index]),
                        outcome_sign(float(rows[row_index]["value"])),
                    )
                ].append(game)
        strata = {key: sorted(set(value)) for key, value in strata.items()}
    # Cycle shuffled complete groups.  This makes the claimed balancing exact up
    # to one sample, rather than relying on a random draw being approximately fair.
    game_cycle: list[int] = []
    stratum_cycle: list[tuple[str, str]] = []
    for draw in range(draws):
        if mode == "game_balanced":
            if not game_cycle:
                game_cycle = list(games)
                rng.shuffle(game_cycle)
            game = game_cycle.pop()
            indexes.append(rng.choice(by_game[game]))
        elif mode == "seat_outcome_balanced":
            available = sorted(strata)
            if not stratum_cycle:
                stratum_cycle = list(available)
                rng.shuffle(stratum_cycle)
            bucket = stratum_cycle.pop()
            game = rng.choice(strata[bucket])
            candidates = [
                i
                for i in by_game[game]
                if (seat(rows[i]), outcome_sign(float(rows[i]["value"]))) == bucket
            ]
            indexes.append(rng.choice(candidates or by_game[game]))
        else:
            raise ValueError(f"unsupported sampler {mode}")
    return np.asarray(indexes, dtype=np.int64)


def pr155_training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match PR #155's 85% game-prefix split before constructing replay indexes."""
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[int(row["game_index"])].append(row)
    game_indexes = sorted(by_game)
    train_game_count = max(1, int(len(game_indexes) * 0.85))
    return [row for game in game_indexes[:train_game_count] for row in by_game[game]]


def realized_sampling_frequencies(
    rows: list[dict[str, Any]], indexes: np.ndarray
) -> dict[str, Any]:
    """Describe actual gradient mass rather than only the sampler's intention."""
    selected = [rows[int(index)] for index in indexes]
    games = Counter(int(row["game_index"]) for row in selected)
    deciles = game_deciles(rows)
    values = list(games.values())

    def counts(key_fn):
        return dict(sorted(Counter(str(key_fn(row)) for row in selected).items()))

    mean = statistics.fmean(values) if values else 0.0
    return {
        "samples": len(selected),
        "sample_count_per_game": dict(sorted(games.items())),
        "minimum_samples_per_game": min(values, default=0),
        "maximum_samples_per_game": max(values, default=0),
        "coefficient_of_variation_across_games": (
            statistics.pstdev(values) / mean if mean else 0.0
        ),
        "sample_count_by_player": counts(seat),
        "sample_count_by_outcome_sign": counts(
            lambda row: outcome_sign(float(row["value"]))
        ),
        "sample_count_by_player_x_outcome": counts(
            lambda row: f"{seat(row)}:{outcome_sign(float(row['value']))}"
        ),
        "sample_count_by_game_length_decile": counts(
            lambda row: deciles[int(row.get("game_length", 0))]
        ),
        "available_player_x_outcome_strata": sorted(
            {f"{seat(row)}:{outcome_sign(float(row['value']))}" for row in rows}
        ),
        "missing_player_x_outcome_strata": sorted(
            {
                f"player_{player}:{outcome}"
                for player in (0, 1)
                for outcome in ("win", "loss", "draw")
            }
            - {f"{seat(row)}:{outcome_sign(float(row['value']))}" for row in rows}
        ),
        "rare_player_x_outcome_strata": sorted(
            key
            for key, count in Counter(
                f"{seat(row)}:{outcome_sign(float(row['value']))}" for row in rows
            ).items()
            if count < 10
        ),
    }


def symmetry_invariants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for row in rows:
        state = row["raw_state"]
        transformed = swap_state(state)
        if swap_state(transformed) != state:
            failures.append("state_transform_or_encoding_bug")
            break
        game = KalahGame.from_state(state)
        mapped = KalahGame.from_state(transformed)
        if sorted(swap_move(m) for m in game.possible_moves()) != sorted(
            mapped.possible_moves()
        ):
            failures.append("state_transform_or_encoding_bug")
            break
        for move in game.possible_moves():
            if swap_state(apply_relative_move(state, move)) != apply_relative_move(
                transformed, swap_move(move)
            ):
                failures.append("state_transform_or_encoding_bug")
                break
        if failures:
            break
        original_features = np.asarray(
            encode_state(state, input_encoding=INPUT_ENCODING)
        )
        expected_features = original_features[
            [*range(6, 12), *range(6), 13, 12, 14, *range(21, 27), *range(15, 21)]
        ]
        expected_features[14] = 1.0 - expected_features[14]
        if not np.allclose(
            encode_state(transformed, input_encoding=INPUT_ENCODING),
            expected_features,
        ):
            failures.append("state_transform_or_encoding_bug")
            break
        winner = row.get("winner")
        player = int(row.get("player", state["current_player"]))
        expected_target = outcome_for_recorded_player(winner, player)
        if not math.isclose(float(row["value"]), expected_target, abs_tol=1e-9):
            failures.append("target_perspective_bug")
            break
        transformed_player = swap_player(player)
        transformed_winner = swap_winner(winner)
        if (
            transformed_player != 1 - player
            or transformed_winner != (None if winner is None else 1 - int(winner))
            or transformed_replay_target(row) != expected_target
        ):
            failures.append("target_perspective_bug")
            break
        margin = float(row.get("final_margin", 0))
        if margin != 0 and outcome_sign(float(row["value"])) != outcome_sign(margin):
            failures.append("terminal_margin_sign_bug")
            break
        checked += 1
    return {
        "checked_rows": checked,
        "passes": not failures,
        "failures": sorted(set(failures)),
    }


def raw_value(
    evaluator: ArtifactEvaluator, state: dict[str, Any]
) -> tuple[np.ndarray, float]:
    return evaluator.evaluate(KalahGame.from_state(state))


def bucket(value: float, width: float = 0.2) -> str:
    lower = math.floor(value / width) * width
    return f"{lower:+.1f}:{lower + width:+.1f}"


def value_symmetry_residual(value: float, transformed_value: float) -> float:
    """Residual for an evaluator invariant under a player-label permutation."""
    return abs(float(value) - float(transformed_value))


def average_symmetry_residual(audit: dict[str, Any], key: str) -> float:
    values = [float(group[key]) for group in audit["breakdowns"]["seat"].values()]
    return statistics.fmean(values) if values else 0.0


def reproduction_metrics(
    rows: list[dict[str, Any]], reproduced: Path, reference: Path
) -> dict[str, Any]:
    """Compare values directly when a byte-identical training run is unavailable."""
    reproduced_evaluator = ArtifactEvaluator(reproduced)
    reference_evaluator = ArtifactEvaluator(reference)
    values = [
        (
            raw_value(reproduced_evaluator, row["raw_state"])[1],
            raw_value(reference_evaluator, row["raw_state"])[1],
        )
        for row in rows
    ]
    return {
        "max_absolute_value_difference": max(
            (abs(left - right) for left, right in values), default=0.0
        ),
        "value_sign_agreement": statistics.fmean(
            float((left > 0) == (right > 0)) for left, right in values
        )
        if values
        else 1.0,
        "value_mae_difference": statistics.fmean(
            abs(left - right) for left, right in values
        )
        if values
        else 0.0,
    }


def symmetry_audit(
    rows: list[dict[str, Any]], current: Path, candidate: Path, *, cap: int = 4096
) -> dict[str, Any]:
    invariant = symmetry_invariants(rows)
    sampled = rows[:cap]
    current_eval, candidate_eval = (
        ArtifactEvaluator(current),
        ArtifactEvaluator(candidate),
    )
    values: list[dict[str, Any]] = []
    deciles = game_deciles(rows)
    for row in sampled:
        transformed = swap_state(row["raw_state"])
        cp, cv = raw_value(current_eval, row["raw_state"])
        ctp, ctv = raw_value(current_eval, transformed)
        pp, pv = raw_value(candidate_eval, row["raw_state"])
        ptp, ptv = raw_value(candidate_eval, transformed)
        mapped_policy = np.asarray(ptp)[[swap_move(i) for i in range(6)]]
        values.append(
            {
                "seat": seat(row),
                "phase": phase(row),
                "outcome": outcome_sign(float(row["value"])),
                "decile": str(deciles[int(row.get("game_length", 0))]),
                "current_bucket": bucket(cv),
                "delta_bucket": bucket(pv - cv),
                "current_residual": value_symmetry_residual(cv, ctv),
                "candidate_residual": value_symmetry_residual(pv, ptv),
                "delta_residual": abs((pv - cv) - (ptv - ctv)),
                "delta": pv - cv,
                "sign_consistent": float((pv > 0) == (float(row["value"]) > 0)),
                "policy_symmetry_l1": float(np.abs(pp - mapped_policy).sum()),
            }
        )

    def grouped(keys: tuple[str, ...]) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in values:
            groups[":".join(item[key] for key in keys)].append(item)
        return {
            key: {
                "rows": len(items),
                "current_symmetry_residual": statistics.fmean(
                    x["current_residual"] for x in items
                ),
                "candidate_symmetry_residual": statistics.fmean(
                    x["candidate_residual"] for x in items
                ),
                "delta_symmetry_residual": statistics.fmean(
                    x["delta_residual"] for x in items
                ),
                "candidate_current_delta_mean": statistics.fmean(
                    x["delta"] for x in items
                ),
                "value_sign_consistency": statistics.fmean(
                    x["sign_consistent"] for x in items
                ),
                "policy_symmetry_l1": statistics.fmean(
                    x["policy_symmetry_l1"] for x in items
                ),
            }
            for key, items in sorted(groups.items())
        }

    return {
        "schema": "azlite_terminal_outcome_symmetry_audit_v1",
        "invariants": invariant,
        "rows": len(values),
        "breakdowns": {
            "seat": grouped(("seat",)),
            "phase": grouped(("phase",)),
            "outcome": grouped(("outcome",)),
            "length_decile": grouped(("decile",)),
            "current_value_bucket": grouped(("current_bucket",)),
            "value_delta_bucket": grouped(("delta_bucket",)),
        },
    }


def freeze_value_head(model: PolicyValueNet) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert model.value_hidden_layer is not None
    for parameter in (
        *model.value_hidden_layer.parameters(),
        *model.value_head.parameters(),
    ):
        parameter.requires_grad = True


def checkpoint_audit(reference: Path, candidate: Path) -> dict[str, Any]:
    with np.load(reference) as old, np.load(candidate) as new:
        changed = [key for key in old.files if not np.array_equal(old[key], new[key])]
    allowed = {"w_value_hidden", "b_value_hidden", "w_value", "b_value"}
    return {
        "passes": set(changed) <= allowed and bool(changed),
        "changed_keys": changed,
        "policy_unchanged": not any("policy" in key for key in changed),
        "trunk_unchanged": not any(
            key.startswith(("w_input", "b_input", "w_residual", "b_residual"))
            for key in changed
        ),
    }


def train_lane(
    rows: list[dict[str, Any]],
    init_checkpoint: Path,
    lane_dir: Path,
    *,
    name: str,
    sampler: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    lane_dir.mkdir(parents=True, exist_ok=True)
    replay = lane_dir / "replay.jsonl"
    replay.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    x, p, v = load_jsonl(replay)
    set_seed(seed)
    model = PolicyValueNet(
        HIDDEN_SIZES, MODEL_TYPE, input_size_for_encoding(INPUT_ENCODING)
    )
    load_checkpoint_into_model(model, init_checkpoint)
    freeze_value_head(model)
    total, trainable = _count_parameters(model)
    indexes = deterministic_replay_indexes(rows, sampler, seed=seed, epochs=epochs)
    policy_loss, value_loss, best = train(
        model,
        x,
        p,
        v,
        indexes,
        epochs=epochs,
        batch_size=512,
        lr=5e-5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        value_loss_weight=1.25,
        value_loss="huber",
        huber_delta=1.0,
        val_split=0.1,
        grad_clip=1.0,
        save_top_k=0,
        lr_scheduler="none",
    )
    checkpoint = lane_dir / "checkpoint.npz"
    np.savez(checkpoint, **checkpoint_from_model(model))
    artifact = lane_dir / "artifact"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(artifact),
            "--version",
            name,
            "--model-type",
            MODEL_TYPE,
            "--input-encoding",
            INPUT_ENCODING,
            "--rules-version",
            "kalah_v3",
            "--policy-loss",
            str(policy_loss),
            "--value-loss",
            str(value_loss),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "name": name,
        "artifact": str(artifact),
        "weights_sha256": sha256(artifact / "weights.json"),
        "checkpoint": str(checkpoint),
        "sampler": sampler,
        "realized_sampling_frequencies": realized_sampling_frequencies(rows, indexes),
        "epochs": epochs,
        "training": {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "best_val_loss": best,
            "total_params": total,
            "trainable_params": trainable,
        },
        "parameter_audit": checkpoint_audit(init_checkpoint, checkpoint),
    }


def value_metrics(
    rows: list[dict[str, Any]], artifact: Path, current: Path | None = None
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact)
    current_eval = ArtifactEvaluator(current) if current else None
    records = []
    for row in rows:
        _, value = raw_value(evaluator, row["raw_state"])
        delta = (
            value - raw_value(current_eval, row["raw_state"])[1]
            if current_eval
            else 0.0
        )
        records.append((row, value, delta))

    def metrics(items):
        targets, values = [float(x[0]["value"]) for x in items], [x[1] for x in items]
        calibration: dict[str, dict[str, float | int]] = {}
        for lower in np.arange(-1.0, 1.0, 0.2):
            selected = [
                (value, target)
                for value, target in zip(values, targets, strict=True)
                if lower <= value < lower + 0.2
            ]
            if selected:
                calibration[f"{lower:+.1f}:{lower + 0.2:+.1f}"] = {
                    "rows": len(selected),
                    "mean_prediction": statistics.fmean(value for value, _ in selected),
                    "mean_target": statistics.fmean(target for _, target in selected),
                }
        return {
            "rows": len(items),
            "mae": statistics.fmean(abs(a - b) for a, b in zip(values, targets))
            if items
            else 0.0,
            "sign_accuracy": statistics.fmean(
                float((a > 0) == (b > 0)) for a, b in zip(values, targets)
            )
            if items
            else 0.0,
            "pearson": correlation(values, targets),
            "spearman": spearman(values, targets),
            "mean_delta": statistics.fmean(x[2] for x in items) if items else 0.0,
            "calibration": calibration,
        }

    groups = {
        "overall": records,
        "player_0": [x for x in records if seat(x[0]) == "player_0"],
        "player_1": [x for x in records if seat(x[0]) == "player_1"],
    }
    groups.update(
        {
            f"phase:{key}": [x for x in records if phase(x[0]) == key]
            for key in ("opening", "mid", "late")
        }
    )
    return {key: metrics(items) for key, items in groups.items()}


def probe_sample(
    rows: list[dict[str, Any]], *, seed: int, cap: int = 256
) -> list[dict[str, Any]]:
    """Select unique probe states across replay covariates, not replay order."""
    deciles = game_deciles(rows)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        key = state_key(row["raw_state"])
        if key in seen:
            continue
        seen.add(key)
        buckets[
            (
                seat(row),
                phase(row),
                outcome_sign(float(row["value"])),
                str(deciles[int(row.get("game_length", 0))]),
                bucket(float(row["value"])),
            )
        ].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    pools = {key: list(value) for key, value in buckets.items()}
    for values in pools.values():
        rng.shuffle(values)
    while len(selected) < min(cap, len(seen)):
        progressed = False
        for key in sorted(pools):
            if pools[key] and len(selected) < cap:
                selected.append(pools[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def wilson_interval(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}
    z = 1.96
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    margin = z * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * total)) / total)
    return {
        "lower": max(0.0, (center - margin) / denominator),
        "upper": min(1.0, (center + margin) / denominator),
    }


def search_probe(
    rows: list[dict[str, Any]], candidate: Path, current: Path, args: argparse.Namespace
) -> dict[str, Any]:
    sample = probe_sample(rows, seed=args.seed)
    schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    ce, pe = ArtifactEvaluator(current), ArtifactEvaluator(candidate)
    out: dict[str, Any] = {}
    for budget in BUDGETS:
        cs, rs = (int(x) for x in budget.split(":"))
        artifact_changed: list[float] = []
        budget_changed: list[float] = []
        kls: list[float] = []
        budget_kls: list[float] = []
        deltas: list[float] = []
        budget_deltas: list[float] = []
        child_q_deltas: list[float] = []
        budget_child_q_deltas: list[float] = []
        by_seat: dict[str, list[float]] = defaultdict(list)
        by_phase: dict[str, list[float]] = defaultdict(list)
        cpuct = resolve_budget_cpuct(
            schedule=schedule,
            challenger_simulations=cs,
            current_simulations=rs,
            default_c_puct=args.default_c_puct,
        )
        for i, row in enumerate(sample):
            opts = {
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "deterministic",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
            }
            left = evaluate_artifact_position(
                evaluator=pe,
                state=row["raw_state"],
                simulations=cs,
                seed=args.seed + i,
                c_puct=cpuct,
                search_options=opts,
            )
            right = evaluate_artifact_position(
                evaluator=ce,
                state=row["raw_state"],
                simulations=cs,
                seed=args.seed + i,
                c_puct=cpuct,
                search_options=opts,
            )
            budget_right = evaluate_artifact_position(
                evaluator=ce,
                state=row["raw_state"],
                simulations=rs,
                seed=args.seed + i,
                c_puct=cpuct,
                search_options=opts,
            )
            flag = float(left["selected_move"] != right["selected_move"])
            budget_flag = float(right["selected_move"] != budget_right["selected_move"])
            artifact_changed.append(flag)
            budget_changed.append(budget_flag)
            by_seat[seat(row)].append(flag)
            by_phase[phase(row)].append(flag)
            a, b = np.asarray(left["visits"], float), np.asarray(right["visits"], float)
            a = (a + 1e-8) / sum(a + 1e-8)
            b = (b + 1e-8) / sum(b + 1e-8)
            kls.append(float(np.sum(b * np.log(b / a))))
            bbudget = np.asarray(budget_right["visits"], float)
            bbudget = (bbudget + 1e-8) / sum(bbudget + 1e-8)
            budget_kls.append(float(np.sum(bbudget * np.log(bbudget / b))))
            deltas.append(
                float(left.get("search_root_value", left["value"]))
                - float(right.get("search_root_value", right["value"]))
            )
            budget_deltas.append(
                float(right.get("search_root_value", right["value"]))
                - float(budget_right.get("search_root_value", budget_right["value"]))
            )
            left_q = {
                int(x["move"]): float(x["q_value"]) for x in left.get("child_stats", [])
            }
            right_q = {
                int(x["move"]): float(x["q_value"])
                for x in right.get("child_stats", [])
            }
            budget_q = {
                int(x["move"]): float(x["q_value"])
                for x in budget_right.get("child_stats", [])
            }
            shared = sorted(set(left_q) & set(right_q))
            if shared:
                child_q_deltas.append(
                    statistics.fmean(abs(left_q[m] - right_q[m]) for m in shared)
                )
            shared_budget = sorted(set(right_q) & set(budget_q))
            if shared_budget:
                budget_child_q_deltas.append(
                    statistics.fmean(
                        abs(right_q[m] - budget_q[m]) for m in shared_budget
                    )
                )
        rate = statistics.fmean(artifact_changed) if artifact_changed else 0.0
        out[budget] = {
            "rows": len(sample),
            "artifact_changed_move_rate": rate,
            "artifact_wilson_95": wilson_interval(
                int(sum(artifact_changed)), len(artifact_changed)
            ),
            "artifact_root_visit_kl": statistics.fmean(kls) if kls else 0.0,
            "artifact_root_value_delta": statistics.fmean(deltas) if deltas else 0.0,
            "artifact_child_q_delta": statistics.fmean(child_q_deltas)
            if child_q_deltas
            else None,
            "budget_only_changed_move_rate": statistics.fmean(budget_changed)
            if budget_changed
            else 0.0,
            "budget_only_wilson_95": wilson_interval(
                int(sum(budget_changed)), len(budget_changed)
            ),
            "budget_only_root_visit_kl": statistics.fmean(budget_kls)
            if budget_kls
            else 0.0,
            "budget_only_root_value_delta": statistics.fmean(budget_deltas)
            if budget_deltas
            else 0.0,
            "budget_only_child_q_delta": statistics.fmean(budget_child_q_deltas)
            if budget_child_q_deltas
            else None,
            "changed_rate_by_player": {
                k: statistics.fmean(v) for k, v in by_seat.items()
            },
            "changed_rate_by_phase": {
                k: statistics.fmean(v) for k, v in by_phase.items()
            },
        }
    return out


def run_stage(
    suite: Path,
    candidates: dict[str, Path],
    current: Path,
    args: argparse.Namespace,
    workdir: Path,
) -> dict[str, Any]:
    entries = read_jsonl(suite)
    result = {}
    for name, candidate in candidates.items():
        by_budget = {}
        for budget in BUDGETS:
            cs, rs = (int(x) for x in budget.split(":"))
            target = workdir / name / budget.replace(":", "_")
            target.mkdir(parents=True, exist_ok=True)
            games = []
            for start in (0, 1):
                opening = target / f"openings_{start}.jsonl"
                opening.write_text(
                    "".join(
                        json.dumps({"prefix_moves": x["prefix_moves"]}) + "\n"
                        for x in entries
                    ),
                    encoding="utf-8",
                )
                run_arena(
                    challenger=str(candidate),
                    current=str(current),
                    challenger_sims=cs,
                    current_sims=rs,
                    games=len(entries) * 2,
                    seed=args.seed,
                    workers=args.workers,
                    out_json=str(target / f"arena_{start}.json"),
                    out_jsonl=str(target / f"games_{start}.jsonl"),
                    opening_prefixes_jsonl=str(opening),
                    challenger_starts=start,
                    games_per_opening=2,
                    root_policy_mode="deterministic",
                    normalize_values=False,
                    c_puct=resolve_budget_cpuct(
                        schedule=parse_cpuct_schedule_json(args.cpuct_schedule),
                        challenger_simulations=cs,
                        current_simulations=rs,
                        default_c_puct=args.default_c_puct,
                    ),
                    tactical_root_bias=0.0,
                )
                games.extend(parse_game_jsonl(str(target / f"games_{start}.jsonl")))
            by_budget[budget] = compute_seat_metrics(games)
        result[name] = {"by_budget": by_budget}
    reference = result["current_ref"]["by_budget"]
    for name, row in result.items():
        for budget, metric in row["by_budget"].items():
            metric["candidate_minus_current_ds"] = (
                metric["ds"] - reference[budget]["ds"]
            )
    return result


def classify(summary: dict[str, Any]) -> str:
    failures = summary["symmetry_audit"]["invariants"]["failures"]
    if failures:
        return failures[0]
    if "pr155_value_training_not_reproducible" in summary.get("stop_reasons", []):
        return "pr155_value_training_not_reproducible"
    lanes = summary.get("lanes", {})
    balanced = [x for x in lanes.values() if x.get("probe_pass")]
    if not balanced:
        return "no_balanced_lane_passed_probes"
    if "medium" not in summary:
        return "replay_balance_experiment_incomplete"
    medium = summary["medium"]
    candidates = [x for name, x in medium.items() if name != "current_ref"]
    if candidates and all(
        x["by_budget"]["384:256"]["candidate_minus_current_ds"] < 0.03
        for x in candidates
    ):
        return "balanced_value_safe_but_strength_neutral"
    if any(
        x["by_budget"]["384:256"]["candidate_minus_current_ds"] > 0
        and any(
            x["by_budget"][budget]["candidate_minus_current_ds"] < limit
            for budget, limit in (
                ("768:768", -0.03),
                ("1200:1200", -0.03),
                ("1200:256", -0.03),
            )
        )
        for x in candidates
    ):
        return "balanced_value_tradeoff"
    if any(x.get("fixed_large_pass") for x in summary.get("fixed_large", {}).values()):
        return "game_balanced_value_candidate"
    return "terminal_outcome_value_head_path_closed"


def medium_pass(metrics: dict[str, Any]) -> bool:
    values = metrics["by_budget"]
    return (
        values["384:256"]["candidate_minus_current_ds"] >= 0.03
        and values["768:768"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:1200"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:256"]["candidate_minus_current_ds"] >= -0.03
    )


def fixed_large_pass(metrics: dict[str, Any]) -> bool:
    values = metrics["by_budget"]
    return (
        values["384:256"]["candidate_minus_current_ds"] >= 0.05
        and values["768:768"]["candidate_minus_current_ds"] >= -0.05
        and values["1200:1200"]["candidate_minus_current_ds"] >= -0.03
        and values["1200:256"]["candidate_minus_current_ds"] >= -0.03
    )


def report(summary: dict[str, Any]) -> str:
    weighting = summary["replay_weighting_audit"]
    lines = [
        "# AlphaZero-Lite Terminal-Outcome Replay-Balance Results",
        "",
        f"- classification: `{summary['classification']}`",
        "",
        "## Replay Weighting",
        "",
        "| stratum | rows | gradient fraction |",
        "|---|---:|---:|",
    ]
    for key, value in weighting["strata"]["player"].items():
        lines.append(
            f"| player {key} | {value['replay_rows']} | {value['gradient_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Symmetry",
        "",
        f"- checked rows: `{summary['symmetry_audit']['invariants']['checked_rows']}`",
        f"- invariant failures: `{', '.join(summary['symmetry_audit']['invariants']['failures']) or 'none'}`",
        "",
        "## Training Lanes",
        "",
        "| lane | sampler | epochs | probe pass | stop reasons |",
        "|---|---|---:|---:|---|",
    ]
    for name, lane in summary.get("lanes", {}).items():
        lines.append(
            f"| {name} | {lane.get('sampler')} | {lane.get('epochs')} | {lane.get('probe_pass', False)} | {', '.join(lane.get('abort_reasons', [])) or 'none'} |"
        )
    reproduction = (
        summary.get("lanes", {}).get("row_uniform_repro", {}).get("reproduction")
    )
    if reproduction:
        lines += [
            "",
            "## Row-Uniform Reproduction",
            "",
            f"- exact weights match: `{reproduction['matches_expected_weights']}`",
            f"- max absolute value difference: `{reproduction['max_absolute_value_difference']:.8f}`",
            f"- value sign agreement: `{reproduction['value_sign_agreement']:.6f}`",
            f"- value MAE difference: `{reproduction['value_mae_difference']:.8f}`",
            f"- failure reasons: `{', '.join(reproduction.get('reasons', [])) or 'none'}`",
        ]
    lines += [
        "",
        "## Value And Search Diagnostics",
        "",
        "| lane | group | MAE | sign accuracy | Pearson | Spearman | mean delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, lane in summary.get("lanes", {}).items():
        for group in (
            "overall",
            "player_0",
            "player_1",
            "phase:opening",
            "phase:mid",
            "phase:late",
        ):
            values = lane["value_metrics"].get(group, {})
            lines.append(
                f"| {name} | {group} | {values.get('mae', 0):.4f} | {values.get('sign_accuracy', 0):.4f} | {values.get('pearson', 0):.4f} | {values.get('spearman', 0):.4f} | {values.get('mean_delta', 0):+.4f} |"
            )
    lines += [
        "",
        "| lane | budget | artifact changed rate (95% CI) | budget-only changed rate (95% CI) | artifact KL | budget KL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, lane in summary.get("lanes", {}).items():
        for budget, probe in lane["search_probes"].items():
            artifact_ci = probe["artifact_wilson_95"]
            budget_ci = probe["budget_only_wilson_95"]
            lines.append(
                f"| {name} | {budget} | {probe['artifact_changed_move_rate']:.4f} ({artifact_ci['lower']:.4f}, {artifact_ci['upper']:.4f}) | {probe['budget_only_changed_move_rate']:.4f} ({budget_ci['lower']:.4f}, {budget_ci['upper']:.4f}) | {probe['artifact_root_visit_kl']:.4f} | {probe['budget_only_root_visit_kl']:.4f} |"
            )
    lines += [
        "",
        "## Realized Sampling",
        "",
        "| lane | samples | game min/max | game CV | player/outcome strata |",
        "|---|---:|---:|---:|---|",
    ]
    for name, lane in summary.get("lanes", {}).items():
        sampling = lane["realized_sampling_frequencies"]
        strata = ", ".join(
            f"{key}={value}"
            for key, value in sampling["sample_count_by_player_x_outcome"].items()
        )
        lines.append(
            f"| {name} | {sampling['samples']} | {sampling['minimum_samples_per_game']}/{sampling['maximum_samples_per_game']} | {sampling['coefficient_of_variation_across_games']:.4f} | {strata} |"
        )
    lines += ["", "## Staged Strength", ""]
    if "medium" in summary:
        lines += [
            "### Medium",
            "",
            "| lane | budget | candidate-current DS |",
            "|---|---|---:|",
        ]
        for name, lane in summary["medium"].items():
            if name == "current_ref":
                continue
            for budget, values in lane["by_budget"].items():
                lines.append(
                    f"| {name} | {budget} | {values['candidate_minus_current_ds']:+.4f} |"
                )
    else:
        lines.append(
            "- medium: not run because no balanced lane passed corrected probes"
        )
    if "fixed_large" not in summary:
        lines.append("- fixed-large and held-out: not reached")
    lines += [
        "",
        "## Stop Reasons",
        "",
        *[f"- {x}" for x in summary.get("stop_reasons", [])],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--pr155-candidate", required=True)
    parser.add_argument("--expected-pr155-candidate-sha256", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument(
        "--lanes",
        default="row_uniform_repro,game_balanced_e1,game_balanced_e2,game_seat_balanced_e1,game_seat_balanced_e2",
    )
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    current = Path(args.current)
    candidate = Path(args.pr155_candidate)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if args.tactical_root_bias != 0.0:
        raise ValueError("tactical_root_bias must be 0")
    if sha256(current / "weights.json") != args.expected_current_weights_sha256:
        raise RuntimeError("current weights hash mismatch")
    if sha256(candidate / "weights.json") != args.expected_pr155_candidate_sha256:
        raise RuntimeError("PR #155 candidate weights hash mismatch")
    rows = read_jsonl(Path(args.replay))
    weighting = replay_weighting_audit(rows)
    write_json(workdir / "replay_weighting_audit.json", weighting)
    symmetry = symmetry_audit(rows, current, candidate)
    write_json(workdir / "symmetry_audit.json", symmetry)
    summary = {
        "schema": "azlite_terminal_outcome_replay_balance_v1",
        "inputs": {
            "current_weights_sha256": sha256(current / "weights.json"),
            "pr155_weights_sha256": sha256(candidate / "weights.json"),
        },
        "replay_weighting_audit": weighting,
        "symmetry_audit": symmetry,
        "lanes": {},
        "stop_reasons": [],
    }
    if not symmetry["invariants"]["passes"]:
        summary["stop_reasons"].append(
            "target or symmetry invariant failed; training skipped"
        )
    else:
        init = workdir / "current_init.npz"
        materialize_weights_json_checkpoint(
            weights_path=current / "weights.json", out_path=init
        )
        lane_map = {
            "row_uniform_repro": ("row_uniform", 2),
            "game_balanced_e1": ("game_balanced", 1),
            "game_balanced_e2": ("game_balanced", 2),
            "game_seat_balanced_e1": ("seat_outcome_balanced", 1),
            "game_seat_balanced_e2": ("seat_outcome_balanced", 2),
        }
        training_rows = pr155_training_rows(rows)
        evaluation_rows = rows[:4096]
        current_metrics = value_metrics(evaluation_rows, current)
        for name in [x.strip() for x in args.lanes.split(",") if x.strip()]:
            sampler, epochs = lane_map[name]
            lane = train_lane(
                training_rows,
                init,
                workdir / name,
                name=name,
                sampler=sampler,
                epochs=epochs,
                seed=args.seed,
            )
            artifact = Path(lane["artifact"])
            lane["value_metrics"] = value_metrics(evaluation_rows, artifact, current)
            lane["symmetry_metrics"] = symmetry_audit(rows, current, artifact)
            lane["search_probes"] = search_probe(rows, artifact, current, args)
            p0 = lane["value_metrics"]["player_0"]
            p1 = lane["value_metrics"]["player_1"]
            search = lane["search_probes"]
            if name == "row_uniform_repro":
                reference_search = search_probe(rows, candidate, current, args)
                lane["reproduction"] = {
                    "expected_weights_sha256": args.expected_pr155_candidate_sha256,
                    "matches_expected_weights": lane["weights_sha256"]
                    == args.expected_pr155_candidate_sha256,
                    **reproduction_metrics(evaluation_rows, artifact, candidate),
                    "search_changed_rate_difference": {
                        budget: abs(
                            search[budget]["artifact_changed_move_rate"]
                            - reference_search[budget]["artifact_changed_move_rate"]
                        )
                        for budget in BUDGETS
                    },
                }
                lane["reproduction"]["reasons"] = [
                    reason
                    for reason, passed in {
                        "weights_not_identical": lane["reproduction"][
                            "matches_expected_weights"
                        ],
                        "value_predictions_not_reproduced": lane["reproduction"][
                            "max_absolute_value_difference"
                        ]
                        <= 1e-5
                        and lane["reproduction"]["value_mae_difference"] <= 1e-6
                        and lane["reproduction"]["value_sign_agreement"] >= 0.999,
                        "search_changed_rate_not_reproduced": max(
                            lane["reproduction"][
                                "search_changed_rate_difference"
                            ].values(),
                            default=0.0,
                        )
                        <= 0.01,
                    }.items()
                    if not passed
                ]
                lane["reproduction"]["passes"] = bool(
                    lane["reproduction"]["matches_expected_weights"]
                    or (
                        lane["reproduction"]["max_absolute_value_difference"] <= 1e-5
                        and lane["reproduction"]["value_mae_difference"] <= 1e-6
                        and lane["reproduction"]["value_sign_agreement"] >= 0.999
                        and max(
                            lane["reproduction"][
                                "search_changed_rate_difference"
                            ].values(),
                            default=0.0,
                        )
                        <= 0.01
                    )
                )
                if not lane["reproduction"]["passes"]:
                    summary["stop_reasons"].append(
                        "pr155_value_training_not_reproducible"
                    )
            imbalance = abs(p0["mean_delta"] - p1["mean_delta"])
            baseline_imbalance = abs(
                summary["lanes"]
                .get("row_uniform_repro", {})
                .get("value_metrics", {})
                .get("player_0", {})
                .get("mean_delta", float("inf"))
                - summary["lanes"]
                .get("row_uniform_repro", {})
                .get("value_metrics", {})
                .get("player_1", {})
                .get("mean_delta", 0.0)
            )
            reasons = []
            checks = {
                "parameter_audit_failed": lane["parameter_audit"]["passes"],
                "symmetry_invariant_failed": lane["symmetry_metrics"]["invariants"][
                    "passes"
                ],
                "terminal_outcome_mae_not_improved": lane["value_metrics"]["overall"][
                    "mae"
                ]
                < current_metrics["overall"]["mae"],
                "player_0_sign_regressed": p0["sign_accuracy"]
                >= current_metrics["player_0"]["sign_accuracy"],
                "player_1_sign_regressed": p1["sign_accuracy"]
                >= current_metrics["player_1"]["sign_accuracy"],
                "correlation_not_improved": lane["value_metrics"]["overall"]["pearson"]
                > current_metrics["overall"]["pearson"]
                or lane["value_metrics"]["overall"]["spearman"]
                > current_metrics["overall"]["spearman"],
                "player_delta_imbalance_not_lower": name == "row_uniform_repro"
                or imbalance < baseline_imbalance,
                "artifact_768_768_changed_rate": search["768:768"][
                    "artifact_changed_move_rate"
                ]
                <= 0.08,
                "artifact_1200_1200_changed_rate": search["1200:1200"][
                    "artifact_changed_move_rate"
                ]
                <= 0.08,
                "artifact_1200_256_changed_rate": search["1200:256"][
                    "artifact_changed_move_rate"
                ]
                <= 0.10,
                "symmetry_residual_materially_worse": average_symmetry_residual(
                    lane["symmetry_metrics"], "candidate_symmetry_residual"
                )
                <= average_symmetry_residual(
                    lane["symmetry_metrics"], "current_symmetry_residual"
                )
                + 0.02,
            }
            reasons = [reason for reason, passed in checks.items() if not passed]
            lane["value_delta_imbalance"] = imbalance
            lane["abort_reasons"] = reasons
            lane["probe_pass"] = not reasons and name != "row_uniform_repro"
            summary["lanes"][name] = lane
        passing = {
            "current_ref": current,
            "row_uniform_repro": Path(
                summary["lanes"]["row_uniform_repro"]["artifact"]
            ),
            **{
                name: Path(x["artifact"])
                for name, x in summary["lanes"].items()
                if x["probe_pass"]
            },
        }
        if len(passing) > 1:
            summary["medium"] = run_stage(
                Path(args.medium_suite), passing, current, args, workdir / "medium"
            )
            carry = [
                name
                for name, metrics in summary["medium"].items()
                if name not in {"current_ref", "row_uniform_repro"}
                and medium_pass(metrics)
            ][:2]
            if carry:
                fixed_candidates = {
                    "current_ref": current,
                    **{
                        name: Path(summary["lanes"][name]["artifact"]) for name in carry
                    },
                }
                summary["fixed_large"] = run_stage(
                    Path(args.fixed_large_suite),
                    fixed_candidates,
                    current,
                    args,
                    workdir / "fixed_large",
                )
                for name in carry:
                    summary["fixed_large"][name]["fixed_large_pass"] = fixed_large_pass(
                        summary["fixed_large"][name]
                    )
            else:
                summary["stop_reasons"].append(
                    "no balanced lane passed medium carry criteria"
                )
        else:
            summary["stop_reasons"].append(
                "no balanced lane passed probes; medium evaluation skipped"
            )
    summary["classification"] = classify(summary)
    write_json(workdir / "summary_metrics.json", summary)
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
