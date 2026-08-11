#!/usr/bin/env python3
# ruff: noqa: E401, E402, E701, E702
"""Paired noisy-versus-denoised PUCT policy-target ablation.

The canonical replay is generated once with noisy gameplay.  Every row then
receives a second, deterministic no-Dirichlet target search; the two training
views differ only in ``policy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    stable_hash,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    build_manifest,
    compare_runs,
    train_fixed_manifest,
    value_metrics,
)
from ml.alphazero_lite.run_distribution_aligned_selfplay_iteration import _decoded_state
from ml.alphazero_lite.run_distribution_aligned_selfplay_iteration import (
    paired_strength_screen,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import bootstrap_ci
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    encode_state,
    policy_from_visits,
    run_self_play_worker,
)

EXPECTED_CURRENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
DEFAULT_WORKDIR = Path("/tmp/azlite_policy_target_noise_ablation")
SCHEMA = "azlite_policy_target_noise_ablation_v1"
RUNTIME_PROFILE = {
    "tactical_root_bias": 0.0,
    "default_c_puct": 1.25,
    "c_puct_schedule": {"768:768": 0.90},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def phase_for_state(state: dict[str, Any]) -> str:
    remaining = sum(state["player_pits"]) + sum(state["opponent_pits"])
    return "opening" if remaining > 24 else "midgame" if remaining > 12 else "late"


def entropy(policy: list[float] | np.ndarray) -> float:
    values = np.asarray(policy, dtype=float)
    return float(-np.sum(values[values > 0] * np.log(values[values > 0])))


def top_move(policy: list[float] | np.ndarray, legal_moves: list[int]) -> int:
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def visit_margin(policy: list[float] | np.ndarray, legal_moves: list[int]) -> float:
    ranked = sorted((float(policy[move]) for move in legal_moves), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]


def _probe_candidates(rows: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    result = []
    for index, row in enumerate(rows):
        state = row.get("raw_state") or (
            _decoded_state(row)
            if "state" in row and isinstance(row["state"], list)
            else row["state"]
        )
        legal = row.get("legal_moves") or KalahGame.from_state(state).possible_moves()
        policy = row.get(
            "policy", [1 / len(legal) if move in legal else 0 for move in range(6)]
        )
        result.append(
            {
                "state": state,
                "state_hash": stable_hash(state),
                "source_domain": domain,
                "source_index": index,
                "player": int(state["current_player"]),
                "phase": phase_for_state(state),
                "legal_moves": [int(move) for move in legal],
                "legal_move_count": len(legal),
                "current_policy_entropy": entropy(policy),
            }
        )
    return result


def select_probe_states(
    pilot_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically select disjoint, approximately stratified diagnostic states."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    # Reserve diagnostic evaluation identities first: standard-start pilot
    # trajectories can revisit evaluation openings, but diagnostics must remain
    # disjoint from the eventual training probe sample.
    for domain, candidates, count in (
        (
            "evaluation_opening_diagnostic",
            _probe_candidates(evaluation_rows, "evaluation_opening_diagnostic"),
            384,
        ),
        (
            "pr176_standard_start_pilot",
            _probe_candidates(pilot_rows, "pr176_standard_start_pilot"),
            384,
        ),
    ):
        buckets: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
        for row in candidates:
            buckets[
                (
                    row["player"],
                    row["phase"],
                    row["legal_move_count"],
                    int(row["current_policy_entropy"] * 2),
                )
            ].append(row)
        rng = random.Random(stable_seed(seed, domain))
        queues = deque()
        for key in sorted(buckets, key=str):
            values = list(buckets[key])
            rng.shuffle(values)
            queues.append(deque(values))
        domain_selected = []
        while queues and len(domain_selected) < count:
            values = queues.popleft()
            while values:
                row = values.popleft()
                if row["state_hash"] not in used:
                    used.add(row["state_hash"])
                    domain_selected.append(row)
                    break
            if values:
                queues.append(values)
        if len(domain_selected) != count:
            raise RuntimeError(
                f"only {len(domain_selected)} unique {domain} states; need {count}"
            )
        selected.extend(domain_selected)
    manifest = {
        "schema": "azlite_target_probe_v1",
        "seed": seed,
        "state_count": len(selected),
        "domain_counts": dict(Counter(row["source_domain"] for row in selected)),
        "state_hashes": [row["state_hash"] for row in selected],
        "strata": dict(
            Counter(
                f"{r['player']}|{r['phase']}|{r['legal_move_count']}" for r in selected
            )
        ),
    }
    return selected, manifest


def teacher_target(
    *,
    evaluator: CheckpointEvaluator,
    state: dict[str, Any],
    simulations: int,
    noisy: bool,
    seed: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=random.Random(seed),
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
    )
    visits, root = search.run(
        game,
        dirichlet_alpha=0.3 if noisy else None,
        dirichlet_epsilon=0.25 if noisy else 0.0,
    )
    policy = np.asarray(
        policy_from_visits(visits, legal_moves=legal, temperature=1.0), dtype=float
    )
    return {
        "visits": [int(value) for value in visits],
        "policy": policy.tolist(),
        "top_move": top_move(policy, legal),
        "entropy": entropy(policy),
        "visit_margin": visit_margin(policy, legal),
        "root_value": float(root.q_value),
        "legal_moves": legal,
        "seed": seed,
        "seed_context_hash": stable_hash(
            {"state": state, "simulations": simulations, "noisy": noisy, "seed": seed}
        ),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def _pair_metrics(
    reference: dict[str, Any], target: dict[str, Any]
) -> dict[str, float | bool]:
    legal = reference["legal_moves"]
    ref = np.asarray(reference["policy"])[legal]
    candidate = np.asarray(target["policy"])[legal]
    ref = np.maximum(ref, 1e-12)
    candidate = np.maximum(candidate, 1e-12)
    midpoint = (ref + candidate) / 2
    return {
        "kl_reference_to_target": float(np.sum(ref * np.log(ref / candidate))),
        "js_divergence": float(
            (
                np.sum(ref * np.log(ref / midpoint))
                + np.sum(candidate * np.log(candidate / midpoint))
            )
            / 2
        ),
        "top1_agreement": reference["top_move"] == target["top_move"],
        "top2_set_agreement": set(np.argsort(-ref)[:2])
        == set(np.argsort(-candidate)[:2]),
        "spearman": float(np.corrcoef(_rank(ref), _rank(candidate))[0, 1])
        if len(legal) > 1
        else 1.0,
        "entropy_difference": target["entropy"] - reference["entropy"],
        "selected_move_visit_margin": target["visit_margin"],
    }


def consistency_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in records:
        noisy, denoised = (
            _pair_metrics(row["reference_d1200"], row["noisy_n384"]),
            _pair_metrics(row["reference_d1200"], row["denoised_d384"]),
        )
        rows.append(
            {
                **row,
                "noisy_metrics": noisy,
                "denoised_metrics": denoised,
                "disagree": row["noisy_n384"]["top_move"]
                != row["denoised_d384"]["top_move"],
            }
        )

    def summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            key: float(np.mean([float(value[key]) for value in values]))
            for key in (
                "kl_reference_to_target",
                "js_divergence",
                "spearman",
                "entropy_difference",
                "selected_move_visit_margin",
            )
        } | {
            key: float(np.mean([bool(value[key]) for value in values]))
            for key in ("top1_agreement", "top2_set_agreement")
        }

    noisy, denoised = (
        summary([r["noisy_metrics"] for r in rows]),
        summary([r["denoised_metrics"] for r in rows]),
    )
    by_stratum = {}
    for key in ("phase", "player", "source_domain"):
        for value in sorted({str(row[key]) for row in rows}):
            subset = [row for row in rows if str(row[key]) == value]
            by_stratum[f"{key}:{value}"] = {
                "noisy": summary([r["noisy_metrics"] for r in subset]),
                "denoised": summary([r["denoised_metrics"] for r in subset]),
            }
    js_improvement = (
        1 - denoised["js_divergence"] / noisy["js_divergence"]
        if noisy["js_divergence"]
        else 0.0
    )
    passes = (
        js_improvement >= 0.15
        and denoised["top1_agreement"] >= noisy["top1_agreement"] + 0.05
        and all(
            s["denoised"]["js_divergence"] <= s["noisy"]["js_divergence"]
            for s in by_stratum.values()
        )
    )
    return {
        "records": rows,
        "noisy_n384": noisy,
        "denoised_d384": denoised,
        "by_stratum": by_stratum,
        "disagreement_fraction": float(np.mean([r["disagree"] for r in rows])),
        "js_improvement_fraction": js_improvement,
        "passes": passes,
    }


def materialize_paired_views(
    canonical_rows: list[dict[str, Any]], *, workdir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    noisy, denoised = [], []
    for row in canonical_rows:
        base = {
            key: value
            for key, value in row.items()
            if key not in {"policy_target_noisy", "policy_target_denoised", "policy"}
        }
        noisy.append(
            {
                **base,
                "policy": row["policy_target_noisy"],
                "policy_target_noise_mode": "noisy",
            }
        )
        denoised.append(
            {
                **base,
                "policy": row["policy_target_denoised"],
                "policy_target_noise_mode": "denoised",
            }
        )
    noisy_path, denoised_path = (
        workdir / "replay_noisy.jsonl",
        workdir / "replay_denoised.jsonl",
    )
    write_jsonl(noisy_path, noisy)
    write_jsonl(denoised_path, denoised)
    non_policy_equal = all(
        {
            k: v
            for k, v in left.items()
            if k not in {"policy", "policy_target_noise_mode"}
        }
        == {
            k: v
            for k, v in right.items()
            if k not in {"policy", "policy_target_noise_mode"}
        }
        for left, right in zip(noisy, denoised)
    )
    state_hashes = [stable_hash(row["state"]) for row in noisy]
    audit = {
        "row_count": len(noisy),
        "game_count": len({row["game_index"] for row in noisy}),
        "state_sequence_sha256": stable_hash(state_hashes),
        "trajectory_hash_sequence_sha256": stable_hash(
            [row.get("trajectory_hash") for row in noisy]
        ),
        "value_target_sha256": stable_hash([row["value"] for row in noisy]),
        "noisy_policy_target_sha256": stable_hash([row["policy"] for row in noisy]),
        "denoised_policy_target_sha256": stable_hash(
            [row["policy"] for row in denoised]
        ),
        "average_target_entropy": {
            "noisy": float(np.mean([entropy(row["policy"]) for row in noisy])),
            "denoised": float(np.mean([entropy(row["policy"]) for row in denoised])),
        },
        "top1_disagreement_rate": float(
            np.mean(
                [
                    top_move(a["policy"], a["legal_moves"])
                    != top_move(b["policy"], b["legal_moves"])
                    for a, b in zip(noisy, denoised)
                ]
            )
        ),
        "pairing_invariants": {
            "state_sequence_identical": non_policy_equal,
            "value_targets_identical": [r["value"] for r in noisy]
            == [r["value"] for r in denoised],
            "game_outcomes_identical": [r["winner"] for r in noisy]
            == [r["winner"] for r in denoised],
            "split_membership_identical": [r["game_index"] for r in noisy]
            == [r["game_index"] for r in denoised],
        },
    }
    if not all(audit["pairing_invariants"].values()):
        raise RuntimeError("paired replay invariant failed")
    return noisy_path, denoised_path, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--pilot-replay", required=True)
    parser.add_argument(
        "--diagnostic-evaluation-suite",
        required=True,
        help="Comma-separated diagnostic-only evaluation suites",
    )
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def _generate_canonical_replay(
    *, workdir: Path, checkpoint: Path, rows: int, seed: int
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    game = 0
    while len(collected) < rows:
        shard = workdir / f"selfplay_{game}.jsonl"
        run_self_play_worker(
            worker_id=0,
            start_index=game,
            games=16,
            seed=seed,
            seed_pool=[seed],
            checkpoint=str(checkpoint),
            input_encoding="kalah_v3",
            simulations=384,
            c_puct=1.25,
            temperature_threshold=8,
            temperature=0.67,
            temperature_late=0.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            max_moves=200,
            shard_path=str(shard),
            root_policy_mode="visit_count",
            write_game_metadata=True,
            write_root_target_telemetry=True,
            policy_target_noise_mode="noisy",
        )
        collected.extend(read_jsonl(shard))
        game += 16
    evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    for index, row in enumerate(collected):
        state = _decoded_state(row)
        denoised = teacher_target(
            evaluator=evaluator,
            state=state,
            simulations=384,
            noisy=False,
            seed=stable_seed(seed, "paired-denoised", index, stable_hash(state)),
        )
        row["policy_target_noisy"] = list(row["policy"])
        row["policy_target_denoised"] = denoised["policy"]
        row["trajectory_hash"] = row.get(
            "trajectory_hash",
            stable_hash({"game": row["game_index"], "winner": row["winner"]}),
        )
    write_jsonl(workdir / "replay_canonical.jsonl", collected)
    return collected


def _forced_continuation(
    *,
    evaluator: CheckpointEvaluator,
    state: dict[str, Any],
    forced_move: int,
    simulations: int,
    seed: int,
) -> tuple[float, int]:
    """Play a deterministic continuation, scored for the player at the probe root."""
    game = KalahGame.from_state(state)
    root_player = game.current_player
    if not game.move(game.pit_index(forced_move)):
        raise RuntimeError("forced teacher move is not legal")
    ply = 1
    while not game.over() and ply < 200:
        legal = game.possible_moves()
        if not legal:
            break
        context = stable_hash(
            {
                "state": game.to_state(),
                "root": root_player,
                "ply": ply,
                "budget": simulations,
            }
        )
        search = PUCT(
            evaluator=evaluator,
            simulations=simulations,
            c_puct=1.25,
            rng=random.Random(stable_seed(seed, context)),
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
        )
        visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
        move = search.select_root_move(root, legal)
        if not game.move(game.pit_index(move)):
            break
        ply += 1
    winner = game.winner
    outcome = 0.0 if winner is None else (1.0 if int(winner) == root_player else -1.0)
    stores = game.captured_seeds
    return outcome, int(stores[root_player] - stores[1 - root_player])


def causal_disagreement_audit(
    records: list[dict[str, Any]], *, evaluator: CheckpointEvaluator, seed: int
) -> dict[str, Any]:
    """Compare forced noisy and denoised moves under both prescribed budgets."""
    disagreements = [
        row
        for row in records
        if row["noisy_n384"]["top_move"] != row["denoised_d384"]["top_move"]
    ][:256]
    results = []
    for row in disagreements:
        result = {
            "state_hash": row["state_hash"],
            "phase": row["phase"],
            "player": row["player"],
            "moves": {
                "noisy": row["noisy_n384"]["top_move"],
                "denoised": row["denoised_d384"]["top_move"],
                "reference": row["reference_d1200"]["top_move"],
            },
            "budgets": {},
        }
        for budget in (768, 1200):
            noisy = _forced_continuation(
                evaluator=evaluator,
                state=row["state"],
                forced_move=result["moves"]["noisy"],
                simulations=budget,
                seed=stable_seed(seed, row["state_hash"], "noisy", budget),
            )
            denoised = _forced_continuation(
                evaluator=evaluator,
                state=row["state"],
                forced_move=result["moves"]["denoised"],
                simulations=budget,
                seed=stable_seed(seed, row["state_hash"], "denoised", budget),
            )
            reference = None
            if result["moves"]["reference"] not in {
                result["moves"]["noisy"],
                result["moves"]["denoised"],
            }:
                reference = _forced_continuation(
                    evaluator=evaluator,
                    state=row["state"],
                    forced_move=result["moves"]["reference"],
                    simulations=budget,
                    seed=stable_seed(seed, row["state_hash"], "reference", budget),
                )
            result["budgets"][str(budget)] = {
                "noisy_outcome": noisy[0],
                "denoised_outcome": denoised[0],
                "reference_outcome": None if reference is None else reference[0],
                "noisy_store_margin": noisy[1],
                "denoised_store_margin": denoised[1],
                "reference_store_margin": None if reference is None else reference[1],
                "outcome_delta": denoised[0] - noisy[0],
                "store_margin_delta": denoised[1] - noisy[1],
            }
        results.append(result)

    def aggregate(budget: str) -> dict[str, Any]:
        values = [row["budgets"][budget]["outcome_delta"] for row in results]
        margins = [row["budgets"][budget]["store_margin_delta"] for row in results]
        interval = (
            bootstrap_ci(values, seed=stable_seed(seed, "causal", budget))
            if values
            else {"mean": 0.0, "lower": 0.0, "upper": 0.0}
        )
        return {
            "mean_outcome_delta": float(np.mean(values)) if values else 0.0,
            "median_outcome_delta": float(np.median(values)) if values else 0.0,
            "bootstrap_95": interval,
            "mean_store_margin_delta": float(np.mean(margins)) if margins else 0.0,
            "fraction_denoised_better": float(np.mean(np.asarray(values) > 0))
            if values
            else 0.0,
            "fraction_noisy_better": float(np.mean(np.asarray(values) < 0))
            if values
            else 0.0,
        }

    budgets = {str(budget): aggregate(str(budget)) for budget in (768, 1200)}
    return {
        "requested_states": 256,
        "available_disagreements": sum(
            row["noisy_n384"]["top_move"] != row["denoised_d384"]["top_move"]
            for row in records
        ),
        "evaluated_states": len(results),
        "sample_limited": len(results) < 256,
        "budgets": budgets,
        "records": results,
        "passes": all(
            metric["mean_outcome_delta"] >= 0 and metric["bootstrap_95"]["lower"] >= 0
            for metric in budgets.values()
        ),
    }


def cross_target_policy_metrics(
    *, checkpoint: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    policy_rows, outputs = [], []
    for row in rows:
        state = row.get("raw_state") or _decoded_state(row)
        policy, value = evaluator.evaluate(KalahGame.from_state(state))
        policy_rows.append(row)
        outputs.append({"policy": policy.tolist(), "value": value})
    targets = np.asarray([row["policy"] for row in policy_rows], dtype=float)
    predictions = np.asarray([row["policy"] for row in outputs], dtype=float)
    kl = np.sum(
        np.maximum(targets, 1e-12)
        * np.log(np.maximum(targets, 1e-12) / np.maximum(predictions, 1e-12)),
        axis=1,
    )
    midpoint = (targets + predictions) / 2
    js = (
        np.sum(
            np.maximum(targets, 1e-12)
            * np.log(np.maximum(targets, 1e-12) / np.maximum(midpoint, 1e-12)),
            axis=1,
        )
        + np.sum(
            np.maximum(predictions, 1e-12)
            * np.log(np.maximum(predictions, 1e-12) / np.maximum(midpoint, 1e-12)),
            axis=1,
        )
    ) / 2
    return {
        "policy": {
            "ce": float(
                -np.mean(
                    np.sum(targets * np.log(np.maximum(predictions, 1e-12)), axis=1)
                )
            ),
            "kl": float(np.mean(kl)),
            "js_divergence": float(np.mean(js)),
            "top1_agreement": float(
                np.mean(np.argmax(targets, 1) == np.argmax(predictions, 1))
            ),
            "entropy": float(np.mean([entropy(row) for row in predictions])),
        },
        "value": value_metrics(policy_rows, outputs),
    }


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current = Path(args.current)
    if sha256_file(current / "weights.json") != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")
    checkpoint = materialize_weights_json_checkpoint(
        weights_path=current / "weights.json", out_path=workdir / "current.npz"
    )
    pilot = read_jsonl(Path(args.pilot_replay))
    evaluation = [
        row
        for path in args.diagnostic_evaluation_suite.split(",")
        if path
        for row in read_jsonl(Path(path))
    ]
    probe, probe_manifest = select_probe_states(pilot, evaluation, seed=args.seed)
    write_jsonl(workdir / "target_probe_states.jsonl", probe)
    write_json(workdir / "target_probe_manifest.json", probe_manifest)
    evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    teachers = []
    for row in probe:
        state_hash = row["state_hash"]
        teachers.append(
            {
                **row,
                "noisy_n384": teacher_target(
                    evaluator=evaluator,
                    state=row["state"],
                    simulations=384,
                    noisy=True,
                    seed=stable_seed(args.seed, "n384", state_hash),
                ),
                "denoised_d384": teacher_target(
                    evaluator=evaluator,
                    state=row["state"],
                    simulations=384,
                    noisy=False,
                    seed=stable_seed(args.seed, "d384", state_hash),
                ),
                "reference_d1200": teacher_target(
                    evaluator=evaluator,
                    state=row["state"],
                    simulations=1200,
                    noisy=False,
                    seed=stable_seed(args.seed, "d1200", state_hash),
                ),
            }
        )
    write_jsonl(workdir / "target_probe_teachers.jsonl", teachers)
    consistency = consistency_audit(teachers)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "current_weights_sha256": sha256_file(current / "weights.json"),
        "seed_contract": SEED_CONTRACT_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "probe_manifest": probe_manifest,
        "target_consistency": {
            key: value for key, value in consistency.items() if key != "records"
        },
        "stop_reasons": [],
    }
    if not consistency["passes"]:
        summary["classification"] = "policy_target_noise_not_primary"
        summary["stop_reasons"].append("target_consistency_gate_failed")
    else:
        causal = causal_disagreement_audit(
            teachers, evaluator=evaluator, seed=args.seed
        )
        write_jsonl(workdir / "causal_disagreement_audit.jsonl", causal["records"])
        summary["causal_disagreement_audit"] = {
            key: value for key, value in causal.items() if key != "records"
        }
        if not causal["passes"]:
            summary["classification"] = "denoised_policy_target_rejected"
            summary["stop_reasons"].append("causal_disagreement_gate_failed")
            write_json(workdir / "summary_metrics.json", summary)
            write_json(
                REPO_ROOT
                / "docs/data/alphazero-lite-policy-target-noise-ablation-summary.json",
                summary,
            )
            (
                REPO_ROOT
                / "docs/alphazero-lite-policy-target-noise-ablation-results.md"
            ).write_text(
                "# AlphaZero-Lite Policy-Target Noise Ablation Results\n\n```json\n"
                + json.dumps(summary, indent=2, sort_keys=True)
                + "\n```\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "classification": summary["classification"],
                        "workdir": str(workdir),
                    }
                )
            )
            return 0
        canonical = _generate_canonical_replay(
            workdir=workdir, checkpoint=checkpoint, rows=args.rows, seed=args.seed
        )
        noisy_path, denoised_path, audit = materialize_paired_views(
            canonical, workdir=workdir
        )
        summary["paired_replay_audit"] = audit
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lanes = {}
        manifests = {}
        for name, path, rows in (
            ("noisy_target_joint_heads_e1", noisy_path, read_jsonl(noisy_path)),
            (
                "denoised_target_joint_heads_e1",
                denoised_path,
                read_jsonl(denoised_path),
            ),
        ):
            lane = workdir / name
            write_json(lane / "replay_audit.json", audit)
            manifests[name] = build_manifest(
                rows=rows,
                workdir=lane,
                current=current,
                replay=path,
                replay_audit=lane / "replay_audit.json",
                seed=args.seed,
                epochs=1,
                batch_size=512,
            )
            first, second = (
                train_fixed_manifest(lane / "training_manifest.json", device, "run_a"),
                train_fixed_manifest(lane / "training_manifest.json", device, "run_b"),
            )
            lanes[name] = {
                "manifest": manifests[name],
                "run_a": {
                    k: v for k, v in first.items() if not isinstance(v, np.ndarray)
                },
                "run_b": {
                    k: v for k, v in second.items() if not isinstance(v, np.ndarray)
                },
                "reproducibility": compare_runs(first, second),
            }
        summary["training"] = lanes
        if not all(lane["reproducibility"]["passes"] for lane in lanes.values()):
            summary["classification"] = "full_scale_training_nondeterministic"
            summary["stop_reasons"].append("same_device_training_diverged")
        else:
            validation_indexes = lanes["noisy_target_joint_heads_e1"]["manifest"][
                "validation_source_row_indexes"
            ]
            noisy_validation, denoised_validation = (
                [read_jsonl(noisy_path)[i] for i in validation_indexes],
                [read_jsonl(denoised_path)[i] for i in validation_indexes],
            )
            reference_probe = [
                {
                    "raw_state": row["state"],
                    "state": encode_state(row["state"], input_encoding="kalah_v3"),
                    "policy": row["reference_d1200"]["policy"],
                    "value": 0.0,
                }
                for row in teachers
            ]
            summary["cross_target_validation"] = {
                name: {
                    "noisy_validation": cross_target_policy_metrics(
                        checkpoint=Path(lane["run_a"]["checkpoint"]),
                        rows=noisy_validation,
                    ),
                    "denoised_validation": cross_target_policy_metrics(
                        checkpoint=Path(lane["run_a"]["checkpoint"]),
                        rows=denoised_validation,
                    ),
                    "d1200_probe_targets": cross_target_policy_metrics(
                        checkpoint=Path(lane["run_a"]["checkpoint"]),
                        rows=reference_probe,
                    ),
                }
                for name, lane in lanes.items()
            }
            noisy_artifact, denoised_artifact = (
                Path(lanes["noisy_target_joint_heads_e1"]["run_a"]["artifact"]),
                Path(lanes["denoised_target_joint_heads_e1"]["run_a"]["artifact"]),
            )
            medium = paired_strength_screen(
                workdir=workdir / "medium",
                suite=Path(args.medium_suite),
                current=current,
                control=noisy_artifact,
                aligned=denoised_artifact,
                seed=args.seed,
                workers=args.workers,
            )
            summary["medium_strength"] = medium
            pair = medium["paired_opening_bootstrap_95"]["aligned_minus_control"]
            current_delta = medium["paired_opening_bootstrap_95"][
                "aligned_minus_current"
            ]
            medium_pass = (
                pair["384:256"]["mean"] >= 0.03
                and pair["384:256"]["lower"] > 0
                and current_delta["384:256"]["mean"] > 0
                and all(
                    current_delta[key]["mean"] >= -0.03
                    for key in ("768:768", "1200:1200", "1200:256")
                )
            )
            if not medium_pass:
                summary["classification"] = (
                    "policy_target_noise_not_primary"
                    if pair["384:256"]["mean"] < 0.03
                    else "denoised_targets_improve_learning_but_not_strength"
                )
                summary["stop_reasons"].append("medium_strength_gate_failed")
            else:
                fixed = paired_strength_screen(
                    workdir=workdir / "fixed_large",
                    suite=Path(args.fixed_large_suite),
                    current=current,
                    control=noisy_artifact,
                    aligned=denoised_artifact,
                    seed=args.seed,
                    workers=args.workers,
                )
                summary["fixed_large_strength"] = fixed
                fixed_current = fixed["paired_opening_bootstrap_95"][
                    "aligned_minus_current"
                ]
                fixed_pass = (
                    fixed_current["384:256"]["mean"] >= 0.05
                    and fixed_current["384:256"]["lower"] > 0.01
                    and fixed_current["768:768"]["mean"] >= -0.05
                    and fixed_current["1200:1200"]["mean"] >= -0.03
                    and fixed_current["1200:256"]["mean"] >= -0.03
                )
                if not fixed_pass:
                    summary["classification"] = (
                        "denoised_targets_improve_learning_but_not_strength"
                    )
                    summary["stop_reasons"].append("fixed_large_qualification_failed")
                else:
                    heldout = {
                        path.name: paired_strength_screen(
                            workdir=workdir / f"heldout_{i}",
                            suite=path,
                            current=current,
                            control=noisy_artifact,
                            aligned=denoised_artifact,
                            seed=args.seed,
                            workers=args.workers,
                        )
                        for i, path in enumerate(
                            Path(item)
                            for item in args.heldout_suites.split(",")
                            if item
                        )
                    }
                    summary["heldout_strength"] = heldout
                    primary = [
                        screen["paired_opening_bootstrap_95"]["aligned_minus_current"]
                        for screen in heldout.values()
                    ]
                    aggregate = {
                        budget: {
                            "mean": float(
                                np.mean([row[budget]["mean"] for row in primary])
                            ),
                            "lower": float(
                                min(row[budget]["lower"] for row in primary)
                            ),
                        }
                        for budget in ("384:256", "768:768", "1200:1200", "1200:256")
                    }
                    heldout_pass = (
                        aggregate["384:256"]["mean"] >= 0.05
                        and aggregate["384:256"]["lower"] > 0.01
                        and aggregate["768:768"]["mean"] >= -0.05
                        and aggregate["1200:1200"]["mean"] >= -0.03
                        and aggregate["1200:256"]["mean"] >= -0.03
                    )
                    summary["heldout_qualification"] = {
                        "budgets": aggregate,
                        "passes": heldout_pass,
                    }
                    summary["classification"] = (
                        "denoised_joint_heads_candidate"
                        if heldout_pass
                        else "denoised_targets_improve_learning_but_not_strength"
                    )
                    if heldout_pass:
                        summary["scientific_classification"] = (
                            "noisy_policy_targets_confirmed_bottleneck"
                        )
                    summary["stop_reasons"].append(
                        "promotion_not_run_by_protocol"
                        if heldout_pass
                        else "heldout_qualification_failed"
                    )
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT
        / "docs/data/alphazero-lite-policy-target-noise-ablation-summary.json",
        summary,
    )
    consistency_table = summary.get("target_consistency", {})
    markdown = [
        "# AlphaZero-Lite Policy-Target Noise Ablation Results",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Stop reasons: `{', '.join(summary['stop_reasons']) or 'none'}`",
        "",
        "## Target Consistency",
        "",
        "| Target | JS divergence | Top-1 agreement |",
        "| --- | ---: | ---: |",
    ]
    for label in ("noisy_n384", "denoised_d384"):
        metrics = consistency_table.get(label, {})
        markdown.append(
            f"| {label} | {metrics.get('js_divergence', 'not run')} | {metrics.get('top1_agreement', 'not run')} |"
        )
    markdown.extend(
        [
            "",
            "## Compact Record",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (
        REPO_ROOT / "docs/alphazero-lite-policy-target-noise-ablation-results.md"
    ).write_text("\n".join(markdown), encoding="utf-8")
    print(
        json.dumps(
            {"classification": summary["classification"], "workdir": str(workdir)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
