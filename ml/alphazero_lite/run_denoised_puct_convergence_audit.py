#!/usr/bin/env python3
# ruff: noqa: E402
"""Audit denoised PUCT target convergence and forced-move quality by budget.

This is diagnostic-only: it creates no replay and never trains or promotes a
model.  Per-state teacher and forced-continuation records remain in ``workdir``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
import time
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
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import bootstrap_ci
from ml.alphazero_lite.run_distribution_aligned_selfplay_iteration import _decoded_state
from ml.alphazero_lite.run_policy_target_noise_causal_closeout import (
    continuation_seed_identity,
)
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    policy_from_visits,
    run_self_play_worker,
)

EXPECTED_CURRENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
DEFAULT_WORKDIR = Path("/tmp/azlite_denoised_puct_convergence")
TEACHER_BUDGETS = (128, 384, 768, 1200)
CONTINUATION_BUDGETS = (768, 1200)
BOOTSTRAP_SAMPLES = 10_000
SCHEMA = "azlite_denoised_puct_convergence_audit_v1"
_WORKER_EVALUATOR: CheckpointEvaluator | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def phase_for_state(state: dict[str, Any]) -> str:
    remaining = sum(state["player_pits"]) + sum(state["opponent_pits"])
    return "opening" if remaining > 24 else "midgame" if remaining > 12 else "late"


def entropy(policy: list[float] | np.ndarray) -> float:
    values = np.asarray(policy, dtype=float)
    values = values[values > 0]
    return float(-np.sum(values * np.log(values)))


def deterministic_move_order(root: Any, legal_moves: list[int]) -> list[int]:
    return sorted(
        legal_moves,
        key=lambda move: (
            -int(root.children[move].visit_count),
            -float(root.children[move].q_value),
            -float(root.children[move].prior),
            int(move),
        ),
    )


def teacher_seed_identity(*, state_hash: str, experiment_seed: int) -> tuple[int, str]:
    """Return a search-budget-independent base identity for all four teachers."""
    context = {
        "schema": "azlite_denoised_puct_teacher_seed_v1",
        "state_hash": str(state_hash),
        "experiment_seed": int(experiment_seed),
        "rng_stream_name": "puct_teacher",
    }
    return stable_seed(context), stable_hash(context)


def _teacher_target(
    *,
    evaluator: CheckpointEvaluator,
    state: dict[str, Any],
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    legal = [int(move) for move in game.possible_moves()]
    started = time.perf_counter()
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=random.Random(seed),
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        normalize_values=False,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    policy = np.asarray(
        policy_from_visits(visits, legal_moves=legal, temperature=1.0), dtype=float
    )
    order = deterministic_move_order(root, legal)
    summary = search.root_summary()
    top_share = float(policy[order[0]]) if order else 0.0
    return {
        "simulations": int(simulations),
        "visits": [int(value) for value in visits],
        "policy": policy.tolist(),
        "legal_moves": legal,
        "top_move": int(order[0]),
        "second_move": None if len(order) < 2 else int(order[1]),
        "entropy": entropy(policy),
        "top1_visit_share": top_share,
        "top1_top2_visit_margin": top_share
        - (float(policy[order[1]]) if len(order) > 1 else 0.0),
        "root_value": float(root.q_value),
        "child_q_values": {
            str(row["move"]): float(row["q_value"]) for row in summary["child_stats"]
        },
        "child_stats": summary["child_stats"],
        "runtime_seconds": time.perf_counter() - started,
    }


def _init_worker(checkpoint: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def _teacher_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("teacher worker is not initialized")
    seed, context_hash = teacher_seed_identity(
        state_hash=task["state_hash"], experiment_seed=task["experiment_seed"]
    )
    teachers = {
        f"D{budget}": _teacher_target(
            evaluator=_WORKER_EVALUATOR,
            state=task["state"],
            simulations=budget,
            seed=seed,
        )
        for budget in TEACHER_BUDGETS
    }
    alternate_d128 = _teacher_target(
        evaluator=_WORKER_EVALUATOR,
        state=task["state"],
        simulations=128,
        seed=stable_seed(seed, "rng-sensitivity"),
    )
    return {key: task[key] for key in task if key not in {"experiment_seed"}} | {
        "search_seed": seed,
        "search_seed_context_hash": context_hash,
        "teachers": teachers,
        "rng_sensitivity_d128": {
            "different_visit_counts": alternate_d128["visits"]
            != teachers["D128"]["visits"],
            "different_top_move": alternate_d128["top_move"]
            != teachers["D128"]["top_move"],
        },
    }


def run_teacher_tasks(
    *, tasks: list[dict[str, Any]], checkpoint: Path, workers: int
) -> list[dict[str, Any]]:
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        _init_worker(str(checkpoint))
        rows = [_teacher_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(str(checkpoint),),
        ) as executor:
            rows = list(executor.map(_teacher_task, tasks))
    return sorted(rows, key=lambda row: row["state_hash"])


def _state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("raw_state") or row.get("state")
    if isinstance(state, list):
        state = _decoded_state(row)
    if not isinstance(state, dict):
        raise ValueError("source row has no decodable state")
    return state


def harvest_additional_selfplay_states(
    *, workdir: Path, checkpoint: Path, seed: int, target_count: int = 512
) -> Path:
    """Harvest state-only standard-start diagnostics without retaining replay rows.

    The temporary worker shards use the established PR #177 standard-start
    self-play settings. They are deleted after extraction; the retained JSONL
    contains only raw states and explicit diagnostic-only provenance.
    """
    output_path = workdir / "additional_standard_start_selfplay_states.jsonl"
    if output_path.is_file():
        rows = read_jsonl(output_path)
        if len({stable_hash(_state_from_row(row)) for row in rows}) >= target_count:
            return output_path
    states: list[dict[str, Any]] = []
    seen: set[str] = set()
    game_start = 0
    while len(states) < target_count:
        shard = workdir / f"_diagnostic_state_harvest_{game_start}.jsonl"
        run_self_play_worker(
            worker_id=0,
            start_index=game_start,
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
            write_game_metadata=False,
            write_root_target_telemetry=False,
            policy_target_noise_mode="noisy",
        )
        try:
            for row in read_jsonl(shard):
                state = _state_from_row(row)
                state_hash = stable_hash(state)
                if state_hash not in seen:
                    seen.add(state_hash)
                    states.append(
                        {
                            "state": state,
                            "state_hash": state_hash,
                            "source": "bounded_standard_start_selfplay_harvest",
                            "diagnostic_only": True,
                            "excluded_from_training_replay": True,
                        }
                    )
                    if len(states) == target_count:
                        break
        finally:
            shard.unlink(missing_ok=True)
        game_start += 16
        if game_start > 512:
            raise RuntimeError(
                "unable to harvest the requested unique standard-start self-play states"
            )
    write_jsonl(output_path, states)
    return output_path


def _rows_for_source(
    rows: list[dict[str, Any]], expected_domain: str | None
) -> list[dict[str, Any]]:
    if expected_domain is None:
        return rows
    filtered = [row for row in rows if row.get("source_domain") == expected_domain]
    if not filtered:
        raise RuntimeError(
            f"source contains no rows for expected domain {expected_domain}"
        )
    return filtered


def candidates(
    rows: list[dict[str, Any]], domain: str, evaluator: CheckpointEvaluator
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        state = _state_from_row(row)
        game = KalahGame.from_state(state)
        legal = [int(move) for move in game.possible_moves()]
        if not legal:
            continue
        policy, _ = evaluator.evaluate(game)
        output.append(
            {
                "state": state,
                "state_hash": stable_hash(state),
                "source_domain": domain,
                "source_index": index,
                "player": int(state["current_player"]),
                "phase": phase_for_state(state),
                "legal_moves": legal,
                "legal_move_count": len(legal),
                "current_policy_entropy": entropy(policy),
            }
        )
    return output


def select_probe_states(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    evaluator: CheckpointEvaluator,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select 256 unique states per domain, balancing phase then other strata."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    requested = {"opening": 128, "midgame": 77, "late": 51}
    for domain, rows in source_rows.items():
        values = candidates(rows, domain, evaluator)
        buckets: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
        for row in values:
            buckets[
                (
                    row["phase"],
                    row["player"],
                    row["legal_move_count"],
                    int(row["current_policy_entropy"] * 4),
                )
            ].append(row)
        rng = random.Random(stable_seed(seed, domain, "probe-selection"))
        queues: dict[str, deque[deque[dict[str, Any]]]] = {
            phase: deque() for phase in requested
        }
        for key in sorted(buckets, key=str):
            bucket = list(buckets[key])
            rng.shuffle(bucket)
            queues[key[0]].append(deque(bucket))
        domain_selected = []
        for phase, target in requested.items():
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
        # Fill shortages from any remaining balanced queue without duplicating a state.
        remainder = deque(bucket for phase in requested for bucket in queues[phase])
        while remainder and len(domain_selected) < 256:
            bucket = remainder.popleft()
            while bucket and bucket[0]["state_hash"] in used:
                bucket.popleft()
            if bucket:
                row = bucket.popleft()
                used.add(row["state_hash"])
                domain_selected.append(row)
            if bucket:
                remainder.append(bucket)
        if len(domain_selected) != 256:
            raise RuntimeError(
                f"only {len(domain_selected)} unique states available for {domain}; need 256"
            )
        selected.extend(domain_selected)
    if len(selected) != 1024 or len({row["state_hash"] for row in selected}) != 1024:
        raise RuntimeError("probe must contain exactly 1,024 unique states")
    manifest = {
        "schema": "azlite_denoised_puct_convergence_probe_v1",
        "selection_seed": seed,
        "state_count": len(selected),
        "state_hashes": [row["state_hash"] for row in selected],
        "source_domain_counts": dict(Counter(row["source_domain"] for row in selected)),
        "player_counts": dict(Counter(str(row["player"]) for row in selected)),
        "phase_counts": dict(Counter(row["phase"] for row in selected)),
        "legal_move_distribution": dict(
            Counter(str(row["legal_move_count"]) for row in selected)
        ),
    }
    return selected, manifest


def _rank(values: np.ndarray) -> np.ndarray:
    # Stable ordinal ranks make tied zero-visit moves deterministic and reproducible.
    ranks = np.empty(len(values), dtype=float)
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values))
    return ranks


def pair_metrics(
    lower: dict[str, Any], higher: dict[str, Any]
) -> dict[str, float | bool]:
    legal = lower["legal_moves"]
    left = np.maximum(np.asarray(lower["policy"])[legal], 1e-12)
    right = np.maximum(np.asarray(higher["policy"])[legal], 1e-12)
    midpoint = (left + right) / 2
    return {
        "js_divergence": float(
            (
                np.sum(left * np.log(left / midpoint))
                + np.sum(right * np.log(right / midpoint))
            )
            / 2
        ),
        "kl_lower_to_higher": float(np.sum(left * np.log(left / right))),
        "kl_higher_to_lower": float(np.sum(right * np.log(right / left))),
        "spearman": float(np.corrcoef(_rank(left), _rank(right))[0, 1])
        if len(legal) > 1
        else 1.0,
        "top2_set_agreement": set(np.argsort(-left)[:2]) == set(np.argsort(-right)[:2]),
        "entropy_delta": float(higher["entropy"] - lower["entropy"]),
        "top1_agreement": lower["top_move"] == higher["top_move"],
        "top_move_change": lower["top_move"] != higher["top_move"],
        "top1_visit_margin_delta": float(
            higher["top1_top2_visit_margin"] - lower["top1_top2_visit_margin"]
        ),
        "root_value_delta": float(higher["root_value"] - lower["root_value"]),
    }


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = (
        "js_divergence",
        "kl_lower_to_higher",
        "kl_higher_to_lower",
        "spearman",
        "top2_set_agreement",
        "entropy_delta",
        "top1_agreement",
        "top_move_change",
        "top1_visit_margin_delta",
        "root_value_delta",
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys
        if key in rows[0]
    }


def _entropy_quartiles(records: list[dict[str, Any]]) -> list[float]:
    return [
        float(x)
        for x in np.quantile(
            [r["current_policy_entropy"] for r in records], [0.25, 0.5, 0.75]
        )
    ]


def convergence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [
        ("D128", "D384"),
        ("D384", "D768"),
        ("D768", "D1200"),
        ("D128", "D1200"),
        ("D384", "D1200"),
        ("D768", "D1200"),
    ]
    quartiles = _entropy_quartiles(records)
    result: dict[str, Any] = {"entropy_quartile_cutoffs": quartiles, "comparisons": {}}
    for lower, higher in comparisons:
        rows = [
            {
                **record,
                **pair_metrics(record["teachers"][lower], record["teachers"][higher]),
            }
            for record in records
        ]
        slices: dict[str, dict[str, Any]] = {}
        for field in ("player", "phase", "source_domain", "legal_move_count"):
            slices[field] = {
                str(value): _mean_metrics(
                    [r for r in rows if str(r[field]) == str(value)]
                )
                for value in sorted({r[field] for r in rows}, key=str)
            }
        slices["current_policy_entropy_quartile"] = {
            str(index + 1): _mean_metrics(
                [
                    r
                    for r in rows
                    if (
                        r["current_policy_entropy"] <= quartiles[index]
                        if index == 0
                        else r["current_policy_entropy"] > quartiles[index - 1]
                        and (
                            index == 3
                            or r["current_policy_entropy"] <= quartiles[index]
                        )
                    )
                ]
            )
            for index in range(4)
        }
        result["comparisons"][f"{lower}->{higher}"] = {
            "global": _mean_metrics(rows),
            "slices": slices,
        }
    return result


def stability_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        moves = [
            row["teachers"][f"D{budget}"]["top_move"] for budget in TEACHER_BUDGETS
        ]
        category = (
            "stable_from_128"
            if len(set(moves)) == 1
            else "stable_from_384"
            if len(set(moves[1:])) == 1
            else "stable_from_768"
            if moves[2] == moves[3]
            else "unstable_through_1200"
        )
        categories[category].append(row)
    summary = {}
    for category in (
        "stable_from_128",
        "stable_from_384",
        "stable_from_768",
        "unstable_through_1200",
    ):
        rows = categories[category]
        summary[category] = {
            "count": len(rows),
            "fraction": len(rows) / len(records),
            "player_counts": dict(Counter(str(r["player"]) for r in rows)),
            "phase_counts": dict(Counter(r["phase"] for r in rows)),
            "mean_policy_entropy": float(
                np.mean([r["current_policy_entropy"] for r in rows])
            )
            if rows
            else 0.0,
            "mean_visit_margin": float(
                np.mean(
                    [r["teachers"]["D1200"]["top1_top2_visit_margin"] for r in rows]
                )
            )
            if rows
            else 0.0,
            "current_network_top_move_agreement": float(
                np.mean(
                    [
                        r["current_network_top_move"]
                        == r["teachers"]["D1200"]["top_move"]
                        for r in rows
                    ]
                )
            )
            if rows
            else 0.0,
        }
    oscillations = [
        r["state_hash"]
        for r in records
        if r["teachers"]["D128"]["top_move"] == r["teachers"]["D768"]["top_move"]
        and r["teachers"]["D384"]["top_move"] == r["teachers"]["D1200"]["top_move"]
        and r["teachers"]["D128"]["top_move"] != r["teachers"]["D384"]["top_move"]
    ]
    return {
        "categories": summary,
        "alternating_128_384_768_1200_count": len(oscillations),
        "alternating_state_hashes": oscillations,
    }


def disagreement_sets(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = (("D128", "D384"), ("D384", "D768"), ("D768", "D1200"))
    sets = {
        f"{a}->{b}": {
            r["state_hash"]
            for r in records
            if r["teachers"][a]["top_move"] != r["teachers"][b]["top_move"]
        }
        for a, b in names
    }
    memberships = Counter(
        "|".join(name for name, values in sets.items() if state in values)
        for state in set().union(*sets.values())
    )
    return {
        "sizes": {name: len(values) for name, values in sets.items()},
        "overlap_patterns": dict(memberships),
        "state_hashes": {name: sorted(values) for name, values in sets.items()},
    }


def forced_continuation(
    *, evaluator: CheckpointEvaluator, task: dict[str, Any], forced_move: int
) -> dict[str, Any]:
    game = KalahGame.from_state(task["state"])
    root_player = int(game.current_player)
    base_seed, context_hash = continuation_seed_identity(
        state_hash=task["state_hash"],
        continuation_budget=task["continuation_budget"],
        root_player=root_player,
        experiment_seed=task["experiment_seed"],
    )
    if forced_move not in game.possible_moves() or not game.move(
        game.pit_index(forced_move)
    ):
        raise RuntimeError(f"illegal forced move {forced_move}")
    trajectory = [forced_move]
    for ply in range(1, 200):
        if game.over() or not game.possible_moves():
            break
        search = PUCT(
            evaluator=evaluator,
            simulations=task["continuation_budget"],
            c_puct=1.25,
            rng=random.Random(
                stable_seed(base_seed, stable_hash(game.to_state()), ply)
            ),
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            normalize_values=False,
        )
        _, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
        move = int(search.select_root_move(root, game.possible_moves()))
        game.move(game.pit_index(move))
        trajectory.append(move)
    stores = game.captured_seeds
    winner = game.winner
    return {
        "forced_move": forced_move,
        "outcome_root": 0.0
        if winner is None
        else (1.0 if int(winner) == root_player else -1.0),
        "store_margin_root": int(stores[root_player] - stores[1 - root_player]),
        "trajectory_hash": stable_hash(trajectory),
        "paired_seed_context_hash": context_hash,
        "paired_base_seed": base_seed,
    }


def _forced_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("continuation worker is not initialized")
    return {
        key: task[key] for key in task if key not in {"state", "experiment_seed"}
    } | {
        "interventions": {
            str(move): forced_continuation(
                evaluator=_WORKER_EVALUATOR, task=task, forced_move=move
            )
            for move in task["moves"]
        }
    }


def run_forced_tasks(
    *, tasks: list[dict[str, Any]], checkpoint: Path, workers: int
) -> list[dict[str, Any]]:
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        _init_worker(str(checkpoint))
        rows = [_forced_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(str(checkpoint),),
        ) as executor:
            rows = list(executor.map(_forced_task, tasks))
    return sorted(
        rows, key=lambda r: (r["comparison"], r["state_hash"], r["continuation_budget"])
    )


def build_forced_tasks(
    records: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    tasks = []
    for lower, higher in (("D128", "D384"), ("D384", "D768"), ("D768", "D1200")):
        for row in records:
            if (
                row["teachers"][lower]["top_move"]
                == row["teachers"][higher]["top_move"]
            ):
                continue
            # All identities form the causal ladder, but duplicate moves need one intervention.
            moves = sorted(
                {
                    int(row["teachers"][f"D{budget}"]["top_move"])
                    for budget in TEACHER_BUDGETS
                }
            )
            for budget in CONTINUATION_BUDGETS:
                tasks.append(
                    {
                        key: row[key]
                        for key in (
                            "state_hash",
                            "state",
                            "player",
                            "phase",
                            "source_domain",
                            "legal_move_count",
                        )
                    }
                    | {
                        "comparison": f"{lower}->{higher}",
                        "lower_teacher": lower,
                        "higher_teacher": higher,
                        "moves": moves,
                        "continuation_budget": budget,
                        "experiment_seed": seed,
                        "teacher_moves": {
                            name: row["teachers"][name]["top_move"]
                            for name in ("D128", "D384", "D768", "D1200")
                        },
                        "teacher_child_q_values": {
                            name: row["teachers"][name]["child_q_values"]
                            for name in ("D128", "D384", "D768", "D1200")
                        },
                    }
                )
    return tasks


def _outcome_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    deltas = np.asarray(
        [r["higher_outcome"] - r["lower_outcome"] for r in rows], dtype=float
    )
    margins = np.asarray(
        [r["higher_margin"] - r["lower_margin"] for r in rows], dtype=float
    )
    ci = (
        bootstrap_ci(deltas.tolist(), seed=seed, samples=BOOTSTRAP_SAMPLES)
        if len(deltas)
        else {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n": 0,
            "samples": BOOTSTRAP_SAMPLES,
        }
    )
    return {
        "unique_states": len({r["state_hash"] for r in rows}),
        "mean_outcome_delta": float(np.mean(deltas)) if len(deltas) else 0.0,
        "median_outcome_delta": float(np.median(deltas)) if len(deltas) else 0.0,
        "mean_store_margin_delta": float(np.mean(margins)) if len(margins) else 0.0,
        "median_store_margin_delta": float(np.median(margins)) if len(margins) else 0.0,
        "paired_bootstrap_95": ci,
        "fraction_higher_better": float(np.mean(deltas > 0)) if len(deltas) else 0.0,
        "fraction_lower_better": float(np.mean(deltas < 0)) if len(deltas) else 0.0,
        "fraction_tied": float(np.mean(deltas == 0)) if len(deltas) else 0.0,
    }


def causal_summary(
    records: list[dict[str, Any]], *, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for record in records:
        interventions = record["interventions"]
        lower_move, higher_move = record["lower_move"], record["higher_move"]
        rows.append(
            {
                **record,
                "lower_outcome": interventions[str(lower_move)]["outcome_root"],
                "higher_outcome": interventions[str(higher_move)]["outcome_root"],
                "lower_margin": interventions[str(lower_move)]["store_margin_root"],
                "higher_margin": interventions[str(higher_move)]["store_margin_root"],
            }
        )
    result = {}
    for comparison in sorted({r["comparison"] for r in rows}):
        result[comparison] = {}
        for budget in CONTINUATION_BUDGETS:
            subset = [
                r
                for r in rows
                if r["comparison"] == comparison and r["continuation_budget"] == budget
            ]
            slices = {
                field: {
                    str(value): _outcome_summary(
                        [r for r in subset if str(r[field]) == str(value)],
                        seed=stable_seed(seed, comparison, budget, field, value),
                    )
                    for value in sorted({r[field] for r in subset}, key=str)
                }
                for field in ("player", "phase", "source_domain")
            }
            result[comparison][str(budget)] = {
                "global": _outcome_summary(
                    subset, seed=stable_seed(seed, comparison, budget)
                ),
                "slices": slices,
            }
    return result, rows


def _correlation(left: list[float], right: list[float]) -> float:
    return (
        float(np.corrcoef(left, right)[0, 1])
        if len(left) > 1 and np.std(left) and np.std(right)
        else 0.0
    )


def q_calibration(causal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for budget in TEACHER_BUDGETS:
        label = f"D{budget}"
        entries = []
        for row in causal_rows:
            q = row["teacher_child_q_values"].get(label, {})
            if str(row["lower_move"]) not in q or str(row["higher_move"]) not in q:
                continue
            predicted = float(q[str(row["higher_move"])]) - float(
                q[str(row["lower_move"])]
            )
            realized = float(row["higher_outcome"] - row["lower_outcome"])
            entries.append((predicted, realized))
        predictions, realized = zip(*entries) if entries else ([], [])
        buckets = {}
        for low, high in (
            (-float("inf"), -0.25),
            (-0.25, -0.05),
            (-0.05, 0.05),
            (0.05, 0.25),
            (0.25, float("inf")),
        ):
            values = [
                actual for predicted, actual in entries if low <= predicted < high
            ]
            buckets[f"{low}:{high}"] = {
                "n": len(values),
                "mean_realized_outcome_delta": float(np.mean(values))
                if values
                else 0.0,
            }
        high_confidence = [
            actual
            for predicted, actual in entries
            if abs(predicted) >= 0.25 and predicted * actual < 0
        ]
        output[label] = {
            "n": len(entries),
            "sign_agreement": float(
                np.mean([np.sign(p) == np.sign(r) for p, r in entries])
            )
            if entries
            else 0.0,
            "pearson": _correlation(list(predictions), list(realized)),
            "spearman": _correlation(
                _rank(np.asarray(predictions, dtype=float)).tolist(),
                _rank(np.asarray(realized, dtype=float)).tolist(),
            )
            if entries
            else 0.0,
            "calibration_buckets": buckets,
            "high_confidence_causally_wrong_fraction": len(high_confidence)
            / sum(abs(p) >= 0.25 for p, _ in entries)
            if entries
            else 0.0,
        }
    return output


def causal_ladder(causal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate each teacher identity's move only through forced continuations."""
    result: dict[str, Any] = {}
    for comparison in sorted({row["comparison"] for row in causal_rows}):
        result[comparison] = {}
        for budget in CONTINUATION_BUDGETS:
            rows = [
                row
                for row in causal_rows
                if row["comparison"] == comparison
                and row["continuation_budget"] == budget
            ]
            result[comparison][str(budget)] = {
                label: {
                    "mean_outcome": float(
                        np.mean(
                            [
                                row["interventions"][str(row["teacher_moves"][label])][
                                    "outcome_root"
                                ]
                                for row in rows
                            ]
                        )
                    )
                    if rows
                    else 0.0,
                    "mean_store_margin": float(
                        np.mean(
                            [
                                row["interventions"][str(row["teacher_moves"][label])][
                                    "store_margin_root"
                                ]
                                for row in rows
                            ]
                        )
                    )
                    if rows
                    else 0.0,
                }
                for label in ("D128", "D384", "D768", "D1200")
            }
    return result


def classify(summary: dict[str, Any]) -> tuple[list[str], str]:
    convergence = summary["distribution_convergence"]["comparisons"]
    js = [
        convergence[f"D{b}->D1200"]["global"]["js_divergence"] for b in (128, 384, 768)
    ]
    rank = [convergence[f"D{b}->D1200"]["global"]["spearman"] for b in (128, 384, 768)]
    d768 = convergence["D768->D1200"]["global"]
    causal = summary["causal_adjacent_budget_results"]
    labels = []
    global_rows = [
        causal.get(name, {}).get(str(budget), {}).get("global", {})
        for name in causal
        for budget in CONTINUATION_BUDGETS
    ]
    adjacent_means = [
        causal.get(name, {})
        .get(str(budget), {})
        .get("global", {})
        .get("mean_outcome_delta", 0.0)
        for name in causal
        for budget in CONTINUATION_BUDGETS
    ]
    if (
        js[0] >= js[1] >= js[2]
        and d768["top1_agreement"] >= 0.95
        and (1 - d768["top1_agreement"]) <= 0.05
        and all(value >= 0 for value in adjacent_means)
        and not any(
            row.get("unique_states", 0) >= 64
            and row["paired_bootstrap_95"]["upper"] < 0
            for row in global_rows
        )
    ):
        labels.append("puct_policy_targets_converge_with_budget")
    higher_steps = [
        causal.get(name, {}).get(str(budget), {}).get("global", {})
        for name in ("D384->D768", "D768->D1200")
        for budget in CONTINUATION_BUDGETS
    ]
    bad_player_phase_slice = any(
        slice_row["unique_states"] >= 24 and slice_row["mean_outcome_delta"] < -0.10
        for comparison in ("D384->D768", "D768->D1200")
        for budget in CONTINUATION_BUDGETS
        for field in ("player", "phase")
        for slice_row in causal.get(comparison, {})
        .get(str(budget), {})
        .get("slices", {})
        .get(field, {})
        .values()
    )
    if (
        higher_steps
        and all(
            r.get("unique_states", 0) >= 64
            and r.get("mean_outcome_delta", 0) > 0
            and r.get("paired_bootstrap_95", {}).get("lower", -1) >= 0
            for r in higher_steps
        )
        and not bad_player_phase_slice
    ):
        labels.append("stronger_puct_teacher_causally_better")
    if (
        js[0] >= js[1] >= js[2]
        and rank[0] <= rank[1] <= rank[2]
        and any(value <= 0 for value in adjacent_means)
    ):
        labels.append("puct_target_distribution_converges_but_moves_do_not")
    if (
        (1 - d768["top1_agreement"]) > 0.10
        or summary["target_stability"]["alternating_128_384_768_1200_count"]
        / summary["probe_manifest"]["state_count"]
        >= 0.05
        or any(value < 0 for value in adjacent_means)
    ):
        labels.append("puct_target_instability_confirmed")
    calibration = summary["search_q_calibration"]
    if any(
        row["pearson"] <= 0.10
        or row["spearman"] <= 0.10
        or row["high_confidence_causally_wrong_fraction"] >= 0.25
        for row in calibration.values()
        if row["n"]
    ):
        labels.append("puct_search_q_miscalibration")
    if summary["target_stability"]["categories"]["stable_from_384"][
        "fraction"
    ] + summary["target_stability"]["categories"]["stable_from_128"][
        "fraction"
    ] >= 0.95 and not any(value > 0 for value in adjacent_means):
        labels.append("search_budget_not_primary_policy_bottleneck")
    next_action = "No training authorization; inspect the recorded convergence and causal evidence."
    if "stronger_puct_teacher_causally_better" in labels:
        next_action = "Run exactly one matched existing-budget versus stronger-denoised-target training experiment using identical trajectories and value targets."
    elif "puct_target_distribution_converges_but_moves_do_not" in labels:
        next_action = "Stop optimizing target KL/JS against stronger search; use direct causal move quality for policy-target work."
    elif "puct_target_instability_confirmed" in labels:
        next_action = "Do not increase self-play search compute; diagnose prior influence, Q calibration, visit margin, and tactical state family."
    elif "puct_search_q_miscalibration" in labels:
        next_action = (
            "Prioritize search-value/Q calibration before new policy training."
        )
    elif "search_budget_not_primary_policy_bottleneck" in labels:
        next_action = "Close policy-target search-budget tuning and shift to search-Q/value quality."
    return labels or ["inconclusive"], next_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--pr176-pilot",
        default="/tmp/azlite_policy_target_noise_ablation/target_probe_states.jsonl",
    )
    parser.add_argument(
        "--pr177-probe-states",
        default="/tmp/azlite_policy_target_noise_ablation/target_probe_states.jsonl",
    )
    parser.add_argument(
        "--additional-selfplay",
        default=None,
        help="Existing diagnostic-only state file; otherwise harvest a bounded state-only file.",
    )
    parser.add_argument(
        "--opening-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Denoised PUCT Convergence Audit",
        "",
        f"- Classifications: `{', '.join(summary['classifications'])}`",
        f"- Next action: {summary['next_action']}",
        "",
        "## Search Configuration",
        "",
        "```json",
        json.dumps(summary["search_configuration"], indent=2, sort_keys=True),
        "```",
        "",
        "## Teacher Runtime",
        "",
        "| Teacher | Total seconds |",
        "| --- | ---: |",
        *[
            f"| {teacher} | {seconds:.3f} |"
            for teacher, seconds in summary["teacher_runtime_seconds"].items()
        ],
        "",
        "## Frozen State Manifest",
        "",
        "```json",
        json.dumps(summary["probe_manifest"], indent=2, sort_keys=True),
        "```",
        "",
        "## Distribution Convergence",
        "",
        "| Comparison | JS | Top-1 agreement | Spearman |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in summary["distribution_convergence"]["comparisons"].items():
        row = value["global"]
        lines.append(
            f"| {name} | {row['js_divergence']:.5f} | {row['top1_agreement']:.4f} | {row['spearman']:.4f} |"
        )
    lines += [
        "",
        "## Top-Move Agreement Matrix",
        "",
        "| Comparison | Agreement | Change rate |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {value['global']['top1_agreement']:.4f} | {value['global']['top_move_change']:.4f} |"
        for name, value in summary["distribution_convergence"]["comparisons"].items()
    )
    lines += [
        "",
        "## Stability And Oscillation",
        "",
        "```json",
        json.dumps(summary["target_stability"], indent=2, sort_keys=True),
        "```",
        "",
        "## Disagreement Sets",
        "",
        "```json",
        json.dumps(
            {
                key: value
                for key, value in summary["disagreement_sets"].items()
                if key != "state_hashes"
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Causal Adjacent-Budget Results",
        "",
        "```json",
        json.dumps(summary["causal_adjacent_budget_results"], indent=2, sort_keys=True),
        "```",
        "",
        "## Absolute Teacher-Quality Ladder",
        "",
        "```json",
        json.dumps(
            summary["absolute_teacher_quality_ladder"], indent=2, sort_keys=True
        ),
        "```",
        "",
        "## Search-Q Calibration",
        "",
        "```json",
        json.dumps(summary["search_q_calibration"], indent=2, sort_keys=True),
        "```",
        "",
        "## Decision Rules",
        "",
        "```json",
        json.dumps(summary["decision_criteria"], indent=2, sort_keys=True),
        "```",
        "",
        "Per-state search and continuation records, including full child-Q values and paired continuation contexts, remain in the workdir. No replay or model training was generated.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current = Path(args.current)
    weights = current / "weights.json"
    if sha256_file(weights) != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")
    checkpoint = materialize_weights_json_checkpoint(
        weights_path=weights, out_path=workdir / "current.npz"
    )
    additional_selfplay = (
        Path(args.additional_selfplay)
        if args.additional_selfplay
        else harvest_additional_selfplay_states(
            workdir=workdir, checkpoint=checkpoint, seed=args.seed
        )
    )
    source_paths = {
        "pr176_standard_start_pilot": Path(args.pr176_pilot),
        "pr177_evaluation_diagnostic": Path(args.pr177_probe_states),
        "additional_standard_start_selfplay": additional_selfplay,
        "independent_opening_suite_diagnostic": Path(args.opening_suite),
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise RuntimeError("all four source files must exist")
    selector = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    source_rows = {
        "pr176_standard_start_pilot": _rows_for_source(
            read_jsonl(source_paths["pr176_standard_start_pilot"]),
            "pr176_standard_start_pilot",
        ),
        "pr177_evaluation_diagnostic": _rows_for_source(
            read_jsonl(source_paths["pr177_evaluation_diagnostic"]),
            "evaluation_opening_diagnostic",
        ),
        "additional_standard_start_selfplay": read_jsonl(additional_selfplay),
        "independent_opening_suite_diagnostic": read_jsonl(
            source_paths["independent_opening_suite_diagnostic"]
        ),
    }
    probe, manifest = select_probe_states(
        source_rows,
        evaluator=selector,
        seed=args.seed,
    )
    manifest["source_hashes"] = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    manifest["current_model_hash"] = sha256_file(weights)
    for row in probe:
        policy, _ = selector.evaluate(KalahGame.from_state(row["state"]))
        row["current_network_top_move"] = int(
            max(row["legal_moves"], key=lambda move: (float(policy[move]), -move))
        )
    write_jsonl(workdir / "convergence_probe_states.jsonl", probe)
    write_json(workdir / "convergence_probe_manifest.json", manifest)
    teacher_records = run_teacher_tasks(
        tasks=[{**row, "experiment_seed": args.seed} for row in probe],
        checkpoint=checkpoint,
        workers=args.workers,
    )
    write_jsonl(workdir / "teacher_search_records.jsonl", teacher_records)
    # Verify every probe under a second seed because zero-noise PUCT is expected
    # to be RNG-invariant with deterministic root extraction.
    rng_affects = any(
        row["rng_sensitivity_d128"]["different_visit_counts"]
        or row["rng_sensitivity_d128"]["different_top_move"]
        for row in teacher_records
    )
    disagreement = disagreement_sets(teacher_records)
    forced_tasks = build_forced_tasks(teacher_records, seed=args.seed)
    for task in forced_tasks:
        task["lower_move"] = next(
            r for r in teacher_records if r["state_hash"] == task["state_hash"]
        )["teachers"][task["lower_teacher"]]["top_move"]
        task["higher_move"] = next(
            r for r in teacher_records if r["state_hash"] == task["state_hash"]
        )["teachers"][task["higher_teacher"]]["top_move"]
    forced = (
        run_forced_tasks(
            tasks=forced_tasks, checkpoint=checkpoint, workers=args.workers
        )
        if forced_tasks
        else []
    )
    write_jsonl(workdir / "forced_continuation_records.jsonl", forced)
    causal, causal_rows = causal_summary(forced, seed=args.seed)
    summary = {
        "schema": SCHEMA,
        "current_weights_sha256": sha256_file(weights),
        "probe_manifest": manifest,
        "search_configuration": {
            "teacher_budgets": list(TEACHER_BUDGETS),
            "continuation_budgets": list(CONTINUATION_BUDGETS),
            "dirichlet_epsilon": 0.0,
            "c_puct": 1.25,
            "tactical_root_bias": 0.0,
            "root_policy_mode": "deterministic",
            "normalize_values": False,
            "value_transform": None,
            "value_trust_override": None,
            "teacher_rng_affects_zero_noise_search": rng_affects,
        },
        "teacher_runtime_seconds": {
            name: float(
                sum(row["teachers"][name]["runtime_seconds"] for row in teacher_records)
            )
            for name in ("D128", "D384", "D768", "D1200")
        },
        "distribution_convergence": convergence_summary(teacher_records),
        "target_stability": stability_summary(teacher_records),
        "disagreement_sets": disagreement,
        "causal_adjacent_budget_results": causal,
        "absolute_teacher_quality_ladder": causal_ladder(causal_rows),
        "search_q_calibration": q_calibration(causal_rows),
        "decision_criteria": {
            "puct_policy_targets_converge_with_budget": "JS(D128,D1200) >= JS(D384,D1200) >= JS(D768,D1200); D768/D1200 top-1 agreement >= 0.95; disagreement <= 0.05; non-negative causal mean at every adjacent step; and no >=64-state adjacent result has upper 95% CI below zero.",
            "stronger_puct_teacher_causally_better": "Both D768-D384 and D1200-D768 have positive means, lower 95% CI >= 0, >=64 disagreement states, and no >=24-state player/phase slice mean below -0.10.",
            "puct_target_distribution_converges_but_moves_do_not": "JS decreases and Spearman rises monotonically toward D1200, but at least one adjacent forced higher-minus-lower mean is non-positive.",
            "puct_target_instability_confirmed": "D768/D1200 disagreement > 0.10, alternating oscillations affect >=5% of probes, or an adjacent forced higher-minus-lower mean is negative.",
            "puct_search_q_miscalibration": "Any available teacher Q comparison has Pearson or Spearman <= 0.10, or >=25% high-Q-margin preferences are causally wrong.",
            "search_budget_not_primary_policy_bottleneck": "At least 95% are stable from D384 or earlier and no adjacent causal mean is positive.",
        },
        "teacher_records_sha256": sha256_file(workdir / "teacher_search_records.jsonl"),
        "forced_records_sha256": sha256_file(
            workdir / "forced_continuation_records.jsonl"
        ),
    }
    labels, next_action = classify(summary)
    summary["classification"] = labels[0]
    summary["classifications"] = labels
    summary["next_action"] = next_action
    write_json(workdir / "summary_metrics.json", summary)
    write_json(
        REPO_ROOT / "docs/data/alphazero-lite-denoised-puct-convergence-summary.json",
        summary,
    )
    (REPO_ROOT / "docs/alphazero-lite-denoised-puct-convergence-results.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"classifications": labels, "workdir": str(workdir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
