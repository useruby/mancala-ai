#!/usr/bin/env python3
# ruff: noqa: E402
"""Canonical, endpoint-exact policy interpolation preflight for PR #182.

This is deliberately diagnostic-only.  It reads the immutable PR #182 replay
and artifacts, never exports weights or generates training data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.evaluation_seed_contract import (
    derive_search_seed,
    stable_hash,
    stable_seed,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
from ml.alphazero_lite.run_policy_prior_search_amplification_audit import EXPECTED

ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
POLICY_KEYS = frozenset({"w_policy", "b_policy", "w_policy_hidden", "b_policy_hidden"})
CACHE_SCHEMA_VERSION = "azlite_canonical_parallel_benchmark_cache_v2"
_WORKER_CURRENT: ArtifactEvaluator | None = None
_WORKER_CANDIDATES: dict[str, ArtifactEvaluator] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cache_manifest(
    *,
    suite_sha256: str,
    challenger_artifact_sha256: str,
    opponent_artifact_sha256: str,
    budget_pairs: tuple[str, ...],
    c_puct_schedule: dict[str, float],
    tactical_root_bias: float,
    root_policy: str,
    seed_contract: str,
    base_seed: int,
    games_per_opening: int,
) -> dict[str, Any]:
    """Identity for a reusable parallel benchmark result.

    Paths and labels are deliberately excluded: a cache is valid only for the
    immutable inputs and runtime treatment that determine game records.
    """
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "suite_sha256": suite_sha256,
        "challenger_artifact_sha256": challenger_artifact_sha256,
        "opponent_artifact_sha256": opponent_artifact_sha256,
        "budget_pairs": list(budget_pairs),
        "c_puct_resolution": dict(sorted(c_puct_schedule.items())),
        "tactical_root_bias": float(tactical_root_bias),
        "root_policy": root_policy,
        "seed_contract": seed_contract,
        "base_seed": int(base_seed),
        "games_per_opening": int(games_per_opening),
        "code_config_schema": CACHE_SCHEMA_VERSION,
    }


def cache_status(cache: Path, expected_manifest: dict[str, Any]) -> tuple[bool, str]:
    """Return whether both the cache identity and its result hash verify."""
    if not cache.is_file():
        return False, "cache_missing"
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "cache_unreadable"
    if payload.get("manifest") != expected_manifest:
        return False, "manifest_mismatch"
    records = payload.get("records")
    if not isinstance(records, list) or payload.get("records_sha256") != stable_hash(
        records
    ):
        return False, "output_hash_mismatch"
    return True, "reused"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class CanonicalInterpolatedPolicyEvaluator:
    """Interpolate legal log probabilities while preserving exact endpoints.

    Returning the artifact array directly at alpha 0/1 is required because an
    otherwise algebraically equivalent log/exp round trip changes float32 bits.
    """

    def __init__(
        self, current: ArtifactEvaluator, candidate: ArtifactEvaluator, alpha: float
    ):
        if alpha not in ALPHAS:
            raise ValueError(f"alpha must be one of {ALPHAS}")
        self.current, self.candidate, self.alpha = current, candidate, alpha

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        current_policy, current_value = self.current.evaluate(game)
        if self.alpha == 0.0:
            return current_policy, float(current_value)
        candidate_policy, _candidate_value = self.candidate.evaluate(game)
        if self.alpha == 1.0:
            return candidate_policy, float(current_value)
        legal = game.possible_moves()
        result = np.zeros_like(current_policy)
        if legal:
            log_policy = (1.0 - self.alpha) * np.log(current_policy[legal])
            log_policy += self.alpha * np.log(candidate_policy[legal])
            log_policy -= np.max(log_policy)
            result[legal] = np.exp(log_policy) / np.exp(log_policy).sum()
        return result, float(current_value)


def run_search(
    evaluator: Any, state: dict[str, Any], budget: int, seed: int, c_puct: float = 1.25
) -> dict[str, Any]:
    # Arena owns the production PUCT invocation.  Keeping reconciliation on
    # this primitive prevents a second direct-search implementation drifting.
    result = evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=budget,
        seed=seed,
        c_puct=c_puct,
        search_options={
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
        },
    )
    move = result["selected_move"]
    visits = result["visits"]
    selected_move = int(move) if move is not None else -1
    root_value = float(result.get("search_root_value", result["value"]))
    return {
        "seed": seed,
        "move": selected_move,
        "visits": [int(value) for value in visits],
        "root_value": root_value,
        "search_hash": stable_hash(
            {
                "move": selected_move,
                "visits": [int(value) for value in visits],
                "root_value": root_value,
            }
        ),
    }


def bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), size=(10_000, len(data)))].mean(axis=1)
    return {
        "n": len(values),
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "samples": 10_000,
    }


def opening_state(prefix: list[int]) -> dict[str, Any]:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in prefix:
        if not game.move(move):
            raise RuntimeError("illegal canonical opening prefix")
    return game.to_state()


def canonical_game(
    *,
    opening: dict[str, Any],
    opening_index: int,
    challenger: Any,
    current: Any,
    challenger_player: int,
    challenger_budget: int,
    current_budget: int,
    seed: int,
    suite_hash: str,
    label: str,
) -> dict[str, Any]:
    game = KalahGame.from_state(opening)
    opening_hash, moves, trace = stable_hash(opening), [], []
    for ply in range(200):
        if game.over() or not game.possible_moves():
            break
        role = "challenger" if game.current_player == challenger_player else "current"
        evaluator = challenger if role == "challenger" else current
        budget = challenger_budget if role == "challenger" else current_budget
        derived_seed, seed_hash = derive_search_seed(
            base_seed=seed,
            suite_sha256=suite_hash,
            opening_index=opening_index,
            opening_state_hash=opening_hash,
            challenger_player=challenger_player,
            # Seat is already represented by challenger_player.  This runner
            # evaluates one game per opening/seat, matching arena's index 0.
            game_within_opening=0,
            ply=ply,
            canonical_current_state_hash=stable_hash(game.to_state()),
            acting_role=role,
        )
        result = run_search(
            evaluator,
            game.to_state(),
            budget,
            derived_seed,
            c_puct=0.90 if challenger_budget == current_budget == 768 else 1.25,
        )
        moves.append(result["move"])
        trace.append(
            {
                "ply": ply,
                "acting_role": role,
                "canonical_current_state_hash": stable_hash(game.to_state()),
                "seed_context_hash": seed_hash,
                "derived_search_seed": result["seed"],
                "selected_move": result["move"],
                # Arena hashes its numpy visit vector, whose JSON form retains
                # float values (for example 4.0 rather than 4).
                "visit_hash": stable_hash([float(value) for value in result["visits"]]),
                "root_value": result["root_value"],
            }
        )
        if not game.move(game.pit_index(result["move"])):
            raise RuntimeError("canonical game selected illegal move")
    margin = int(
        game.captured_seeds[challenger_player]
        - game.captured_seeds[1 - challenger_player]
    )
    return {
        "label": label,
        "opening_index": opening_index,
        "challenger_player": challenger_player,
        "score": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "margin": margin,
        "trajectory_hash": stable_hash(moves),
        "search_hash": stable_hash(trace),
        "trace": trace,
        "final_score": [int(value) for value in game.captured_seeds],
    }


def canonical_benchmark(
    *,
    openings: list[dict[str, Any]],
    current: Any,
    challenger: Any,
    budgets: tuple[str, ...],
    seed: int,
    suite_hash: str,
    label: str,
) -> list[dict[str, Any]]:
    records = []
    for budget in budgets:
        challenger_budget, current_budget = (int(x) for x in budget.split(":"))
        for index, opening in enumerate(openings):
            for seat in (0, 1):
                records.append(
                    canonical_game(
                        opening=opening,
                        opening_index=index,
                        challenger=challenger,
                        current=current,
                        challenger_player=seat,
                        challenger_budget=challenger_budget,
                        current_budget=current_budget,
                        seed=seed,
                        suite_hash=suite_hash,
                        label=label,
                    )
                    | {"budget": budget}
                )
    return records


def _init_benchmark_worker(current_path: str, candidates: dict[str, str]) -> None:
    global _WORKER_CURRENT, _WORKER_CANDIDATES
    _WORKER_CURRENT = ArtifactEvaluator(Path(current_path))
    _WORKER_CANDIDATES = {
        name: ArtifactEvaluator(Path(path)) for name, path in candidates.items()
    }


def _benchmark_task(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CURRENT is None:
        raise RuntimeError("benchmark worker was not initialized")
    candidate = _WORKER_CANDIDATES[task["direction"]]
    alpha = float(task["alpha"])
    challenger: Any = (
        candidate
        if task["direct"]
        else CanonicalInterpolatedPolicyEvaluator(_WORKER_CURRENT, candidate, alpha)
    )
    challenger_budget, current_budget = (
        int(value) for value in task["budget"].split(":")
    )
    return canonical_game(
        opening=task["opening"],
        opening_index=task["opening_index"],
        challenger=challenger,
        current=_WORKER_CURRENT,
        challenger_player=task["challenger_player"],
        challenger_budget=challenger_budget,
        current_budget=current_budget,
        seed=42,
        suite_hash=task["suite_hash"],
        label=task["label"],
    ) | {"budget": task["budget"]}


def parallel_benchmark(
    *,
    executor: concurrent.futures.ProcessPoolExecutor,
    workdir: Path,
    label: str,
    direction: str,
    alpha: float,
    direct: bool,
    openings: list[dict[str, Any]],
    budgets: tuple[str, ...],
    suite_hash: str,
    challenger_artifact_sha256: str,
    opponent_artifact_sha256: str,
) -> list[dict[str, Any]]:
    cache = workdir / f"{label}_records.json"
    manifest = cache_manifest(
        suite_sha256=suite_hash,
        challenger_artifact_sha256=challenger_artifact_sha256,
        opponent_artifact_sha256=opponent_artifact_sha256,
        budget_pairs=budgets,
        c_puct_schedule={"default": 1.25, "768:768": 0.90},
        tactical_root_bias=0.0,
        root_policy="deterministic",
        seed_contract="azlite_eval_seed_v2",
        base_seed=42,
        games_per_opening=1,
    )
    reusable, _reason = cache_status(cache, manifest)
    if reusable:
        return json.loads(cache.read_text(encoding="utf-8"))["records"]
    tasks = [
        {
            "label": label,
            "direction": direction,
            "alpha": alpha,
            "direct": direct,
            "opening": opening,
            "opening_index": index,
            "challenger_player": seat,
            "budget": budget,
            "suite_hash": suite_hash,
        }
        for budget in budgets
        for index, opening in enumerate(openings)
        for seat in (0, 1)
    ]
    records = list(executor.map(_benchmark_task, tasks, chunksize=1))
    write_json(
        cache,
        {
            "manifest": manifest,
            "records": records,
            "records_sha256": stable_hash(records),
        },
    )
    return records


def opening_effect(
    left: list[dict[str, Any]], right: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    left_by_key = {
        (r["budget"], r["opening_index"], r["challenger_player"]): r for r in left
    }
    right_by_key = {
        (r["budget"], r["opening_index"], r["challenger_player"]): r for r in right
    }
    result = {}
    for budget in sorted({r["budget"] for r in left}):
        opening_deltas, seat = [], {"0": [], "1": []}
        for index in sorted(
            {r["opening_index"] for r in left if r["budget"] == budget}
        ):
            values = []
            for player in (0, 1):
                delta = (
                    left_by_key[budget, index, player]["score"]
                    - right_by_key[budget, index, player]["score"]
                )
                values.append(delta)
                seat[str(player)].append(delta)
            opening_deltas.append(float(np.mean(values)))
        result[budget] = {
            **bootstrap(opening_deltas, stable_seed(seed, budget, "opening")),
            "positive_openings": sum(v > 0 for v in opening_deltas),
            "zero_openings": sum(v == 0 for v in opening_deltas),
            "negative_openings": sum(v < 0 for v in opening_deltas),
            "player_seat_decomposition": {
                key: float(np.mean(value)) for key, value in seat.items()
            },
            "orientation_decomposition": {
                "candidate_challenger": float(np.mean(opening_deltas))
            },
        }
    return result


def phase_e_classification(benchmarks: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    evidence: dict[str, Any] = {}
    confirmed = False
    for direction, report in benchmarks.items():
        evidence[direction] = {}
        for alpha, comparisons in report["intermediate_minus_alpha1"].items():
            low = comparisons["384:256"]
            versus_current = report["intermediate_vs_current"][alpha]["384:256"]
            regressions = {
                budget: report["intermediate_vs_current"][alpha][budget]["mean"]
                for budget in ("768:768", "1200:1200", "1200:256")
            }
            seats = low["player_seat_decomposition"]
            passes = (
                low["lower_95"] > 0.0
                and versus_current["mean"] >= 0.0
                and all(value >= -0.03 for value in regressions.values())
                and seats["0"] >= 0.0
                and seats["1"] >= 0.0
            )
            evidence[direction][alpha] = {
                "passes": passes,
                "regressions": regressions,
                "seat_effects": seats,
            }
            confirmed = confirmed or passes
    return (
        "policy_update_overshoot_confirmed"
        if confirmed
        else "policy_update_overshoot_not_established",
        evidence,
    )


def write_final_outputs(summary: dict[str, Any]) -> None:
    output = (
        REPO_ROOT
        / "docs/data/alphazero-lite-canonical-policy-interpolation-summary.json"
    )
    write_json(output, summary)
    markdown = (
        "# Canonical Policy Interpolation Reconciliation\n\n"
        f"- Classification: `{summary['classification']}`\n"
        f"- Next action: {summary['next_action']}\n\n"
        "## Results\n\n```json\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n```\n"
    )
    (
        REPO_ROOT / "docs/alphazero-lite-canonical-policy-interpolation-results.md"
    ).write_text(markdown, encoding="utf-8")


def preflight_direction(
    *,
    current_path: Path,
    candidate_path: Path,
    current_checkpoint: Path,
    states: list[dict[str, Any]],
    direction: str,
) -> dict[str, Any]:
    current, candidate = (
        ArtifactEvaluator(current_path),
        ArtifactEvaluator(candidate_path),
    )
    candidate_checkpoint = candidate_path.parent / "checkpoint.npz"
    current_arrays, candidate_arrays = (
        dict(np.load(current_checkpoint)),
        dict(np.load(candidate_checkpoint)),
    )
    changed = sorted(
        key
        for key in current_arrays
        if not np.array_equal(current_arrays[key], candidate_arrays[key])
    )
    nonpolicy = sorted(set(changed) - POLICY_KEYS)
    endpoint_failures = []
    for index, state in enumerate(states):
        game = KalahGame.from_state(state)
        current_policy, current_value = current.evaluate(game)
        candidate_policy, _candidate_value = candidate.evaluate(game)
        for alpha in ALPHAS:
            policy, value = CanonicalInterpolatedPolicyEvaluator(
                current, candidate, alpha
            ).evaluate(game)
            if not np.array_equal(value, current_value):
                endpoint_failures.append(
                    {"state": index, "check": f"value_alpha_{alpha}"}
                )
            if alpha == 0.0 and not np.array_equal(policy, current_policy):
                endpoint_failures.append({"state": index, "check": "alpha0_policy"})
            if alpha == 1.0 and not np.array_equal(policy, candidate_policy):
                endpoint_failures.append({"state": index, "check": "alpha1_policy"})
            legal = game.possible_moves()
            if legal and not np.array_equal(
                policy[np.setdiff1d(np.arange(6), legal)],
                np.zeros(6 - len(legal), dtype=policy.dtype),
            ):
                endpoint_failures.append(
                    {"state": index, "check": f"legal_mask_alpha_{alpha}"}
                )
        seed = stable_seed(
            "canonical_interpolation_preflight", direction, index, stable_hash(state)
        )
        direct = run_search(candidate, state, 384, seed)
        wrapped = run_search(
            CanonicalInterpolatedPolicyEvaluator(current, candidate, 1.0),
            state,
            384,
            seed,
        )
        if direct != wrapped:
            endpoint_failures.append(
                {
                    "state": index,
                    "check": "alpha1_search",
                    "direct": direct,
                    "wrapper": wrapped,
                }
            )
    return {
        "states": len(states),
        "artifact_sha256": sha256_file(candidate_checkpoint),
        "trunk_and_value_hashes_current": not nonpolicy,
        "changed_checkpoint_keys": changed,
        "failures": endpoint_failures,
        "passes": not endpoint_failures and not nonpolicy,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_canonical_policy_interpolation"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--source-workdir", default="/tmp/azlite_stronger_policy_teacher"
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--run-benchmarks", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    current, source, workdir = (
        Path(args.current),
        Path(args.source_workdir),
        Path(args.workdir),
    )
    current_checkpoint = materialize_weights_json_checkpoint(
        weights_path=current / "weights.json", out_path=workdir / "current.npz"
    )
    if sha256_file(current / "weights.json") != EXPECTED["current"]:
        raise RuntimeError("current artifact SHA256 mismatch")
    corpus = read_jsonl(source / "trajectory_replay.jsonl")
    states = [row["raw_state"] for row in corpus[: args.states]]
    if len(states) != args.states:
        raise RuntimeError("insufficient immutable PR #182 states")
    results = {}
    for direction in ("D384", "D1200"):
        artifact = source / f"d{direction[1:]}_policy_teacher_e1_run_a" / "artifact"
        results[direction] = preflight_direction(
            current_path=current,
            candidate_path=artifact,
            current_checkpoint=current_checkpoint,
            states=states,
            direction=direction,
        )
        if results[direction]["artifact_sha256"] != EXPECTED[direction]:
            raise RuntimeError(f"{direction} artifact SHA256 mismatch")
    passed = all(value["passes"] for value in results.values())
    benchmarks: dict[str, Any] = {}
    if passed and args.run_benchmarks:
        suite_rows = read_jsonl(Path(args.medium_suite))
        if len(suite_rows) != 128:
            raise RuntimeError(
                "canonical medium suite must contain exactly 128 openings"
            )
        openings = [
            opening_state([int(move) for move in row["prefix_moves"]])
            for row in suite_rows
        ]
        suite_hash = sha256_file(Path(args.medium_suite))
        phase_b_budgets = (
            "384:256",
            "768:256",
            "768:768",
            "1200:1200",
            "1200:256",
            "256:768",
        )
        phase_d_budgets = ("384:256", "768:768", "1200:1200", "1200:256")
        candidate_paths = {
            direction: str(
                source / f"d{direction[1:]}_policy_teacher_e1_run_a" / "artifact"
            )
            for direction in ("D384", "D1200")
        }
        current_artifact_sha256 = sha256_file(current / "weights.json")
        candidate_artifact_sha256 = {
            direction: sha256_file(Path(path) / "weights.json")
            for direction, path in candidate_paths.items()
        }
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_benchmark_worker,
            initargs=(str(current), candidate_paths),
        ) as executor:
            current_records = parallel_benchmark(
                executor=executor,
                workdir=workdir,
                label="current",
                direction="D384",
                alpha=0.0,
                direct=False,
                openings=openings,
                budgets=phase_b_budgets,
                suite_hash=suite_hash,
                challenger_artifact_sha256=current_artifact_sha256,
                opponent_artifact_sha256=current_artifact_sha256,
            )
            for direction in ("D384", "D1200"):
                direct = parallel_benchmark(
                    executor=executor,
                    workdir=workdir,
                    label=f"direct_{direction}",
                    direction=direction,
                    alpha=1.0,
                    direct=True,
                    openings=openings,
                    budgets=phase_b_budgets,
                    suite_hash=suite_hash,
                    challenger_artifact_sha256=candidate_artifact_sha256[direction],
                    opponent_artifact_sha256=current_artifact_sha256,
                )
                wrapped = parallel_benchmark(
                    executor=executor,
                    workdir=workdir,
                    label=f"alpha1_{direction}",
                    direction=direction,
                    alpha=1.0,
                    direct=False,
                    openings=openings,
                    budgets=phase_b_budgets,
                    suite_hash=suite_hash,
                    challenger_artifact_sha256=candidate_artifact_sha256[direction],
                    opponent_artifact_sha256=current_artifact_sha256,
                )
                equivalence = [
                    {
                        key: left[key] == right[key]
                        for key in ("score", "margin", "trajectory_hash", "search_hash")
                    }
                    for left, right in zip(direct, wrapped)
                ]
                if not all(all(check.values()) for check in equivalence):
                    raise RuntimeError(
                        "interpolation_evaluator_not_artifact_equivalent"
                    )
                direction_result: dict[str, Any] = {
                    "direct_vs_current": opening_effect(direct, current_records, 42),
                    "alpha1_vs_current": opening_effect(wrapped, current_records, 42),
                    "direct_equals_alpha1_per_opening": True,
                }
                treatments = {"1.00": wrapped}
                for alpha in (0.25, 0.50, 0.75):
                    treatments[f"{alpha:.2f}"] = parallel_benchmark(
                        executor=executor,
                        workdir=workdir,
                        label=f"{direction}@{alpha:.2f}",
                        direction=direction,
                        alpha=alpha,
                        direct=False,
                        openings=openings,
                        budgets=phase_d_budgets,
                        suite_hash=suite_hash,
                        challenger_artifact_sha256=candidate_artifact_sha256[direction],
                        opponent_artifact_sha256=current_artifact_sha256,
                    )
                direction_result["intermediate_vs_current"] = {
                    alpha: opening_effect(records, current_records, 42)
                    for alpha, records in treatments.items()
                }
                direction_result["intermediate_minus_alpha1"] = {
                    alpha: opening_effect(records, wrapped, 42)
                    for alpha, records in treatments.items()
                    if alpha != "1.00"
                }
                benchmarks[direction] = direction_result
    summary = {
        "schema": "azlite_canonical_policy_interpolation_v1",
        "guardrails": {
            "training": False,
            "promotion": False,
            "new_replay": False,
            "new_teacher_budget": False,
        },
        "canonical_profile": {
            "tactical_root_bias": 0.0,
            "default_c_puct": 1.25,
            "c_puct_schedule": {"768:768": 0.90},
            "seed_contract": "azlite_eval_seed_v2",
        },
        "phase_a": results,
        "canonical_benchmark": benchmarks or None,
        "classification": "canonical_interpolation_evaluator_reconciled"
        if passed
        else "interpolation_evaluator_not_artifact_equivalent",
        "next_action": "Proceed to canonical benchmark phases."
        if passed
        else "Stop before trust-region or benchmark conclusions.",
    }
    if benchmarks:
        overshoot, overshoot_evidence = phase_e_classification(benchmarks)
        summary["phase_e"] = {
            "method": "10,000-sample opening-clustered bootstrap over 128 unique openings",
            "classification": overshoot,
            "evidence": overshoot_evidence,
        }
        summary["phases_f_to_h"] = {
            "classification": "interpolation_results_statistically_inconclusive",
            "reason": "The PR #183 raw first-divergence and forced-continuation manifests are not present in this workspace; its committed aggregate summary cannot be re-clustered or provenance-verified.",
        }
        summary["classification"] = overshoot
        summary["next_action"] = (
            "Authorize only the preregistered smaller deterministic policy update."
            if overshoot == "policy_update_overshoot_confirmed"
            else "Do not train another policy-head lane; close policy-step-size tuning."
        )
    write_json(workdir / "summary_metrics.json", summary)
    if not passed or args.run_benchmarks:
        write_final_outputs(summary)
    print(
        json.dumps({"classification": summary["classification"], "states": args.states})
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
