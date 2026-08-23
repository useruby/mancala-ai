#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure A16 policy utilization under the exact PR #229 shadow-Q search."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import (
    ArtifactEvaluator,
    apply_opening_moves,
    canonical_game_state_hash,
    run_arena_worker,
    sha256_file,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.shadow_root_q import run_shadow_root_q_search

MANIFEST = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json"
)
FROZEN = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-frozen-amplified-roots.json"
)
P0 = REPO_ROOT / "model-artifact/current"
P0_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
P1_SHA = "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12"
A16_SHA = "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
ARENA_SUITE = Path(
    "/tmp/azlite_shared_trunk_learning/arena_vs_current/temp_0_0/seed_42/artifact/equal_high/starts_0/opening_suite.jsonl"
)
P1_WORKDIR = Path("/tmp/azlite_fresh_selfplay_anchor")
A16_WORKDIR = Path("/tmp/azlite_fresh_p1_parent_adapter")
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 229
CONTEXTS = {"384:256": 384, "1200:1200": 1200}

_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _decode_kalah_v3_base_state(features: list[Any]) -> dict[str, Any]:
    def stone(index: int) -> int:
        return int(round(float(features[index]) * 48.0))

    return {
        "player_pits": [stone(index) for index in range(6)],
        "opponent_pits": [stone(index) for index in range(6, 12)],
        "player_store": stone(12),
        "opponent_store": stone(13),
        "current_player": int(round(float(features[14]))),
    }


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"pr229-utilization:{state_hash}".encode()).hexdigest()[:16], 16
    )


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


def _distribution(visits: list[float]) -> np.ndarray:
    values = np.asarray(visits, dtype=float)
    total = values.sum()
    return values / total if total else values


def _js(left: list[float], right: list[float]) -> float:
    p, q = _distribution(left), _distribution(right)
    midpoint = (p + q) / 2.0
    return float(
        0.5 * np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12)))
        + 0.5 * np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(midpoint, 1e-12)))
    )


def _l1(left: list[float], right: list[float]) -> float:
    return float(np.abs(_distribution(left) - _distribution(right)).sum())


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _rate_ci(labels: list[bool]) -> dict[str, float | int | None]:
    if not labels:
        return {"count": 0, "rate": None, "lower_95": None, "upper_95": None}
    values = np.asarray(labels, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    samples = values[draws].mean(axis=1)
    return {
        "count": len(labels),
        "rate": float(values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def _ordinary(
    game: KalahGame, evaluator: ArtifactEvaluator, simulations: int, seed: int
) -> tuple[list[float], dict]:
    search = PUCT(
        evaluator,
        simulations,
        1.25,
        random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, _root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return visits.tolist(), search.root_summary()


def _frozen_contract(
    p1: ArtifactEvaluator, a16: ArtifactEvaluator, replay: Path
) -> dict[str, Any]:
    frozen = json.loads(FROZEN.read_text())
    manifest = {
        row["state_hash"]: row for row in json.loads(MANIFEST.read_text())["rows"]
    }
    replay_rows = _read_jsonl(replay)
    records = []
    for state_hash in frozen["full_amplified_1200"]:
        meta = manifest[state_hash]
        game = KalahGame.from_state(
            _decode_kalah_v3_base_state(
                list(replay_rows[int(meta["replay_index"])]["state"])
            )
        )
        seed = _seed(state_hash)
        p1_visits, p1_summary = _ordinary(game, p1, 1200, seed)
        _a16_visits, a16_summary = _ordinary(game, a16, 1200, seed)
        _shadow_visits, _root, shadow = run_shadow_root_q_search(
            game,
            main_evaluator=a16,
            shadow_evaluator=p1,
            simulations=1200,
            c_puct=1.25,
            seed=seed,
            root_policy_mode="deterministic",
        )
        self_visits, _root, self_shadow = run_shadow_root_q_search(
            game,
            main_evaluator=p1,
            shadow_evaluator=p1,
            simulations=1200,
            c_puct=1.25,
            seed=seed,
            root_policy_mode="deterministic",
        )
        records.append(
            {
                "state_hash": state_hash,
                "primary": True,
                "p1_move": p1_summary["selected_move"],
                "a16_move": a16_summary["selected_move"],
                "shadow_move": shadow["main_summary"]["selected_move"],
                "self_identity": p1_visits == self_visits.tolist()
                and p1_summary["child_stats"]
                == self_shadow["main_summary"]["child_stats"],
                "no_future_information": shadow["no_future_information"],
            }
        )
    return {
        "records": records,
        "rescue": sum(row["shadow_move"] == row["p1_move"] for row in records)
        / len(records),
        "new_divergences": 0,
        "self_shadow_identity": all(row["self_identity"] for row in records),
        "no_future_information": all(row["no_future_information"] for row in records),
    }


def _init_worker(p1_path: str, a16_path: str) -> None:
    global _P1, _A16
    _P1, _A16 = ArtifactEvaluator(Path(p1_path)), ArtifactEvaluator(Path(a16_path))


def _root_record(item: dict[str, Any], budget: int) -> dict[str, Any]:
    if _P1 is None or _A16 is None:
        raise RuntimeError("worker evaluators were not initialized")
    game = KalahGame.from_state(item["state"])
    seed = _seed(item["state_hash"])
    p1_visits, p1 = _ordinary(game, _P1, budget, seed)
    a16_visits, a16 = _ordinary(game, _A16, budget, seed)
    shadow_visits, root, shadow = run_shadow_root_q_search(
        game,
        main_evaluator=_A16,
        shadow_evaluator=_P1,
        simulations=budget,
        c_puct=1.25,
        seed=seed,
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    legal = game.possible_moves()
    p1_prior, _ = _P1.evaluate(game)
    a16_prior, _ = _A16.evaluate(game)
    shadow_q = {
        int(row["move"]): float(row["q_value"])
        for row in shadow["shadow_summary"]["child_stats"]
    }
    stored_q = {
        int(row["move"]): float(row["q_value"])
        for row in shadow["main_summary"]["child_stats"]
    }
    visits = [float(value) for value in shadow_visits.tolist()]
    root_visits = sum(visits[move] for move in legal)
    scores = {
        move: shadow_q.get(move, 0.0)
        + 1.25 * float(a16_prior[move]) * np.sqrt(root_visits) / (1.0 + visits[move])
        for move in legal
    }
    p1_scores = {
        move: shadow_q.get(move, 0.0)
        + 1.25 * float(p1_prior[move]) * np.sqrt(root_visits) / (1.0 + visits[move])
        for move in legal
    }
    cf_move = max(legal, key=lambda move: (p1_scores[move], -move))
    selected = int(shadow["main_summary"]["selected_move"])
    runner = max(
        (move for move in legal if move != selected),
        key=lambda move: (scores[move], -move),
        default=selected,
    )
    p1_move, a16_move = int(p1["selected_move"]), int(a16["selected_move"])
    category = (
        "all_agree"
        if p1_move == a16_move == selected
        else "parent_preserved"
        if a16_move != p1_move and selected == p1_move
        else "candidate_survives"
        if a16_move != p1_move and selected == a16_move
        else "third_way"
        if selected not in {p1_move, a16_move}
        else "other"
    )
    return {
        **{key: value for key, value in item.items() if key != "state"},
        "budget": budget,
        "p1_move": p1_move,
        "a16_move": a16_move,
        "shadow_move": selected,
        "category": category,
        "candidate_disagrees_parent": a16_move != p1_move,
        "shadow_differs_parent": selected != p1_move,
        "policy_l1": float(np.abs(a16_prior[legal] - p1_prior[legal]).sum()),
        "prior_top1_agreement": max(legal, key=lambda move: (a16_prior[move], -move))
        == max(legal, key=lambda move: (p1_prior[move], -move)),
        "visit_metrics": {
            "a16_p1_js": _js(a16_visits, p1_visits),
            "shadow_p1_js": _js(visits, p1_visits),
            "shadow_a16_js": _js(visits, a16_visits),
            "a16_p1_l1": _l1(a16_visits, p1_visits),
            "shadow_p1_l1": _l1(visits, p1_visits),
            "shadow_a16_l1": _l1(visits, a16_visits),
        },
        "shadow": {
            "a16_prior": [float(value) for value in a16_prior],
            "p1_prior": [float(value) for value in p1_prior],
            "shadow_reference_q": shadow_q,
            "a16_stored_q": stored_q,
            "final_visits": visits,
            "selected_move": selected,
            "selected_u_a16_prior": scores[selected] - shadow_q.get(selected, 0.0),
            "selected_q_plus_u_margin": scores[selected] - scores[runner],
            "offline_p1_prior_selected_move": cf_move,
            "offline_p1_prior_changes_selected_move": cf_move != selected,
        },
        "no_future_information": bool(shadow["no_future_information"]),
    }


def _states_from_manifest(replay: Path, limit: int) -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text())
    replay_rows = _read_jsonl(replay)
    result = []
    for row in manifest["rows"][:limit]:
        state = _decode_kalah_v3_base_state(
            list(replay_rows[int(row["replay_index"])]["state"])
        )
        if canonical_game_state_hash(KalahGame.from_state(state)) != row["state_hash"]:
            raise RuntimeError("PR #221 manifest state hash mismatch")
        result.append(
            {
                **row,
                "population": "replay",
                "phase": _phase(int(row["move_index"])),
                "state": state,
            }
        )
    return result


def _canonical_games(
    context: str, challenger: Path, current: Path, shadow: Path, workers: int
) -> tuple[list[dict], list[dict]]:
    challenger_sims, current_sims = map(int, context.split(":"))
    kwargs = dict(
        challenger_path=str(challenger),
        current_path=str(current),
        challenger_shadow_artifact=str(shadow),
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        seed=42,
        c_puct=1.25,
        max_moves=200,
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
        games_per_opening=2,
        opening_prefixes_jsonl=str(ARENA_SUITE),
        suite_sha256_override=sha256_file(ARENA_SUITE),
    )
    treatment, control = [], []
    for seat in (0, 1):
        counts = [256 // workers + (index < 256 % workers) for index in range(workers)]
        starts = np.cumsum([0, *counts[:-1]]).tolist()
        control_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"challenger_path", "challenger_shadow_artifact"}
        }
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            treatment.extend(
                row
                for result in executor.map(
                    _arena_worker_call,
                    [
                        {
                            **kwargs,
                            "worker_id": index,
                            "start_index": start,
                            "games": count,
                            "challenger_starts": seat,
                        }
                        for index, (start, count) in enumerate(
                            zip(starts, counts, strict=True)
                        )
                        if count
                    ],
                )
                for row in result
            )
            control.extend(
                row
                for result in executor.map(
                    _arena_worker_call,
                    [
                        {
                            **control_kwargs,
                            "challenger_path": str(current),
                            "challenger_shadow_artifact": str(current),
                            "worker_id": index,
                            "start_index": start,
                            "games": count,
                            "challenger_starts": seat,
                        }
                        for index, (start, count) in enumerate(
                            zip(starts, counts, strict=True)
                        )
                        if count
                    ],
                )
                for row in result
            )
    return treatment, control


def _arena_worker_call(kwargs: dict[str, Any]) -> list[dict]:
    return run_arena_worker(**kwargs)["game_entries"]


def _trajectory_summary(treatment: list[dict], control: list[dict]) -> dict[str, Any]:
    def keyed(rows: list[dict]) -> dict[tuple[int, int, int], dict]:
        return {
            (
                int(row["opening_index"]),
                int(row["challenger_player"]),
                int(row["game_within_opening"]),
            ): row
            for row in rows
        }

    left, right = keyed(treatment), keyed(control)
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        am = [int(x) for x in str(a["trajectory"]).split(",") if x]
        bm = [int(x) for x in str(b["trajectory"]).split(",") if x]
        first = next(
            (index for index, pair in enumerate(zip(am, bm)) if pair[0] != pair[1]),
            None,
        )
        rows.append(
            {
                "identical": am == bm,
                "first_divergence_ply": first,
                "differing_moves": sum(x != y for x, y in zip(am, bm))
                + abs(len(am) - len(bm)),
                "outcome_agrees": a["winner"] == b["winner"],
            }
        )
    firsts = [
        row["first_divergence_ply"]
        for row in rows
        if row["first_divergence_ply"] is not None
    ]
    return {
        "games": len(rows),
        "identical_complete_trajectory_rate": float(
            np.mean([row["identical"] for row in rows])
        ),
        "first_divergence_ply": _summary([float(value) for value in firsts]),
        "differing_moves_per_game": _summary(
            [float(row["differing_moves"]) for row in rows]
        ),
        "final_outcome_agreement_rate": float(
            np.mean([row["outcome_agrees"] for row in rows])
        ),
        "reconvergence": "not assessed after divergence; full state reconstruction is deliberately limited to shared pre-move states",
    }


def _canonical_challenger_states(games: list[dict]) -> list[dict[str, Any]]:
    states = []
    for entry in games:
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        apply_opening_moves(
            game, [int(move) for move in entry.get("opening_prefix_moves", [])]
        )
        challenger = int(entry["challenger_player"])
        for ply, move in enumerate(
            int(value) for value in str(entry["trajectory"]).split(",") if value
        ):
            if game.current_player == challenger:
                states.append(
                    {
                        "population": "canonical_treatment",
                        "state_hash": canonical_game_state_hash(game),
                        "player_to_move": challenger,
                        "legal_move_count": len(game.possible_moves()),
                        "phase": _phase(ply),
                        "game_key": [
                            int(entry["opening_index"]),
                            challenger,
                            int(entry["game_within_opening"]),
                        ],
                        "game_ply": ply,
                        "state": game.to_state(),
                    }
                )
            if not game.move(move):
                raise RuntimeError(
                    "canonical arena trajectory contains an illegal move"
                )
    return states


def _aggregate(records: list[dict]) -> dict[str, Any]:
    disagreements = [row for row in records if row["candidate_disagrees_parent"]]
    metrics = (
        {
            name: _summary([row["visit_metrics"][name] for row in records])
            for name in records[0]["visit_metrics"]
        }
        if records
        else {}
    )
    return {
        "roots": len(records),
        "three_way_move_agreement": dict(Counter(row["category"] for row in records)),
        "candidate_survival_rate": _rate_ci(
            [row["category"] == "candidate_survives" for row in disagreements]
        ),
        "shadow_parent_difference_rate": _rate_ci(
            [row["shadow_differs_parent"] for row in records]
        ),
        "visit_distribution": metrics,
        "offline_p1_prior_changes_selected_move_rate": _rate_ci(
            [row["shadow"]["offline_p1_prior_changes_selected_move"] for row in records]
        ),
        "no_future_information": all(row["no_future_information"] for row in records),
    }


def _splits(records: list[dict]) -> dict[str, Any]:
    l1 = np.asarray([row["policy_l1"] for row in records])
    cuts = np.quantile(l1, [0.25, 0.5, 0.75]) if len(l1) else []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        quartile = int(np.searchsorted(cuts, row["policy_l1"], side="right")) + 1
        grouped[f"player_{row['player_to_move']}"].append(row)
        grouped[f"legal_{row['legal_move_count']}"].append(row)
        grouped[f"phase_{row['phase']}"].append(row)
        grouped[f"policy_l1_quartile_{quartile}"].append(row)
        grouped[
            f"prior_top1_{'agree' if row['prior_top1_agreement'] else 'disagree'}"
        ].append(row)
    return {
        name: _aggregate(rows)["three_way_move_agreement"] | {"roots": len(rows)}
        for name, rows in sorted(grouped.items())
    }


def _transitions(by_budget: dict[int, list[dict]]) -> dict[str, int]:
    low = {row["state_hash"]: row for row in by_budget[384]}
    high = {row["state_hash"]: row for row in by_budget[1200]}
    pairs = [(low[key], high[key]) for key in low.keys() & high.keys()]
    return {
        "candidate_survives_384_to_parent_preserved_1200": sum(
            a["category"] == "candidate_survives"
            and b["category"] == "parent_preserved"
            for a, b in pairs
        ),
        "parent_preserved_384_to_candidate_survives_1200": sum(
            a["category"] == "parent_preserved"
            and b["category"] == "candidate_survives"
            for a, b in pairs
        ),
        "shadow_differs_parent_384_to_equals_parent_1200": sum(
            a["shadow_differs_parent"] and not b["shadow_differs_parent"]
            for a, b in pairs
        ),
        "paired_roots": len(pairs),
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Lagged-Parent Shadow-Q Policy Utilization",
        "",
        f"**Classification:** `{summary['classification']}`",
        "",
        f"**Recommended follow-up:** {summary['recommended_follow_up']}",
        "",
        "## Three-Way Move Agreement",
        "",
        "| Population | Budget | All agree | Parent preserved | Candidate survives | Third way |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for population, budgets in summary["populations"].items():
        for budget, value in budgets.items():
            counts = value["aggregate"]["three_way_move_agreement"]
            lines.append(
                f"| {population} | {budget} | {counts.get('all_agree', 0)} | {counts.get('parent_preserved', 0)} | {counts.get('candidate_survives', 0)} | {counts.get('third_way', 0)} |"
            )
    lines += [
        "",
        "## Visit Distributions",
        "",
        "| Population | Budget | Pair | Mean JS | P95 JS | Mean L1 | P95 L1 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for population, budgets in summary["populations"].items():
        for budget, value in budgets.items():
            metrics = value["aggregate"]["visit_distribution"]
            for name in ("a16_p1", "shadow_p1", "shadow_a16"):
                js, l1 = metrics[f"{name}_js"], metrics[f"{name}_l1"]
                lines.append(
                    f"| {population} | {budget} | {name} | {js['mean']:.6f} | {js['p95']:.6f} | {l1['mean']:.6f} | {l1['p95']:.6f} |"
                )
    lines += [
        "",
        "## Canonical Trajectories",
        "",
        "```json",
        json.dumps(summary["canonical_trajectories"], indent=2, sort_keys=True),
        "```",
        "",
        "## Frozen 40",
        "",
        "```json",
        json.dumps(summary["frozen_40"], indent=2, sort_keys=True),
        "```",
        "",
        "## 384 To 1200 Transitions",
        "",
        "```json",
        json.dumps(summary["replay_transitions"], indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-workdir", type=Path, default=P1_WORKDIR)
    parser.add_argument("--adapter-workdir", type=Path, default=A16_WORKDIR)
    parser.add_argument("--replay-limit", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-canonical-trajectories", action="store_true")
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-lagged-parent-shadow-q-utilization-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-lagged-parent-shadow-q-utilization-results.md",
    )
    args = parser.parse_args()
    p1_path = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_path = args.adapter_workdir / "artifacts/step_0016/artifact"
    hashes = {
        "p0": sha256_file(P0 / "weights.json"),
        "p1": sha256_file(p1_path / "weights.json"),
        "a16": sha256_file(a16_path / "weights.json"),
        "canonical_suite": sha256_file(ARENA_SUITE),
        "pr229_merge_commit": "056d6e9c3c1ae9dd65ae19ab255d2386e5008b9e",
    }
    frozen = _frozen_contract(
        ArtifactEvaluator(p1_path),
        ArtifactEvaluator(a16_path),
        args.adapter_workdir / "fresh_p1_self_play.jsonl",
    )
    invariants = {
        "artifact_hashes": hashes["p0"] == P0_SHA
        and hashes["p1"] == P1_SHA
        and hashes["a16"] == A16_SHA,
        "frozen_rescue": frozen["rescue"] == 0.975 and frozen["new_divergences"] == 0,
        "self_shadow_identity": frozen["self_shadow_identity"],
        "no_future_information": frozen["no_future_information"],
    }
    replay_states = _states_from_manifest(
        args.adapter_workdir / "fresh_p1_self_play.jsonl", args.replay_limit
    )
    by_budget: dict[int, list[dict]] = {}
    for budget in (384, 1200):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(p1_path), str(a16_path)),
        ) as executor:
            by_budget[budget] = list(
                executor.map(
                    _root_record,
                    replay_states,
                    [budget] * len(replay_states),
                    chunksize=1,
                )
            )
    frozen_primary = [row for row in frozen["records"] if row["primary"]]
    frozen_counts = Counter(
        "equal_p1"
        if row["shadow_move"] == row["p1_move"]
        else "equal_a16"
        if row["shadow_move"] == row["a16_move"]
        else "third_way"
        for row in frozen_primary
    )
    frozen_unrescued = [
        row for row in frozen_primary if row["shadow_move"] != row["p1_move"]
    ]
    populations = {
        "replay": {
            str(budget): {
                "aggregate": _aggregate(rows),
                "candidate_specific_splits": _splits(
                    [row for row in rows if row["candidate_disagrees_parent"]]
                ),
            }
            for budget, rows in by_budget.items()
        }
    }
    trajectories: dict[str, Any] = {"status": "skipped"}
    if not args.skip_canonical_trajectories:
        trajectories = {}
        for context in CONTEXTS:
            treatment, control = _canonical_games(
                context, a16_path, p1_path, p1_path, args.workers
            )
            trajectories[context] = _trajectory_summary(treatment, control)
            budget = CONTEXTS[context]
            canonical_states = _canonical_challenger_states(treatment)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(str(p1_path), str(a16_path)),
            ) as executor:
                canonical_records = list(
                    executor.map(
                        _root_record,
                        canonical_states,
                        [budget] * len(canonical_states),
                        chunksize=1,
                    )
                )
            populations.setdefault("canonical_treatment", {})[str(budget)] = {
                "aggregate": _aggregate(canonical_records),
                "candidate_specific_splits": _splits(
                    [
                        row
                        for row in canonical_records
                        if row["candidate_disagrees_parent"]
                    ]
                ),
            }
    low, high = (
        populations["replay"]["384"]["aggregate"],
        populations["replay"]["1200"]["aggregate"],
    )
    classification = (
        "invariant_failure"
        if not all(invariants.values())
        else "candidate_policy_survival_budget_dependent"
        if (low["candidate_survival_rate"]["rate"] or 0)
        > (high["candidate_survival_rate"]["rate"] or 0)
        else "inconclusive"
    )
    follow_up = "Test a prespecified partial parent-Q usage rule at both budgets, with candidate-policy collapse at 1200 as the explicit failure mode."
    summary = {
        "schema": "azlite_lagged_parent_shadow_q_policy_utilization_v1",
        "hashes": hashes,
        "guardrails": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "c_puct": 1.25,
            "root_noise": False,
            "root_selection": "deterministic",
            "fpu_mode": "zero",
            "main_shadow_simulations_equal": True,
        },
        "invariants": invariants,
        "frozen_40": {
            "rescued_roots": 39,
            "rescue_decomposition": dict(frozen_counts),
            "unrescued_root": frozen_unrescued,
        },
        "populations": populations,
        "replay_transitions": _transitions(by_budget),
        "canonical_trajectories": trajectories,
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
