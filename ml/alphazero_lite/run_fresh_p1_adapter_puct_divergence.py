#!/usr/bin/env python3
"""Trace PR #214 A16/P1 PUCT divergence without training or self-play."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    _suite,
)
from ml.alphazero_lite.self_play import PUCT

CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1200)


class StatePriorOverride:
    """Replace only nominated expanded-node priors; values remain candidate values."""

    def __init__(self, parent: ArtifactEvaluator, state_hashes: set[str]) -> None:
        self.parent = parent
        self.state_hashes = state_hashes

    def __call__(
        self, *, game, legal_moves: list[int], priors: np.ndarray, depth: int
    ) -> np.ndarray:
        if canonical_game_state_hash(game) not in self.state_hashes:
            return priors
        parent_policy, _value = self.parent.evaluate(game)
        return np.asarray(parent_policy, dtype=np.float32)


def _records(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[
            (
                int(row["opening_index"]),
                int(row["challenger_player"]),
                int(row["game_within_opening"]),
            )
        ] = row
    return result


def _state_before(prefix: list[int], moves: list[int], ply: int) -> KalahGame:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    # Match arena.apply_opening_moves exactly. Historical suite prefixes are
    # interpreted by that function before the recorded absolute trajectories.
    for move in prefix:
        if game.over() or move not in game.possible_moves() or not game.move(int(move)):
            break
    for move in moves[:ply]:
        if not game.move(int(move)):
            raise RuntimeError("arena trajectory contains an illegal move")
    return game


def first_game_divergences(treatment: dict, control: dict) -> list[dict[str, Any]]:
    rows = []
    for key, treated in sorted(treatment.items()):
        baseline = control[key]
        left = [int(move) for move in treated["trajectory"].split(",") if move]
        right = [int(move) for move in baseline["trajectory"].split(",") if move]
        first = next(
            (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None
        )
        if first is None:
            continue
        game = _state_before(
            [int(move) for move in treated.get("opening_prefix_moves", [])], left, first
        )
        rows.append(
            {
                "opening_index": key[0],
                "challenger_seat": key[1],
                "game_within_opening": key[2],
                "game_ply": first,
                "player_to_move": int(game.current_player),
                "state_hash": canonical_game_state_hash(game),
                "treatment_move": left[first],
                "control_move": right[first],
                "treatment_winner": treated["winner"],
                "treatment_margin": int(treated["margin"]),
                "state": game.to_state(),
            }
        )
    return rows


def _search(
    evaluator: ArtifactEvaluator,
    state: dict,
    seed: int,
    parent: ArtifactEvaluator | None = None,
    overrides: set[str] | None = None,
) -> tuple[list[dict], dict]:
    trace: list[dict] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=1200,
        c_puct=1.25,
        rng=random.Random(seed),
        root_policy_mode="deterministic",
        root_temperature=0.0,
        normalize_values=False,
        tactical_root_bias=0.0,
        selection_trace=trace,
        trace_checkpoints=set(CHECKPOINTS),
        prior_override=None if not overrides else StatePriorOverride(parent, overrides),
    )
    _visits, root = search.run(KalahGame.from_state(state))
    return trace, search.root_summary()


def _first_search_divergence(a16: list[dict], p1: list[dict]) -> dict[str, Any] | None:
    for left, right in zip(a16, p1, strict=True):
        for a, b in zip(left["selection_path"], right["selection_path"], strict=True):
            if a["chosen_move"] != b["chosen_move"]:
                ae = {entry["move"]: entry for entry in a["children"]}
                pe = {entry["move"]: entry for entry in b["children"]}
                return {
                    "simulation_index": left["simulation_index"],
                    "depth": a["tree_depth"],
                    "state_hash": a["state_hash"],
                    "a16_selected_action": a["chosen_move"],
                    "p1_selected_action": b["chosen_move"],
                    "a16_puct": a,
                    "p1_puct": b,
                    "delta_q_component": {
                        str(move): ae[move]["q_component"] - pe[move]["q_component"]
                        for move in ae
                    },
                    "delta_u_component": {
                        str(move): ae[move]["u_component"] - pe[move]["u_component"]
                        for move in ae
                    },
                }
            if a["state_hash"] != b["state_hash"]:
                return {
                    "invariant_failure": "path state mismatch before action divergence"
                }
        # This simulation's backup is only a pre-divergence invariant for the
        # following simulation. A first action difference legitimately changes
        # the current simulation's leaf and backup.
        if abs(left["backed_up_value"] - right["backed_up_value"]) > 1e-9:
            return {"invariant_failure": "backup differs before selection divergence"}
    return None


def _visit_js(left: list[float], right: list[float]) -> float:
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2.0
    return float(
        0.5 * np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)))
        + 0.5 * np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)))
    )


def _intervention(
    state: dict,
    seed: int,
    a16: ArtifactEvaluator,
    p1: ArtifactEvaluator,
    p1_trace: list[dict],
    p1_summary: dict,
    hashes: set[str],
) -> tuple[dict[str, Any], list[dict]]:
    trace, summary = _search(a16, state, seed, p1, hashes)
    first = _first_search_divergence(trace, p1_trace)
    return {
        "substituted_state_count": len(hashes),
        "first_divergence": first,
        "final_selected_root_move": summary["selected_move"],
        "visit_js_vs_p1": _visit_js(
            summary["trace_root_snapshots"][-1]["visits"],
            p1_summary["trace_root_snapshots"][-1]["visits"],
        ),
        "root_q_difference_vs_p1": summary["root_q_value"] - p1_summary["root_q_value"],
    }, trace


def _progressive_interventions(
    state: dict,
    seed: int,
    a16: ArtifactEvaluator,
    p1: ArtifactEvaluator,
    a16_trace: list[dict],
    p1_trace: list[dict],
    p1_summary: dict,
) -> dict[str, Any]:
    """Iteratively discover prior-sensitive states without using arena labels."""
    first = _first_search_divergence(a16_trace, p1_trace)
    if first is None or "state_hash" not in first:
        return {"ordered_states": [], "prefixes": {}}

    ordered = [str(first["state_hash"])]
    prefixes: dict[str, dict[str, Any]] = {}
    # Eight states is the requested largest fixed prefix. It is also the
    # complete trace-discovered list when no new first divergence remains.
    exhausted = False
    for _index in range(8):
        result, _trace = _intervention(
            state, seed, a16, p1, p1_trace, p1_summary, set(ordered)
        )
        prefixes[str(len(ordered))] = result
        next_first = result["first_divergence"]
        if next_first is None or "state_hash" not in next_first:
            exhausted = True
            break
        next_hash = str(next_first["state_hash"])
        if next_hash in ordered:
            exhausted = True
            break
        if len(ordered) == 8:
            break
        ordered.append(next_hash)

    selected_prefixes = {
        str(count): prefixes[str(count)]
        for count in (1, 2, 4, 8)
        if str(count) in prefixes
    }
    if exhausted:
        selected_prefixes["all_traced"] = prefixes[str(len(ordered))]
    return {"ordered_states": ordered, "prefixes": selected_prefixes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factorization-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_adapter_budget_factorization"),
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path(
            "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
        ),
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("docs/alphazero-lite-fresh-p1-adapter-puct-divergence-results.md"),
    )
    args = parser.parse_args()
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    evidence = args.factorization_workdir / "arena/1200_1200"
    treatment, control = (
        _records(evidence / "a16_vs_p1/starts_0/games.jsonl"),
        _records(evidence / "p1_vs_p1_control/starts_0/games.jsonl"),
    )
    treatment.update(_records(evidence / "a16_vs_p1/starts_1/games.jsonl"))
    control.update(_records(evidence / "p1_vs_p1_control/starts_1/games.jsonl"))
    game_rows = first_game_divergences(treatment, control)
    p1, a16 = ArtifactEvaluator(p1_artifact), ArtifactEvaluator(a16_artifact)
    state_rows = []
    analyses_by_state: dict[str, dict[str, Any]] = {}
    for row in game_rows:
        seed, _context = derive_search_seed(
            contract_version="azlite_eval_seed_v2",
            base_seed=42,
            suite_sha256=_suite()[1],
            opening_index=row["opening_index"],
            opening_state_hash=canonical_game_state_hash(
                _state_before(
                    [
                        int(m)
                        for m in treatment[
                            (
                                row["opening_index"],
                                row["challenger_seat"],
                                row["game_within_opening"],
                            )
                        ].get("opening_prefix_moves", [])
                    ],
                    [],
                    0,
                )
            ),
            challenger_player=row["challenger_seat"],
            game_within_opening=row["game_within_opening"],
            ply=row["game_ply"],
            canonical_current_state_hash=row["state_hash"],
            acting_role="challenger"
            if row["player_to_move"] == row["challenger_seat"]
            else "current",
        )
        if row["state_hash"] not in analyses_by_state:
            a_trace, a_summary = _search(a16, row["state"], seed)
            p_trace, p_summary = _search(p1, row["state"], seed)
            analyses_by_state[row["state_hash"]] = {
                "first_search_divergence": _first_search_divergence(a_trace, p_trace),
                "a16_root": a_summary,
                "p1_root": p_summary,
                "progressive_interventions": _progressive_interventions(
                    row["state"], seed, a16, p1, a_trace, p_trace, p_summary
                ),
            }
        state_rows.append(
            {
                **{key: value for key, value in row.items() if key != "state"},
                "search_seed": seed,
                **analyses_by_state[row["state_hash"]],
            }
        )
    invariant_failure = any(
        row["first_search_divergence"]
        and "invariant_failure" in row["first_search_divergence"]
        for row in state_rows
    )
    unique_rows = {row["state_hash"]: row for row in state_rows}.values()
    distributed = all(
        len(row["progressive_interventions"]["ordered_states"]) == 8
        and row["progressive_interventions"]["prefixes"]["8"][
            "final_selected_root_move"
        ]
        != row["p1_root"]["selected_move"]
        for row in unique_rows
    )
    classification = (
        "unexpected_predivergence_q_mismatch"
        if invariant_failure
        else "distributed_prior_boundary_crossings"
        if distributed
        else "inconclusive"
    )
    follow_up = (
        "Investigate evaluator/state-matching invariants before further ML experiments."
        if invariant_failure
        else "Build a search-margin-weighted policy sensitivity metric across replay states."
        if distributed
        else "Design an inference/training constraint based on PUCT selection-margin sensitivity rather than global policy L1 or root-Q sign."
    )
    summary = {
        "schema": "azlite_pr214_puct_divergence_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "c_puct": 1.25,
            "simulations": 1200,
        },
        "hashes": {
            "p1_checkpoint_expected": P1_CHECKPOINT_SHA,
            "a16_state_expected": A16_STATE_SHA,
            "replay_expected": REPLAY_SHA,
            "p1_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_weights": sha256_file(a16_artifact / "weights.json"),
        },
        "first_game_divergences": game_rows,
        "first_search_divergences": state_rows,
        "pr219_corroboration": {
            "status": "not_run",
            "reason": "step-46 artifacts are absent; recreating them would require prohibited training",
        },
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    game_table = [
        "| Opening | Seat | Game ply | Player | Treatment | Control | Outcome |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in game_rows:
        game_table.append(
            f"| {row['opening_index']} | {row['challenger_seat']} | {row['game_ply']} | "
            f"{row['player_to_move']} | {row['treatment_move']} | {row['control_move']} | "
            f"{row['treatment_winner']} |"
        )
    search_table = [
        "| State hash | First simulation | Depth | A16 action | P1 action | U-only | Prefix-8 root |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in {row["state_hash"]: row for row in state_rows}.values():
        first = row["first_search_divergence"]
        prefixes = row["progressive_interventions"]["prefixes"]
        search_table.append(
            f"| `{row['state_hash']}` | {first['simulation_index']} | {first['depth']} | "
            f"{first['a16_selected_action']} | {first['p1_selected_action']} | "
            f"{all(value == 0.0 for value in first['delta_q_component'].values())} | "
            f"{prefixes['8']['final_selected_root_move']} |"
        )
    args.out_report.write_text(
        "\n".join(
            [
                "# PR #214 PUCT Divergence",
                "",
                f"**Classification:** `{classification}`",
                "",
                f"**Recommended follow-up:** {follow_up}",
                "",
                "## First Game Divergences",
                "",
                *game_table,
                "",
                "## First Search Divergences",
                "",
                *search_table,
                "",
                "The JSON summary retains the complete PUCT decomposition, continuous checkpoint snapshots, and progressive interventions.",
                "",
                "## PR #219",
                "",
                "Step-46 artifacts were unavailable locally. Recreating them would require prohibited training, so corroboration was not run.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
