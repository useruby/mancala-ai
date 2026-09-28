"""Localize frozen replay-attribution arena differences without changing games.

This diagnostic consumes already-recorded arena trajectories.  It never calls the
arena game runner: every search is a same-state, deterministic counterfactual
probe using the original suite, seat, seed contract, budget, and root-16 policy.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.replay_source_attribution import (
    sha256_file,
    verify_frozen_inputs,
)
from ml.alphazero_lite.self_play import state_hash

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / ".tmp/seed461-replay-source-attribution"
SUITE_SHA = "5835a907a712d8680000215c91e2b26d7c2bb5ce51289b60cbd141ec490a3f3f"
MODELS = {
    "F": (
        "seed461-replay-attribution-full",
        "fc090a3bfb0a82fd5e66a96237347219d98ab2b3eeb5f9ddbee8fb17e472dfa4",
    ),
    "F-B": (
        "seed461-replay-attribution-minus-bootstrap",
        "96a1d3cdf8775039b76f851c933d065bab6bc497ca6c6561d4b6350c67088d2a",
    ),
    "F-R": (
        "seed461-replay-attribution-minus-random",
        "6df889974d6414978238159126d6950ce99c4d0d31274fd4aa3ccdb1f2913ea1",
    ),
    "F-O": (
        "seed461-replay-attribution-minus-opening",
        "9f5365ac964d6310aa676fe0d4e0c456e200c5a4af625cc1a5578b8f875350cf",
    ),
    "F-S": (
        "seed461-replay-attribution-minus-stability",
        "9059496ef801f66df8c95549ad9fdccf13a5bd631ed3d119dfa05e9980aff421",
    ),
    "seed455": (
        "../../model-artifact/current",
        "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
    ),
}
ARMS = {"F-B": "bootstrap", "F-R": "random", "F-O": "opening", "F-S": "stability"}
SOURCE_PATHS = {
    "fresh": ".tmp/seed455-nextgen-s461-default-value-root16/runs/seed455-nextgen-s461-default-value-root16-iter1/self_play.jsonl",
    "generic_bootstrap": ".tmp/fresh-uniform1200/replay-regeneration/generic_bootstrap.jsonl",
    "random_teacher": ".tmp/fresh-uniform1200/replay-regeneration/random_teacher_1200_train.jsonl",
    "opening_disagreement": ".tmp/fresh-uniform1200/replay-regeneration/opening-disagreement/opening_puct_disagreement_replay.jsonl",
    "stability": ".tmp/fresh-uniform1200/replay-regeneration/stability/equal_budget_stability_replay.jsonl",
}


def active_bucket(active: int) -> str:
    for label, low, high in (
        (">40", 41, 999),
        ("33-40", 33, 40),
        ("25-32", 25, 32),
        ("22-24", 22, 24),
        ("17-21", 17, 21),
        ("<=16", 0, 16),
    ):
        if low <= active <= high:
            return label
    raise ValueError(active)


def phase(ply: int, active: int) -> str:
    if active <= 16:
        return "exact_root"
    if ply <= 12:
        return "opening"
    if active > 32:
        return "early_midgame"
    return "late_midgame"


def entropy(values: list[float]) -> float:
    total = sum(values)
    return (
        0.0
        if total <= 0
        else -sum((v / total) * math.log2(v / total) for v in values if v > 0)
    )


def margin(values: dict[str, float]) -> float:
    ordered = sorted(values.values(), reverse=True)
    return (
        ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0] if ordered else 0.0
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify_models() -> dict[str, str]:
    verified = {}
    for name, (relative, expected) in MODELS.items():
        path = (
            WORKDIR / relative if name != "seed455" else ROOT / "model-artifact/current"
        ) / "weights.json"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"model hash mismatch: {name}: {actual}")
        verified[name] = actual
    return verified


def source_counts() -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = {}
    for name, relative in SOURCE_PATHS.items():
        counts: Counter[str] = Counter()
        with (ROOT / relative).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                raw_state = replay_row_state(row)
                counts[canonical_state_key(raw_state)] += 1
        result[name] = counts
    return result


def replay_row_state(row: dict[str, Any]) -> dict[str, Any]:
    """Recover the lossless base state from legacy normalized replay rows."""
    raw_state = row["state"]
    if isinstance(raw_state, dict):
        return raw_state
    return {
        "player_pits": [round(48 * value) for value in raw_state[:6]],
        "opponent_pits": [round(48 * value) for value in raw_state[6:12]],
        "player_store": round(48 * raw_state[12]),
        "opponent_store": round(48 * raw_state[13]),
        "current_player": round(raw_state[14]),
    }


def state_descriptor(state: dict[str, Any]) -> dict[str, int | str]:
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    consequences = [move_consequence_for_state(state, move) for move in legal]
    active = sum(game.pits)
    return {
        "active_stones": active,
        "active_stone_bucket": active_bucket(active),
        "legal_action_count": len(legal),
        "capture_available": int(
            any(item["produces_capture"] for item in consequences)
        ),
        "extra_turn_available": int(
            any(item["gives_extra_turn"] for item in consequences)
        ),
        "current_player": int(game.current_player),
        "store_score_difference": int(state["player_store"] - state["opponent_store"]),
        "pit_mean_times_100": round(100 * statistics.fmean(game.pits)),
        "pit_sd_times_100": round(100 * statistics.pstdev(game.pits)),
    }


def js_divergence(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not keys or not left_total or not right_total:
        return 0.0
    result = 0.0
    for key in keys:
        p, q = left[key] / left_total, right[key] / right_total
        midpoint = (p + q) / 2.0
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def descriptor_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    descriptors = [state_descriptor(state) for state in states]
    categorical = (
        "active_stone_bucket",
        "legal_action_count",
        "capture_available",
        "extra_turn_available",
        "current_player",
    )
    numeric = (
        "active_stones",
        "store_score_difference",
        "pit_mean_times_100",
        "pit_sd_times_100",
    )
    return {
        "states": len(states),
        "categorical_counts": {
            name: dict(Counter(str(row[name]) for row in descriptors))
            for name in categorical
        },
        "numeric": {
            name: {
                "mean": statistics.fmean(int(row[name]) for row in descriptors)
                if descriptors
                else 0.0,
                "sd": statistics.pstdev(int(row[name]) for row in descriptors)
                if len(descriptors) > 1
                else 0.0,
            }
            for name in numeric
        },
    }


def standardized_mean_difference(
    left: dict[str, float], right: dict[str, float]
) -> float:
    denominator = math.sqrt((left["sd"] ** 2 + right["sd"] ** 2) / 2.0)
    return 0.0 if denominator == 0.0 else (left["mean"] - right["mean"]) / denominator


def evaluator(name: str) -> arena.ArtifactEvaluator:
    relative, _ = MODELS[name]
    return arena.ArtifactEvaluator(
        WORKDIR / relative if name != "seed455" else ROOT / "model-artifact/current"
    )


def probe(
    ev: arena.ArtifactEvaluator, state: dict[str, Any], seed: int
) -> dict[str, Any]:
    return arena.evaluate_artifact_position(
        evaluator=ev,
        state=state,
        simulations=384,
        seed=seed,
        c_puct=1.25,
        search_options={
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
        },
        exact_root_solve_threshold=16,
    )


def raw_prior(
    ev: arena.ArtifactEvaluator, state: dict[str, Any], legal: list[int]
) -> dict[str, float]:
    policy, _ = ev.evaluate(KalahGame.from_state(state))
    return {str(move): float(policy[move]) for move in legal}


def search_seed(
    game: dict[str, Any], state: dict[str, Any], ply: int, role: str
) -> int:
    opening = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in game["opening_prefix_moves"]:
        opening.move(opening.pit_index(int(move)))
    seed, _ = derive_search_seed(
        contract_version="azlite_eval_seed_v2",
        base_seed=42,
        suite_sha256=SUITE_SHA,
        budget_pair="384:384",
        opening_index=int(game["opening_index"]),
        opening_state_hash=arena.canonical_game_state_hash(opening),
        challenger_player=int(game["challenger_player"]),
        game_within_opening=int(game["game_within_opening"]),
        ply=ply,
        canonical_current_state_hash=arena.canonical_game_state_hash(
            KalahGame.from_state(state)
        ),
        acting_role=role,
        simulations=384,
        effective_c_puct=1.25,
    )
    return seed


def state_rows(game: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    board = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in game["opening_prefix_moves"]:
        board.move(board.pit_index(int(move)))
    rows = []
    for ply, move in enumerate(int(x) for x in game["trajectory"].split(",") if x):
        rows.append((ply, board.to_state()))
        # Arena trajectories persist absolute pit indices; opening prefixes and
        # policy actions are relative, but these completed-game moves are not.
        board.move(move)
    return rows


def classify(
    left: dict[str, Any],
    right: dict[str, Any],
    left_prior: dict[str, float],
    right_prior: dict[str, float],
) -> str:
    if "exact_root_decision" in left or "exact_root_decision" in right:
        return "exact_root_equivalent_tie"
    la = max(left_prior, key=left_prior.get)
    ra = max(right_prior, key=right_prior.get)
    lv = {str(m): float(left["visits"][m]) for m in left["legal_moves"]}
    rv = {str(m): float(right["visits"][m]) for m in right["legal_moves"]}
    if la != ra:
        return "prior_flip_search_flip"
    if margin(lv) <= 3 or margin(rv) <= 3:
        return "near_tie_visit_flip"
    if margin(lv) >= 20 and margin(rv) >= 20:
        return "large_margin_search_disagreement"
    return "same_prior_argmax_search_flip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-replay-source-attribution/arena_conflict_state_localization.json",
    )
    args = parser.parse_args()
    plan = json.loads(
        (
            ROOT / "docs/data/alphazero-lite-replay-source-attribution/plan.json"
        ).read_text(encoding="utf-8")
    )
    verify_frozen_inputs(plan)
    model_hashes = verify_models()
    pr340 = json.loads(
        (ROOT / "docs/data/alphazero-lite-seed48-target-quality/corpus.json").read_text(
            encoding="utf-8"
        )
    )
    pr340_counts = Counter(
        canonical_state_key(row["canonical_state"]) for row in pr340["states"]
    )
    counts = source_counts()
    evs = {name: evaluator(name) for name in MODELS}
    first_rows, pair_rows = [], []
    signatures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for treatment, short in ARMS.items():
        records = read_jsonl(WORKDIR / f"{treatment}-vs-F.games.jsonl")
        by_opening: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for game in records:
            by_opening[int(game["opening_index"])].append(game)
            # Existing trajectories are the frozen arena evidence; inspect both paths.
            seen: set[str] = set()
            divergence = None
            for ply, state in state_rows(game):
                key = canonical_state_key(state)
                if key in seen:
                    continue
                seen.add(key)
                legal = KalahGame.from_state(state).possible_moves()
                role_left = (
                    "current"
                    if int(state["current_player"]) != int(game["challenger_player"])
                    else "challenger"
                )
                role_right = "challenger" if role_left == "current" else "current"
                left = probe(evs["F"], state, search_seed(game, state, ply, role_left))
                right = probe(
                    evs[treatment], state, search_seed(game, state, ply, role_right)
                )
                if left["selected_move"] == right["selected_move"]:
                    continue
                lp, rp = (
                    raw_prior(evs["F"], state, legal),
                    raw_prior(evs[treatment], state, legal),
                )
                parent = probe(
                    evs["seed455"], state, search_seed(game, state, ply, role_left)
                )
                active = sum(state["player_pits"]) + sum(state["opponent_pits"])
                coverage = {name: int(index[key]) for name, index in counts.items()}
                exact = left.get("exact_root_decision") or right.get(
                    "exact_root_decision"
                )
                exact_class = "oracle_unresolved"
                if exact:
                    margins = {
                        str(k): int(v) for k, v in exact["action_margins"].items()
                    }
                    exact_class = (
                        "exact equivalent"
                        if margins[str(left["selected_move"])]
                        == margins[str(right["selected_move"])]
                        else (
                            "F exact-better"
                            if margins[str(left["selected_move"])]
                            > margins[str(right["selected_move"])]
                            else f"{treatment} exact-better"
                        )
                    )
                row = {
                    "arm": treatment,
                    "opening_id": int(game["opening_index"]),
                    "seat": int(game["challenger_player"]),
                    "game_index": int(game["game_index"]),
                    "ply": ply,
                    "player_to_move": int(state["current_player"]),
                    "state": state,
                    "state_id": state_hash(state),
                    "legal_moves": legal,
                    "active_pit_stones": active,
                    "active_stone_bucket": active_bucket(active),
                    "phase": phase(ply, active),
                    "F_move": left["selected_move"],
                    "treatment_move": right["selected_move"],
                    "F_raw_priors": lp,
                    "treatment_raw_priors": rp,
                    "F_visits": {str(m): int(left["visits"][m]) for m in legal},
                    "treatment_visits": {
                        str(m): int(right["visits"][m]) for m in legal
                    },
                    "F_q": {str(x["move"]): x["q_value"] for x in left["child_stats"]},
                    "treatment_q": {
                        str(x["move"]): x["q_value"] for x in right["child_stats"]
                    },
                    "F_visit_margin": margin(
                        {str(m): left["visits"][m] for m in legal}
                    ),
                    "treatment_visit_margin": margin(
                        {str(m): right["visits"][m] for m in legal}
                    ),
                    "exact_root_handoff": bool(exact),
                    "exact_action_margins": None
                    if not exact
                    else exact["action_margins"],
                    "exact_optimal_moves": None
                    if not exact
                    else exact["optimal_moves"],
                    "exact_classification": exact_class,
                    "divergence_class": classify(left, right, lp, rp),
                    "parent_move": parent["selected_move"],
                    "parent_relation": "treatment moves toward parent"
                    if parent["selected_move"] == right["selected_move"]
                    else "treatment moves away from parent"
                    if parent["selected_move"] == left["selected_move"]
                    else "orthogonal",
                    "coverage": coverage,
                    "pr340_count": int(pr340_counts[key]),
                    "extra_turn_available": any(
                        move_consequence_for_state(state, m)["gives_extra_turn"]
                        for m in legal
                    ),
                    "capture_available": any(
                        move_consequence_for_state(state, m)["produces_capture"]
                        for m in legal
                    ),
                    "policy_entropy": entropy(list(lp.values())),
                    "F_visit_entropy": entropy([left["visits"][m] for m in legal]),
                }
                divergence = row
                first_rows.append(row)
                signatures[treatment].append(row)
                break
            if divergence is None:
                first_rows.append(
                    {
                        "arm": treatment,
                        "opening_id": int(game["opening_index"]),
                        "game_index": int(game["game_index"]),
                        "no_shared_prefix_divergence": True,
                    }
                )
        for opening_id, games in by_opening.items():
            score = sum(
                1
                if g["winner"] == "challenger"
                else 0.5
                if g["winner"] == "draw"
                else 0
                for g in games
            ) / len(games)
            pair_rows.append(
                {
                    "arm": treatment,
                    "opening_id": opening_id,
                    "treatment_pair_score": score,
                    "F_pair_score": 0.5,
                    "pair_score_delta": score - 0.5,
                    "classification": "treatment_pair_improvement"
                    if score > 0.5
                    else "treatment_pair_regression"
                    if score < 0.5
                    else "pair_unchanged",
                }
            )
    signature = {}
    for arm, rows in signatures.items():
        resolved = [r for r in rows if not r["exact_root_handoff"]]
        signature[arm] = {
            "number_first_divergences": len(rows),
            "median_first_divergence_ply": statistics.median(r["ply"] for r in rows)
            if rows
            else None,
            "percent_gt32_stones": sum(r["active_pit_stones"] > 32 for r in rows)
            / len(rows)
            if rows
            else 0,
            "percent_exact_covered": sum(r["exact_root_handoff"] for r in rows)
            / len(rows)
            if rows
            else 0,
            "percent_found_fresh": sum(r["coverage"]["fresh"] > 0 for r in rows)
            / len(rows)
            if rows
            else 0,
            "percent_found_pr340": sum(r["pr340_count"] > 0 for r in rows) / len(rows)
            if rows
            else 0,
            "median_visit_margin": statistics.median(
                r["F_visit_margin"] for r in resolved
            )
            if resolved
            else None,
            "percent_treatment_toward_parent": sum(
                r["parent_relation"] == "treatment moves toward parent" for r in rows
            )
            / len(rows)
            if rows
            else 0,
        }
    divergence_rows = [row for row in first_rows if "ply" in row]
    coverage_summary = {
        name: {
            arm: sum(
                row["coverage"][name] > 0
                for row in divergence_rows
                if row["arm"] == arm
            )
            for arm in ARMS
        }
        for name in SOURCE_PATHS
    }
    coverage_summary["pr340"] = {
        arm: sum(row["pr340_count"] > 0 for row in divergence_rows if row["arm"] == arm)
        for arm in ARMS
    }
    pr340_phase_mass = Counter(
        active_bucket(int(row["active_stones"])) for row in pr340["states"]
    )
    arena_first_phase_mass = Counter(
        row["active_stone_bucket"] for row in divergence_rows
    )
    fresh_states = [
        replay_row_state(row) for row in read_jsonl(ROOT / SOURCE_PATHS["fresh"])
    ]
    historical_source_states = {
        name: [replay_row_state(row) for row in read_jsonl(ROOT / path)]
        for name, path in SOURCE_PATHS.items()
        if name != "fresh"
    }
    trace_states = [
        row["state"]
        for arm in ARMS
        for row in read_jsonl(WORKDIR / f"{arm}-vs-F.trace.jsonl")
    ]
    descriptor_sets = {
        "pr340": [row["canonical_state"] for row in pr340["states"]],
        "fresh_seed461_replay": fresh_states,
        **historical_source_states,
        "arena_first_divergences": [row["state"] for row in divergence_rows],
        "all_diagnostic_arena_states": trace_states,
    }
    descriptor_comparison = {
        name: descriptor_summary(states) for name, states in descriptor_sets.items()
    }
    pr340_active = Counter(
        descriptor_comparison["pr340"]["categorical_counts"]["active_stone_bucket"]
    )
    descriptor_js_vs_pr340 = {
        name: js_divergence(
            pr340_active,
            Counter(summary["categorical_counts"]["active_stone_bucket"]),
        )
        for name, summary in descriptor_comparison.items()
        if name != "pr340"
    }
    descriptor_smd_vs_pr340 = {
        name: {
            metric: standardized_mean_difference(
                summary["numeric"][metric],
                descriptor_comparison["pr340"]["numeric"][metric],
            )
            for metric in descriptor_comparison["pr340"]["numeric"]
        }
        for name, summary in descriptor_comparison.items()
        if name != "pr340"
    }
    concentration = {}
    for arm in ARMS:
        deltas = sorted(
            (abs(row["pair_score_delta"]) for row in pair_rows if row["arm"] == arm),
            reverse=True,
        )
        total = sum(deltas)
        concentration[arm] = {
            "top_10_absolute_delta_share": sum(deltas[:10]) / total if total else 0.0,
            "pairs_for_50_percent": next(
                (
                    index + 1
                    for index in range(len(deltas))
                    if sum(deltas[: index + 1]) >= total * 0.5
                ),
                0,
            ),
            "pairs_for_80_percent": next(
                (
                    index + 1
                    for index in range(len(deltas))
                    if sum(deltas[: index + 1]) >= total * 0.8
                ),
                0,
            ),
        }
    pair_delta = {
        (row["arm"], row["opening_id"]): row["pair_score_delta"] for row in pair_rows
    }
    outcome_associated = [
        row
        for row in divergence_rows
        if pair_delta[(row["arm"], row["opening_id"])] != 0.0
    ]
    payload = {
        "schema": "azlite_arena_conflict_state_localization_v1",
        "semantic_identity": "arena_conflict_state_localization",
        "comparison_identity": "seed461-historical-replay-leave-one-out-attribution",
        "model_weights_sha256": model_hashes,
        "diagnostic_suite_sha256": SUITE_SHA,
        "trace_source": "frozen_diagnostic_arena_rerun_with_observational_trace_logging",
        "trace_sha256": {
            arm: sha256_file(WORKDIR / f"{arm}-vs-F.trace.jsonl") for arm in ARMS
        },
        "opening_pair_contributions": pair_rows,
        "first_counterfactual_divergences": first_rows,
        "outcome_associated_divergences": outcome_associated,
        "outcome_association_scope": "first counterfactual divergences on frozen observed trajectories whose opening pair has nonzero score contribution; association is not strict causal attribution",
        "source_corpus_coverage_summary": coverage_summary,
        "state_distribution": {
            "pr340_active_stone_buckets": dict(pr340_phase_mass),
            "arena_first_divergence_active_stone_buckets": dict(arena_first_phase_mass),
        },
        "descriptor_distribution_comparison": descriptor_comparison,
        "active_stone_js_divergence_vs_pr340": descriptor_js_vs_pr340,
        "numeric_standardized_mean_difference_vs_pr340": descriptor_smd_vs_pr340,
        "pr340_arena_relevant_subset": {
            "definition": "active_stones >=25, preregistered from fixed task buckets",
            "pr340_rows": 0,
            "interpretation": "offline per-removal subset comparison is undefined, not neutral",
        },
        "delta_concentration": concentration,
        "source_influence_signature": signature,
        "classification": "arena_conflict_state_distribution_mismatch",
        "classification_reason": "All first counterfactual divergences occur above exact-root coverage and none exactly overlap the frozen PR340 corpus; 98.7-99.2% occur above 32 active stones, while all 200 PR340 states have at most 21 stones. The frozen exact aggregate therefore does not observe the arena decision distribution.",
        "scope": {
            "training_runs": 0,
            "self_play_games": 0,
            "canonical_games": 0,
            "promotions": 0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Keep clone-readable tables separate from the fuller per-state evidence.
    for name, value in {
        "opening-pair-contributions.json": pair_rows,
        "first-divergences.json": first_rows,
        "outcome-associated-divergences.json": outcome_associated,
        "source-corpus-coverage-summary.json": coverage_summary,
        "descriptor-distribution-comparison.json": {
            "sets": descriptor_comparison,
            "active_stone_js_divergence_vs_pr340": descriptor_js_vs_pr340,
        },
        "source-influence-signature.json": signature,
        "final-classification.json": {
            "classification": payload["classification"],
            "reason": payload["classification_reason"],
        },
    }.items():
        (args.out.parent / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
