#!/usr/bin/env python3
"""Evaluate frozen seed48 against the unchanged incumbent on a new holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import (  # noqa: E402
    apply_opening_moves,
    canonical_game_state_hash,
)
from ml.alphazero_lite.evaluation_seed_contract import stable_hash  # noqa: E402
from ml.alphazero_lite.kalah_rules import (  # noqa: E402
    KalahGame,
    move_consequence_for_state,
)
from ml.alphazero_lite.run_uniform1200_high_power_arena import (  # noqa: E402
    canonical_unique_opening_population,
    enumerate_four_ply_openings,
    initial_game,
)

SCHEMA = "azlite_uniform1200_incumbent_opening_holdout_v1"
SUITE_VERSION = "uniform1200_incumbent_holdout_unique_v1"
SELECTION_SEED = 42
OPENING_COUNT = 256
OPENING_PLIES = 4
GAMES_PER_OPENING = 2
CHALLENGER_SIMULATIONS = 384
CURRENT_SIMULATIONS = 256
CANDIDATE_SHA256 = "935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c"
INCUMBENT_SHA256 = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
PR327_SUITE_SHA256 = "c4f74fe141cae1ea1aebe0fb97865f96864172cdd806ed9d7984184c6c8d9b9d"
HOLDOUT_SUITE_SHA256 = (
    "5d1f5982c990d00bce0a57161c1ae710bed6b8ced831ac2c1551af5217d2716c"
)
PRODUCTION_THRESHOLD = 0.55
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
CANDIDATE = (
    ROOT
    / ".tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1"
)
INCUMBENT = ROOT / "model-artifact/current"
PR327_SUITE = ROOT / "docs/data/alphazero-lite-uniform1200-high-power-openings-v2.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def candidate_preflight() -> dict[str, Any]:
    metadata_path = CANDIDATE / "metadata.json"
    if (
        not metadata_path.is_file()
        or sha256(CANDIDATE / "weights.json") != CANDIDATE_SHA256
    ):
        raise ValueError("incumbent_holdout_candidate_artifact_mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("architecture", {}).get("model_type") != "residual_v3"
        or metadata.get("input_encoding") != "kalah_v3"
    ):
        raise ValueError("incumbent_holdout_candidate_artifact_mismatch")
    return {
        "path": str(CANDIDATE.relative_to(ROOT)),
        "weights_sha256": CANDIDATE_SHA256,
        "model_type": "residual_v3",
        "input_encoding": "kalah_v3",
    }


def incumbent_preflight() -> dict[str, Any]:
    metadata_path = INCUMBENT / "metadata.json"
    if (
        not metadata_path.is_file()
        or sha256(INCUMBENT / "weights.json") != INCUMBENT_SHA256
    ):
        raise ValueError("incumbent_holdout_incumbent_changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "azlite-balanced-w8s4-policy-head-e1":
        raise ValueError("incumbent_holdout_incumbent_changed")
    return {
        "path": "model-artifact/current",
        "weights_sha256": INCUMBENT_SHA256,
        "version": metadata["version"],
    }


def load_pr327_hashes() -> set[str]:
    if sha256(PR327_SUITE) != PR327_SUITE_SHA256:
        raise ValueError("incumbent_holdout_suite_invalid")
    rows = [
        json.loads(line)
        for line in PR327_SUITE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    hashes = {row["canonical_resulting_state_hash"] for row in rows}
    if len(rows) != OPENING_COUNT or len(hashes) != OPENING_COUNT:
        raise ValueError("incumbent_holdout_suite_invalid")
    return hashes


def build_suite() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_hashes = load_pr327_hashes()
    openings, terminal_excluded = enumerate_four_ply_openings()
    population = canonical_unique_opening_population(openings)
    if len(openings) != 942 or len(population) != 942 or terminal_excluded != 0:
        raise ValueError("incumbent_holdout_suite_invalid")
    remaining = [
        row
        for row in population
        if row["canonical_resulting_state_hash"] not in excluded_hashes
    ]
    if len(remaining) != 686:
        raise ValueError("incumbent_holdout_suite_invalid")
    ranked = sorted(
        (
            stable_hash(
                {
                    "suite_version": SUITE_VERSION,
                    "selection_seed": SELECTION_SEED,
                    "canonical_state_hash": row["canonical_resulting_state_hash"],
                }
            ),
            row["canonical_resulting_state_hash"],
            row,
        )
        for row in remaining
    )
    suite = [
        {
            "opening_index": index,
            "prefix_moves": row["prefix_moves"],
            "canonical_resulting_state_hash": state_hash,
            "selection_key": selection_key,
            "suite_version": SUITE_VERSION,
        }
        for index, (selection_key, state_hash, row) in enumerate(ranked[:OPENING_COUNT])
    ]
    preflight_suite(suite, excluded_hashes)
    return suite, {
        "total_legal_four_ply_prefixes": len(openings),
        "total_canonical_population": len(population),
        "terminal_or_short_excluded": terminal_excluded,
        "pr327_excluded_count": len(excluded_hashes),
        "remaining_population_count": len(remaining),
        "selected_count": len(suite),
        "selection_seed": SELECTION_SEED,
        "selection_salt": SUITE_VERSION,
        "model_or_checkpoint_input": False,
    }


def preflight_suite(suite: list[dict[str, Any]], excluded_hashes: set[str]) -> None:
    if len(suite) != OPENING_COUNT or [row["opening_index"] for row in suite] != list(
        range(OPENING_COUNT)
    ):
        raise ValueError("incumbent_holdout_suite_invalid")
    hashes = []
    for row in suite:
        game = initial_game()
        if (
            len(row["prefix_moves"]) != OPENING_PLIES
            or apply_opening_moves(game, row["prefix_moves"]) != OPENING_PLIES
            or game.over()
        ):
            raise ValueError("incumbent_holdout_suite_invalid")
        state_hash = canonical_game_state_hash(game)
        expected_key = stable_hash(
            {
                "suite_version": SUITE_VERSION,
                "selection_seed": SELECTION_SEED,
                "canonical_state_hash": state_hash,
            }
        )
        if (
            state_hash != row["canonical_resulting_state_hash"]
            or row.get("selection_key") != expected_key
            or row.get("suite_version") != SUITE_VERSION
        ):
            raise ValueError("incumbent_holdout_suite_invalid")
        hashes.append(state_hash)
    if len(set(hashes)) != OPENING_COUNT or set(hashes) & excluded_hashes:
        raise ValueError("incumbent_holdout_suite_invalid")


def persist_suite(
    suite_path: Path, preflight_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite, preflight = build_suite()
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in suite
    )
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(payload, encoding="utf-8")
    suite_sha = sha256(suite_path)
    if (
        suite_path
        == ROOT
        / "docs/data/alphazero-lite-uniform1200-incumbent-holdout-openings-v1.jsonl"
        and suite_sha != HOLDOUT_SUITE_SHA256
    ):
        raise ValueError("incumbent_holdout_suite_invalid")
    repeat_suite, _ = build_suite()
    repeat_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in repeat_suite
    )
    if hashlib.sha256(repeat_payload.encode()).hexdigest() != suite_sha:
        raise ValueError("incumbent_holdout_suite_invalid")
    report = {
        **preflight,
        "schema": SCHEMA,
        "suite_version": SUITE_VERSION,
        "suite_sha256": suite_sha,
        "canonical_unique_states": OPENING_COUNT,
        "canonical_unique_fraction": 1.0,
        "pr327_overlap": 0,
        "repeated_construction_identical_suite_sha": True,
    }
    write_json(preflight_path, report)
    return suite, report


def arena_command(suite_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(CANDIDATE),
        "--current",
        str(INCUMBENT),
        "--games",
        str(OPENING_COUNT * GAMES_PER_OPENING),
        "--games-per-opening",
        str(GAMES_PER_OPENING),
        "--opening-prefixes-jsonl",
        str(suite_path),
        "--suite-sha256",
        sha256(suite_path),
        "--challenger-simulations",
        str(CHALLENGER_SIMULATIONS),
        "--current-simulations",
        str(CURRENT_SIMULATIONS),
        "--seed",
        str(SELECTION_SEED),
        "--c-puct",
        "1.25",
        "--fpu-mode",
        "zero",
        "--root-policy-mode",
        "deterministic",
        "--tactical-root-bias",
        "0.0",
        "--min-score",
        "0.0",
        "--seed-ledger-output",
        str(output_dir / "seed-ledger.jsonl"),
        "--search-configuration-ledger-output",
        str(output_dir / "search-configuration-ledger.jsonl"),
        "--search-outcome-ledger-output",
        str(output_dir / "search-outcome-ledger.jsonl"),
        "--game-jsonl",
        str(output_dir / "games.jsonl"),
        "--out",
        str(output_dir / "arena.json"),
    ]


def score(row: dict[str, Any]) -> float:
    return {"challenger": 1.0, "draw": 0.5, "current": 0.0}[row["winner"]]


def opening_pairs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["opening_index"])].append(entry)
    if set(grouped) != set(range(OPENING_COUNT)):
        raise ValueError("incumbent_holdout_arena_inconclusive")
    pairs = []
    for index in range(OPENING_COUNT):
        games = sorted(grouped[index], key=lambda row: int(row["game_within_opening"]))
        if len(games) != GAMES_PER_OPENING or {
            int(row["challenger_player"]) for row in games
        } != {0, 1}:
            raise ValueError("incumbent_holdout_arena_inconclusive")
        pairs.append(
            {
                "opening_index": index,
                "pair_score": statistics.fmean(score(row) for row in games),
                "seat_pair_balance": sum(score(row) for row in games),
                "games": games,
            }
        )
    return pairs


def bootstrap(scores: list[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    means = (
        np.random.default_rng(SELECTION_SEED)
        .choice(values, size=(20_000, len(values)), replace=True)
        .mean(axis=1)
    )
    return {
        "method": "opening_pair_percentile_bootstrap",
        "replicates": 20_000,
        "rng_seed": SELECTION_SEED,
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
    }


def descriptor(state: dict[str, Any]) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    pits = state["player_pits"] + state["opponent_pits"]
    consequences = [
        move_consequence_for_state(state, move) for move in game.possible_moves()
    ]
    return {
        "current_player": state["current_player"],
        "legal_action_count": len(game.possible_moves()),
        "store_difference": state["player_store"] - state["opponent_store"],
        "pit_stones_total": sum(pits),
        "pit_stones_min": min(pits),
        "pit_stones_max": max(pits),
        "pit_stones_mean": statistics.fmean(pits),
        "capture_available": any(row["produces_capture"] for row in consequences),
        "extra_turn_available": any(row["gives_extra_turn"] for row in consequences),
    }


def descriptor_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "current_player": dict(Counter(row["current_player"] for row in rows)),
        "legal_action_count": dict(Counter(row["legal_action_count"] for row in rows)),
        "capture_available": sum(row["capture_available"] for row in rows),
        "extra_turn_available": sum(row["extra_turn_available"] for row in rows),
        "mean_store_difference": statistics.fmean(
            row["store_difference"] for row in rows
        ),
        "mean_pit_stones_total": statistics.fmean(
            row["pit_stones_total"] for row in rows
        ),
        "mean_pit_stones_min": statistics.fmean(row["pit_stones_min"] for row in rows),
        "mean_pit_stones_max": statistics.fmean(row["pit_stones_max"] for row in rows),
        "mean_pit_stones_mean": statistics.fmean(
            row["pit_stones_mean"] for row in rows
        ),
    }


def classify(metrics: dict[str, Any], shadow_passed: bool) -> str:
    mean, ci, classes = (
        metrics["mean_opening_pair_score"],
        metrics["opening_pair_ci95"],
        metrics["opening_strength_distribution"],
    )
    if ci["upper"] < 0.5 or (
        mean < 0.45 and classes["incumbent_favored"] > classes["candidate_favored"]
    ):
        return "incumbent_holdout_candidate_weaker"
    if (
        mean >= 0.55
        and classes["candidate_favored"] < 64
        and classes["candidate_favored"]
        < classes["neutral"] + classes["incumbent_favored"]
    ):
        return "incumbent_holdout_opening_specific"
    if (
        mean >= 0.55
        and ci["lower"] > 0.5
        and classes["candidate_favored"] > classes["incumbent_favored"]
        and min(metrics["score_by_challenger_seat"].values()) >= 0.45
        and shadow_passed
    ):
        return "production_start_state_arena_disagreement_confirmed"
    if mean > 0.5 and ci["lower"] > 0.5 and mean < 0.55:
        return "incumbent_holdout_positive_below_production_threshold"
    if mean > 0.5 and ci["lower"] <= 0.5 <= ci["upper"]:
        return "incumbent_holdout_strength_uncertain"
    if (
        abs(mean - 0.5) <= 0.02
        and classes["candidate_favored"] == classes["incumbent_favored"]
    ):
        return "incumbent_holdout_no_strength_gain"
    return "incumbent_holdout_arena_inconclusive"


def next_experiment(classification: str) -> str:
    return {
        "production_start_state_arena_disagreement_confirmed": "Audit the production arena prefilter against canonical-unique suites across frozen historical pairs with known strength ordering.",
        "incumbent_holdout_positive_below_production_threshold": "Calibrate the 0.55 threshold on frozen historical model pairs under canonical-unique opening evaluation.",
        "incumbent_holdout_strength_uncertain": "Create one second disjoint 256-state four-ply holdout block from the unused canonical population and combine both blocks.",
        "incumbent_holdout_no_strength_gain": "Compare seed48 matched-control and incumbent strength through frozen pairwise results or one calibrated three-model ranking experiment.",
        "incumbent_holdout_candidate_weaker": "Audit seed48 state-visitation and teacher-transfer differences explaining matched-control improvement but incumbent underperformance.",
        "incumbent_holdout_opening_specific": "Compare the candidate-dominant opening families against natural self-play visitation frequency.",
        "incumbent_holdout_arena_inconclusive": "Reproduce this frozen holdout arena without changing checkpoints or search semantics.",
    }[classification]


def render_report(result: dict[str, Any]) -> str:
    metrics, suite, ci = (
        result["metrics"],
        result["suite"],
        result["metrics"]["opening_pair_ci95"],
    )
    return f"""# Uniform1200 Incumbent Opening Holdout

Inherited classifications: PR #327 `uniform1200_high_power_strength_confirmed`; PR #328 `uniform1200_promotion_arena_failed`.

Candidate SHA256: `{CANDIDATE_SHA256}`. Incumbent SHA256: `{INCUMBENT_SHA256}` (`azlite-balanced-w8s4-policy-head-e1`).

The immutable PR #327 suite SHA256 was verified as `{PR327_SUITE_SHA256}`. Its 256 hashes were excluded from the mechanically enumerated 942 four-ply canonical states, leaving {suite["remaining_population_count"]}; selected-suite overlap is {suite["pr327_overlap"]}/256. The new suite SHA256 is `{suite["suite_sha256"]}`. It ranks the remainder by stable hash of `{SUITE_VERSION}`, seed 42, and canonical state hash.

Arena semantics exactly preserve PR #328: candidate/incumbent simulations {CHALLENGER_SIMULATIONS}/{CURRENT_SIMULATIONS}, c_puct 1.25, zero FPU, no subtree reuse or normalization, deterministic root policy, zero tactical bias and root temperature, no root-prior transform, no opening cache, no value transform, base seed 42, seed contract `{result["arena"]["seed_contract"]}`.

## Results

- W/D/L: {metrics["wins"]}/{metrics["draws"]}/{metrics["losses"]} across 512 games; raw score `{metrics["raw_score"]:.4f}`.
- Candidate P0 W/D/L: {metrics["seat_wdl"]["player_0"]}; P1: {metrics["seat_wdl"]["player_1"]}. Scores P0/P1: `{metrics["score_by_challenger_seat"]["player_0"]:.4f}`/`{metrics["score_by_challenger_seat"]["player_1"]:.4f}`.
- Mean/median opening-pair score: `{metrics["mean_opening_pair_score"]:.4f}`/`{metrics["median_opening_pair_score"]:.4f}`; paired bootstrap CI95 `[{ci["lower"]:.4f}, {ci["upper"]:.4f}]`.
- Mean/median candidate stone margin: `{metrics["mean_stone_margin"]:.2f}`/`{metrics["median_stone_margin"]:.2f}`. Unique/duplicate trajectories: {metrics["unique_trajectories"]}/{metrics["duplicate_trajectories"]} ({metrics["duplicate_fraction"]:.4f}).
- Opening pairs: candidate wins both `{metrics["opening_outcomes"]["candidate_wins_both_seats"]}`, incumbent wins both `{metrics["opening_outcomes"]["incumbent_wins_both_seats"]}`, one win each `{metrics["opening_outcomes"]["one_win_each"]}`, one/both draws `{metrics["opening_outcomes"]["one_or_both_draws"]}`.
- Candidate/neutral/incumbent-favored openings: {metrics["opening_strength_distribution"]["candidate_favored"]}/{metrics["opening_strength_distribution"]["neutral"]}/{metrics["opening_strength_distribution"]["incumbent_favored"]}. Seat-pair balance 0/0.5/1/1.5/2: {metrics["seat_pair_balance_distribution"]}.
- Existing exact-reference coverage: `{len(result["exact_reference_coverage"])}` holdout openings; no new tablebase was generated.
- Difficulty descriptors, candidate-favored/neutral/incumbent-favored: counts `{metrics["opening_difficulty_descriptors"]["candidate_favored"]["count"]}`/`{metrics["opening_difficulty_descriptors"]["neutral"]["count"]}`/`{metrics["opening_difficulty_descriptors"]["incumbent_favored"]["count"]}`; capture available `{metrics["opening_difficulty_descriptors"]["candidate_favored"]["capture_available"]}`/`{metrics["opening_difficulty_descriptors"]["neutral"]["capture_available"]}`/`{metrics["opening_difficulty_descriptors"]["incumbent_favored"]["capture_available"]}`; extra turn available `{metrics["opening_difficulty_descriptors"]["candidate_favored"]["extra_turn_available"]}`/`{metrics["opening_difficulty_descriptors"]["neutral"]["extra_turn_available"]}`/`{metrics["opening_difficulty_descriptors"]["incumbent_favored"]["extra_turn_available"]}`. Full current-player, legal-action, store, and pit-stone distributions are frozen in the aggregate JSON.
- Threshold margin: `{metrics["threshold_margin"]:+.4f}`. Point estimate exceeds 0.55: `{metrics["exceeds_055"]}`; exceeds 0.50: `{metrics["exceeds_050"]}`; CI lower bound above 0.50: `{metrics["ci_lower_above_050"]}`.

Compared descriptively with frozen PR #328 start-state W/D/L 60/0/60, score 0.5000 and threshold margin -0.05, diversity gain is `{metrics["diversity_gain"]:+.4f}`.

Frozen W/D/L shadow safety remains valid: candidate/incumbent top1 0.9296/0.8732, policy mass 0.4078/0.3577, regret 0.1174/0.2207, blunder rate 0.0704/0.1268, win-to-draw 5/5, win-to-loss 10/20.

## Classification

`{result["classification"]}`

Next experiment: {next_experiment(result["classification"])}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-uniform1200-incumbent-holdout",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-uniform1200-incumbent-holdout-openings-v1.jsonl",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-uniform1200-incumbent-holdout-opening-preflight-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/alphazero-lite-uniform1200-incumbent-opening-holdout.md",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Aggregate an existing frozen game JSONL without executing another arena.",
    )
    args = parser.parse_args(argv)
    try:
        candidate, incumbent = candidate_preflight(), incumbent_preflight()
        suite, suite_info = persist_suite(args.suite, args.preflight)
    except ValueError as error:
        write_json(
            args.out_dir / "aggregate.json",
            {"schema": SCHEMA, "classification": str(error), "arena_executed": False},
        )
        return 0
    if args.prepare_only:
        write_json(
            args.out_dir / "plan.json",
            {
                "schema": SCHEMA,
                "candidate": candidate,
                "incumbent": incumbent,
                "suite": suite_info,
                "candidate_count": 1,
                "command": arena_command(args.suite, args.out_dir),
                "training_invoked": False,
                "replay_mutation": False,
                "promotion": {"performed": False},
            },
        )
        return 0
    if not args.summarize_existing:
        subprocess.run(arena_command(args.suite, args.out_dir), cwd=ROOT, check=True)
    elif (
        not (args.out_dir / "games.jsonl").is_file()
        or not (args.out_dir / "arena.json").is_file()
    ):
        raise FileNotFoundError("incumbent_holdout_arena_inconclusive")
    entries = [
        json.loads(line)
        for line in (args.out_dir / "games.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    pairs = opening_pairs(entries)
    pair_scores = [row["pair_score"] for row in pairs]
    outcomes = Counter(row["winner"] for row in entries)
    seats = {
        str(player): [row for row in entries if row["challenger_player"] == player]
        for player in (0, 1)
    }
    suite_by_index = {row["opening_index"]: row for row in suite}
    classes = {"candidate_favored": [], "neutral": [], "incumbent_favored": []}
    for pair in pairs:
        classes[
            "candidate_favored"
            if pair["pair_score"] > 0.5
            else "incumbent_favored"
            if pair["pair_score"] < 0.5
            else "neutral"
        ].append(pair)
    descriptors = {name: [] for name in classes}
    for name, class_pairs in classes.items():
        for pair in class_pairs:
            game = initial_game()
            apply_opening_moves(
                game, suite_by_index[pair["opening_index"]]["prefix_moves"]
            )
            descriptors[name].append(descriptor(game.to_state()))
    ci = bootstrap(pair_scores)
    metrics = {
        "wins": outcomes["challenger"],
        "draws": outcomes["draw"],
        "losses": outcomes["current"],
        "raw_score": statistics.fmean(score(row) for row in entries),
        "mean_opening_pair_score": statistics.fmean(pair_scores),
        "median_opening_pair_score": statistics.median(pair_scores),
        "opening_pair_ci95": ci,
        "score_by_challenger_seat": {
            f"player_{player}": statistics.fmean(score(row) for row in rows)
            for player, rows in ((0, seats["0"]), (1, seats["1"]))
        },
        "seat_wdl": {
            f"player_{player}": {
                "wins": sum(row["winner"] == "challenger" for row in rows),
                "draws": sum(row["winner"] == "draw" for row in rows),
                "losses": sum(row["winner"] == "current" for row in rows),
            }
            for player, rows in ((0, seats["0"]), (1, seats["1"]))
        },
        "mean_stone_margin": statistics.fmean(row["margin"] for row in entries),
        "median_stone_margin": statistics.median(row["margin"] for row in entries),
        "opening_outcomes": {
            "candidate_wins_both_seats": sum(
                all(game["winner"] == "challenger" for game in pair["games"])
                for pair in pairs
            ),
            "incumbent_wins_both_seats": sum(
                all(game["winner"] == "current" for game in pair["games"])
                for pair in pairs
            ),
            "one_win_each": sum(
                {game["winner"] for game in pair["games"]} == {"challenger", "current"}
                for pair in pairs
            ),
            "one_or_both_draws": sum(
                any(game["winner"] == "draw" for game in pair["games"])
                for pair in pairs
            ),
        },
        "unique_trajectories": len({row["trajectory"] for row in entries}),
        "duplicate_trajectories": len(entries)
        - len({row["trajectory"] for row in entries}),
        "opening_strength_distribution": {
            name: len(rows) for name, rows in classes.items()
        },
        "seat_pair_balance_distribution": {
            str(value): sum(pair["seat_pair_balance"] == value for pair in pairs)
            for value in (0, 0.5, 1, 1.5, 2)
        },
        "opening_difficulty_descriptors": {
            name: descriptor_distribution(rows) for name, rows in descriptors.items()
        },
    }
    metrics["duplicate_fraction"] = metrics["duplicate_trajectories"] / len(entries)
    metrics["threshold_margin"] = (
        metrics["mean_opening_pair_score"] - PRODUCTION_THRESHOLD
    )
    metrics["exceeds_055"] = metrics["mean_opening_pair_score"] > PRODUCTION_THRESHOLD
    metrics["exceeds_050"] = metrics["mean_opening_pair_score"] > 0.5
    metrics["ci_lower_above_050"] = ci["lower"] > 0.5
    metrics["diversity_gain"] = metrics["mean_opening_pair_score"] - 0.5
    references = json.loads(
        (
            ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
        ).read_text(encoding="utf-8")
    )
    exact_by_hash = {
        canonical_game_state_hash(KalahGame.from_state(row["state"])): row
        for row in references["rows"]
        if row.get("exact_status") == "exact_solved"
    }
    exact_coverage = []
    for pair in pairs:
        reference = exact_by_hash.get(
            suite_by_index[pair["opening_index"]]["canonical_resulting_state_hash"]
        )
        if reference is None:
            continue
        candidate_game = next(
            game for game in pair["games"] if game["challenger_player"] == 0
        )
        incumbent_game = next(
            game for game in pair["games"] if game["challenger_player"] == 1
        )
        values = reference.get("exact_action_values", {})
        candidate_value = values.get(str(candidate_game["first_move_challenger"]))
        incumbent_value = values.get(str(incumbent_game["first_move_current"]))
        exact_coverage.append(
            {
                "opening_index": pair["opening_index"],
                "exact_wdl_value": reference.get("exact_root_value"),
                "candidate_selected_move": candidate_game["first_move_challenger"],
                "candidate_selected_move_outcome": None
                if candidate_value is None
                else int((candidate_value > 0) - (candidate_value < 0)),
                "incumbent_selected_move": incumbent_game["first_move_current"],
                "incumbent_selected_move_outcome": None
                if incumbent_value is None
                else int((incumbent_value > 0) - (incumbent_value < 0)),
            }
        )
    pair_path = args.out_dir / "opening-pairs.jsonl"
    pair_path.write_text(
        "".join(
            json.dumps(
                {key: value for key, value in pair.items() if key != "games"},
                sort_keys=True,
            )
            + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )
    arena = json.loads((args.out_dir / "arena.json").read_text(encoding="utf-8"))
    result = {
        "schema": SCHEMA,
        "candidate": candidate,
        "incumbent": incumbent,
        "candidate_count": 1,
        "suite": suite_info,
        "arena": {
            "games": 512,
            "games_per_opening": 2,
            "challenger_simulations": 384,
            "current_simulations": 256,
            "c_puct": 1.25,
            "search_options": SEARCH_OPTIONS,
            "base_seed": 42,
            "seed_contract": arena["notes"]["seed_contract"],
            "suite_sha256": suite_info["suite_sha256"],
            "seed_ledger_hash": arena["notes"]["seed_ledger_hash"],
            "search_configuration_ledger_hash": arena["notes"][
                "search_configuration_ledger_sha256"
            ],
            "search_outcome_ledger_hash": arena["notes"][
                "search_outcome_ledger_sha256"
            ],
        },
        "metrics": metrics,
        "frozen_pr328_start_state": {
            "score": 0.5,
            "wins": 60,
            "draws": 0,
            "losses": 60,
            "games": 120,
            "threshold_margin": -0.05,
        },
        "exact_reference_coverage": exact_coverage,
        "wdl_shadow_safety_remains_valid": True,
        "training_invoked": False,
        "replay_mutation": False,
        "promotion": {"performed": False},
    }
    result["classification"] = classify(
        metrics, result["wdl_shadow_safety_remains_valid"]
    )
    write_json(args.out_dir / "aggregate.json", result)
    args.report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
