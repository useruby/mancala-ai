#!/usr/bin/env python3
"""Run the frozen high-power paired-opening uniform1200 arena.

This evaluation-only runner never invokes training, self-play, replay mutation,
promotion, or checkpoint creation.  It delegates all game/search execution to
``arena.py`` and only creates frozen opening and result artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
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
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.seat_aware_arena import compute_seat_split_metrics  # noqa: E402

SCHEMA = "azlite_uniform1200_high_power_arena_v2"
SEEDS = (47, 48, 49, 50, 51, 52)
OPENING_COUNT = 256
OPENING_PLIES = 4
BASE_SEED = 90417
SUITE_VERSION = "uniform1200_high_power_unique_v2"
GAMES_PER_OPENING = 2
SIMULATIONS = 384
WORKERS = 24
SEED_CONTRACT = "azlite_eval_seed_v1"
EXPECTED_SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
OLD_SCORES = dict(zip(SEEDS, (0.50, 0.50, 1.00, 0.50, 0.75, 1.00)))
EXPECTED_SHA256 = {
    47: {
        "control": "4d2217c4509cd84ee408ed380c9a32616e698e11a8fea2841bcc1c4c32c38cc6",
        "uniform1200": "8eb1ad896e4ab3d228275a65da137c425710ddba96dd5005b8cef0bac2984f56",
    },
    48: {
        "control": "e1516c4db1a7c7e7a733d8e982087231b5a2395549172a4d8991cd901da04598",
        "uniform1200": "935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c",
    },
    49: {
        "control": "068e4b11a0b34c7de439d9b4c6af8979ff8ce858bf8bcc62baccd91c4cc29b2c",
        "uniform1200": "1de880981885b174aa57166a0431a050565968e181a9931f15e93cd7180c205e",
    },
    50: {
        "control": "7a9a733d0b24c3e424a596e2b14781a691c57a795885e0a6b23ab7b697b214a8",
        "uniform1200": "5da18b268aa1cc80b2e5a0a3342f5ae1a3e9a13f4867d0a2da87567c51fb0ecf",
    },
    51: {
        "control": "e462e6bfbad2a78a66d02c0cb66326d7e6360c89fc771ca2f209855edd983ea9",
        "uniform1200": "e6f5ce2c184676b2c0306616b4a865e9ba2a599169b8013af69c680ac36d07bc",
    },
    52: {
        "control": "d2d7faacbea919c12af3461295f7f189db40ef1beb34b70a5b74cc270272f403",
        "uniform1200": "78084a42915f75fa6b3d66d01cd7da386078c343711094a84f25033bb148ed4e",
    },
}


def checkpoint_path(seed: int, lane: str) -> Path:
    parent = (
        ROOT / ".tmp/fresh-uniform1200/runs"
        if seed <= 49
        else ROOT / ".tmp/fresh-uniform1200-extension/runs"
    )
    return parent / f"seed{seed}" / lane / f"fresh-uniform1200-s{seed}-{lane}-iter1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def initial_game() -> KalahGame:
    return KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )


def enumerate_four_ply_openings() -> tuple[list[dict[str, Any]], int]:
    """Enumerate legal action sequences, preserving Kalah extra turns."""
    openings: list[dict[str, Any]] = []
    excluded = 0

    def visit(game: KalahGame, prefix_moves: tuple[int, ...]) -> None:
        nonlocal excluded
        if len(prefix_moves) == OPENING_PLIES:
            if game.over():
                excluded += 1
                return
            openings.append(
                {
                    "prefix_moves": list(prefix_moves),
                    "resulting_state": game.to_state(),
                    "canonical_resulting_state_hash": canonical_game_state_hash(game),
                    "current_player": game.current_player,
                    "terminal": False,
                    "legal_action_count": len(game.possible_moves()),
                }
            )
            return
        if game.over():
            excluded += 1
            return
        for move in game.possible_moves():
            child = game.clone()
            if not child.move(child.pit_index(move)):
                raise ValueError("four-ply opening enumeration applied an illegal move")
            visit(child, prefix_moves + (move,))

    visit(initial_game(), ())
    return openings, excluded


def canonical_unique_opening_population(
    openings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for opening in openings:
        grouped[str(opening["canonical_resulting_state_hash"])].append(opening)
    population = []
    for state_hash, prefixes in grouped.items():
        representative = min(prefixes, key=lambda row: tuple(row["prefix_moves"]))
        population.append(
            {
                **representative,
                "prefix_multiplicity": len(prefixes),
                "canonical_resulting_state_hash": state_hash,
            }
        )
    return sorted(population, key=lambda row: row["canonical_resulting_state_hash"])


def select_unique_opening_suite(
    population: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(population) < OPENING_COUNT:
        raise ValueError("four_ply_unique_population_insufficient")
    ranked = []
    for row in population:
        state_hash = row["canonical_resulting_state_hash"]
        selection_key = stable_hash(
            {
                "suite_version": SUITE_VERSION,
                "selection_seed": BASE_SEED,
                "canonical_state_hash": state_hash,
            }
        )
        ranked.append((selection_key, state_hash, row))
    return [
        {
            "opening_index": opening_index,
            "prefix_moves": row["prefix_moves"],
            "canonical_resulting_state_hash": state_hash,
            "prefix_multiplicity": row["prefix_multiplicity"],
            "selection_key": selection_key,
        }
        for opening_index, (selection_key, state_hash, row) in enumerate(
            sorted(ranked, key=lambda item: (item[0], item[1]))[:OPENING_COUNT]
        )
    ]


def build_opening_suite() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    openings, excluded = enumerate_four_ply_openings()
    population = canonical_unique_opening_population(openings)
    return select_unique_opening_suite(population), openings, excluded


def suite_preflight(suite: list[dict[str, Any]]) -> dict[str, Any]:
    if len(suite) != OPENING_COUNT or [row["opening_index"] for row in suite] != list(
        range(OPENING_COUNT)
    ):
        raise ValueError("canonical_unique_suite_preflight_failed")
    hashes = []
    for row in suite:
        game = initial_game()
        if (
            len(row["prefix_moves"]) != OPENING_PLIES
            or apply_opening_moves(game, row["prefix_moves"]) != OPENING_PLIES
            or game.over()
        ):
            raise ValueError("canonical_unique_suite_preflight_failed")
        state_hash = canonical_game_state_hash(game)
        if (
            state_hash != row["canonical_resulting_state_hash"]
            or row.get("selection_key")
            != stable_hash(
                {
                    "suite_version": SUITE_VERSION,
                    "selection_seed": BASE_SEED,
                    "canonical_state_hash": state_hash,
                }
            )
            or int(row.get("prefix_multiplicity", 0)) < 1
        ):
            raise ValueError("canonical_unique_suite_preflight_failed")
        hashes.append(state_hash)
    unique = len(set(hashes))
    if (
        unique != OPENING_COUNT
        or len({tuple(row["prefix_moves"]) for row in suite}) != OPENING_COUNT
    ):
        raise ValueError("canonical_unique_suite_preflight_failed")
    return {
        "opening_count": OPENING_COUNT,
        "opening_plies": OPENING_PLIES,
        "canonical_unique_states": unique,
        "canonical_unique_fraction": unique / OPENING_COUNT,
        "model_or_checkpoint_input": False,
    }


def persist_suite(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite, _openings, _excluded = build_opening_suite()
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in suite
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    preflight = suite_preflight(suite)
    return suite, {**preflight, "sha256": sha256_file(path)}


def population_report(
    openings: list[dict[str, Any]], population: list[dict[str, Any]], excluded: int
) -> dict[str, Any]:
    multiplicities = sorted(row["prefix_multiplicity"] for row in population)
    return {
        "schema": SCHEMA,
        "suite_version": SUITE_VERSION,
        "total_legal_four_ply_prefixes": len(openings),
        "canonical_unique_population": len(population),
        "terminal_or_short_prefixes_excluded": excluded,
        "prefix_multiplicity": {
            "min": min(multiplicities),
            "median": statistics.median(multiplicities),
            "p90": float(np.percentile(multiplicities, 90)),
            "max": max(multiplicities),
        },
        "current_player_distribution": dict(
            sorted(Counter(row["current_player"] for row in openings).items())
        ),
        "legal_action_count_distribution": dict(
            sorted(Counter(row["legal_action_count"] for row in openings).items())
        ),
        "model_or_checkpoint_input": False,
    }


def persist_population_artifacts(
    population_path: Path,
    preflight_path: Path,
    suite: list[dict[str, Any]],
    suite_sha256: str,
) -> dict[str, Any]:
    _selected, openings, excluded = build_opening_suite()
    population = canonical_unique_opening_population(openings)
    population_path.parent.mkdir(parents=True, exist_ok=True)
    population_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in openings
        ),
        encoding="utf-8",
    )
    report = population_report(openings, population, excluded)
    report["population_sha256"] = sha256_file(population_path)
    report["suite_preflight"] = suite_preflight(suite)
    report["suite_sha256"] = suite_sha256
    write_json(preflight_path, report)
    return report


def checkpoint_preflight() -> dict[str, dict[str, str]]:
    verified = {}
    for seed in SEEDS:
        verified[str(seed)] = {}
        for lane, expected in EXPECTED_SHA256[seed].items():
            weights = checkpoint_path(seed, lane) / "weights.json"
            if not weights.is_file() or sha256_file(weights) != expected:
                raise FileNotFoundError("high_power_arena_artifact_missing")
            verified[str(seed)][lane] = expected
    return verified


def score(entry: dict[str, Any]) -> float:
    return {"challenger": 1.0, "draw": 0.5, "current": 0.0}[entry["winner"]]


def bootstrap_ci(
    scores: list[float], *, seed: int, replicates: int = 20_000
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(replicates, len(values)), replace=True).mean(
        axis=1
    )
    return {
        "method": "opening_pair_percentile_bootstrap",
        "replicates": replicates,
        "rng_seed": seed,
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
    }


def aggregate_opening_pairs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["opening_index"])].append(entry)
    if set(grouped) != set(range(OPENING_COUNT)):
        raise ValueError("every frozen opening must be used")
    rows = []
    for opening_index in range(OPENING_COUNT):
        games = sorted(
            grouped[opening_index], key=lambda row: int(row["game_within_opening"])
        )
        if len(games) != GAMES_PER_OPENING or {
            int(row["challenger_player"]) for row in games
        } != {0, 1}:
            raise ValueError("each opening must have opposite challenger seats")
        rows.append(
            {
                "opening_index": opening_index,
                "pair_score": statistics.fmean(score(row) for row in games),
                "games": games,
            }
        )
    return rows


def seed_metrics(
    entries: list[dict[str, Any]], pairs: list[dict[str, Any]], wall_seconds: float
) -> dict[str, Any]:
    outcomes = Counter(row["winner"] for row in entries)
    pair_scores = [float(row["pair_score"]) for row in pairs]
    seats = compute_seat_split_metrics(entries)
    return {
        "games": len(entries),
        "wins": outcomes["challenger"],
        "draws": outcomes["draw"],
        "losses": outcomes["current"],
        "raw_challenger_score": statistics.fmean(score(row) for row in entries),
        "mean_opening_pair_score": statistics.fmean(pair_scores),
        "median_opening_pair_score": statistics.median(pair_scores),
        "opening_pair_ci95": bootstrap_ci(pair_scores, seed=BASE_SEED),
        "score_by_challenger_seat": {
            "player_0": seats["challenger_starts_0"]["score"],
            "player_1": seats["challenger_starts_1"]["score"],
        },
        "mean_stone_margin": seats["margin_mean"],
        "median_stone_margin": seats["margin_median"],
        "opening_outcomes": {
            "uniform_wins_both_seats": sum(
                all(row["winner"] == "challenger" for row in pair["games"])
                for pair in pairs
            ),
            "uniform_wins_one_loses_one": sum(
                {row["winner"] for row in pair["games"]} == {"challenger", "current"}
                for pair in pairs
            ),
            "control_wins_both_seats": sum(
                all(row["winner"] == "current" for row in pair["games"])
                for pair in pairs
            ),
            "one_or_both_draws": sum(
                any(row["winner"] == "draw" for row in pair["games"]) for pair in pairs
            ),
        },
        "unique_trajectory_count": seats["unique_trajectories"],
        "duplicate_trajectory_count": seats["duplicate_trajectory_count"],
        "duplicate_trajectory_fraction": seats["duplicate_trajectory_count"]
        / len(entries),
        "low_effective_arena_diversity": seats["unique_trajectories"] < 128,
        "wall_seconds": round(wall_seconds, 3),
    }


def hierarchical_ci(pair_scores_by_seed: list[list[float]]) -> dict[str, Any]:
    values = np.asarray(pair_scores_by_seed, dtype=np.float64)
    rng = np.random.default_rng(BASE_SEED + 1)
    seed_indices = rng.integers(0, len(values), size=(20_000, len(values)))
    opening_indices = rng.integers(
        0, OPENING_COUNT, size=(20_000, len(values), OPENING_COUNT)
    )
    samples = values[seed_indices[:, :, None], opening_indices].mean(axis=(1, 2))
    return {
        "method": "hierarchical_percentile_bootstrap",
        "replicates": 20_000,
        "rng_seed": BASE_SEED + 1,
        "lower": float(np.percentile(samples, 2.5)),
        "upper": float(np.percentile(samples, 97.5)),
    }


def classify(
    effects: list[float],
    hierarchical: dict[str, Any],
    repeated_strength: int,
    repeated_weakness: int,
) -> str:
    scores = [effect + 0.5 for effect in effects]
    if statistics.fmean(scores) <= 0.5 or sum(score > 0.5 for score in scores) < 3:
        return "uniform1200_exact_gain_not_game_strength"
    if (
        sum(effect > 0 for effect in effects) >= 4
        and statistics.fmean(effects) > 0
        and hierarchical["lower"] > 0.5
        and sum(score < 0.5 for score in scores) <= 1
        and repeated_strength > repeated_weakness
    ):
        return "uniform1200_high_power_strength_confirmed"
    if (
        sum(effect > 0 for effect in effects) >= 4
        and statistics.fmean(effects) > 0
        and hierarchical["lower"] <= 0.5
    ):
        return "uniform1200_high_power_positive_but_uncertain"
    if (
        sum(score < 0.5 for score in scores) >= 2
        and sum(score > 0.5 for score in scores) >= 2
    ):
        return "uniform1200_high_power_seed_heterogeneous"
    return "uniform1200_high_power_arena_inconclusive"


def arena_command(seed: int, suite_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(checkpoint_path(seed, "uniform1200")),
        "--current",
        str(checkpoint_path(seed, "control")),
        "--games",
        str(OPENING_COUNT * GAMES_PER_OPENING),
        "--games-per-opening",
        str(GAMES_PER_OPENING),
        "--opening-prefixes-jsonl",
        str(suite_path),
        "--suite-sha256",
        sha256_file(suite_path),
        "--challenger-simulations",
        str(SIMULATIONS),
        "--current-simulations",
        str(SIMULATIONS),
        "--seed",
        str(BASE_SEED),
        "--seed-contract",
        SEED_CONTRACT,
        "--c-puct",
        "1.25",
        "--fpu-mode",
        "zero",
        "--root-policy-mode",
        "deterministic",
        "--tactical-root-bias",
        "0.0",
        "--workers",
        str(WORKERS),
        "--min-score",
        "0.0",
        "--game-jsonl",
        str(output_dir / f"seed{seed}-games.jsonl"),
        "--out",
        str(output_dir / f"seed{seed}-arena.json"),
    ]


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Uniform1200 High-Power Paired Arena",
        "",
        f"Inherited PR #325 classification: `{result['inherited_classification']}`.",
        "",
        f"Hard classification: `{result['classification']}`.",
        "",
        "## Frozen Design",
        "",
        f"- Opening suite: {OPENING_COUNT} model-independent four-ply prefixes, SHA256 `{result['opening_suite']['sha256']}`, canonical uniqueness {result['opening_suite']['canonical_unique_states']}/{OPENING_COUNT}.",
        f"- V1 failed before games: 233/256 canonical states (91.02%). V2 enumerates all legal four-ply prefixes, groups by canonical resulting-state hash, retains the lexicographically smallest prefix, then ranks states by stable hash of `{SUITE_VERSION}`, seed `{BASE_SEED}`, and state hash without replacement.",
        f"- Four-ply population: {result['opening_population']['total_legal_four_ply_prefixes']} legal prefixes and {result['opening_population']['canonical_unique_population']} canonical states; multiplicity min/median/p90/max {result['opening_population']['prefix_multiplicity']['min']}/{result['opening_population']['prefix_multiplicity']['median']}/{result['opening_population']['prefix_multiplicity']['p90']}/{result['opening_population']['prefix_multiplicity']['max']}.",
        f"- V1/V2 canonical-state overlap: {result['opening_population']['v1_v2_canonical_overlap']}.",
        "- Checkpoint preflight: all 12 frozen control/uniform1200 weight SHA256 values match the PR #326 aggregate.",
        f"- Arena: {GAMES_PER_OPENING} seat-swapped games/opening, {SIMULATIONS}/{SIMULATIONS} simulations, c_puct 1.25, deterministic PUCT, zero FPU, seed `{BASE_SEED}`, contract `{SEED_CONTRACT}`, {WORKERS} workers.",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | W/D/L | Pair score | Pair CI95 | Median pair | Seat P0/P1 | Margin mean/median | Trajectories unique/duplicate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed in SEEDS:
        row = result["seeds"][str(seed)]
        metric = row["metrics"]
        ci = metric["opening_pair_ci95"]
        lines.append(
            f"| {seed} | {metric['wins']}/{metric['draws']}/{metric['losses']} | {metric['mean_opening_pair_score']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {metric['median_opening_pair_score']:.4f} | {metric['score_by_challenger_seat']['player_0']:.4f}/{metric['score_by_challenger_seat']['player_1']:.4f} | {metric['mean_stone_margin']:.2f}/{metric['median_stone_margin']:.2f} | {metric['unique_trajectory_count']}/{metric['duplicate_trajectory_count']} |"
        )
    aggregate = result["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"Effects: {', '.join(f'{value:+.4f}' for value in aggregate['effects'])}. Mean `{aggregate['mean_effect']:+.4f}`, median `{aggregate['median_effect']:+.4f}`; positive/neutral/negative `{aggregate['positive']}/{aggregate['neutral']}/{aggregate['negative']}`.",
            f"Hierarchical 95% CI for grand score: `[{aggregate['hierarchical_ci95']['lower']:.4f}, {aggregate['hierarchical_ci95']['upper']:.4f}]`.",
            f"Repeated strength openings: `{len(aggregate['repeated_strength_opening_ids'])}` IDs `{aggregate['repeated_strength_opening_ids']}`. Repeated weakness openings: `{len(aggregate['repeated_weakness_opening_ids'])}` IDs `{aggregate['repeated_weakness_opening_ids']}`.",
            "",
            "## Frozen Exact Context",
            "",
            "PR #325 exact W/D/L improved in all six matched pairs; no true win-to-loss regression repeated at 4/6; the outcome-aligned shadow gate passed 6/6. These exact results were reused, not rerun. PR #326 did not execute arena games because V1 preflight failed.",
            "",
            "## Next Experiment",
            "",
        ]
    )
    next_experiment = {
        "uniform1200_high_power_strength_confirmed": "Run ONE promotion-candidate confirmation using a pre-registered candidate-selection rule over the six frozen uniform1200 checkpoints against the unchanged incumbent with the full production battery.",
        "uniform1200_high_power_positive_but_uncertain": "Create ONE second independently selected 256-state canonical-unique opening block using a pre-registered different selection-domain salt, then combine both blocks.",
        "uniform1200_high_power_seed_heterogeneous": "Compare teacher/replay-transfer statistics between strongest and weakest fresh seed pairs.",
        "uniform1200_exact_gain_not_game_strength": "Measure exact-reference coverage and error specifically on states actually visited by these high-power arena games.",
        "uniform1200_high_power_arena_inconclusive": "Reproduce the frozen high-power arena execution without changing search or checkpoints.",
    }
    lines.append(next_experiment[result["classification"]])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-uniform1200-high-power-arena-v2",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-uniform1200-high-power-openings-v2.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/alphazero-lite-uniform1200-high-power-arena-v2.md",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    population_path = (
        ROOT
        / "docs/data/alphazero-lite-uniform1200-high-power-opening-population-v2.jsonl"
    )
    preflight_path = (
        ROOT
        / "docs/data/alphazero-lite-uniform1200-high-power-opening-preflight-v2.json"
    )
    checkpoints: dict[str, dict[str, str]] = {}
    try:
        suite, _openings, _excluded = build_opening_suite()
        suite_info = persist_suite(args.suite)[1]
        opening_population = persist_population_artifacts(
            population_path, preflight_path, suite, suite_info["sha256"]
        )
        v1_hashes = {
            json.loads(line)["canonical_resulting_state_hash"]
            for line in (
                ROOT / "docs/data/alphazero-lite-uniform1200-high-power-openings.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        }
        opening_population["v1_v2_canonical_overlap"] = len(
            v1_hashes & {row["canonical_resulting_state_hash"] for row in suite}
        )
        write_json(preflight_path, opening_population)
        checkpoints = checkpoint_preflight()
    except FileNotFoundError:
        write_json(
            args.output_dir / "aggregate.json",
            {"schema": SCHEMA, "classification": "high_power_arena_artifact_missing"},
        )
        return 0
    except ValueError as error:
        classification = str(error)
        write_json(
            args.output_dir / "aggregate.json",
            {
                "schema": SCHEMA,
                "classification": classification,
                "arena_executed": False,
            },
        )
        return 0
    if args.dry_run:
        write_json(
            args.output_dir / "plan.json",
            {
                "schema": SCHEMA,
                "checkpoints": checkpoints,
                "opening_suite": suite_info,
                "opening_population": opening_population,
                "commands": [
                    arena_command(seed, args.suite, args.output_dir) for seed in SEEDS
                ],
                "promotion": {"performed": False},
                "replay_mutation": False,
                "training_invoked": False,
            },
        )
        return 0
    per_seed: dict[str, Any] = {}
    all_pairs = []
    for seed in SEEDS:
        started = time.monotonic()
        game_path = args.output_dir / f"seed{seed}-games.jsonl"
        if not args.summarize_existing:
            subprocess.run(
                arena_command(seed, args.suite, args.output_dir), cwd=ROOT, check=True
            )
        elif not game_path.is_file():
            raise FileNotFoundError("high_power_arena_artifact_missing")
        entries = [
            json.loads(line)
            for line in game_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        pairs = aggregate_opening_pairs(entries)
        pair_path = args.output_dir / f"seed{seed}-opening-pairs.jsonl"
        pair_path.write_text(
            "".join(
                json.dumps(
                    {
                        "opening_index": row["opening_index"],
                        "pair_score": row["pair_score"],
                    },
                    sort_keys=True,
                )
                + "\n"
                for row in pairs
            ),
            encoding="utf-8",
        )
        metrics = seed_metrics(entries, pairs, time.monotonic() - started)
        per_seed[str(seed)] = {
            "checkpoint_sha256": checkpoints[str(seed)],
            "metrics": metrics,
            "game_jsonl": str(args.output_dir / f"seed{seed}-games.jsonl"),
            "opening_pairs_jsonl": str(pair_path),
        }
        all_pairs.append([float(row["pair_score"]) for row in pairs])
    effects = [statistics.fmean(values) - 0.5 for values in all_pairs]
    by_opening = list(zip(*all_pairs))
    opening_consistency = [
        {
            "opening_index": index,
            "uniform_gt_0_5": sum(value > 0.5 for value in values),
            "equal_0_5": sum(value == 0.5 for value in values),
            "uniform_lt_0_5": sum(value < 0.5 for value in values),
        }
        for index, values in enumerate(by_opening)
    ]
    repeated_strength = [
        row["opening_index"]
        for row in opening_consistency
        if row["uniform_gt_0_5"] >= 4
    ]
    repeated_weakness = [
        row["opening_index"]
        for row in opening_consistency
        if row["uniform_lt_0_5"] >= 4
    ]
    hierarchical = hierarchical_ci(all_pairs)
    aggregate = {
        "experimental_unit": "matched_training_seed_pair",
        "effects": effects,
        "mean_effect": statistics.fmean(effects),
        "median_effect": statistics.median(effects),
        "min_effect": min(effects),
        "max_effect": max(effects),
        "positive": sum(value > 0 for value in effects),
        "neutral": sum(value == 0 for value in effects),
        "negative": sum(value < 0 for value in effects),
        "hierarchical_ci95": hierarchical,
        "repeated_strength_opening_ids": repeated_strength,
        "repeated_weakness_opening_ids": repeated_weakness,
        "opening_cross_seed_consistency": opening_consistency,
        "opening_cross_seed_scores": [
            {"opening_index": index, "scores": list(values)}
            for index, values in enumerate(by_opening)
        ],
    }
    result = {
        "schema": SCHEMA,
        "inherited_classification": "uniform1200_six_seed_exact_gain_arena_uncertain",
        "opening_suite": suite_info,
        "opening_population": opening_population,
        "checkpoint_sha256": checkpoints,
        "arena": {
            "games_per_pair": 512,
            "games_per_opening": 2,
            "challenger_simulations": 384,
            "current_simulations": 384,
            "c_puct": 1.25,
            "search_options": EXPECTED_SEARCH_OPTIONS,
            "base_seed": BASE_SEED,
            "seed_contract": SEED_CONTRACT,
            "workers": WORKERS,
        },
        "seeds": per_seed,
        "aggregate": aggregate,
        "promotion": {"performed": False},
        "replay_mutation": False,
        "training_invoked": False,
    }
    result["classification"] = classify(
        effects, hierarchical, len(repeated_strength), len(repeated_weakness)
    )
    write_json(args.output_dir / "aggregate.json", result)
    args.report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
