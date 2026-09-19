#!/usr/bin/env python3
"""Calibrate the production repeated-start arena against a third frozen suite.

This is evaluation-only. It never calls the promotion gate, training, or self-play.
"""

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

from ml.alphazero_lite.arena import apply_opening_moves, canonical_game_state_hash  # noqa: E402
from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    stable_hash,
)
from ml.alphazero_lite.run_uniform1200_high_power_arena import (  # noqa: E402
    EXPECTED_SHA256,
    canonical_unique_opening_population,
    checkpoint_path,
    enumerate_four_ply_openings,
    initial_game,
)
from ml.alphazero_lite.run_uniform1200_incumbent_opening_holdout import (  # noqa: E402
    HOLDOUT_SUITE_SHA256,
    PR327_SUITE_SHA256,
)

SCHEMA = "azlite_production_arena_prefilter_calibration_v1"
SUITE_VERSION = "production_prefilter_calibration_unique_v1"
SELECTION_SEED = 330
OPENING_COUNT = 256
OPENING_PLIES = 4
START_GAMES = 120
GAMES_PER_OPENING = 2
CHALLENGER_SIMULATIONS = 384
CURRENT_SIMULATIONS = 256
PRODUCTION_SEED = 42
PRODUCTION_THRESHOLD = 0.55
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
PR327_SUITE = ROOT / "docs/data/alphazero-lite-uniform1200-high-power-openings-v2.jsonl"
PR329_SUITE = (
    ROOT / "docs/data/alphazero-lite-uniform1200-incumbent-holdout-openings-v1.jsonl"
)
INCUMBENT = ROOT / "model-artifact/current"
INCUMBENT_SHA256 = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
PR327_AGGREGATE = (
    ROOT / "docs/data/alphazero-lite-uniform1200-high-power-arena-v2/aggregate.json"
)
HISTORICAL_SCORES = dict(
    zip(
        (47, 48, 49, 50, 51, 52),
        (
            0.6396484375,
            0.681640625,
            0.6572265625,
            0.6328125,
            0.6240234375,
            0.6357421875,
        ),
    )
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_suite_hashes(path: Path, expected_sha: str) -> set[str]:
    if sha256(path) != expected_sha:
        raise ValueError("production_prefilter_calibration_invalid")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    hashes = {str(row["canonical_resulting_state_hash"]) for row in rows}
    if len(rows) != OPENING_COUNT or len(hashes) != OPENING_COUNT:
        raise ValueError("production_prefilter_calibration_invalid")
    return hashes


def frozen_pairs() -> list[dict[str, Any]]:
    """Return labels fixed solely from PR #327 and PR #329 evidence."""
    pairs = []
    for seed in (47, 48, 49, 50, 51, 52):
        pairs.append(
            {
                "pair_id": f"P{seed}",
                "challenger_path": str(
                    checkpoint_path(seed, "uniform1200").relative_to(ROOT)
                ),
                "challenger_sha256": EXPECTED_SHA256[seed]["uniform1200"],
                "current_path": str(checkpoint_path(seed, "control").relative_to(ROOT)),
                "current_sha256": EXPECTED_SHA256[seed]["control"],
                "historical_evidence_source": "PR #327 uniform1200_high_power_strength_confirmed",
                "historical_independent_strength_score": HISTORICAL_SCORES[seed],
                "frozen_label": "known_positive",
                "identity_control": False,
            }
        )
    candidate = checkpoint_path(48, "uniform1200")
    pairs.append(
        {
            "pair_id": "P_INC",
            "challenger_path": str(candidate.relative_to(ROOT)),
            "challenger_sha256": EXPECTED_SHA256[48]["uniform1200"],
            "current_path": str(INCUMBENT.relative_to(ROOT)),
            "current_sha256": INCUMBENT_SHA256,
            "historical_evidence_source": "PR #329 production_start_state_arena_disagreement_confirmed",
            "historical_independent_strength_score": 0.8837890625,
            "frozen_label": "known_positive",
            "identity_control": False,
        }
    )
    pairs.extend(
        [
            {
                "pair_id": "I_INC",
                "challenger_path": str(INCUMBENT.relative_to(ROOT)),
                "challenger_sha256": INCUMBENT_SHA256,
                "current_path": str(INCUMBENT.relative_to(ROOT)),
                "current_sha256": INCUMBENT_SHA256,
                "historical_evidence_source": "model_identity_control",
                "frozen_label": None,
                "identity_control": True,
            },
            {
                "pair_id": "I_48",
                "challenger_path": str(candidate.relative_to(ROOT)),
                "challenger_sha256": EXPECTED_SHA256[48]["uniform1200"],
                "current_path": str(candidate.relative_to(ROOT)),
                "current_sha256": EXPECTED_SHA256[48]["uniform1200"],
                "historical_evidence_source": "model_identity_control",
                "frozen_label": None,
                "identity_control": True,
            },
        ]
    )
    return pairs


def verify_manifest(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if len([pair for pair in pairs if pair["frozen_label"] == "known_positive"]) != 7:
        raise ValueError("production_prefilter_calibration_invalid")
    if len([pair for pair in pairs if pair["identity_control"]]) != 2:
        raise ValueError("production_prefilter_calibration_invalid")
    pr327 = json.loads(PR327_AGGREGATE.read_text(encoding="utf-8"))
    for seed in (47, 48, 49, 50, 51, 52):
        historical = pr327["seeds"][str(seed)]
        if (
            historical["checkpoint_sha256"] != EXPECTED_SHA256[seed]
            or float(historical["metrics"]["mean_opening_pair_score"])
            != HISTORICAL_SCORES[seed]
        ):
            raise ValueError("production_prefilter_calibration_invalid")
    verified = []
    for pair in pairs:
        for side in ("challenger", "current"):
            path = ROOT / pair[f"{side}_path"]
            if (
                not (path / "weights.json").is_file()
                or sha256(path / "weights.json") != pair[f"{side}_sha256"]
            ):
                raise ValueError("production_prefilter_calibration_invalid")
        verified.append(pair["pair_id"])
    return {
        "schema": SCHEMA,
        "frozen_before_calibration_results": True,
        "pairs": pairs,
        "pr327_aggregate_path": str(PR327_AGGREGATE.relative_to(ROOT)),
        "pr327_aggregate_sha256": sha256(PR327_AGGREGATE),
        "verified_pair_ids": verified,
    }


def production_contract() -> dict[str, Any]:
    """Assert the current gate and arena defaults still reproduce PR #328 semantics."""
    gate = (ROOT / "script/ai/local_promotion_gate").read_text(encoding="utf-8")
    required_gate_fragments = (
        'parser.add_argument("--arena-games", default=120',
        'parser.add_argument("--min-arena-score", default=0.55',
        '"ml/alphazero_lite/arena.py"',
        '"--games",',
        '"--min-score",',
    )
    arena = (ROOT / "ml/alphazero_lite/arena.py").read_text(encoding="utf-8")
    required_arena_fragments = (
        'parser.add_argument("--challenger-simulations", type=int, default=384)',
        'parser.add_argument("--current-simulations", type=int, default=256)',
        'parser.add_argument("--seed", type=int, default=42)',
        'parser.add_argument("--random-opening-plies", type=int, default=0)',
        'parser.add_argument("--c-puct", type=float, default=1.25)',
    )
    if not all(fragment in gate for fragment in required_gate_fragments) or not all(
        fragment in arena for fragment in required_arena_fragments
    ):
        raise ValueError("production_prefilter_contract_changed")
    return {
        "arena_games": START_GAMES,
        "minimum_score": PRODUCTION_THRESHOLD,
        "challenger_simulations": CHALLENGER_SIMULATIONS,
        "current_simulations": CURRENT_SIMULATIONS,
        "seed": PRODUCTION_SEED,
        "seed_contract": SEED_CONTRACT_VERSION,
        "random_opening_plies": 0,
        "c_puct": 1.25,
        "search_options": SEARCH_OPTIONS,
        "root_prior_transform": None,
        "value_transform": None,
        "opening_cache": None,
    }


def build_suite() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pr327 = load_suite_hashes(PR327_SUITE, PR327_SUITE_SHA256)
    pr329 = load_suite_hashes(PR329_SUITE, HOLDOUT_SUITE_SHA256)
    if pr327 & pr329:
        raise ValueError("production_prefilter_calibration_invalid")
    openings, terminal_excluded = enumerate_four_ply_openings()
    population = canonical_unique_opening_population(openings)
    if len(openings) != 942 or len(population) != 942 or terminal_excluded != 0:
        raise ValueError("production_prefilter_calibration_invalid")
    remaining = [
        row
        for row in population
        if row["canonical_resulting_state_hash"] not in pr327 | pr329
    ]
    if len(remaining) < OPENING_COUNT:
        raise ValueError("production_prefilter_calibration_population_insufficient")
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
    hashes = []
    for row in suite:
        game = initial_game()
        if (
            len(row["prefix_moves"]) != OPENING_PLIES
            or apply_opening_moves(game, row["prefix_moves"]) != OPENING_PLIES
            or game.over()
        ):
            raise ValueError("production_prefilter_calibration_invalid")
        hashes.append(canonical_game_state_hash(game))
    if len(set(hashes)) != OPENING_COUNT or set(hashes) & (pr327 | pr329):
        raise ValueError("production_prefilter_calibration_invalid")
    return suite, {
        "schema": SCHEMA,
        "suite_version": SUITE_VERSION,
        "selection_seed": SELECTION_SEED,
        "total_legal_four_ply_prefixes": len(openings),
        "total_canonical_population": len(population),
        "pr327_state_count": len(pr327),
        "pr329_state_count": len(pr329),
        "historical_suite_overlap": len(pr327 & pr329),
        "remaining_population": len(remaining),
        "opening_count": len(suite),
        "canonical_unique_states": len(set(hashes)),
        "canonical_unique_fraction": len(set(hashes)) / len(suite),
        "pr327_overlap": len(set(hashes) & pr327),
        "pr329_overlap": len(set(hashes) & pr329),
        "every_prefix_legal": True,
        "every_prefix_four_plies": True,
        "no_terminal_resulting_state": True,
        "model_or_checkpoint_input": False,
    }


def persist_suite(path: Path, preflight_path: Path) -> dict[str, Any]:
    suite, preflight = build_suite()
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in suite
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    repeat, _ = build_suite()
    repeat_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in repeat
    )
    suite_sha = sha256(path)
    if hashlib.sha256(repeat_payload.encode()).hexdigest() != suite_sha:
        raise ValueError("production_prefilter_calibration_invalid")
    report = {
        **preflight,
        "suite_sha256": suite_sha,
        "deterministic_reconstruction_sha": suite_sha,
    }
    write_json(preflight_path, report)
    return report


def arena_command(
    pair: dict[str, Any], output: Path, *, suite: Path | None = None
) -> list[str]:
    command = [
        sys.executable,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(ROOT / pair["challenger_path"]),
        "--current",
        str(ROOT / pair["current_path"]),
        "--games",
        str(START_GAMES if suite is None else OPENING_COUNT * GAMES_PER_OPENING),
        "--challenger-simulations",
        str(CHALLENGER_SIMULATIONS),
        "--current-simulations",
        str(CURRENT_SIMULATIONS),
        "--seed",
        str(PRODUCTION_SEED),
        "--seed-contract",
        SEED_CONTRACT_VERSION,
        "--c-puct",
        "1.25",
        "--fpu-mode",
        "zero",
        "--root-policy-mode",
        "deterministic",
        "--tactical-root-bias",
        "0.0",
        "--min-score",
        str(PRODUCTION_THRESHOLD if suite is None else 0.0),
        "--seed-ledger-output",
        str(output / "seed-ledger.jsonl"),
        "--search-configuration-ledger-output",
        str(output / "search-configuration-ledger.jsonl"),
        "--search-outcome-ledger-output",
        str(output / "search-outcome-ledger.jsonl"),
        "--game-jsonl",
        str(output / "games.jsonl"),
        "--out",
        str(output / "arena.json"),
    ]
    if suite is not None:
        command.extend(
            [
                "--games-per-opening",
                str(GAMES_PER_OPENING),
                "--opening-prefixes-jsonl",
                str(suite),
                "--suite-sha256",
                sha256(suite),
            ]
        )
    return command


def score(row: dict[str, Any]) -> float:
    return {"challenger": 1.0, "draw": 0.5, "current": 0.0}[row["winner"]]


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


def seat_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for seat in (0, 1):
        rows = [row for row in entries if int(row["challenger_player"]) == seat]
        outcomes = Counter(row["winner"] for row in rows)
        result[f"player_{seat}"] = {
            "wins": outcomes["challenger"],
            "draws": outcomes["draw"],
            "losses": outcomes["current"],
            "score": statistics.fmean(score(row) for row in rows),
        }
    return result


def repeated_metrics(
    entries: list[dict[str, Any]], arena: dict[str, Any]
) -> dict[str, Any]:
    outcomes, trajectories = (
        Counter(row["winner"] for row in entries),
        Counter(str(row["trajectory"]) for row in entries),
    )
    first_sequences = Counter(
        (row.get("first_move_challenger"), row.get("first_move_current"))
        for row in entries
    )
    roots = Counter(row.get("first_move_challenger") for row in entries)
    unique = len(trajectories)
    return {
        "games": len(entries),
        "wins": outcomes["challenger"],
        "draws": outcomes["draw"],
        "losses": outcomes["current"],
        "score": statistics.fmean(score(row) for row in entries),
        "production_start_pass": statistics.fmean(score(row) for row in entries)
        >= PRODUCTION_THRESHOLD,
        "challenger_seats": seat_metrics(entries),
        "unique_start_states": 1,
        "unique_trajectories": unique,
        "trajectory_multiplicities": dict(sorted(trajectories.items())),
        "unique_first_move_sequences": len(first_sequences),
        "unique_root_decisions": len(roots),
        "trajectory_repetition_factor": len(entries) / max(unique, 1),
        "effective_unique_trajectory_fraction": unique / len(entries),
        "seed_ledger_hash": arena["notes"].get("seed_ledger_hash"),
        "search_configuration_ledger_hash": arena["notes"].get(
            "search_configuration_ledger_sha256"
        ),
        "search_outcome_ledger_hash": arena["notes"].get(
            "search_outcome_ledger_sha256"
        ),
    }


def opening_pairs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["opening_index"])].append(entry)
    if set(grouped) != set(range(OPENING_COUNT)):
        raise ValueError("production_prefilter_calibration_invalid")
    pairs = []
    for index in range(OPENING_COUNT):
        games = sorted(grouped[index], key=lambda row: int(row["game_within_opening"]))
        if len(games) != 2 or {int(row["challenger_player"]) for row in games} != {
            0,
            1,
        }:
            raise ValueError("production_prefilter_calibration_invalid")
        pairs.append(
            {
                "opening_index": index,
                "pair_score": statistics.fmean(score(row) for row in games),
                "games": games,
            }
        )
    return pairs


def canonical_metrics(
    entries: list[dict[str, Any]], arena: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs, outcomes = opening_pairs(entries), Counter(row["winner"] for row in entries)
    pair_scores = [float(pair["pair_score"]) for pair in pairs]
    classes = {
        "challenger_favored": sum(value > 0.5 for value in pair_scores),
        "neutral": sum(value == 0.5 for value in pair_scores),
        "current_favored": sum(value < 0.5 for value in pair_scores),
    }
    metrics = {
        "games": len(entries),
        "wins": outcomes["challenger"],
        "draws": outcomes["draw"],
        "losses": outcomes["current"],
        "raw_score": statistics.fmean(score(row) for row in entries),
        "canonical_pair_score": statistics.fmean(pair_scores),
        "median_pair_score": statistics.median(pair_scores),
        "canonical_pair_ci95": bootstrap(pair_scores),
        "canonical_threshold_pass": statistics.fmean(pair_scores)
        >= PRODUCTION_THRESHOLD,
        "canonical_positive_strength_evidence": False,
        "challenger_seats": seat_metrics(entries),
        "mean_stone_margin": statistics.fmean(float(row["margin"]) for row in entries),
        "unique_trajectories": len({str(row["trajectory"]) for row in entries}),
        "opening_distribution": {
            **classes,
            "challenger_wins_both_seats": sum(
                all(game["winner"] == "challenger" for game in pair["games"])
                for pair in pairs
            ),
            "current_wins_both_seats": sum(
                all(game["winner"] == "current" for game in pair["games"])
                for pair in pairs
            ),
            "split_seats": sum(
                {game["winner"] for game in pair["games"]} == {"challenger", "current"}
                for pair in pairs
            ),
            "draws": sum(
                any(game["winner"] == "draw" for game in pair["games"])
                for pair in pairs
            ),
        },
        "seed_ledger_hash": arena["notes"].get("seed_ledger_hash"),
        "search_configuration_ledger_hash": arena["notes"].get(
            "search_configuration_ledger_sha256"
        ),
        "search_outcome_ledger_hash": arena["notes"].get(
            "search_outcome_ledger_sha256"
        ),
    }
    metrics["canonical_positive_strength_evidence"] = (
        metrics["canonical_pair_score"] > 0.5
        and metrics["canonical_pair_ci95"]["lower"] > 0.5
    )
    return metrics, pairs


def spearman(left: list[float], right: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        return [
            statistics.fmean(
                index + 1
                for index, candidate in enumerate(sorted(values))
                if candidate == value
            )
            for value in values
        ]

    a, b = ranks(left), ranks(right)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    denominator = sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    return (
        0.0
        if denominator == 0
        else sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
        / denominator**0.5
    )


def classify(summary: dict[str, Any]) -> str:
    if summary.get("invalid"):
        return "production_prefilter_calibration_invalid"
    if summary.get("contract_changed"):
        return "production_prefilter_contract_changed"
    if summary.get("population_insufficient"):
        return "production_prefilter_calibration_population_insufficient"
    known = summary["known_positive"]
    primary = summary["primary_rule"]
    if primary["passes"]:
        return "production_prefilter_start_state_systematic_false_negative"
    if any(
        row["identity_material_budget_advantage"]
        for row in summary["identity_controls"]
    ):
        return "production_prefilter_budget_advantage_material"
    if known["positive_evidence_count"] >= 6 and known["canonical_pass_count"] < 5:
        return "production_prefilter_threshold_misaligned"
    if (
        summary["pair_inc_strong_disagreement"]
        and summary["uniform_matching_decisions"] >= 5
    ):
        return "production_prefilter_start_state_pair_specific"
    if (
        known["start_pass_count"] >= 6
        and known["matching_decision_count"] >= 6
        and known["meaningful_start_diversity"]
    ):
        return "production_prefilter_calibration_supports_current_design"
    return "production_prefilter_known_ordering_not_reproduced"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Production Arena Prefilter Calibration",
        "",
        "Inherited: PR #327 `uniform1200_high_power_strength_confirmed`; PR #328 `uniform1200_promotion_arena_failed`; PR #329 `production_start_state_arena_disagreement_confirmed`.",
        "",
        "## Known-Positive Calibration",
        "",
        "| Pair | Label | Start score | Start pass | Canonical score | CI95 | Canonical 0.55 pass | Positive evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["known_positive_rows"]:
        canonical, start = row["canonical"], row["start"]
        ci = canonical["canonical_pair_ci95"]
        lines.append(
            f"| {row['pair_id']} | known_positive | {start['score']:.4f} | {start['production_start_pass']} | {canonical['canonical_pair_score']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {canonical['canonical_threshold_pass']} | {canonical['canonical_positive_strength_evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Identity Controls",
            "",
            "| Pair | Start 384/256 | Canonical 384/256 | CI95 | Pure search-budget effect | Material |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["identity_controls"]:
        ci = row["canonical"]["canonical_pair_ci95"]
        lines.append(
            f"| {row['pair_id']} | {row['start']['score']:.4f} | {row['canonical']['canonical_pair_score']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {row['pure_search_budget_effect']:+.4f} | {row['identity_material_budget_advantage']} |"
        )
    known = result["known_positive"]
    lines.extend(
        [
            "",
            "## Production Repeated Start",
            "",
            "| Pair | W/D/L | Score | P0 score | P1 score | Unique trajectories | Repetition factor | Unique fraction | Seat explanation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["pairs"]:
        start = row["start"]
        p0, p1 = (
            start["challenger_seats"]["player_0"],
            start["challenger_seats"]["player_1"],
        )
        explanation = (
            "deterministic P0 win / P1 loss"
            if p0["score"] == 1 and p1["score"] == 0
            else "deterministic P0 loss / P1 win"
            if p0["score"] == 0 and p1["score"] == 1
            else "both same"
        )
        lines.append(
            f"| {row['pair_id']} | {start['wins']}/{start['draws']}/{start['losses']} | {start['score']:.4f} | {p0['score']:.4f} | {p1['score']:.4f} | {start['unique_trajectories']} | {start['trajectory_repetition_factor']:.1f} | {start['effective_unique_trajectory_fraction']:.4f} | {explanation} |"
        )
    lines.extend(
        [
            "",
            "Every repeated-start arena has one start state. Its nominal 120 games therefore do not correspond to 120 distinct position/game observations; the table reports direct observed trajectory duplication only, not an effective-sample-size estimate.",
            "",
            "## Canonical Results",
            "",
            "| Pair | W/D/L | Pair score | Median | CI95 | P0/P1 | Margin | Unique trajectories |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["pairs"]:
        canonical = row["canonical"]
        seats = canonical["challenger_seats"]
        ci = canonical["canonical_pair_ci95"]
        lines.append(
            f"| {row['pair_id']} | {canonical['wins']}/{canonical['draws']}/{canonical['losses']} | {canonical['canonical_pair_score']:.4f} | {canonical['median_pair_score']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {seats['player_0']['score']:.4f}/{seats['player_1']['score']:.4f} | {canonical['mean_stone_margin']:.2f} | {canonical['unique_trajectories']} |"
        )
    lines.extend(
        [
            "",
            "## Opening-Pair Distribution",
            "",
            "| Pair | Challenger favored | Neutral | Current favored | Challenger both | Current both | Split seats | Draw pairs |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["known_positive_rows"]:
        distribution = row["canonical"]["opening_distribution"]
        lines.append(
            f"| {row['pair_id']} | {distribution['challenger_favored']} | {distribution['neutral']} | {distribution['current_favored']} | {distribution['challenger_wins_both_seats']} | {distribution['current_wins_both_seats']} | {distribution['split_seats']} | {distribution['draws']} |"
        )
    consistency = result["uniform_opening_favored_count_distribution"]
    lines.extend(
        [
            "",
            "Cross-pair opening consistency for only the six uniform-vs-control pairs, kept separate from P_INC: "
            + ", ".join(f"{count}/6={consistency[str(count)]}" for count in range(7))
            + ".",
            "",
            "## Design And Calibration",
            "",
            "Frozen manifest has seven known-positive labels before results: P47-P52 inherited from PR #327 and P_INC from PR #329. The two identity controls have no positive/negative label and are excluded from every /7 sensitivity denominator.",
            f"Third suite SHA256: `{result['suite']['suite_sha256']}`; population 942, historical overlap {result['suite']['historical_suite_overlap']}, remaining {result['suite']['remaining_population']}, selected overlap PR #327/#329 {result['suite']['pr327_overlap']}/{result['suite']['pr329_overlap']}. It uses lexicographic representatives and stable-hash selection with version `{SUITE_VERSION}` and seed {SELECTION_SEED}, without model input.",
            "Production contract: 120 games, threshold 0.55, 384/256, seed 42, seed contract "
            + result["production_contract"]["seed_contract"]
            + ", zero opening plies, deterministic PUCT, c_puct 1.25, zero FPU, no subtree reuse/normalization, root temperature and tactical bias 0, no root-prior/value transforms or opening cache.",
            f"False negatives: `{known['false_negative_count']}/7`; both pass `{sum(row['start']['production_start_pass'] and row['canonical']['canonical_threshold_pass'] for row in result['known_positive_rows'])}/7`; start fail/canonical pass `{known['start_fail_canonical_pass']}/7`; start pass/canonical fail `{sum(row['start']['production_start_pass'] and not row['canonical']['canonical_threshold_pass'] for row in result['known_positive_rows'])}/7`; both fail `{sum(not row['start']['production_start_pass'] and not row['canonical']['canonical_threshold_pass'] for row in result['known_positive_rows'])}/7`.",
            f"Start/canonical distinct scores: {known['start_distinct_scores']}/{known['canonical_distinct_scores']}; variances {known['start_score_variance']:.6f}/{known['canonical_score_variance']:.6f}. Descriptive Spearman historical-vs-start/historical-vs-canonical/start-vs-canonical: {known['spearman_descriptive']['historical_vs_start']:.4f}/{known['spearman_descriptive']['historical_vs_canonical']:.4f}/{known['spearman_descriptive']['start_vs_canonical']:.4f}; historical populations differ, so these are descriptive only.",
            f"Median repeated-start unique-trajectory fraction: `{known['median_effective_unique_trajectory_fraction']:.4f}`. Both canonical seats exceed 0.50 for `{known['both_seats_above_050_count']}/7` known-positive pairs.",
            "",
            "## Classification",
            "",
            f"`{result['classification']}`",
            "",
            f"Next experiment: {result['next_experiment']}",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-production-arena-prefilter-calibration",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-production-prefilter-calibration-openings-v1.jsonl",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-production-prefilter-calibration-opening-preflight-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/alphazero-lite-production-arena-prefilter-calibration.md",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        contract = production_contract()
        manifest = verify_manifest(frozen_pairs())
        write_json(args.out_dir / "frozen-known-pair-manifest.json", manifest)
        suite = persist_suite(args.suite, args.preflight)
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
                "manifest": manifest,
                "production_contract": contract,
                "suite": suite,
                "commands": {
                    pair["pair_id"]: {
                        "start": arena_command(
                            pair, args.out_dir / pair["pair_id"] / "start"
                        ),
                        "canonical": arena_command(
                            pair,
                            args.out_dir / pair["pair_id"] / "canonical",
                            suite=args.suite,
                        ),
                    }
                    for pair in manifest["pairs"]
                },
                "training_invoked": False,
                "promotion": {"performed": False},
            },
        )
        return 0
    rows = []
    for pair in manifest["pairs"]:
        pair_dir = args.out_dir / pair["pair_id"]
        start_dir, canonical_dir = pair_dir / "start", pair_dir / "canonical"
        if not args.summarize_existing and not (start_dir / "arena.json").is_file():
            subprocess.run(arena_command(pair, start_dir), cwd=ROOT, check=True)
        if not args.summarize_existing and not (canonical_dir / "arena.json").is_file():
            subprocess.run(
                arena_command(pair, canonical_dir, suite=args.suite),
                cwd=ROOT,
                check=True,
            )
        for directory in (start_dir, canonical_dir):
            if (
                not (directory / "games.jsonl").is_file()
                or not (directory / "arena.json").is_file()
            ):
                raise FileNotFoundError("production_prefilter_calibration_invalid")
        start_entries = [
            json.loads(line)
            for line in (start_dir / "games.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        canonical_entries = [
            json.loads(line)
            for line in (canonical_dir / "games.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        start = repeated_metrics(
            start_entries,
            json.loads((start_dir / "arena.json").read_text(encoding="utf-8")),
        )
        canonical, pairs = canonical_metrics(
            canonical_entries,
            json.loads((canonical_dir / "arena.json").read_text(encoding="utf-8")),
        )
        (canonical_dir / "opening-pairs.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "opening_index": item["opening_index"],
                        "pair_score": item["pair_score"],
                    },
                    sort_keys=True,
                )
                + "\n"
                for item in pairs
            ),
            encoding="utf-8",
        )
        rows.append(
            {**pair, "start": start, "canonical": canonical, "opening_pairs": pairs}
        )
    known_rows = [row for row in rows if row["frozen_label"] == "known_positive"]
    identities = []
    for row in rows:
        if row["identity_control"]:
            canonical = row["canonical"]
            identities.append(
                {
                    "pair_id": row["pair_id"],
                    "start": row["start"],
                    "canonical": canonical,
                    "pure_search_budget_effect": canonical["canonical_pair_score"]
                    - 0.5,
                    "identity_material_budget_advantage": canonical[
                        "canonical_threshold_pass"
                    ]
                    and canonical["canonical_pair_ci95"]["lower"] > 0.5,
                }
            )
    false_negatives = [
        row
        for row in known_rows
        if row["start"]["score"] < 0.55
        and row["canonical"]["canonical_positive_strength_evidence"]
        and row["canonical"]["canonical_pair_score"] >= 0.55
    ]
    start_fail_canonical_pass = [
        row
        for row in known_rows
        if not row["start"]["production_start_pass"]
        and row["canonical"]["canonical_threshold_pass"]
    ]
    matching = sum(
        row["start"]["production_start_pass"]
        == row["canonical"]["canonical_threshold_pass"]
        for row in known_rows
    )
    positive = {
        "false_negative_count": len(false_negatives),
        "start_fail_canonical_pass": len(start_fail_canonical_pass),
        "start_pass_count": sum(
            row["start"]["production_start_pass"] for row in known_rows
        ),
        "canonical_pass_count": sum(
            row["canonical"]["canonical_threshold_pass"] for row in known_rows
        ),
        "positive_evidence_count": sum(
            row["canonical"]["canonical_positive_strength_evidence"]
            for row in known_rows
        ),
        "matching_decision_count": matching,
        "start_distinct_scores": len({row["start"]["score"] for row in known_rows}),
        "canonical_distinct_scores": len(
            {row["canonical"]["canonical_pair_score"] for row in known_rows}
        ),
        "start_score_variance": statistics.pvariance(
            row["start"]["score"] for row in known_rows
        ),
        "canonical_score_variance": statistics.pvariance(
            row["canonical"]["canonical_pair_score"] for row in known_rows
        ),
        "median_effective_unique_trajectory_fraction": statistics.median(
            row["start"]["effective_unique_trajectory_fraction"] for row in known_rows
        ),
        "both_seats_above_050_count": sum(
            min(
                row["canonical"]["challenger_seats"]["player_0"]["score"],
                row["canonical"]["challenger_seats"]["player_1"]["score"],
            )
            > 0.5
            for row in known_rows
        ),
        "meaningful_start_diversity": statistics.median(
            row["start"]["effective_unique_trajectory_fraction"] for row in known_rows
        )
        > 0.10,
    }
    historical = [
        float(row["historical_independent_strength_score"]) for row in known_rows
    ]
    positive["spearman_descriptive"] = {
        "historical_vs_start": spearman(
            historical, [row["start"]["score"] for row in known_rows]
        ),
        "historical_vs_canonical": spearman(
            historical, [row["canonical"]["canonical_pair_score"] for row in known_rows]
        ),
        "start_vs_canonical": spearman(
            [row["start"]["score"] for row in known_rows],
            [row["canonical"]["canonical_pair_score"] for row in known_rows],
        ),
    }
    primary = {
        "positive_evidence_at_least_6": positive["positive_evidence_count"] >= 6,
        "canonical_pass_at_least_5": positive["canonical_pass_count"] >= 5,
        "start_pass_at_most_3": positive["start_pass_count"] <= 3,
        "flips_at_least_4": positive["start_fail_canonical_pass"] >= 4,
        "median_unique_trajectory_fraction_at_most_010": positive[
            "median_effective_unique_trajectory_fraction"
        ]
        <= 0.10,
        "both_canonical_seats_above_050_at_least_5": positive[
            "both_seats_above_050_count"
        ]
        >= 5,
    }
    primary["passes"] = all(primary.values())
    uniform = [row for row in known_rows if row["pair_id"] != "P_INC"]
    consistency = Counter(
        sum(pair["pair_score"] > 0.5 for pair in opening_index)
        for opening_index in zip(*(row["opening_pairs"] for row in uniform))
    )
    result = {
        "schema": SCHEMA,
        "manifest": manifest,
        "production_contract": contract,
        "suite": suite,
        "pairs": [
            {key: value for key, value in row.items() if key != "opening_pairs"}
            for row in rows
        ],
        "known_positive_rows": known_rows,
        "identity_controls": identities,
        "known_positive": positive,
        "primary_rule": primary,
        "pair_inc_strong_disagreement": next(
            row for row in known_rows if row["pair_id"] == "P_INC"
        )["start"]["score"]
        < 0.55
        and next(row for row in known_rows if row["pair_id"] == "P_INC")["canonical"][
            "canonical_pair_score"
        ]
        >= 0.55,
        "uniform_matching_decisions": sum(
            row["start"]["production_start_pass"]
            == row["canonical"]["canonical_threshold_pass"]
            for row in uniform
        ),
        "uniform_opening_favored_count_distribution": {
            str(count): consistency[count] for count in range(7)
        },
        "training_invoked": False,
        "replay_mutation": False,
        "promotion": {"performed": False},
    }
    result["classification"] = classify(result)
    next_experiments = {
        "production_prefilter_start_state_systematic_false_negative": "Implement a SHADOW canonical-opening prefilter mode in local_promotion_gate, disabled by default, using a fixed canonical-unique holdout suite and opening-pair scoring. Then run the complete downstream gate for seed48 once through that shadow mode.",
        "production_prefilter_budget_advantage_material": "Compare the same frozen calibration pairs under 384/256 versus equal-budget 384/384 on the SAME frozen calibration suite.",
        "production_prefilter_threshold_misaligned": "Calibrate candidate thresholds on the frozen known-positive pairs plus identity controls.",
        "production_prefilter_start_state_pair_specific": "Compare seed48/incumbent start-state trajectory against the six incumbent-holdout openings with the strongest candidate advantage to identify what is exceptional about the initial state.",
        "production_prefilter_calibration_supports_current_design": "Return to the frozen seed48 promotion failure and diagnose why it is the exceptional pair before changing the gate.",
        "production_prefilter_known_ordering_not_reproduced": "Expand the independent canonical evaluation population before drawing any production-gate conclusion.",
    }
    result["next_experiment"] = next_experiments[result["classification"]]
    write_json(args.out_dir / "aggregate.json", result)
    args.report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
