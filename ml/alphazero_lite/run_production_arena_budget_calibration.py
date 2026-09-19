#!/usr/bin/env python3
"""Measure the causal effect of the production 384/256 search asymmetry.

This evaluation-only runner reuses the immutable PR #330 manifest, suite, and
384/256 outcomes.  It runs only the equal-budget 384/384 counterpart.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import apply_opening_moves, canonical_game_state_hash  # noqa: E402
from ml.alphazero_lite.run_production_arena_prefilter_calibration import (  # noqa: E402
    CHALLENGER_SIMULATIONS,
    GAMES_PER_OPENING,
    OPENING_COUNT,
    OPENING_PLIES,
    PRODUCTION_SEED,
    canonical_metrics,
    frozen_pairs,
    sha256,
    spearman,
    verify_manifest,
)
from ml.alphazero_lite.run_uniform1200_high_power_arena import initial_game  # noqa: E402

SCHEMA = "azlite_production_arena_budget_calibration_v1"
ASYMMETRIC_SIMULATIONS = 256
EQUAL_CURRENT_SIMULATIONS = 384
SUITE_SHA256 = "811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf"
ASYMMETRIC_ROOT = (
    ROOT / "docs/data/alphazero-lite-production-arena-prefilter-calibration"
)
SUITE = (
    ROOT / "docs/data/alphazero-lite-production-prefilter-calibration-openings-v1.jsonl"
)
PREFLIGHT = (
    ROOT
    / "docs/data/alphazero-lite-production-prefilter-calibration-opening-preflight-v1.json"
)
ASYMMETRIC_AGGREGATE = ASYMMETRIC_ROOT / "aggregate.json"
BOOTSTRAP_REPLICATES = 20_000


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_frozen_suite() -> list[dict[str, Any]]:
    """Validate the committed suite, never reconstruct or replace it."""
    if sha256(SUITE) != SUITE_SHA256:
        raise ValueError("budget_calibration_suite_mismatch")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    required = {
        "suite_sha256": SUITE_SHA256,
        "deterministic_reconstruction_sha": SUITE_SHA256,
        "opening_count": OPENING_COUNT,
        "canonical_unique_states": OPENING_COUNT,
        "every_prefix_four_plies": True,
        "every_prefix_legal": True,
        "no_terminal_resulting_state": True,
        "pr327_overlap": 0,
        "pr329_overlap": 0,
    }
    if any(preflight.get(key) != value for key, value in required.items()):
        raise ValueError("budget_calibration_suite_mismatch")
    rows = [
        json.loads(line)
        for line in SUITE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != OPENING_COUNT or {row.get("opening_index") for row in rows} != set(
        range(OPENING_COUNT)
    ):
        raise ValueError("budget_calibration_suite_mismatch")
    hashes = set()
    for row in rows:
        game = initial_game()
        if (
            len(row.get("prefix_moves", [])) != OPENING_PLIES
            or apply_opening_moves(game, row["prefix_moves"]) != OPENING_PLIES
            or game.over()
        ):
            raise ValueError("budget_calibration_suite_mismatch")
        state_hash = canonical_game_state_hash(game)
        if state_hash != row.get("canonical_resulting_state_hash"):
            raise ValueError("budget_calibration_suite_mismatch")
        hashes.add(state_hash)
    if len(hashes) != OPENING_COUNT:
        raise ValueError("budget_calibration_suite_mismatch")
    return rows


def load_asymmetric_results(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load PR #330 results as immutable input and bind them to the manifest."""
    aggregate = json.loads(ASYMMETRIC_AGGREGATE.read_text(encoding="utf-8"))
    if (
        aggregate.get("classification")
        != "production_prefilter_budget_advantage_material"
    ):
        raise ValueError("budget_calibration_artifact_mismatch")
    rows = {row["pair_id"]: row for row in aggregate.get("pairs", [])}
    if set(rows) != {pair["pair_id"] for pair in manifest["pairs"]}:
        raise ValueError("budget_calibration_artifact_mismatch")
    result = {}
    for pair in manifest["pairs"]:
        row = rows[pair["pair_id"]]
        if any(
            row.get(key) != pair.get(key)
            for key in (
                "challenger_sha256",
                "current_sha256",
                "challenger_path",
                "current_path",
            )
        ):
            raise ValueError("budget_calibration_artifact_mismatch")
        path = ASYMMETRIC_ROOT / pair["pair_id"] / "canonical" / "games.jsonl"
        arena_path = path.with_name("arena.json")
        if not path.is_file() or not arena_path.is_file():
            raise ValueError("budget_calibration_artifact_mismatch")
        arena = json.loads(arena_path.read_text(encoding="utf-8"))
        notes = arena.get("notes", {})
        if (
            notes.get("suite_sha256") != SUITE_SHA256
            or notes.get("challenger_simulations") != CHALLENGER_SIMULATIONS
            or notes.get("current_simulations") != ASYMMETRIC_SIMULATIONS
        ):
            raise ValueError("budget_calibration_artifact_mismatch")
        entries = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        metrics, pairs = canonical_metrics(entries, arena)
        if metrics["canonical_pair_score"] != row["canonical"]["canonical_pair_score"]:
            raise ValueError("budget_calibration_artifact_mismatch")
        result[pair["pair_id"]] = {
            "metrics": metrics,
            "opening_pairs": pairs,
            "artifact_sha256": sha256(path),
        }
    return result


def arena_command(pair: dict[str, Any], output: Path) -> list[str]:
    return [
        sys.executable,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(ROOT / pair["challenger_path"]),
        "--current",
        str(ROOT / pair["current_path"]),
        "--games",
        str(OPENING_COUNT * GAMES_PER_OPENING),
        "--challenger-simulations",
        str(CHALLENGER_SIMULATIONS),
        "--current-simulations",
        str(EQUAL_CURRENT_SIMULATIONS),
        "--seed",
        str(PRODUCTION_SEED),
        "--seed-contract",
        "azlite_eval_seed_v2",
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
        "--games-per-opening",
        str(GAMES_PER_OPENING),
        "--opening-prefixes-jsonl",
        str(SUITE),
        "--suite-sha256",
        SUITE_SHA256,
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


def command_delta(pair: dict[str, Any]) -> dict[str, Any]:
    """Record byte-for-byte equivalent options except the opponent budget."""
    from ml.alphazero_lite.run_production_arena_prefilter_calibration import (
        arena_command as old_command,
    )

    old = old_command(pair, Path("asymmetric"), suite=SUITE)
    new = arena_command(pair, Path("equal"))

    # Output ledger paths differ by design; named settings must otherwise agree.
    def options(command: list[str]) -> dict[str, str | None]:
        return {
            command[index]: command[index + 1]
            if index + 1 < len(command) and not command[index + 1].startswith("--")
            else None
            for index in range(len(command))
            if command[index].startswith("--")
            and command[index]
            not in {
                "--current-simulations",
                "--seed-ledger-output",
                "--search-configuration-ledger-output",
                "--search-outcome-ledger-output",
                "--game-jsonl",
                "--out",
            }
        }

    if options(old) != options(new) or old.count("--challenger-simulations") != 1:
        raise ValueError("budget_calibration_artifact_mismatch")
    return {
        "only_semantic_change": "current_simulations: 256 -> 384",
        "asymmetric_current_simulations": ASYMMETRIC_SIMULATIONS,
        "equal_current_simulations": EQUAL_CURRENT_SIMULATIONS,
        "unchanged_options": options(new),
    }


def bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    samples = (
        np.random.default_rng(seed)
        .choice(
            np.asarray(values), size=(BOOTSTRAP_REPLICATES, len(values)), replace=True
        )
        .mean(axis=1)
    )
    return {
        "method": "opening_pair_percentile_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": seed,
        "lower": float(np.percentile(samples, 2.5)),
        "upper": float(np.percentile(samples, 97.5)),
    }


def budget_effect(
    asymmetric: list[dict[str, Any]], equal: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_index = {row["opening_index"]: float(row["pair_score"]) for row in asymmetric}
    equal_by_index = {row["opening_index"]: float(row["pair_score"]) for row in equal}
    if set(by_index) != set(range(OPENING_COUNT)) or set(equal_by_index) != set(
        by_index
    ):
        raise ValueError("budget_calibration_inconclusive")
    rows = [
        {
            "opening_index": index,
            "asymmetric_pair_score": by_index[index],
            "equal_pair_score": equal_by_index[index],
            "budget_effect": by_index[index] - equal_by_index[index],
        }
        for index in range(OPENING_COUNT)
    ]
    values = [row["budget_effect"] for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "budget_effect_ci95": bootstrap(values, 331),
    }, rows


def transition_counts(
    asymmetric: list[dict[str, Any]], equal: list[dict[str, Any]]
) -> dict[str, int]:
    def state(value: float) -> str:
        return (
            "challenger_favored"
            if value > 0.5
            else "current_favored"
            if value < 0.5
            else "neutral"
        )

    counts: Counter[str] = Counter()
    for old, new in zip(asymmetric, equal, strict=True):
        before, after = state(float(new["pair_score"])), state(float(old["pair_score"]))
        counts["unchanged" if before == after else f"{before}_to_{after}"] += 1
    return dict(sorted(counts.items()))


def identity_rule(
    asymmetric_score: float, equal: dict[str, Any], effect: dict[str, Any]
) -> bool:
    ci = equal["canonical_pair_ci95"]
    effect_ci = effect["budget_effect_ci95"]
    return (
        0.48 <= equal["canonical_pair_score"] <= 0.52
        and ci["lower"] <= 0.5 <= ci["upper"]
        and asymmetric_score - equal["canonical_pair_score"] > 0
        and effect_ci["lower"] > 0
    )


def classify(summary: dict[str, Any]) -> str:
    if summary.get("artifact_mismatch"):
        return "budget_calibration_artifact_mismatch"
    if summary.get("suite_mismatch"):
        return "budget_calibration_suite_mismatch"
    if summary.get("execution_failed"):
        return "budget_calibration_inconclusive"
    if summary["identity_failed"]:
        return "equal_budget_identity_control_failed"
    if summary["budget_causal_rule"] and summary["strength_retention_rule"]:
        return (
            "production_prefilter_budget_asymmetry_confound_confirmed_strength_retained"
        )
    if (
        summary["budget_causal_rule"]
        and summary["asymmetry_dependent_055_pass_count"] >= 2
    ):
        return "production_prefilter_budget_asymmetry_confound_confirmed_threshold_impacted"
    if summary["budget_causal_rule"] and summary["positive_evidence_count"] < 5:
        return "production_prefilter_budget_asymmetry_explains_known_ordering"
    if summary["identity_budget_effect_material"] and not summary["budget_causal_rule"]:
        return "production_prefilter_budget_model_interaction"
    return "production_prefilter_budget_effect_small"


def next_experiment(classification: str) -> str:
    return {
        "production_prefilter_budget_asymmetry_confound_confirmed_strength_retained": "Implement an evaluation-only SHADOW canonical-opening prefilter in local_promotion_gate, disabled by default, using equal 384/384 search, the frozen calibration methodology, and opening-pair scoring. Run seed48 once through the complete downstream shadow gate.",
        "production_prefilter_budget_asymmetry_confound_confirmed_threshold_impacted": "Calibrate an equal-budget canonical score threshold using ONLY the frozen seven known-positive pairs plus identity controls. Do not implement a new production threshold yet.",
        "production_prefilter_budget_asymmetry_explains_known_ordering": "Re-establish model-only ordering for the six uniform/control pairs plus seed48/incumbent under an additional independent equal-budget canonical holdout. Do not change gate policy.",
        "production_prefilter_budget_effect_small": "Return to the repeated-start state-diversity problem and implement a shadow canonical-opening prefilter while preserving 384/256 budgets. Do not alter production yet.",
        "production_prefilter_budget_model_interaction": "Audit search-budget sensitivity on the small set of openings with the strongest cross-model interaction. No training and no threshold change.",
        "equal_budget_identity_control_failed": "Audit evaluation seed/seat/search symmetry before interpreting any model-strength comparison.",
    }[classification]


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Production Arena Budget Calibration",
        "",
        "Inherited PR #330 classification: `production_prefilter_budget_advantage_material`.",
        "",
        f"Frozen canonical suite SHA256: `{SUITE_SHA256}`. Frozen pairs: P47, P48, P49, P50, P51, P52, P_INC, I_INC, I_48.",
        "",
        "Only semantic arena change: `current_simulations: 256 -> 384`; challenger remains 384 and all other search settings, suite, seed contract, opening-pair setup, and seats are unchanged.",
        "",
        "## Scores",
        "",
        "| Pair | 384/256 | 384/384 CI95 | Budget effect CI95 | 0.55 asymmetric/equal |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["pairs"]:
        equal, effect = row["equal"], row["budget_effect"]
        ci, effect_ci = equal["canonical_pair_ci95"], effect["budget_effect_ci95"]
        lines.append(
            f"| {row['pair_id']} | {row['asymmetric']['canonical_pair_score']:.4f} | {equal['canonical_pair_score']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] | {effect['mean']:+.4f} [{effect_ci['lower']:+.4f}, {effect_ci['upper']:+.4f}] | {row['asymmetric']['canonical_pair_score'] >= 0.55}/{equal['canonical_pair_score'] >= 0.55} |"
        )
    lines.extend(
        [
            "",
            "## Equal-Budget Arena Metrics",
            "",
            "| Pair | W/D/L | Raw/pair/median | P0/P1 | Margin | Trajectories | Favored/neutral/current | Both challenger/current/split/draw |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["pairs"]:
        equal, seats, distribution = (
            row["equal"],
            row["equal"]["challenger_seats"],
            row["equal"]["opening_distribution"],
        )
        lines.append(
            f"| {row['pair_id']} | {equal['wins']}/{equal['draws']}/{equal['losses']} | {equal['raw_score']:.4f}/{equal['canonical_pair_score']:.4f}/{equal['median_pair_score']:.4f} | {seats['player_0']['score']:.4f}/{seats['player_1']['score']:.4f} | {equal['mean_stone_margin']:.2f} | {equal['unique_trajectories']} | {distribution['challenger_favored']}/{distribution['neutral']}/{distribution['current_favored']} | {distribution['challenger_wins_both_seats']}/{distribution['current_wins_both_seats']}/{distribution['split_seats']}/{distribution['draws']} |"
        )
    lines.extend(
        [
            "",
            "## Identity Controls",
            "",
            "| Pair | Equal score | Seat P0/P1 | Bias removed |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in result["identity_controls"]:
        seats = row["equal"]["challenger_seats"]
        lines.append(
            f"| {row['pair_id']} | {row['equal']['canonical_pair_score']:.4f} | {seats['player_0']['score']:.4f}/{seats['player_1']['score']:.4f} | {row['identity_budget_bias_removed']} |"
        )
    lines.extend(
        [
            "",
            "Equal controls use deterministic trajectory pairing and their paired games reverse winners by seat; seed/search ledgers are retained in the equal-budget arena artifacts.",
            "",
            "## Causal Effects",
            "",
            "| Pair | Mean | Median | P10/P90 | Positive/zero/negative | Margin shift |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["pairs"]:
        effect = row["budget_effect"]
        lines.append(
            f"| {row['pair_id']} | {effect['mean']:+.4f} | {effect['median']:+.4f} | {effect['p10']:+.4f}/{effect['p90']:+.4f} | {effect['positive']}/{effect['zero']}/{effect['negative']} | {row['stone_margin_shift']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "Observed asymmetric advantage is `score_384_256 - 0.5`; equal-budget model advantage is `score_384_384 - 0.5`; budget shift is their difference. These are descriptive paired comparisons, not an exact additive correction.",
            "",
            "## Retention And Threshold",
            "",
            "| Pair | Direction | Asymmetric >=0.55 | Equal >=0.55 | Equal positive evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in result["known_positive"]:
        lines.append(
            f"| {row['pair_id']} | {row['direction']} | {row['asymmetric']['canonical_pair_score'] >= 0.55} | {row['equal']['canonical_pair_score'] >= 0.55} | {row['equal_budget_positive_evidence']} |"
        )
    lines.extend(
        [
            "",
            "P_INC independent PR #329 holdout 384/256 score: `0.8838`; PR #330 calibration 384/256 score: `0.8896`; this suite's 384/384 score and paired effect are shown above. `seed48_incumbent_equal_budget_confirmed`: `"
            + str(result["seed48_incumbent_equal_budget_confirmed"])
            + "`.",
            f"Known positive evidence at equal budget: `{result['positive_evidence_count']}/7`; equal-budget 0.55 pass: `{result['equal_budget_055_pass_count']}/7`; asymmetry-dependent 0.55 pass: `{result['asymmetry_dependent_055_pass_count']}/7`.",
            "",
            "## Opening Sensitivity",
            "",
            "Positive budget-effect histogram across seven known-positive pairs: "
            + ", ".join(
                f"{index}/7={count}"
                for index, count in result[
                    "opening_budget_sensitivity_histogram"
                ].items()
            )
            + ".",
            f"Repeated budget-sensitive openings (>=5/7): `{result['repeated_budget_sensitive_opening']['count']}`: {result['repeated_budget_sensitive_opening']['opening_ids']}.",
            "Outcome-transition counts and all per-opening effects are retained per pair in the aggregate/artifacts. Budget-shift heterogeneity mean/median/range: "
            + "/".join(
                f"{value:+.4f}" if isinstance(value, float) else str(value)
                for value in (
                    result["budget_shift_heterogeneity"]["mean"],
                    result["budget_shift_heterogeneity"]["median"],
                )
            )
            + f"/[{result['budget_shift_heterogeneity']['range'][0]:+.4f}, {result['budget_shift_heterogeneity']['range'][1]:+.4f}].",
            "",
            "## Rules",
            "",
            f"Primary budget-causal rule: `{result['budget_causal_rule']}`. Model-strength retention rule: `{result['strength_retention_rule']}`.",
            "",
            "## Classification",
            "",
            f"`{result['classification']}`",
            "",
            f"Exactly one next experiment: {result['next_experiment']}",
            "",
        ]
    )
    lines.extend(["", "## Outcome Transitions", ""])
    for row in result["pairs"]:
        lines.append(
            f"`{row['pair_id']}`: {json.dumps(row['outcome_transitions'], sort_keys=True)}"
        )
    lines.append(
        "Descriptive Spearman equal-budget score versus budget shift: "
        f"`{result['equal_score_budget_shift_spearman']:.4f}`; no mechanism is inferred."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-production-arena-budget-calibration",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/alphazero-lite-production-arena-budget-calibration.md",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = verify_manifest(frozen_pairs())
    except (OSError, ValueError):
        write_json(
            args.out_dir / "aggregate.json",
            {
                "schema": SCHEMA,
                "classification": "budget_calibration_artifact_mismatch",
                "arena_executed": False,
            },
        )
        return 0
    try:
        suite = load_frozen_suite()
        asymmetric = load_asymmetric_results(manifest)
        del suite
        proof = command_delta(manifest["pairs"][0])
    except ValueError as error:
        write_json(
            args.out_dir / "aggregate.json",
            {"schema": SCHEMA, "classification": str(error), "arena_executed": False},
        )
        return 0
    write_json(
        args.out_dir / "frozen-pr330-manifest.json",
        {
            **manifest,
            "asymmetric_aggregate_path": str(ASYMMETRIC_AGGREGATE.relative_to(ROOT)),
            "asymmetric_aggregate_sha256": sha256(ASYMMETRIC_AGGREGATE),
            "immutable": True,
        },
    )
    if args.prepare_only:
        write_json(
            args.out_dir / "plan.json",
            {
                "schema": SCHEMA,
                "manifest": manifest,
                "suite_sha256": SUITE_SHA256,
                "search_contract_proof": proof,
                "commands": {
                    pair["pair_id"]: arena_command(
                        pair, args.out_dir / pair["pair_id"] / "equal-384-384"
                    )
                    for pair in manifest["pairs"]
                },
                "training_invoked": False,
                "promotion": {"performed": False},
            },
        )
        return 0
    rows = []
    for pair in manifest["pairs"]:
        directory = args.out_dir / pair["pair_id"] / "equal-384-384"
        if not args.summarize_existing and not (directory / "arena.json").is_file():
            subprocess.run(arena_command(pair, directory), cwd=ROOT, check=True)
        games_path, arena_path = directory / "games.jsonl", directory / "arena.json"
        if not games_path.is_file() or not arena_path.is_file():
            raise RuntimeError("budget_calibration_inconclusive")
        arena = json.loads(arena_path.read_text(encoding="utf-8"))
        if (
            arena.get("notes", {}).get("current_simulations")
            != EQUAL_CURRENT_SIMULATIONS
        ):
            raise RuntimeError("budget_calibration_inconclusive")
        entries = [
            json.loads(line)
            for line in games_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        equal, equal_pairs = canonical_metrics(entries, arena)
        equal["canonical_pair_ci95"] = bootstrap(
            [item["pair_score"] for item in equal_pairs], 330
        )
        effect, effect_rows = budget_effect(
            asymmetric[pair["pair_id"]]["opening_pairs"], equal_pairs
        )
        write_json(directory / "metrics.json", equal)
        (directory / "opening-pairs.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in equal_pairs),
            encoding="utf-8",
        )
        (args.out_dir / pair["pair_id"] / "paired-budget-effects.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in effect_rows),
            encoding="utf-8",
        )
        rows.append(
            {
                **pair,
                "asymmetric": asymmetric[pair["pair_id"]]["metrics"],
                "equal": equal,
                "budget_effect": effect,
                "outcome_transitions": transition_counts(
                    asymmetric[pair["pair_id"]]["opening_pairs"], equal_pairs
                ),
                "stone_margin_shift": asymmetric[pair["pair_id"]]["metrics"][
                    "mean_stone_margin"
                ]
                - equal["mean_stone_margin"],
                "equal_pairs": equal_pairs,
                "effect_rows": effect_rows,
            }
        )
    known = [row for row in rows if row["frozen_label"] == "known_positive"]
    for row in known:
        equal_score = row["equal"]["canonical_pair_score"]
        ci_lower = row["equal"]["canonical_pair_ci95"]["lower"]
        row["equal_budget_positive_evidence"] = equal_score > 0.5 and ci_lower > 0.5
        row["direction"] = (
            "direction_reversed"
            if equal_score < 0.5
            else "direction_statistically_supported"
            if row["equal_budget_positive_evidence"]
            else "direction_retained"
            if equal_score > 0.5
            else "direction_lost"
        )
    controls = []
    for row in rows:
        if row["identity_control"]:
            controls.append(
                {
                    "pair_id": row["pair_id"],
                    "asymmetric": row["asymmetric"],
                    "equal": row["equal"],
                    "identity_budget_bias_removed": identity_rule(
                        row["asymmetric"]["canonical_pair_score"],
                        row["equal"],
                        row["budget_effect"],
                    ),
                    "deterministic_trajectory_pairing": True,
                    "reverse_winners_by_seat": row["equal"]["canonical_pair_score"]
                    == 0.5,
                }
            )
    shifts = [row["budget_effect"]["mean"] for row in known]
    positive_count = sum(row["equal_budget_positive_evidence"] for row in known)
    equal_055 = sum(row["equal"]["canonical_pair_score"] >= 0.55 for row in known)
    dependent = sum(
        row["asymmetric"]["canonical_pair_score"] >= 0.55
        and row["equal"]["canonical_pair_score"] < 0.55
        for row in known
    )
    sensitivity = Counter(
        sum(row["effect_rows"][index]["budget_effect"] > 0 for row in known)
        for index in range(OPENING_COUNT)
    )
    sensitive_ids = [
        index
        for index in range(OPENING_COUNT)
        if sum(row["effect_rows"][index]["budget_effect"] > 0 for row in known) >= 5
    ]
    identity_shifts = [
        row["budget_effect"]["mean"] for row in rows if row["identity_control"]
    ]
    seed48 = next(row for row in known if row["pair_id"] == "P_INC")
    summary = {
        "schema": SCHEMA,
        "manifest": manifest,
        "suite_sha256": SUITE_SHA256,
        "search_contract_proof": proof,
        "pairs": [
            {
                key: value
                for key, value in row.items()
                if key not in {"equal_pairs", "effect_rows"}
            }
            for row in rows
        ],
        "known_positive": [
            {
                key: value
                for key, value in row.items()
                if key not in {"equal_pairs", "effect_rows"}
            }
            for row in known
        ],
        "identity_controls": controls,
        "positive_evidence_count": positive_count,
        "equal_budget_055_pass_count": equal_055,
        "asymmetry_dependent_055_pass_count": dependent,
        "seed48_incumbent_equal_budget_confirmed": seed48["equal"][
            "canonical_pair_score"
        ]
        > 0.55
        and seed48["equal"]["canonical_pair_ci95"]["lower"] > 0.5,
        "budget_shift_heterogeneity": {
            "mean": statistics.fmean(shifts + identity_shifts),
            "median": statistics.median(shifts + identity_shifts),
            "range": [min(shifts + identity_shifts), max(shifts + identity_shifts)],
        },
        "opening_budget_sensitivity_histogram": {
            str(index): sensitivity[index] for index in range(8)
        },
        "repeated_budget_sensitive_opening": {
            "count": len(sensitive_ids),
            "opening_ids": sensitive_ids,
        },
        "identity_failed": any(
            not row["identity_budget_bias_removed"] for row in controls
        ),
        "identity_budget_effect_material": statistics.fmean(identity_shifts) >= 0.05,
        "budget_causal_rule": all(
            row["identity_budget_bias_removed"] for row in controls
        )
        and statistics.fmean(identity_shifts) >= 0.05
        and sum(shift > 0 for shift in shifts) >= 5
        and (dependent >= 1 or statistics.median(shifts) >= 0.05),
        "strength_retention_rule": sum(
            row["equal"]["canonical_pair_score"] > 0.5 for row in known
        )
        >= 6
        and positive_count >= 5
        and seed48["equal"]["canonical_pair_score"] > 0.55
        and seed48["equal"]["canonical_pair_ci95"]["lower"] > 0.5,
        "training_invoked": False,
        "promotion": {"performed": False},
    }
    summary["equal_score_budget_shift_spearman"] = spearman(
        [row["equal"]["canonical_pair_score"] for row in known], shifts
    )
    summary["classification"] = classify(summary)
    summary["next_experiment"] = next_experiment(summary["classification"])
    write_json(args.out_dir / "aggregate.json", summary)
    args.report.write_text(render_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
