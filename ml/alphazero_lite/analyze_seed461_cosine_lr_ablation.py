"""Validate and analyze the preregistered fixed-E4 cosine-LR arena."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-cosine-lr-ablation-registration.json"
SUITE = DATA / "seed461-cosine-lr-ablation-openings.jsonl"
BINDING = DATA / "seed461-cosine-lr-ablation-evaluation-binding.json"
RESULT = DATA / "seed461-cosine-lr-ablation-results.json"
TRAINING = ROOT / ".tmp/seed461-cosine-lr-ablation/training.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registration, binding = json.loads(REG.read_text()), json.loads(BINDING.read_text())
    training = json.loads(TRAINING.read_text())
    openings = [json.loads(line) for line in SUITE.read_text().splitlines() if line]
    if len(openings) != 256 or sha(SUITE) != registration["holdout"]["sha256"]:
        raise ValueError("registered_suite_mismatch")
    vectors: dict[str, np.ndarray] = {}
    for run, paths in binding["reports"].items():
        report, games = Path(paths["report"]), Path(paths["games"])
        if sha(report) != paths["report_sha256"] or sha(games) != paths["games_sha256"]:
            raise ValueError(f"cached_evidence_hash_mismatch:{run}")
        report_data = json.loads(report.read_text())
        notes = report_data.get("notes", {})
        candidate = binding["candidates"][run]
        for key, expected in (
            ("challenger_path", candidate["artifact"]),
            ("current_path", binding["opponent"]["artifact"]),
            ("suite_sha256", sha(SUITE)),
            ("seed", 386),
            ("seed_contract", "azlite_eval_seed_v2"),
            ("challenger_simulations", 384),
            ("current_simulations", 384),
        ):
            if notes.get(key) != expected:
                raise ValueError(f"report_identity_mismatch:{run}:{key}")
        profile = notes.get("search_profile", {})
        if profile.get("c_puct") != 1.25 or profile.get("simulations") != 384:
            raise ValueError(f"report_search_contract_mismatch:{run}")
        for key, expected in registration["evaluation"]["runtime_contract"].items():
            if notes.get(key) != expected:
                raise ValueError(f"report_runtime_contract_mismatch:{run}:{key}")
        rows = [json.loads(line) for line in games.read_text().splitlines() if line]
        if len(rows) != 512:
            raise ValueError(f"game_count_mismatch:{run}")
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("winner") not in {"challenger", "current", "draw"}:
                raise ValueError(f"unknown_winner:{run}")
            index = int(row["opening_index"])
            if (
                not 0 <= index < 256
                or row.get("opening_prefix_moves") != openings[index]["prefix_moves"]
            ):
                raise ValueError(f"opening_identity_mismatch:{run}:{index}")
            grouped.setdefault(index, []).append(row)
        if set(grouped) != set(range(256)):
            raise ValueError(f"opening_coverage_mismatch:{run}")
        score = np.empty(256, dtype=np.float64)
        for index, pair in grouped.items():
            if len(pair) != 2 or sorted(
                int(row["challenger_player"]) for row in pair
            ) != [0, 1]:
                raise ValueError(f"seat_pairing_mismatch:{run}:{index}")
            score[index] = (
                sum(
                    1.0
                    if row["winner"] == "challenger"
                    else 0.5
                    if row["winner"] == "draw"
                    else 0.0
                    for row in pair
                )
                / 2
            )
        if not np.isclose(score.mean(), report_data["score"]):
            raise ValueError(f"report_score_mismatch:{run}")
        vectors[run] = score
    seeds = range(38611, 38616)
    pairs = {
        str(seed): vectors[f"order_{seed}_B"] - vectors[f"order_{seed}_A"]
        for seed in seeds
    }
    effects = {seed: float(values.mean()) for seed, values in pairs.items()}
    matrix = np.stack(list(pairs.values()))
    rng = np.random.default_rng(386)
    sample_indices = rng.integers(0, 256, size=(10_000, 256))
    draws = matrix[:, sample_indices].mean(axis=2).mean(axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    arm_scores = {
        arm: {str(seed): float(vectors[f"order_{seed}_{arm}"].mean()) for seed in seeds}
        for arm in ("A", "B")
    }
    ranges = {
        arm: max(value.values()) - min(value.values())
        for arm, value in arm_scores.items()
    }
    minima = {arm: min(value.values()) for arm, value in arm_scores.items()}
    mean_effect = float(matrix.mean())
    criteria = {
        "mean_effect_at_least_0.03": mean_effect >= 0.03,
        "lower_interval_above_zero": float(low) > 0,
        "at_least_four_nonnegative_pairs": sum(
            effect >= 0 for effect in effects.values()
        )
        >= 4,
        "worst_pair_at_least_minus_0.05": min(effects.values()) >= -0.05,
        "B_score_range_smaller": ranges["B"] < ranges["A"],
        "B_minimum_no_lower": minima["B"] >= minima["A"],
    }
    result = {
        "schema": "seed461-cosine-lr-ablation-results-v1",
        "status": "completed_fixed_e4",
        "inference_scope": "conditional on these five orders and this frozen dataset",
        "per_order_scores": {
            str(seed): {arm: arm_scores[arm][str(seed)] for arm in ("A", "B")}
            for seed in seeds
        },
        "best_validation_selections_descriptive_only": {
            run: next(
                f"E{row['epoch']}"
                for row in item["history"]
                if item["epochs"][f"E{row['epoch']}"] == item["selected_sha256"]
            )
            for run, item in training["trajectories"].items()
        },
        "paired_effects": effects,
        "mean_effect": mean_effect,
        "worst_pair": min(effects.values()),
        "minimum_score": minima,
        "between_order_range": ranges,
        "bootstrap_95_percentile_interval": [float(low), float(high)],
        "bootstrap": {"resamples": 10000, "seed": 386, "cluster": "shared opening"},
        "criteria": criteria,
        "advance": all(criteria.values()),
        "decision": "advance_to_independent_generation_confirmation"
        if all(criteria.values())
        else "retain_constant_lr_0.001",
        "failed_criteria": [key for key, passed in criteria.items() if not passed],
        "preserves_pr384_rejection": True,
        "no_model_promotion": True,
        "evidence_sha256": {
            "registration": sha(REG),
            "suite": sha(SUITE),
            "binding": sha(BINDING),
        },
        "game_accounting": {
            "games": sum(512 for _ in binding["reports"]),
            "expected_games": 5120,
            "outcome_dependent_extensions": 0,
        },
    }
    if result["game_accounting"]["games"] != 5120:
        raise ValueError("total_game_count_mismatch")
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    table = [
        "# Seed461 cosine learning-rate decay at fixed E4",
        "",
        "Primary evaluation uses E4 from all ten trajectories; best-validation selections are descriptive only.",
        "",
        "| Order | A score | B score | B−A |",
        "|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        table.append(
            f"| {seed} | {arm_scores['A'][str(seed)]:.4f} | {arm_scores['B'][str(seed)]:.4f} | {effects[str(seed)]:+.4f} |"
        )
    table += [
        "",
        f"Mean effect: **{mean_effect:+.4f}** (95% shared-opening bootstrap interval {low:+.4f} to {high:+.4f}).",
        f"Worst pair: {min(effects.values()):+.4f}. Minimum scores A/B: {minima['A']:.4f}/{minima['B']:.4f}; ranges A/B: {ranges['A']:.4f}/{ranges['B']:.4f}.",
        f"Decision: **{result['decision']}**. Failed criteria: {', '.join(result['failed_criteria']) or 'none'}.",
        "",
        "Best-validation selections (descriptive only; arena results did not select checkpoints):",
    ]
    for seed in seeds:
        table.append(
            f"- {seed}: A {result['best_validation_selections_descriptive_only'][f'order_{seed}_A']}, B {result['best_validation_selections_descriptive_only'][f'order_{seed}_B']}"
        )
    table += [
        "",
        "Reproduce from the repository root:",
        "```sh",
        ".venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_ablation register",
        ".venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_ablation train",
        ".venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_ablation publish-training",
        ".venv/bin/python -m ml.alphazero_lite.run_seed461_cosine_lr_arena",
        ".venv/bin/python -m ml.alphazero_lite.analyze_seed461_cosine_lr_ablation",
        "```",
        "",
        "Inference is conditional on these five training orders and this frozen evaluation dataset. PR #384's rejection remains in force; no model is promoted.",
    ]
    (DATA / "seed461-cosine-lr-ablation-results.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
