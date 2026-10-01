"""Validate and analyze the registered seed461 checkpoint-averaging arena."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-e2-e4-average-registration.json"
SUITE = DATA / "seed461-e2-e4-average-openings.jsonl"
CANDIDATES = DATA / "seed461-e2-e4-average-candidate-binding.json"
BINDING = DATA / "seed461-e2-e4-average-evaluation-binding.json"
RESULT = DATA / "seed461-e2-e4-average-results.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_game_records(
    rows: list[dict[str, Any]], openings: list[dict[str, Any]], run: str
) -> np.ndarray:
    if len(rows) != 512:
        raise ValueError(f"game_count_mismatch:{run}")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("winner") not in {"challenger", "current", "draw"}:
            raise ValueError(f"unknown_winner:{run}")
        index = row.get("opening_index")
        if (
            not isinstance(index, int)
            or not 0 <= index < 256
            or row.get("opening_prefix_moves") != openings[index]["prefix_moves"]
        ):
            raise ValueError(f"opening_identity_mismatch:{run}")
        grouped.setdefault(index, []).append(row)
    if set(grouped) != set(range(256)):
        raise ValueError(f"opening_coverage_mismatch:{run}")
    scores = np.empty(256, dtype=np.float64)
    for index, pair in grouped.items():
        seats = [row.get("challenger_player") for row in pair]
        if (
            len(pair) != 2
            or any(seat not in (0, 1) for seat in seats)
            or sorted(seats) != [0, 1]
        ):
            raise ValueError(f"seat_pairing_mismatch:{run}:{index}")
        scores[index] = (
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
    return scores


def main() -> None:
    reg, candidates, binding = (
        json.loads(path.read_text()) for path in (REG, CANDIDATES, BINDING)
    )
    suite_hash = sha(SUITE)
    if (
        suite_hash != reg["evaluation"]["suite"]["sha256"]
        or binding["suite_sha256"] != suite_hash
    ):
        raise ValueError("registered_suite_mismatch")
    if binding.get("registration_sha256") != sha(REG) or binding.get(
        "candidate_binding_sha256"
    ) != sha(CANDIDATES):
        raise ValueError("evaluation_binding_chain_mismatch")
    if binding.get("candidates") != candidates["candidates"]:
        raise ValueError("evaluation_candidate_identity_mismatch")
    openings = [json.loads(line) for line in SUITE.read_text().splitlines() if line]
    if len(openings) != 256:
        raise ValueError("opening_count_mismatch")
    expected_runs = [
        f"order_{order}_{arm}"
        for order in reg["reused_training"]["orders"]
        for arm in ("A", "B")
    ]
    if set(binding["reports"]) != set(expected_runs):
        raise ValueError("evaluation_set_incomplete")
    vectors: dict[str, np.ndarray] = {}
    for run in expected_runs:
        paths, candidate = binding["reports"][run], candidates["candidates"][run]
        report_path, games_path = Path(paths["report"]), Path(paths["games"])
        if (
            sha(report_path) != paths["report_sha256"]
            or sha(games_path) != paths["games_sha256"]
        ):
            raise ValueError(f"cached_evidence_hash_mismatch:{run}")
        report = json.loads(report_path.read_text())
        notes = report.get("notes", {})
        evaluation = reg["evaluation"]
        for key, expected in (
            ("challenger_path", candidate["artifact"]),
            ("current_path", binding["opponent"]["artifact"]),
            ("suite_sha256", suite_hash),
            ("seed", evaluation["arena_seed"]),
            ("seed_contract", evaluation["seed_contract"]),
            ("challenger_simulations", evaluation["simulations_per_side"]),
            ("current_simulations", evaluation["simulations_per_side"]),
        ):
            if notes.get(key) != expected:
                raise ValueError(f"report_identity_mismatch:{run}:{key}")
        if notes.get("search_profile", {}).get("c_puct") != evaluation["c_puct"]:
            raise ValueError(f"report_search_contract_mismatch:{run}")
        for key, expected in candidate["runtime_contract"].items():
            if notes.get(key) != expected:
                raise ValueError(f"report_runtime_contract_mismatch:{run}:{key}")
        scores = validate_game_records(
            [json.loads(line) for line in games_path.read_text().splitlines() if line],
            openings,
            run,
        )
        if not np.isclose(scores.mean(), report.get("score", float("nan"))):
            raise ValueError(f"report_score_mismatch:{run}")
        vectors[run] = scores
    orders = reg["reused_training"]["orders"]
    pairs = {
        str(order): vectors[f"order_{order}_B"] - vectors[f"order_{order}_A"]
        for order in orders
    }
    effects = {order: float(values.mean()) for order, values in pairs.items()}
    matrix = np.stack(list(pairs.values()))
    boot = reg["analysis"]["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    sampled = rng.integers(0, 256, size=(boot["resamples"], 256))
    draws = matrix[:, sampled].mean(axis=2).mean(axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    scores = {
        arm: {
            str(order): float(vectors[f"order_{order}_{arm}"].mean())
            for order in orders
        }
        for arm in ("A", "B")
    }
    ranges = {
        arm: max(values.values()) - min(values.values())
        for arm, values in scores.items()
    }
    minima = {arm: min(values.values()) for arm, values in scores.items()}
    t = reg["analysis"]["decision_thresholds"]
    criteria = {
        "mean_effect_at_least_0.03": float(matrix.mean()) >= t["mean_effect_at_least"],
        "lower_interval_above_zero": float(low) > t["lower_interval_strictly_above"],
        "at_least_four_nonnegative_pairs": sum(value >= 0 for value in effects.values())
        >= t["nonnegative_paired_effects_at_least"],
        "worst_pair_at_least_minus_0.05": min(effects.values())
        >= t["worst_paired_effect_at_least"],
        "B_score_range_smaller": ranges["B"] < ranges["A"],
        "B_minimum_no_lower": minima["B"] >= minima["A"],
    }
    advance = all(criteria.values())
    result = {
        "schema": "seed461-e2-e4-average-results-v1",
        "status": "completed_fixed_5120_games",
        "inference_scope": reg["analysis"]["inference_scope"],
        "per_order_scores": {
            str(order): {arm: scores[arm][str(order)] for arm in ("A", "B")}
            for order in orders
        },
        "paired_effects": effects,
        "mean_effect": float(matrix.mean()),
        "worst_pair": min(effects.values()),
        "minimum_score": minima,
        "between_order_range": ranges,
        "bootstrap_95_percentile_interval": [float(low), float(high)],
        "bootstrap": boot,
        "criteria": criteria,
        "advance": advance,
        "decision": "advance_to_independent_generation_confirmation"
        if advance
        else "retain_constant_lr_0.001",
        "failed_criteria": [key for key, passed in criteria.items() if not passed],
        "preserves_pr386_cosine_rejection": True,
        "no_model_promotion": True,
        "evidence_sha256": {
            "registration": sha(REG),
            "suite": suite_hash,
            "candidate_binding": sha(CANDIDATES),
            "evaluation_binding": sha(BINDING),
        },
        "game_accounting": {
            "games": 5120,
            "expected_games": reg["evaluation"]["games"],
            "outcome_dependent_extensions": 0,
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Seed461 E2–E4 checkpoint averaging",
        "",
        "| Order | A score | B score | B−A |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {order} | {scores['A'][str(order)]:.4f} | {scores['B'][str(order)]:.4f} | {effects[str(order)]:+.4f} |"
        for order in orders
    )
    lines += [
        "",
        f"Mean paired effect: **{matrix.mean():+.4f}** (95% shared-opening bootstrap interval {low:+.4f} to {high:+.4f}).",
        f"Worst pair: {min(effects.values()):+.4f}; minimum A/B: {minima['A']:.4f}/{minima['B']:.4f}; range A/B: {ranges['A']:.4f}/{ranges['B']:.4f}.",
        f"Decision: **{result['decision']}**. Failed criteria: {', '.join(result['failed_criteria']) or 'none'}.",
        "",
        "Parameter distances are descriptive only. Inference is conditional on these five orders and this dataset; PR #386's cosine rejection remains unchanged. No model is promoted.",
        "",
        "Reproduce from repository root:",
        "```sh",
        ".venv/bin/python -m ml.alphazero_lite.run_seed461_e2_e4_average_arena",
        ".venv/bin/python -m ml.alphazero_lite.analyze_seed461_e2_e4_average",
        "```",
        "",
    ]
    (DATA / "seed461-e2-e4-average-results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
