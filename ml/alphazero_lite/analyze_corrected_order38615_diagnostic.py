"""Reproduce preregistered cluster-bootstrap analysis and execution summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed461_arena_validation as validation
from ml.alphazero_lite.build_opening_suite import load_suite_jsonl
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.build_opening_suite import INITIAL_STATE, canonical_key
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/order38615-corrected-diagnostic"
REG = DATA / "registration.json"
BIND = DATA / "evaluation-binding.json"
WORK = ROOT / ".tmp/order38615-corrected-diagnostic"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze() -> dict[str, Any]:
    registration = json.loads(REG.read_text())
    binding = json.loads(BIND.read_text())
    if binding.get("status") != "completed_2048_games":
        raise ValueError("diagnostic_not_complete")
    if binding.get("registration_sha256") != sha(REG):
        raise ValueError("registration_hash_mismatch")
    candidate = registration["candidate"]
    opponent = registration["opponent"]
    all_scores: dict[str, np.ndarray] = {}
    outcomes: dict[str, dict[str, int | float]] = {}
    runtime: dict[str, dict[str, float | int]] = {}
    raw_hashes: dict[str, dict[str, str]] = {}
    for seed_text, spec in registration["evaluation"]["suites"].items():
        report_path = WORK / f"seed{seed_text}.json"
        games_path = WORK / f"seed{seed_text}-games.jsonl"
        bound = binding["reports"][seed_text]
        if (
            sha(report_path) != bound["report_sha256"]
            or sha(games_path) != bound["games_sha256"]
        ):
            raise ValueError(f"raw_evidence_hash_mismatch:{seed_text}")
        openings = load_suite_jsonl(str(ROOT / spec["path"]))
        report = json.loads(report_path.read_text())
        rows = [
            json.loads(line) for line in games_path.read_text().splitlines() if line
        ]
        scores = validation.validate_arena_evidence(
            report,
            rows,
            openings,
            seed_text,
            {
                "artifact": candidate["artifact"],
                "runtime_contract": candidate["runtime_contract"],
            },
            opponent,
            {
                **registration["evaluation"],
                "games_per_candidate": registration["evaluation"]["games_per_suite"],
                "suite": spec,
                "arena_seed": int(seed_text),
                "seed_contract": registration["evaluation"]["seed_contract"],
            },
        )
        all_scores[seed_text] = scores
        outcomes[seed_text] = {
            "games": len(rows),
            "wins": int(report["wins"]),
            "draws": int(report["draws"]),
            "losses": int(report["losses"]),
            "score": float(scores.mean()),
        }
        runtime[seed_text] = {
            "games": len(rows),
            "workers_used": int(report["notes"]["workers_used"]),
            "simulations_per_side": int(report["notes"]["challenger_simulations"]),
            "move_time_mean_ms": float(report["notes"]["move_time_mean_ms"]),
            "move_time_p95_ms": float(report["notes"]["move_time_p95_ms"]),
            "summed_normal_puct_ms": float(
                sum(row.get("normal_puct_time_ms", 0.0) for row in rows)
            ),
            "summed_exact_root_solve_ms": float(
                sum(row.get("exact_root_solve_time_ms", 0.0) for row in rows)
            ),
        }
        raw_hashes[seed_text] = {
            "report_sha256": sha(report_path),
            "games_sha256": sha(games_path),
        }

    bootstrap = registration["analysis"]
    per_suite: dict[str, dict[str, Any]] = {}
    for seed_text, values in all_scores.items():
        seed = int(seed_text)
        rng = np.random.default_rng(seed)
        samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(
            axis=1
        )
        per_suite[seed_text] = {
            **outcomes[seed_text],
            "bootstrap_seed": seed,
            "resamples": 10000,
            "interval_95_percentile": [
                float(x) for x in np.quantile(samples, [0.025, 0.975])
            ],
        }
    rng = np.random.default_rng(393)
    left = rng.integers(0, 512, size=(10000, 512))
    right = rng.integers(0, 512, size=(10000, 512))
    pooled_samples = (
        all_scores["393"][left].mean(axis=1) + all_scores["394"][right].mean(axis=1)
    ) / 2
    pooled_score = float((all_scores["393"].mean() + all_scores["394"].mean()) / 2)
    pooled_interval = [float(x) for x in np.quantile(pooled_samples, [0.025, 0.975])]
    thresholds = bootstrap["success_rule"]
    suite_pass = all(
        row["score"] >= thresholds["each_score_at_least"]
        and row["interval_95_percentile"][0]
        > thresholds["each_interval_lower_strictly_above"]
        for row in per_suite.values()
    )
    pooled_pass = (
        pooled_score >= thresholds["pooled_score_at_least"]
        and pooled_interval[0] > thresholds["pooled_interval_lower_strictly_above"]
    )
    prior_declared: set[str] = set()
    prior_actual: set[str] = set()
    for suite_record in registration["exclusion_proof"]["prior_evaluations"].values():
        prior_path = ROOT / suite_record["path"]
        for opening in load_suite_jsonl(str(prior_path)):
            if "state" in opening:
                prior_declared.add(canonical_key(opening["state"]))
            game = KalahGame.from_state(INITIAL_STATE)
            apply_opening_moves(game, [int(move) for move in opening["prefix_moves"]])
            prior_actual.add(canonical_key(game.to_state()))
    selected_identities = {
        seed: set(values)
        for seed, values in registration["exclusion_proof"][
            "selected_state_identities"
        ].items()
    }
    declared_intersections = {
        seed: len(values & prior_declared)
        for seed, values in selected_identities.items()
    }
    actual_intersections = {
        seed: len(values & prior_actual) for seed, values in selected_identities.items()
    }
    protocol_valid = not any(declared_intersections.values()) and not any(
        actual_intersections.values()
    )
    matrix = {
        "schema": "order38615-corrected-diagnostic-score-matrix-v1",
        "registration_sha256": sha(REG),
        "evaluation_binding_sha256": sha(BIND),
        "raw_evidence_sha256": raw_hashes,
        "opening_scores": {
            seed: [float(score) for score in scores]
            for seed, scores in all_scores.items()
        },
    }
    matrix_path = DATA / "opening-score-matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    result = {
        "schema": "order38615-corrected-diagnostic-results-v1",
        "registration_sha256": sha(REG),
        "evaluation_binding_sha256": sha(BIND),
        "opening_score_matrix_sha256": sha(matrix_path),
        "suite_results": per_suite,
        "pooled_score": pooled_score,
        "pooled_interval_95_percentile": pooled_interval,
        "pooled_bootstrap_seed": 393,
        "pooled_resamples": 10000,
        "per_suite_and_pooled_pass": bool(suite_pass and pooled_pass),
        "suite_pass": bool(suite_pass),
        "pooled_pass": bool(pooled_pass),
        "canonical_promotion_failure_overridden": False,
        "validity_status": "valid"
        if protocol_valid
        else "invalid_for_preregistered_exclusion_contract",
        "prior_declared_state_union": len(prior_declared),
        "prior_actual_state_union": len(prior_actual),
        "prior_declared_intersections": declared_intersections,
        "prior_actual_intersections": actual_intersections,
        "outcomes": outcomes,
        "runtime": runtime,
        "total_games": sum(row["games"] for row in outcomes.values()),
        "total_search_simulations": 2048 * 2 * 384,
        "training_or_export_cost": "none",
    }
    (DATA / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not protocol_valid:
        validity = {
            "schema": "order38615-corrected-diagnostic-post-run-validity-v1",
            "registration_sha256": sha(REG),
            "evaluation_binding_sha256": sha(BIND),
            "status": "invalid_for_preregistered_exclusion_contract",
            "prior_consumed_suite_count": len(
                registration["exclusion_proof"]["prior_evaluations"]
            ),
            "prior_declared_state_union": len(prior_declared),
            "prior_reconstructed_actual_state_union": len(prior_actual),
            "suite_declared_state_intersections": declared_intersections,
            "suite_actual_state_intersections": actual_intersections,
            "suite_393_suite_394_actual_overlap": 0,
            "observed_scores": {
                "393": per_suite["393"]["score"],
                "394": per_suite["394"]["score"],
                "pooled": pooled_score,
            },
            "observed_decision_rule_result": "pass"
            if suite_pass and pooled_pass
            else "fail",
            "interpretation": "Observed scores are preserved, but this run cannot establish corrected-distribution performance because declared prior-suite states were not excluded. No outcome-dependent replacement games were run.",
        }
        (DATA / "post-run-validity.json").write_text(
            json.dumps(validity, indent=2, sort_keys=True) + "\n"
        )
    lines = [
        "# Corrected frozen order 38615 E4 diagnostic",
        "",
        f"Observed score-threshold outcome: **{'PASS' if result['per_suite_and_pooled_pass'] else 'FAIL'}**. Protocol validity: **{result['validity_status']}**.",
        "",
        "| Suite | Score | Wins | Draws | Losses | 95% opening-cluster interval | Rule |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for seed, row in per_suite.items():
        passed = row["score"] >= 0.55 and row["interval_95_percentile"][0] > 0.50
        lines.append(
            f"| {seed} | {row['score']:.4f} | {row['wins']} | {row['draws']} | {row['losses']} | {row['interval_95_percentile'][0]:.4f}–{row['interval_95_percentile'][1]:.4f} | {'PASS' if passed else 'FAIL'} |"
        )
    lines += [
        "",
        f"Pooled equal-weight score: **{pooled_score:.4f}**; stratified 95% interval {pooled_interval[0]:.4f}–{pooled_interval[1]:.4f} ({'PASS' if pooled_pass else 'FAIL'}).",
        "",
        "This diagnostic estimates only the corrected opening distribution and does not override the canonical promotion failure.",
        "",
        "The score matrix contains 512 paired-opening cluster scores per suite; `analyze_corrected_order38615_diagnostic.py` verifies raw evidence hashes and reproduces 10,000-resample intervals.",
        "",
        f"Execution: {result['total_games']} games, {result['total_search_simulations']:,} requested search simulations, 384 per side, 24 workers; no training or model export.",
        "",
        "| Suite | Move mean (ms) | Move p95 (ms) | Summed PUCT (ms) | Summed exact-root (ms) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, item in runtime.items():
        lines.append(
            f"| {seed} | {item['move_time_mean_ms']:.2f} | {item['move_time_p95_ms']:.2f} | {item['summed_normal_puct_ms']:.1f} | {item['summed_exact_root_solve_ms']:.1f} |"
        )
    (DATA / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    analyze()
