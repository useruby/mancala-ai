"""Reproduce preregistered cluster-bootstrap analysis and execution summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed461_arena_validation as validation
from ml.alphazero_lite.build_opening_suite import load_suite_jsonl
from ml.alphazero_lite.build_opening_suite import validate_arena_entries
from ml.alphazero_lite.opening_exclusion_contract import (
    validate_suite_against_manifest,
    verify_manifest,
)
from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/order38615-a5-frozen-diagnostic-v4"
REG = DATA / "registration.json"
BIND = DATA / "evaluation-binding.json"
SOURCE_BIND = ROOT / "docs/data/order38615-a5-confirmation-candidate-binding.json"
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
    source = json.loads(SOURCE_BIND.read_text())
    if binding.get("source_candidate_binding_sha256") != sha(SOURCE_BIND):
        raise ValueError("candidate_source_binding_mismatch")
    manifest_path = ROOT / registration["exclusion_proof"]["manifest_path"]
    if (
        binding.get("exclusion_manifest_sha256") != sha(manifest_path)
        or sha(manifest_path) != registration["exclusion_proof"]["manifest_sha256"]
    ):
        raise ValueError("exclusion_manifest_binding_mismatch")
    manifest = verify_manifest(manifest_path)
    candidate = registration["candidate"]
    opponent = registration["opponent"]
    if (
        source["candidate"]["checkpoint_sha256"] != candidate["checkpoint_sha256"]
        or source["candidate"]["artifact"] != candidate["artifact"]
    ):
        raise ValueError("candidate_source_identity_mismatch")
    for filename, digest in candidate["artifact_sha256"].items():
        if sha(Path(candidate["artifact"]) / filename) != digest:
            raise ValueError(f"candidate_artifact_hash_mismatch:{filename}")
    validate_frozen_opponent_identity(
        Path(opponent["artifact"]), opponent, candidate["runtime_contract"]
    )
    all_scores: dict[str, np.ndarray] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    runtime: dict[str, dict[str, float | int]] = {}
    raw_hashes: dict[str, dict[str, str]] = {}
    public_game_rows: list[dict[str, Any]] = []
    all_suite_states: set[str] = set()
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
        if sha(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"suite_hash_mismatch:{seed_text}")
        identities = set(validate_arena_entries(openings))
        if identities != validate_suite_against_manifest(openings, manifest):
            raise ValueError(f"suite_exclusion_mismatch:{seed_text}")
        if identities & all_suite_states:
            raise ValueError("cross_suite_opening_overlap")
        all_suite_states |= identities
        report = json.loads(report_path.read_text())
        rows = [
            json.loads(line) for line in games_path.read_text().splitlines() if line
        ]
        public_game_rows.extend(
            {
                "seed": int(seed_text),
                "game_index": int(row["game_index"]),
                "opening_index": int(row["opening_index"]),
                "game_within_opening": int(row["game_within_opening"]),
                "challenger_player": int(row["challenger_player"]),
                "winner": str(row["winner"]),
                "margin": int(row["margin"]),
                "game_length": int(row["game_length"]),
                "opening_state_hash": str(row["opening_state_hash"]),
            }
            for row in rows
        )
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
        seat_outcomes: dict[str, dict[str, int | float]] = {}
        for seat in (0, 1):
            seat_rows = [row for row in rows if int(row["challenger_player"]) == seat]
            wins = sum(row["winner"] == "challenger" for row in seat_rows)
            draws = sum(row["winner"] == "draw" for row in seat_rows)
            losses = sum(row["winner"] == "current" for row in seat_rows)
            seat_outcomes[str(seat)] = {
                "games": len(seat_rows),
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score": (wins + 0.5 * draws) / len(seat_rows),
            }
        all_scores[seed_text] = scores
        outcomes[seed_text] = {
            "games": len(rows),
            "wins": int(report["wins"]),
            "draws": int(report["draws"]),
            "losses": int(report["losses"]),
            "score": float(scores.mean()),
            "seat_outcomes": seat_outcomes,
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
    rng = np.random.default_rng(395)
    left = rng.integers(0, 512, size=(10000, 512))
    right = rng.integers(0, 512, size=(10000, 512))
    pooled_samples = (
        all_scores["395"][left].mean(axis=1) + all_scores["396"][right].mean(axis=1)
    ) / 2
    pooled_score = float((all_scores["395"].mean() + all_scores["396"].mean()) / 2)
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
    protocol_valid = True
    matrix = {
        "schema": "order38615-corrected-diagnostic-score-matrix-v1",
        "registration_sha256": sha(REG),
        "evaluation_binding_sha256": sha(BIND),
        "raw_evidence_sha256": raw_hashes,
        "exclusion_manifest_sha256": sha(manifest_path),
        "bootstrap": {
            "per_suite_seeds": bootstrap["per_suite_seeds"],
            "pooled_seed": bootstrap["pooled_seed"],
            "resamples": bootstrap["per_suite_resamples"],
        },
        "opening_scores": {
            seed: [float(score) for score in scores]
            for seed, scores in all_scores.items()
        },
    }
    matrix_path = DATA / "opening-score-matrix.json"
    accounting_path = DATA / "game-outcome-accounting.jsonl"
    accounting_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in sorted(
                public_game_rows,
                key=lambda row: (row["seed"], row["game_index"]),
            )
        )
    )
    accounting_sha = sha(accounting_path)
    matrix["game_outcome_accounting_sha256"] = accounting_sha
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    result = {
        "schema": "order38615-corrected-diagnostic-results-v1",
        "registration_sha256": sha(REG),
        "evaluation_binding_sha256": sha(BIND),
        "opening_score_matrix_sha256": sha(matrix_path),
        "game_outcome_accounting_sha256": accounting_sha,
        "suite_results": per_suite,
        "pooled_score": pooled_score,
        "pooled_interval_95_percentile": pooled_interval,
        "pooled_bootstrap_seed": 395,
        "pooled_resamples": 10000,
        "per_suite_and_pooled_pass": bool(suite_pass and pooled_pass),
        "suite_pass": bool(suite_pass),
        "pooled_pass": bool(pooled_pass),
        "canonical_promotion_failure_overridden": False,
        "validity_status": "valid"
        if protocol_valid
        else "invalid_for_preregistered_exclusion_contract",
        "exclusion_manifest_sha256": sha(manifest_path),
        "excluded_state_count": manifest["excluded_state_count"],
        "outcomes": outcomes,
        "runtime": runtime,
        "total_games": sum(row["games"] for row in outcomes.values()),
        "total_search_simulations": 2048 * 2 * 384,
        "training_or_export_cost": "none",
    }
    (DATA / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
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
        "| Suite | Challenger seat | Games | Wins | Draws | Losses | Score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, outcome in outcomes.items():
        for seat, seat_result in outcome["seat_outcomes"].items():
            lines.append(
                f"| {seed} | {seat} | {seat_result['games']} | {seat_result['wins']} | {seat_result['draws']} | {seat_result['losses']} | {seat_result['score']:.4f} |"
            )
    lines += [
        "",
        f"Pooled equal-weight score: **{pooled_score:.4f}**; stratified 95% interval {pooled_interval[0]:.4f}–{pooled_interval[1]:.4f} ({'PASS' if pooled_pass else 'FAIL'}).",
        "",
        "This diagnostic estimates only the corrected opening distribution and does not override the canonical promotion failure.",
        "",
        "The score matrix contains 512 paired-opening cluster scores per suite; `analyze_corrected_order38615_diagnostic.py` verifies raw evidence hashes and reproduces 10,000-resample intervals.",
        "",
        f"Published per-game outcome accounting: `game-outcome-accounting.jsonl` (SHA-256 `{accounting_sha}`; {result['total_games']} rows). Registration, immutable exclusion manifest, suite bytes, raw run evidence, and runtime identities are hash-bound in the accompanying JSON records.",
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
