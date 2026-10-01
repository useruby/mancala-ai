"""Analyze only the two prospectively registered order 38615 A5 suites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed461_arena_validation import validate_arena_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "order38615-a5-confirmation-registration.json"
CANDIDATE = DATA / "order38615-a5-confirmation-candidate-binding.json"
BINDING = DATA / "order38615-a5-confirmation-evaluation-binding.json"
MATRIX = DATA / "order38615-a5-confirmation-opening-score-matrix.json"
RESULTS = DATA / "order38615-a5-confirmation-results.json"
RESULTS_MD = DATA / "order38615-a5-confirmation-results.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(row: dict[str, Any]) -> float:
    return {"challenger": 1.0, "draw": 0.5, "current": 0.0}[row["winner"]]


def analyze() -> dict[str, Any]:
    reg, candidate, binding = (
        json.loads(p.read_text()) for p in (REG, CANDIDATE, BINDING)
    )
    if binding.get("status") != "completed_2048_games":
        raise ValueError("evaluation_not_complete")
    ev = reg["evaluation"]
    scores: dict[int, np.ndarray] = {}
    outcome_counts = {}
    matrix_suites = {}
    suite_results = {}
    for seed_text, suite in ev["suites"].items():
        seed = int(seed_text)
        evidence = binding["reports"][seed_text]
        report_path, games_path = Path(evidence["report"]), Path(evidence["games"])
        if (
            sha(report_path) != evidence["report_sha256"]
            or sha(games_path) != evidence["games_sha256"]
        ):
            raise ValueError(f"evidence_hash_mismatch:{seed}")
        openings = [
            json.loads(row)
            for row in Path(suite["path"]).read_text().splitlines()
            if row
        ]
        report = json.loads(report_path.read_text())
        rows = [json.loads(row) for row in games_path.read_text().splitlines() if row]
        evaluation = {
            **ev,
            "games_per_candidate": ev["games_per_suite"],
            "suite": suite,
            "arena_seed": seed,
            "seed_contract": ev["seed_contract"],
        }
        values = validate_arena_evidence(
            report,
            rows,
            openings,
            seed_text,
            candidate["candidate"],
            ev["opponent_binding"],
            evaluation,
        )
        if len(values) != 512:
            raise ValueError(f"opening_count_mismatch:{seed}")
        scores[seed] = values
        wins = sum(row["winner"] == "challenger" for row in rows)
        draws = sum(row["winner"] == "draw" for row in rows)
        losses = sum(row["winner"] == "current" for row in rows)
        outcome_counts[seed_text] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "games": len(rows),
        }
        rng = np.random.default_rng(seed)
        draws_boot = values[rng.integers(0, 512, size=(10000, 512))].mean(axis=1)
        lower, upper = np.quantile(draws_boot, [0.025, 0.975])
        suite_results[seed_text] = {
            "score": float(values.mean()),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "interval_95_percentile": [float(lower), float(upper)],
            "bootstrap_seed": seed,
            "resamples": 10000,
        }
        matrix_suites[seed_text] = {
            "opening_scores": [float(x) for x in values],
            "suite_sha256": suite["sha256"],
            "report_sha256": evidence["report_sha256"],
            "games_sha256": evidence["games_sha256"],
        }

    rng = np.random.default_rng(391)
    indexes_391 = rng.integers(0, 512, size=(10000, 512))
    indexes_392 = rng.integers(0, 512, size=(10000, 512))
    pooled_draws = (
        scores[391][indexes_391].mean(axis=1) + scores[392][indexes_392].mean(axis=1)
    ) / 2
    pooled = float((scores[391].mean() + scores[392].mean()) / 2)
    pooled_interval = [float(x) for x in np.quantile(pooled_draws, [0.025, 0.975])]
    thresholds = reg["analysis"]["decision_rule"]
    passed = (
        all(
            row["score"] >= thresholds["each_suite_score_at_least"]
            for row in suite_results.values()
        )
        and all(
            row["interval_95_percentile"][0]
            > thresholds["each_suite_interval_lower_strictly_above"]
            for row in suite_results.values()
        )
        and pooled >= thresholds["pooled_score_at_least"]
        and pooled_interval[0] > thresholds["pooled_interval_lower_strictly_above"]
    )
    matrix = {
        "schema": "order38615-a5-confirmation-opening-score-matrix-v1",
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": sha(CANDIDATE),
        "evaluation_binding_sha256": sha(BINDING),
        "suites": matrix_suites,
    }
    matrix_text = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    if MATRIX.exists() and MATRIX.read_text() != matrix_text:
        raise ValueError("immutable_score_matrix_conflict")
    if not MATRIX.exists():
        MATRIX.write_text(matrix_text)
    results = {
        "schema": "order38615-a5-confirmation-results-v1",
        "suite_results": suite_results,
        "outcome_counts": outcome_counts,
        "pooled_score": pooled,
        "pooled_interval_95_percentile": pooled_interval,
        "pooled_bootstrap_seed": 391,
        "pooled_resamples": 10000,
        "decision": "advance_to_promotion_review"
        if passed
        else "reject_candidate_for_advancement",
        "all_conditions_passed": passed,
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": sha(CANDIDATE),
        "evaluation_binding_sha256": sha(BINDING),
        "opening_score_matrix_sha256": sha(MATRIX),
        "scope": reg["analysis"]["scope"],
        "no_additional_training_cost": True,
    }
    RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Order 38615 A E4 fresh confirmation",
        "",
        "| Suite | Score | Wins | Draws | Losses | 95% opening-cluster interval |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in ("391", "392"):
        row = suite_results[seed]
        lines.append(
            f"| {seed} | {row['score']:.4f} | {row['wins']} | {row['draws']} | {row['losses']} | {row['interval_95_percentile'][0]:.4f}–{row['interval_95_percentile'][1]:.4f} |"
        )
    lines += [
        "",
        f"Equally weighted pooled score: **{pooled:.4f}** (stratified opening bootstrap 95% interval {pooled_interval[0]:.4f}–{pooled_interval[1]:.4f}).",
        "",
        f"Decision: **{results['decision']}**.",
        "",
        reg["analysis"]["scope"],
        "",
        f"Raw evidence is hash-bound by `order38615-a5-confirmation-opening-score-matrix.json` (SHA-256 `{sha(MATRIX)}`).",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines))
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    analyze()
