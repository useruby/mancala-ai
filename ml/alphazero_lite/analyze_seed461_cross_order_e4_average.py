"""Analyze the preregistered six-model cross-order E4 averaging arena."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed461_arena_validation import validate_arena_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-cross-order-e4-average-registration.json"
SUITE = DATA / "seed461-cross-order-e4-average-openings.jsonl"
CANDIDATES = DATA / "seed461-cross-order-e4-average-candidate-binding.json"
BINDING = DATA / "seed461-cross-order-e4-average-evaluation-binding.json"
MATRIX = DATA / "seed461-cross-order-e4-average-opening-score-matrix.json"
RESULT = DATA / "seed461-cross-order-e4-average-results.json"
REPORT = DATA / "seed461-cross-order-e4-average-results.md"
MODELS = ("P", "A1", "A2", "A3", "A4", "A5")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_immutable(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f"immutable_analysis_conflict:{path.name}")
    if not path.exists():
        path.write_text(encoded)


def analyze() -> None:
    reg, candidates, binding = (
        json.loads(path.read_text()) for path in (REG, CANDIDATES, BINDING)
    )
    if (
        len(binding.get("reports", {})) != len(MODELS)
        or binding.get("status") != "completed_fixed_3072_games"
    ):
        raise ValueError("evaluation_incomplete")
    if (
        binding["registration_sha256"] != sha(REG)
        or binding["candidate_binding_sha256"] != sha(CANDIDATES)
        or binding["suite_sha256"] != sha(SUITE)
    ):
        raise ValueError("evaluation_binding_chain_mismatch")
    if candidates["candidates"] != binding["candidates"]:
        raise ValueError("evaluation_candidate_identity_mismatch")
    openings = [json.loads(line) for line in SUITE.read_text().splitlines() if line]
    if len(openings) != 256:
        raise ValueError("suite_opening_count_mismatch")

    vectors: dict[str, np.ndarray] = {}
    evidence: dict[str, dict[str, str]] = {}
    for model in MODELS:
        row = binding["reports"][model]
        report_path, games_path = Path(row["report"]), Path(row["games"])
        if (
            sha(report_path) != row["report_sha256"]
            or sha(games_path) != row["games_sha256"]
        ):
            raise ValueError(f"evidence_hash_mismatch:{model}")
        report = json.loads(report_path.read_text())
        game_rows = [
            json.loads(line) for line in games_path.read_text().splitlines() if line
        ]
        vectors[model] = validate_arena_evidence(
            report,
            game_rows,
            openings,
            model,
            candidates["candidates"][model],
            binding["opponent"],
            reg["evaluation"],
        )
        evidence[model] = {
            "report_sha256": row["report_sha256"],
            "games_sha256": row["games_sha256"],
        }

    baseline_matrix = np.stack([vectors[f"A{i}"] for i in range(1, 6)])
    treatment = vectors["P"]
    effects = {
        f"P-A{i}": float(np.mean(treatment - vectors[f"A{i}"])) for i in range(1, 6)
    }
    primary_opening_effect = treatment - baseline_matrix.mean(axis=0)
    primary = float(primary_opening_effect.mean())
    scores = {model: float(vector.mean()) for model, vector in vectors.items()}
    boot = reg["analysis"]["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    sampled = rng.integers(0, len(openings), size=(boot["resamples"], len(openings)))
    primary_draws = primary_opening_effect[sampled].mean(axis=1)
    absolute_draws = treatment[sampled].mean(axis=1)
    primary_ci = np.percentile(primary_draws, [2.5, 97.5]).astype(float).tolist()
    absolute_ci = np.percentile(absolute_draws, [2.5, 97.5]).astype(float).tolist()
    worst = min(effects.values())
    thresholds = reg["analysis"]["decision_thresholds"]
    criteria = {
        "primary_mean_improvement_at_least_0.03": primary
        >= thresholds["primary_mean_improvement_at_least"],
        "primary_interval_lower_above_zero": primary_ci[0]
        > thresholds["primary_interval_lower_strictly_above"],
        "P_absolute_score_interval_lower_above_0.5": absolute_ci[0]
        > thresholds["P_absolute_score_interval_lower_strictly_above"],
        "P_score_at_least_four_of_five_baselines": sum(
            scores["P"] >= scores[f"A{i}"] for i in range(1, 6)
        )
        >= thresholds["P_score_at_least_baselines"],
        "worst_P_minus_Ai_at_least_minus_0.05": worst
        >= thresholds["worst_P_minus_Ai_at_least"],
    }
    advance = all(criteria.values())
    rows = [
        {"opening_index": i, **{model: float(vectors[model][i]) for model in MODELS}}
        for i in range(len(openings))
    ]
    matrix_data = {
        "schema": "seed461-cross-order-e4-average-opening-score-matrix-v1",
        "suite_sha256": sha(SUITE),
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": sha(CANDIDATES),
        "evaluation_binding_sha256": sha(BINDING),
        "evidence_sha256": evidence,
        "score_definition": "candidate points across two seat-paired games for an opening divided by two",
        "rows": rows,
    }
    save_immutable(MATRIX, matrix_data)
    result = {
        "schema": "seed461-cross-order-e4-average-results-v1",
        "status": "completed_fixed_3072_games",
        "scores": scores,
        "P_minus_Ai_effects": effects,
        "primary_effect": primary,
        "primary_95_percentile_interval": primary_ci,
        "P_absolute_score_95_percentile_interval": absolute_ci,
        "worst_comparison": {"name": min(effects, key=effects.get), "effect": worst},
        "baseline_scores": {f"A{i}": scores[f"A{i}"] for i in range(1, 6)},
        "P_score": scores["P"],
        "P_score_at_least_baselines_count": sum(
            scores["P"] >= scores[f"A{i}"] for i in range(1, 6)
        ),
        "criteria": criteria,
        "advance_to_independent_confirmation": advance,
        "decision": "advance_to_independent_confirmation"
        if advance
        else "reject_cross_order_E4_averaging",
        "failed_criteria": [key for key, passed in criteria.items() if not passed],
        "bootstrap": boot,
        "inference_scope": reg["analysis"]["inference_scope"],
        "training_runs_performed": 0,
        "future_training_runs_required": 5,
        "single_network_inference": True,
        "no_model_promotion": True,
        "preserves_rejections": reg["analysis"]["preserve_rejections"],
        "game_accounting": {
            "games": 3072,
            "models": 6,
            "games_per_model": 512,
            "outcome_dependent_extensions": 0,
        },
        "evidence_sha256": {
            "registration": sha(REG),
            "suite": sha(SUITE),
            "candidate_binding": sha(CANDIDATES),
            "evaluation_binding": sha(BINDING),
            "matrix": sha(MATRIX),
        },
    }
    save_immutable(RESULT, result)
    lines = [
        "# Cross-order constant-LR E4 checkpoint averaging",
        "",
        "Prospective evaluation: one model P, the uniform arithmetic mean of five order-independent A-arm E4 checkpoints. A1–A5 are the individual E4 baselines. Each model played 512 games (two seat-paired games per shared opening) against frozen seed455; total 3,072 games.",
        "",
        "| Model | Mean opening score | P−model |",
        "|---|---:|---:|",
        f"| P | {scores['P']:.4f} | — |",
    ]
    lines.extend(
        f"| A{i} | {scores[f'A{i}']:.4f} | {effects[f'P-A{i}']:+.4f} |"
        for i in range(1, 6)
    )
    lines += [
        "",
        f"Primary P−mean(A1–A5): **{primary:+.4f}** (95% shared-opening bootstrap interval {primary_ci[0]:+.4f} to {primary_ci[1]:+.4f}).",
        f"P absolute score: **{scores['P']:.4f}** (95% interval {absolute_ci[0]:.4f} to {absolute_ci[1]:.4f}). Worst comparison: **{min(effects, key=effects.get)} {worst:+.4f}**.",
        f"Decision: **{result['decision']}**. Failed conditions: {', '.join(result['failed_criteria']) or 'none'}.",
        "",
        "Inference is scoped to these five source trajectories and this dataset. P is one constructed model; its score vector is reused once in all comparisons and shared-opening bootstrap resamples. There is no order-stability claim. Construction used zero training runs; future construction requires five runs. Inference remains single-network.",
        "No model is promoted. PR #386, #388, and #389 rejections remain preserved.",
        "",
        f"The compact six-model opening-score matrix is bound to report/game hashes in `{MATRIX.relative_to(ROOT)}` (SHA-256 `{sha(MATRIX)}`).",
        "",
        "Independently reproduce the point estimates and intervals from the bound score matrix:",
        "```sh",
        ".venv/bin/python -m ml.alphazero_lite.reproduce_seed461_cross_order_e4_average docs/data/seed461-cross-order-e4-average-opening-score-matrix.json --seed 390",
        "```",
        "",
    ]
    report_text = "\n".join(lines)
    if REPORT.exists() and REPORT.read_text() != report_text:
        raise ValueError("immutable_report_conflict")
    if not REPORT.exists():
        REPORT.write_text(report_text)


if __name__ == "__main__":
    analyze()
