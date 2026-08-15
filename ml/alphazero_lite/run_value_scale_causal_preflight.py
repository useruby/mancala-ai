#!/usr/bin/env python3
# ruff: noqa: E402
"""Causal preflight for the PR #188 current-value affine rescaling.

This is diagnostic-only: it never trains, writes an artifact, or changes policy.
Every resumable payload is guarded by a complete, canonical provenance manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.evaluation_seed_contract import stable_hash, stable_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_frozen_trunk_value_sufficiency_audit import (
    EVALUATION_SOURCE,
    EXPECTED_CURRENT_SHA256,
    SOURCES,
    assign_splits,
    continuation,
    read_rows,
    select_corpus,
    state_rows,
    supplemental_standard_start_states,
)
from ml.alphazero_lite.self_play import Evaluator, build_eval_search_options

CACHE_SCHEMA = "azlite_value_scale_causal_preflight_v1"
DEFAULT_WORKDIR = Path("/tmp/azlite_value_scale_causal_preflight")
BUDGETS = (384, 768, 1200)
CONTINUATION_BUDGETS = (768, 1200)
TREATMENT_NAMES = ("margin_affine", "outcome_affine")
SEARCH_OPTIONS = build_eval_search_options()
EXPECTED_PR188_SOURCE_SHA256 = {
    "standard_start_selfplay": "485ea65c8ca1e9a3416bbccbf97f7e212a9f6082da759fd4c5cc8452c612385b",
    "opening_family_diagnostic": "ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4",
    "additional_selfplay": "63df5bf39601cc2c44bd47a92d310e02aef6232c04bdf4bed525576200a5751c",
}
_LABEL_WORKER_EVALUATOR: ArtifactEvaluator | None = None
_SEARCH_WORKER_CURRENT: ArtifactEvaluator | None = None
_SEARCH_WORKER_TREATMENTS: dict[str, AffineValueEvaluator] = {}


def _init_label_worker(artifact_dir: str) -> None:
    global _LABEL_WORKER_EVALUATOR
    _LABEL_WORKER_EVALUATOR = ArtifactEvaluator(Path(artifact_dir))


def _label_worker(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    row, seed = task
    if _LABEL_WORKER_EVALUATOR is None:
        raise RuntimeError("continuation worker evaluator was not initialized")
    item = dict(row)
    for budget in CONTINUATION_BUDGETS:
        item[f"D{budget}"] = continuation(
            item["state"],
            _LABEL_WORKER_EVALUATOR,
            budget=budget,
            seed=stable_seed(seed, item["state_hash"], "label", budget),
        )
    return item


def _init_search_worker(
    artifact_dir: str,
    margin_a: float,
    margin_b: float,
    outcome_a: float,
    outcome_b: float,
) -> None:
    global _SEARCH_WORKER_CURRENT, _SEARCH_WORKER_TREATMENTS
    _SEARCH_WORKER_CURRENT = ArtifactEvaluator(Path(artifact_dir))
    _SEARCH_WORKER_TREATMENTS = {
        "margin_affine": AffineValueEvaluator(
            _SEARCH_WORKER_CURRENT, margin_a, margin_b
        ),
        "outcome_affine": AffineValueEvaluator(
            _SEARCH_WORKER_CURRENT, outcome_a, outcome_b
        ),
    }


def _search_worker(
    task: tuple[dict[str, Any], int, int],
) -> tuple[str, dict[str, dict]]:
    row, budget, seed = task
    if _SEARCH_WORKER_CURRENT is None:
        raise RuntimeError("search worker evaluator was not initialized")
    results = {
        "baseline": search_position(
            _SEARCH_WORKER_CURRENT, row, budget=budget, seed=seed
        )
    }
    for name, evaluator in _SEARCH_WORKER_TREATMENTS.items():
        treatment = search_position(evaluator, row, budget=budget, seed=seed)
        if treatment["policy"] != results["baseline"]["policy"]:
            raise RuntimeError("policy output changed under value-only treatment")
        results[name] = treatment
    return row["state_hash"], results


def _causal_worker(
    task: tuple[dict[str, Any], int, int, int, int],
) -> tuple[int, dict[str, Any]]:
    row, baseline_move, treatment_move, continuation_budget, seed = task
    if _SEARCH_WORKER_CURRENT is None:
        raise RuntimeError("search worker evaluator was not initialized")
    left = forced_continuation(
        row["state"],
        _SEARCH_WORKER_CURRENT,
        baseline_move,
        budget=continuation_budget,
        seed=seed,
    )
    right = forced_continuation(
        row["state"],
        _SEARCH_WORKER_CURRENT,
        treatment_move,
        budget=continuation_budget,
        seed=seed,
    )
    return continuation_budget, {
        "state_hash": row["state_hash"],
        "normalized_final_margin_delta": right["normalized_final_margin"]
        - left["normalized_final_margin"],
        "binary_outcome_delta": right["outcome"] - left["outcome"],
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_manifest(**fields: Any) -> dict[str, Any]:
    """Build a cache identity; callers must include every causal input."""
    return {"schema": CACHE_SCHEMA, **fields}


def cache_matches(actual: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    return isinstance(actual, dict) and actual == expected


def load_cache(
    path: Path, expected_manifest: dict[str, Any]
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not cache_matches(payload.get("manifest"), expected_manifest):
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None


def save_cache(
    path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"manifest": manifest, "rows": rows}, sort_keys=True),
        encoding="utf-8",
    )


class AffineValueEvaluator(Evaluator):
    """Return bit-identical current policy while changing only leaf value scale."""

    def __init__(self, current: ArtifactEvaluator, a: float, b: float) -> None:
        self.current, self.a, self.b = current, float(a), float(b)

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        policy, value = self.current.evaluate(game)
        return policy, float(np.clip((self.a * value) + self.b, -1.0, 1.0))


def corpus_manifest(
    rows: list[dict[str, Any]],
    *,
    weights_hash: str,
    seed: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return canonical_manifest(
        kind="state_corpus",
        current_weights_sha256=weights_hash,
        base_seed=seed,
        state_corpus_hash=stable_hash([r["state_hash"] for r in rows]),
        source_provenance=provenance,
        split_counts={
            name: sum(r["split"] == name for r in rows)
            for name in ("train", "validation", "test")
        },
    )


def reconstruct_corpus(
    current: ArtifactEvaluator, *, weights_hash: str, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, provenance = [], {}
    for name, path in SOURCES.items():
        if not path.is_file():
            raise RuntimeError(f"missing PR #188 source: {path}")
        source_hash = sha256(path)
        if source_hash != EXPECTED_PR188_SOURCE_SHA256[name]:
            raise RuntimeError(f"PR #188 source hash mismatch: {name}")
        provenance[name] = {"path": str(path), "sha256": source_hash}
        candidates.extend(state_rows(read_rows(path), name, current))
    retained = {row["state_hash"] for row in candidates}
    candidates.extend(
        row
        for row in supplemental_standard_start_states(current, seed=seed, count=1024)
        if row["state_hash"] not in retained
    )
    provenance["generated_opening_family_diagnostic"] = {
        "families_requested": 1024,
        "generation": "seeded disjoint standard-start opening prefixes",
    }
    rows = select_corpus(candidates, seed=seed)
    assign_splits(rows, seed)
    manifest = corpus_manifest(
        rows, weights_hash=weights_hash, seed=seed, provenance=provenance
    )
    return rows, manifest


def label_manifest(
    *, corpus_hash: str, weights_hash: str, seed: int, budget: int
) -> dict[str, Any]:
    return canonical_manifest(
        kind="continuation_labels",
        current_weights_sha256=weights_hash,
        state_corpus_hash=corpus_hash,
        base_seed=seed,
        search_budget=budget,
        c_puct=1.25,
        search_options=SEARCH_OPTIONS,
    )


def labels_for(
    rows: list[dict[str, Any]],
    evaluator: Evaluator,
    *,
    weights_hash: str,
    seed: int,
    cache: Path,
    domain: str,
    artifact_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    corpus_hash = stable_hash([r["state_hash"] for r in rows])
    manifest = canonical_manifest(
        kind="continuation_labels",
        current_weights_sha256=weights_hash,
        state_corpus_hash=corpus_hash,
        base_seed=seed,
        budgets=list(CONTINUATION_BUDGETS),
        c_puct=1.25,
        search_options=SEARCH_OPTIONS,
        domain=domain,
    )
    cached = load_cache(cache, manifest)
    if cached is not None:
        return cached
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        labeled = [_label_worker_with_evaluator(row, evaluator, seed) for row in rows]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_label_worker,
            initargs=(str(artifact_dir),),
        ) as executor:
            labeled = list(
                executor.map(_label_worker, ((row, seed) for row in rows), chunksize=1)
            )
    save_cache(cache, manifest, labeled)
    return labeled


def _label_worker_with_evaluator(
    row: dict[str, Any], evaluator: Evaluator, seed: int
) -> dict[str, Any]:
    item = dict(row)
    for budget in CONTINUATION_BUDGETS:
        item[f"D{budget}"] = continuation(
            item["state"],
            evaluator,
            budget=budget,
            seed=stable_seed(seed, item["state_hash"], "label", budget),
        )
    return item


def fit_affine(rows: list[dict[str, Any]], target: str) -> tuple[float, float]:
    x = np.asarray([r["current_value"] for r in rows], dtype=float)
    y = np.asarray([r["D1200"][target] for r in rows], dtype=float)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(float)


def concordance(prediction: np.ndarray, target: np.ndarray) -> float:
    left, right = np.tril_indices(len(prediction), k=-1)
    delta = target[left] - target[right]
    mask = delta != 0
    return (
        0.5
        if not np.any(mask)
        else float(
            np.mean(
                np.sign(prediction[left][mask] - prediction[right][mask])
                == np.sign(delta[mask])
            )
        )
    )


def prediction_metrics(
    rows: list[dict[str, Any]], prediction: np.ndarray
) -> dict[str, Any]:
    margin = np.asarray([r["D1200"]["normalized_final_margin"] for r in rows])
    outcome = np.asarray([r["D1200"]["outcome"] for r in rows])
    return {
        "n": len(rows),
        "margin": {
            "mae": float(np.mean(abs(prediction - margin))),
            "rmse": float(np.sqrt(np.mean((prediction - margin) ** 2))),
            "pearson": float(np.corrcoef(prediction, margin)[0, 1])
            if np.std(prediction) * np.std(margin)
            else 0.0,
            "spearman": float(np.corrcoef(ranks(prediction), ranks(margin))[0, 1])
            if np.std(prediction) * np.std(margin)
            else 0.0,
            "pairwise_concordance": concordance(prediction, margin),
        },
        "outcome": {
            "mae": float(np.mean(abs(prediction - outcome))),
            "brier_style_error": float(np.mean((prediction - outcome) ** 2)),
            "sign_accuracy": float(np.mean(np.sign(prediction) == outcome)),
        },
    }


def js(left: list[float], right: list[float]) -> float:
    p, q = np.asarray(left, float), np.asarray(right, float)
    p, q = p / max(p.sum(), 1), q / max(q.sum(), 1)
    mean = (p + q) / 2
    return float(
        0.5 * np.sum(np.where(p > 0, p * np.log(p / np.maximum(mean, 1e-12)), 0))
        + 0.5 * np.sum(np.where(q > 0, q * np.log(q / np.maximum(mean, 1e-12)), 0))
    )


def _child(result: dict[str, Any], move: int) -> dict[str, Any]:
    return next(item for item in result["child_stats"] if int(item["move"]) == move)


def q_rank(result: dict[str, Any]) -> list[int]:
    """Return the deterministic Q ordering without child-stat payloads."""
    return [
        int(item["move"])
        for item in sorted(
            result["child_stats"],
            key=lambda item: (-float(item["q_value"]), int(item["move"])),
        )
    ]


def search_position(
    evaluator: Evaluator, row: dict[str, Any], *, budget: int, seed: int
) -> dict[str, Any]:
    return evaluate_artifact_position(
        evaluator=evaluator,
        state=row["state"],
        simulations=budget,
        seed=stable_seed(seed, row["state_hash"], "puct", budget),
        c_puct=1.25,
        search_options=SEARCH_OPTIONS,
    )


def q_u_ratio(result: dict[str, Any]) -> float:
    entries = result.get("selection_breakdown", {}).get("moves", [])
    q = sum(abs(float(x["q_component"])) for x in entries)
    u = sum(abs(float(x["u_component"])) for x in entries)
    return q / u if u else 0.0


def visit_margin(result: dict[str, Any]) -> float:
    visits = sorted((float(value) for value in result["visits"]), reverse=True)
    return visits[0] - visits[1] if len(visits) > 1 else (visits[0] if visits else 0.0)


def forced_continuation(
    state: dict[str, Any], evaluator: Evaluator, move: int, *, budget: int, seed: int
) -> dict[str, Any]:
    game, root = KalahGame.from_state(state), KalahGame.from_state(state).current_player
    if not game.move(game.pit_index(move)):
        raise RuntimeError("forced move is illegal")
    result = continuation(game.to_state(), evaluator, budget=budget, seed=seed)
    if game.current_player != root:
        for key in ("outcome", "final_store_margin", "normalized_final_margin"):
            result[key] *= -1
    return result


def bootstrap_clustered(
    rows: list[dict[str, Any]], key: str, *, seed: int
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "unique_states": 0,
            "mean": 0.0,
            "median": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "affine_better_fraction": 0.0,
            "current_better_fraction": 0.0,
            "tie_fraction": 0.0,
        }
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["state_hash"]].append(float(row[key]))
    values = np.asarray([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(1000, len(values)))].mean(axis=1)
    return {
        "n": len(rows),
        "unique_states": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "lower": float(np.quantile(sample, 0.025)),
        "upper": float(np.quantile(sample, 0.975)),
        "affine_better_fraction": float(np.mean(values > 0)),
        "current_better_fraction": float(np.mean(values < 0)),
        "tie_fraction": float(np.mean(values == 0)),
    }


def intervention(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    domain: str,
    artifact_dir: Path,
    margin_a: float,
    margin_b: float,
    outcome_a: float,
    outcome_b: float,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    result: dict[str, Any] = {}
    initargs = (str(artifact_dir), margin_a, margin_b, outcome_a, outcome_b)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init_search_worker, initargs=initargs
    ) as executor:
        for budget in BUDGETS:
            searches = dict(
                executor.map(
                    _search_worker,
                    ((row, budget, seed) for row in rows),
                    chunksize=1,
                )
            )
            diagnostics = {name: [] for name in TREATMENT_NAMES}
            for row in rows:
                pair = searches[row["state_hash"]]
                baseline = pair["baseline"]
                for name in diagnostics:
                    treatment = pair[name]
                    base_move, treated_move = (
                        int(baseline["selected_move"]),
                        int(treatment["selected_move"]),
                    )
                    diagnostics[name].append(
                        {
                            "changed": base_move != treated_move,
                            "visit_js": js(baseline["visits"], treatment["visits"]),
                            "root_value_change": float(
                                treatment.get("search_root_value", 0)
                                - baseline.get("search_root_value", 0)
                            ),
                            "selected_child_q_change": float(
                                _child(treatment, treated_move)["q_value"]
                                - _child(baseline, base_move)["q_value"]
                            ),
                            "visit_margin_change": visit_margin(treatment)
                            - visit_margin(baseline),
                            "q_u_ratio_change": q_u_ratio(treatment)
                            - q_u_ratio(baseline),
                            "q_ranking_changed": q_rank(baseline) != q_rank(treatment),
                        }
                    )
            entry: dict[str, Any] = {
                "domain": domain,
                "states": len(rows),
                "treatments": {
                    name: {
                        key: float(np.mean([x[key] for x in values]))
                        for key in (
                            "visit_js",
                            "root_value_change",
                            "selected_child_q_change",
                            "visit_margin_change",
                            "q_u_ratio_change",
                        )
                    }
                    | {
                        "selected_move_disagreement": float(
                            np.mean([x["changed"] for x in values])
                        ),
                        "q_ranking_changed": float(
                            np.mean([x["q_ranking_changed"] for x in values])
                        ),
                    }
                    for name, values in diagnostics.items()
                },
            }
            for name in diagnostics:
                tasks = []
                for row in rows:
                    pair = searches[row["state_hash"]]
                    if pair["baseline"]["selected_move"] == pair[name]["selected_move"]:
                        continue
                    for cont in CONTINUATION_BUDGETS:
                        tasks.append(
                            (
                                row,
                                int(pair["baseline"]["selected_move"]),
                                int(pair[name]["selected_move"]),
                                cont,
                                stable_seed(
                                    seed,
                                    row["state_hash"],
                                    "forced-continuation",
                                    budget,
                                    cont,
                                ),
                            )
                        )
                causal = {str(cont): [] for cont in CONTINUATION_BUDGETS}
                for cont, item in executor.map(_causal_worker, tasks, chunksize=1):
                    causal[str(cont)].append(item)
                entry["treatments"][name]["causal"] = {
                    cont: {
                        "normalized_final_margin_delta": bootstrap_clustered(
                            items,
                            "normalized_final_margin_delta",
                            seed=stable_seed(seed, domain, budget, name, cont),
                        ),
                        "binary_outcome_delta": bootstrap_clustered(
                            items,
                            "binary_outcome_delta",
                            seed=stable_seed(seed, domain, budget, name, cont),
                        ),
                    }
                    for cont, items in causal.items()
                }
            result[f"D{budget}"] = entry
    return result


def classify(
    evaluation: dict[str, Any], intervention_report: dict[str, Any] | None
) -> str:
    baseline_mae = evaluation["current_value"]["margin"]["mae"]
    margin = evaluation["margin_affine"]["margin"]["mae"] <= baseline_mae * 0.9
    if not margin:
        return "evaluation_domain_generalization_failed"
    if intervention_report is None:
        return "evaluation_domain_generalization_failed"
    margin_treatment = [
        v["treatments"]["margin_affine"]
        for domain in intervention_report.values()
        for v in domain.values()
    ]
    outcome_treatment = [
        v["treatments"]["outcome_affine"]
        for domain in intervention_report.values()
        for v in domain.values()
    ]
    changed = max(
        (
            x["causal"]["768"]["normalized_final_margin_delta"]["unique_states"]
            for x in margin_treatment
        ),
        default=0,
    )
    effects = [
        x["causal"][str(cont)]["normalized_final_margin_delta"]
        for x in margin_treatment
        for cont in CONTINUATION_BUDGETS
    ]
    outcome_effects = [
        x["causal"][str(cont)]["normalized_final_margin_delta"]
        for x in outcome_treatment
        for cont in CONTINUATION_BUDGETS
    ]
    if any(x["upper"] < 0 for x in effects):
        return "margin_value_semantics_rejected_for_search"
    if any(
        left["mean"] * right["mean"] < 0
        for left, right in zip(effects, outcome_effects, strict=True)
    ):
        return "value_target_semantics_unresolved"
    if changed >= 64 and all(x["mean"] > 0 and x["lower"] >= 0 for x in effects):
        return "search_value_scale_mismatch_confirmed"
    return "margin_calibration_not_search_relevant"


def results_markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation_domain"]
    current = evaluation["current_value"]["margin"]["mae"]
    affine = evaluation["margin_affine"]["margin"]["mae"]
    improvement = 100 * (current - affine) / current
    return (
        "# Value-Scale Causal Preflight\n\n"
        f"**Classification:** `{report['classification']}`\n\n"
        "The full aggregate evidence and provenance manifest are in "
        "`docs/data/alphazero-lite-value-scale-causal-preflight-summary.json`.\n\n"
        "| Independent evaluation domain (n=240) | Current | Margin affine |\n"
        "| --- | ---: | ---: |\n"
        f"| Normalized-margin MAE | {current:.6f} | {affine:.6f} |\n\n"
        f"Margin-affine MAE improved by {improvement:.1f}%. The outcome-affine "
        "semantic control produced materially conflicting causal effects, so no "
        "value-head training is authorized.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    weights = Path(args.current) / "weights.json"
    weights_hash = sha256(weights)
    if weights_hash != EXPECTED_CURRENT_SHA256:
        raise RuntimeError("current weights hash mismatch")
    workdir, current = Path(args.workdir), ArtifactEvaluator(Path(args.current))
    workdir.mkdir(parents=True, exist_ok=True)
    rows, manifest = reconstruct_corpus(
        current, weights_hash=weights_hash, seed=args.seed
    )
    if manifest["split_counts"] != {"train": 2411, "validation": 804, "test": 881}:
        raise RuntimeError("reconstructed corpus does not match PR #188 split counts")
    save_cache(workdir / "corpus.json", manifest, rows)
    rows = labels_for(
        rows,
        current,
        weights_hash=weights_hash,
        seed=args.seed,
        cache=workdir / "corpus_labels.json",
        domain="pr188",
        artifact_dir=Path(args.current),
        workers=args.workers,
    )
    train = [r for r in rows if r["split"] == "train"]
    margin_a, margin_b = fit_affine(train, "normalized_final_margin")
    outcome_a, outcome_b = fit_affine(train, "outcome")
    if not EVALUATION_SOURCE.is_file():
        raise RuntimeError(f"missing PR #188 evaluation source: {EVALUATION_SOURCE}")
    evaluation_rows = state_rows(
        read_rows(EVALUATION_SOURCE), "evaluation_domain", current
    )
    used = {r["state_hash"] for r in rows}
    evaluation_rows = [r for r in evaluation_rows if r["state_hash"] not in used][:240]
    if len(evaluation_rows) != 240:
        raise RuntimeError("expected exactly 240 independent evaluation-domain states")
    evaluation_rows = labels_for(
        evaluation_rows,
        current,
        weights_hash=weights_hash,
        seed=args.seed,
        cache=workdir / "evaluation_labels.json",
        domain="evaluation",
        artifact_dir=Path(args.current),
        workers=args.workers,
    )
    predictions = {
        "current_value": np.asarray([r["current_value"] for r in evaluation_rows]),
        "margin_affine": np.clip(
            margin_a * np.asarray([r["current_value"] for r in evaluation_rows])
            + margin_b,
            -1,
            1,
        ),
        "outcome_affine": np.clip(
            outcome_a * np.asarray([r["current_value"] for r in evaluation_rows])
            + outcome_b,
            -1,
            1,
        ),
    }
    evaluation = {
        name: prediction_metrics(evaluation_rows, value)
        for name, value in predictions.items()
    }
    report: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "current_weights_sha256": weights_hash,
        "seed": args.seed,
        "corpus_manifest": manifest,
        "train_row_manifest_hash": stable_hash([r["state_hash"] for r in train]),
        "affine_a": margin_a,
        "affine_b": margin_b,
        "outcome_affine_a": outcome_a,
        "outcome_affine_b": outcome_b,
        "D1200_label_configuration": label_manifest(
            corpus_hash=manifest["state_corpus_hash"],
            weights_hash=weights_hash,
            seed=args.seed,
            budget=1200,
        ),
        "continuation_configuration": {
            "budgets": list(CONTINUATION_BUDGETS),
            "c_puct": 1.25,
            "search_options": SEARCH_OPTIONS,
        },
        "evaluation_domain": evaluation,
    }
    if (
        evaluation["margin_affine"]["margin"]["mae"]
        < evaluation["current_value"]["margin"]["mae"]
    ):
        report["intervention"] = {
            "pr188_test": intervention(
                [r for r in rows if r["split"] == "test"],
                seed=args.seed,
                domain="pr188_test",
                artifact_dir=Path(args.current),
                margin_a=margin_a,
                margin_b=margin_b,
                outcome_a=outcome_a,
                outcome_b=outcome_b,
                workers=args.workers,
            ),
            "evaluation_domain": intervention(
                evaluation_rows,
                seed=args.seed,
                domain="evaluation_domain",
                artifact_dir=Path(args.current),
                margin_a=margin_a,
                margin_b=margin_b,
                outcome_a=outcome_a,
                outcome_b=outcome_b,
                workers=args.workers,
            ),
        }
    report["classification"] = classify(evaluation, report.get("intervention"))
    payload = json.dumps(report, indent=2, sort_keys=True)
    (workdir / "summary.json").write_text(payload, encoding="utf-8")
    (
        REPO_ROOT / "docs/data/alphazero-lite-value-scale-causal-preflight-summary.json"
    ).write_text(payload, encoding="utf-8")
    (
        REPO_ROOT / "docs/alphazero-lite-value-scale-causal-preflight-results.md"
    ).write_text(results_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
