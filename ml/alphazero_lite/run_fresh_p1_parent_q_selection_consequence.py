#!/usr/bin/env python3
# ruff: noqa: E402
"""Frozen PR #234 parent-Q selection-consequence evaluation.

This observer evaluates immutable P1/A1/A4/A16 artifacts only.  It neither
trains, generates self-play, nor changes PUCT or its selection rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.root_q_trust_region import (
    parent_q_counterfactual,
    root_q_diagnostics,
    select_guard_indexes,
)
from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.run_fresh_p1_adapter_margin_sensitivity import HELD_OUT_HASHES
from ml.alphazero_lite.self_play import PUCT

# PR #234's guard manifest is defined with this seed; do not substitute a
# new evaluation seed when reconstructing the primary and secondary sets.
SEED = 235
SIMULATIONS = 1200
GUARD_COUNT = 32
BOOTSTRAP_SAMPLES = 10_000
WORKDIR = Path("/tmp/azlite_parent_q_selection_consequence")
OUT_SUMMARY = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-parent-q-selection-consequence-summary.json"
)
OUT_REPORT = (
    REPO_ROOT / "docs/alphazero-lite-fresh-p1-parent-q-selection-consequence-results.md"
)
PR220_SUMMARY = (
    REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
)


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(
            f"parent-q-selection-consequence:{state_hash}".encode()
        ).hexdigest()[:16],
        16,
    )


def _root_snapshot(simulation: int, root: Any) -> dict[str, Any]:
    """Capture only evidence available immediately before a PUCT simulation."""
    return {
        "simulation": int(simulation),
        "root_visit_count": int(root.visit_count),
        "children": {
            int(move): {
                "visits": int(child.visit_count),
                "q_value": float(child.q_value),
            }
            for move, child in root.children.items()
        },
    }


def _search(
    state: dict[str, Any], evaluator: ArtifactEvaluator, state_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run unchanged ordinary PUCT with root selection tracing and P1 snapshots."""
    trace: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=SIMULATIONS,
        c_puct=1.25,
        rng=random.Random(_seed(state_hash)),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
        pre_simulation_hook=lambda simulation, root: snapshots.append(
            _root_snapshot(simulation, root)
        ),
        selection_trace=trace,
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return search.root_summary(), trace, snapshots


def _policy_l1(
    state: dict[str, Any], left: ArtifactEvaluator, right: ArtifactEvaluator
) -> float:
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    left_policy, _ = left.evaluate(game)
    right_policy, _ = right.evaluate(game)
    return float(np.abs(left_policy[legal] - right_policy[legal]).sum())


def _trace_rows(
    candidate_trace: list[dict[str, Any]],
    parent_trace: list[dict[str, Any]],
    parent_snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every traced candidate root selection using pre-simulation P1 Q."""
    if not (
        len(candidate_trace)
        == len(parent_trace)
        == len(parent_snapshots)
        == SIMULATIONS
    ):
        raise RuntimeError("incomplete ordinary-PUCT trace or P1 snapshot sequence")
    rows = []
    for candidate, parent, snapshot in zip(
        candidate_trace, parent_trace, parent_snapshots, strict=True
    ):
        candidate_root = candidate["selection_path"][0]
        parent_root = parent["selection_path"][0]
        simulation = int(candidate["simulation_index"])
        if simulation != int(parent["simulation_index"]) or simulation != int(
            snapshot["simulation"]
        ):
            raise RuntimeError("candidate/P1 traces are not simulation aligned")
        if candidate_root["state_hash"] != parent_root["state_hash"]:
            raise RuntimeError("candidate/P1 root state mismatch")
        counterfactual = parent_q_counterfactual(
            candidate_root["children"],
            snapshot["children"],
            candidate_root["chosen_move"],
        )
        rows.append(
            {
                "simulation": simulation,
                "actual_move": int(candidate_root["chosen_move"]),
                "p1_move": int(parent_root["chosen_move"]),
                "p1_root_visits": int(snapshot["root_visit_count"]),
                **{
                    key: value for key, value in counterfactual.items() if key != "rows"
                },
            }
        )
    first_actual_flip = next(
        (row for row in rows if row["actual_move"] != row["p1_move"]), None
    )
    # A causal Q-sync explanation must use P1 evidence available at the first
    # observed selection flip, and must select the contemporaneous P1 action.
    qsync_invariant = first_actual_flip is None or (
        first_actual_flip["actual_has_p1_q"]
        and first_actual_flip["cf_winner_has_p1_q"]
        and first_actual_flip["cf_move"] == first_actual_flip["p1_move"]
    )
    return rows, {
        "first_actual_selection_flip": first_actual_flip,
        "first_flip_qsync_invariant": bool(qsync_invariant),
    }


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    order = np.argsort(np.asarray(scores), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    del values
    for group, count in enumerate(counts):
        if count > 1:
            indexes = np.where(inverse == group)[0]
            ranks[indexes] = ranks[indexes].mean()
    return float(
        (ranks[np.asarray(labels)].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    hits = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positives


def _bootstrap(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Root-resample all requested consequence rates and mean regret."""
    if not records:
        return {"unit": "root", "samples": BOOTSTRAP_SAMPLES, "metrics": {}}
    rng = np.random.default_rng(SEED)
    values = {
        "any_counterfactual_flip_rate": np.asarray(
            [any(row["selection_flip"] for row in item["trace"]) for item in records],
            float,
        ),
        "actual_selection_flip_rate": np.asarray(
            [
                any(row["actual_move"] != row["p1_move"] for row in item["trace"])
                for item in records
            ],
            float,
        ),
        "mean_selection_regret": np.asarray(
            [
                np.mean([row["selection_regret"] for row in item["trace"]])
                for item in records
            ],
            float,
        ),
    }
    indexes = rng.integers(0, len(records), size=(BOOTSTRAP_SAMPLES, len(records)))
    return {
        "unit": "root",
        "samples": BOOTSTRAP_SAMPLES,
        "metrics": {
            name: {
                "estimate": float(value.mean()),
                "lower_95": float(np.quantile(value[indexes].mean(axis=1), 0.025)),
                "upper_95": float(np.quantile(value[indexes].mean(axis=1), 0.975)),
            }
            for name, value in values.items()
        },
    }


def _distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for record in records for row in record["trace"]]
    return {
        "roots": len(records),
        "selections": len(rows),
        "counterfactual_flip_rate": float(
            np.mean([row["selection_flip"] for row in rows])
        ),
        "actual_p1_flip_rate": float(
            np.mean([row["actual_move"] != row["p1_move"] for row in rows])
        ),
        "mean_selection_regret": float(
            np.mean([row["selection_regret"] for row in rows])
        ),
        "synchronized_action_count": dict(
            Counter(row["synchronized_actions"] for row in rows)
        ),
        "first_flip_qsync_invariant": all(
            record["first_flip_qsync_invariant"] for record in records
        ),
        "bootstrap": _bootstrap(records),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [any(row["selection_flip"] for row in item["trace"]) for item in records]
    policy_l1 = [float(item["policy_l1"]) for item in records]
    max_regret = [
        max(row["selection_regret"] for row in item["trace"]) for item in records
    ]
    return {
        "per_root_primary": [
            {
                "state_hash": item["state_hash"],
                "parent_q_regret_auc": _metric(item),
                "parent_q_flip_fraction": _flip_fraction(item),
                "max_parent_q_regret": max(
                    row["selection_regret"] for row in item["trace"]
                ),
                "first_parent_q_flip_sim": next(
                    (
                        row["simulation"]
                        for row in item["trace"]
                        if row["selection_flip"]
                    ),
                    None,
                ),
                "regret_auc_1_384": _metric(item, 1, 384),
                "regret_auc_385_1200": _metric(item, 385, 1200),
                "flip_fraction_1_384": _flip_fraction(item, 1, 384),
                "flip_fraction_385_1200": _flip_fraction(item, 385, 1200),
            }
            for item in records
        ],
        "distribution": _distribution(records),
        "policy_l1_vs_p1": {
            "mean": float(np.mean(policy_l1)),
            "p50": float(np.median(policy_l1)),
            "p90": float(np.percentile(policy_l1, 90)),
        },
        "q_diagnostics_vs_p1": {
            key: float(np.mean([item["q_diagnostics"][key] for item in records]))
            for key in records[0]["q_diagnostics"]
        },
        "predicting_any_counterfactual_flip": {
            "outcome_positive_roots": int(sum(labels)),
            "policy_l1": {
                "auroc": _auc(labels, policy_l1),
                "auprc": _average_precision(labels, policy_l1),
            },
            "max_selection_regret": {
                "auroc": _auc(labels, max_regret),
                "auprc": _average_precision(labels, max_regret),
            },
        },
    }


def _record(
    item: dict[str, Any],
    name: str,
    candidate: ArtifactEvaluator,
    p1: ArtifactEvaluator,
    p1_observation: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
) -> dict[str, Any]:
    p1_summary, p1_trace, p1_snapshots = p1_observation
    candidate_summary, candidate_trace, _unused = _search(
        item["state"], candidate, item["state_hash"]
    )
    trace, first_flip = _trace_rows(candidate_trace, p1_trace, p1_snapshots)
    return {
        "state_hash": item["state_hash"],
        "population": item["population"],
        "artifact": name,
        "policy_l1": _policy_l1(item["state"], candidate, p1),
        "q_diagnostics": root_q_diagnostics(candidate_summary, p1_summary),
        "trace": trace,
        **first_flip,
    }


def _metric(record: dict[str, Any], start: int = 1, end: int = SIMULATIONS) -> float:
    values = [
        row["selection_regret"]
        for row in record["trace"]
        if start <= row["simulation"] <= end
    ]
    return float(np.mean(values))


def _flip_fraction(
    record: dict[str, Any], start: int = 1, end: int = SIMULATIONS
) -> float:
    values = [
        row["selection_flip"]
        for row in record["trace"]
        if start <= row["simulation"] <= end
    ]
    return float(np.mean(values))


def _mean_difference_bootstrap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    a, b = (
        np.asarray([_metric(row) for row in left]),
        np.asarray([_metric(row) for row in right]),
    )
    samples = a[rng.integers(0, len(a), (BOOTSTRAP_SAMPLES, len(a)))].mean(1) - b[
        rng.integers(0, len(b), (BOOTSTRAP_SAMPLES, len(b)))
    ].mean(1)
    return {
        "estimate": float(a.mean() - b.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "samples": BOOTSTRAP_SAMPLES,
    }


def _class_predictors(
    amplified: list[dict[str, Any]], washed: list[dict[str, Any]]
) -> dict[str, Any]:
    records = amplified + washed
    labels = [True] * len(amplified) + [False] * len(washed)
    predictors = {
        "parent_q_regret_auc": [_metric(row) for row in records],
        "q_direction_error": [
            row["q_diagnostics"]["q_direction_error"] for row in records
        ],
        "per_action_q_rank_disagreement": [
            row["q_diagnostics"]["per_action_q_rank_disagreement"] for row in records
        ],
        "policy_l1": [row["policy_l1"] for row in records],
    }
    return {
        name: {
            "auroc": _auc(labels, values),
            "auprc": _average_precision(labels, values),
            "amplified_mean": float(np.mean(values[: len(amplified)])),
            "amplified_median": float(np.median(values[: len(amplified)])),
            "washed_mean": float(np.mean(values[len(amplified) :])),
            "washed_median": float(np.median(values[len(amplified) :])),
            "amplified_minus_washed_bootstrap": _mean_difference_bootstrap(
                amplified, washed
            )
            if name == "parent_q_regret_auc"
            else None,
        }
        for name, values in predictors.items()
    }


def _first_flip_invariant(
    item: dict[str, Any],
    candidate: ArtifactEvaluator,
    ordinary: list[dict],
    snapshots: list[dict],
) -> dict[str, Any]:
    """Compare the predicted first root intervention to an actual root_qsync run."""
    root_hash = ordinary[0]["selection_path"][0]["state_hash"]

    def override(
        simulation: int, state_hash: str, move: int, raw_q: float, visits: int
    ) -> float | None:
        reference = snapshots[simulation - 1]["children"].get(move)
        if (
            state_hash == root_hash
            and visits > 0
            and reference is not None
            and reference["visits"] > 0
        ):
            return float(reference["q_value"])
        return None

    trace: list[dict] = []
    search = PUCT(
        candidate,
        SIMULATIONS,
        1.25,
        random.Random(_seed(item["state_hash"])),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
        selection_q_override=override,
        selection_trace=trace,
    )
    search.run(
        KalahGame.from_state(item["state"]), dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    predicted = _trace_rows(ordinary, ordinary, snapshots)[0]
    predicted_first = next((row for row in predicted if row["selection_flip"]), None)
    actual = next(
        (
            {
                "simulation": int(left["simulation_index"]),
                "ordinary_action": int(left["selection_path"][0]["chosen_move"]),
                "root_qsync_action": int(right["selection_path"][0]["chosen_move"]),
            }
            for left, right in zip(ordinary, trace, strict=True)
            if left["selection_path"][0]["chosen_move"]
            != right["selection_path"][0]["chosen_move"]
        ),
        None,
    )
    passed = (
        actual is None
        and predicted_first is None
        or (
            actual is not None
            and predicted_first is not None
            and actual["simulation"] == predicted_first["simulation"]
            and actual["ordinary_action"] == predicted_first["actual_move"]
            and actual["root_qsync_action"] == predicted_first["cf_move"]
        )
    )
    return {
        "state_hash": item["state_hash"],
        "predicted": predicted_first,
        "actual": actual,
        "passed": bool(passed),
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Parent-Q Selection Consequence",
        "",
        "Frozen post-PR #234 observation. No training, self-play, or PUCT modification was performed.",
        "",
        "## Guard Distributions",
        "",
        "| Artifact | Guard | Roots | CF flip rate | Actual P1 flip rate | Mean regret | First-flip Q-sync |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for artifact, groups in summary["results"].items():
        for group in ("primary_guard", "secondary_guard"):
            value = groups[group]["distribution"]
            lines.append(
                f"| {artifact} | {group} | {value['roots']} | {value['counterfactual_flip_rate']:.6f} | {value['actual_p1_flip_rate']:.6f} | {value['mean_selection_regret']:.6f} | {value['first_flip_qsync_invariant']} |"
            )
    lines += [
        "",
        "## Full Results",
        "",
        "```json",
        json.dumps(
            {key: summary[key] for key in ("invariants", "artifacts", "results")},
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    replay = pr233.A16_WORKDIR / "fresh_p1_self_play.jsonl"
    rows = pr233.read_jsonl(replay)
    population, exclusions = pr233._population(rows)
    cache_path = Path("/tmp/azlite_shadow_target_distillation/target_cache.npz")
    if sha256_file(cache_path) != pr233.TARGET_CACHE_SHA:
        raise RuntimeError("immutable PR #233 sensitivity cache mismatch")
    cache_file = np.load(cache_path, allow_pickle=False)
    sensitivity = [
        pr233._js(
            cache_file["ordinary_a16"][index] / SIMULATIONS,
            cache_file["shadow_a16"][index] / SIMULATIONS,
        )
        for index in range(len(population))
    ]
    primary_indexes, secondary_indexes, guard_manifest = select_guard_indexes(
        population, sensitivity, GUARD_COUNT, SEED
    )
    frozen_amplified, frozen_washed = pr233._frozen_hashes()
    groups = {
        "primary_guard": [
            population[int(index)] | {"population": "primary_guard"}
            for index in primary_indexes
        ],
        "secondary_guard": [
            population[int(index)] | {"population": "secondary_guard"}
            for index in secondary_indexes
        ],
    }
    manifest = {
        row["state_hash"]: row for row in json.loads(pr233.MANIFEST.read_text())["rows"]
    }

    def frozen_items(keys: set[str], group: str) -> list[dict[str, Any]]:
        if any(key not in manifest for key in keys):
            raise RuntimeError(
                "frozen or PR220 held-out root missing from PR221 manifest"
            )
        return [
            {
                "state_hash": key,
                "state": pr233.decode_kalah_v3_base_state(
                    list(rows[int(manifest[key]["replay_index"])]["state"])
                ),
                "population": group,
            }
            for key in sorted(keys)
        ]

    def held_out_items() -> list[dict[str, Any]]:
        by_hash = {
            row["state_hash"]: {
                "state_hash": row["state_hash"],
                "state": row["state"],
                "population": "held_out_pr220",
            }
            for row in json.loads(PR220_SUMMARY.read_text())["first_game_divergences"]
            if row["state_hash"] in HELD_OUT_HASHES
        }
        if set(by_hash) != HELD_OUT_HASHES:
            raise RuntimeError("PR220 held-out root missing from replay")
        return [by_hash[key] for key in sorted(by_hash)]

    groups |= {
        "frozen_amplified": frozen_items(frozen_amplified, "frozen_amplified"),
        "frozen_washed": frozen_items(frozen_washed, "frozen_washed"),
        "held_out_pr220": held_out_items(),
    }
    artifacts = {
        "P1": pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact",
        "A1": Path("/tmp/azlite_root_q_trust_region/artifacts/step_0001/artifact"),
        "A4": Path("/tmp/azlite_root_q_trust_region/artifacts/step_0004/artifact"),
        "A16": Path("/tmp/azlite_root_q_trust_region/artifacts/step_0016/artifact"),
    }
    if any(not (path / "weights.json").is_file() for path in artifacts.values()):
        raise RuntimeError("missing immutable PR #234 P1/A1/A4/A16 artifact")
    p1 = ArtifactEvaluator(artifacts["P1"])
    p1_observations = {
        item["state_hash"]: _search(item["state"], p1, item["state_hash"])
        for items in groups.values()
        for item in items
    }
    cache_payload = {
        state_hash: {"summary": observation[0], "pre_simulation_root_q": observation[2]}
        for state_hash, observation in p1_observations.items()
    }
    cache_path_out = args.workdir / "p1_pre_simulation_root_q_cache.json"
    cache_path_out.write_text(json.dumps(cache_payload, sort_keys=True) + "\n")
    results: dict[str, dict[str, Any]] = {}
    raw_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, path in artifacts.items():
        evaluator = ArtifactEvaluator(path)
        raw_records[name] = {
            group: [
                _record(item, name, evaluator, p1, p1_observations[item["state_hash"]])
                for item in items
            ]
            for group, items in groups.items()
        }
        results[name] = {
            group: _aggregate(records) for group, records in raw_records[name].items()
        }
    first_flip_validation = [
        _first_flip_invariant(
            item,
            ArtifactEvaluator(artifacts["A16"]),
            _search(
                item["state"], ArtifactEvaluator(artifacts["A16"]), item["state_hash"]
            )[1],
            p1_observations[item["state_hash"]][2],
        )
        for item in groups["frozen_amplified"]
    ]
    invariants = {
        "no_training": True,
        "no_self_play": True,
        "ordinary_puct_unchanged": True,
        "p1_pre_simulation_snapshots": True,
        "first_flip_qsync": all(row["passed"] for row in first_flip_validation),
    }
    a16 = raw_records["A16"]
    predictors = _class_predictors(a16["frozen_amplified"], a16["frozen_washed"])
    guard_difference = _mean_difference_bootstrap(
        a16["primary_guard"], a16["secondary_guard"]
    )
    useful_validation = (
        predictors["parent_q_regret_auc"]["auroc"] >= 0.80
        and predictors["parent_q_regret_auc"]["amplified_minus_washed_bootstrap"][
            "lower_95"
        ]
        > 0
    )
    guard_exposed = guard_difference["lower_95"] > 0
    grows = _metric(raw_records["A1"]["primary_guard"][0]) >= 0 and (
        np.mean([_metric(row) for row in raw_records["A16"]["primary_guard"]])
        > np.mean([_metric(row) for row in raw_records["A1"]["primary_guard"]])
    )
    classification = (
        "selection_consequence_metric_generalizes"
        if all(invariants.values()) and useful_validation and guard_exposed and grows
        else "causal_metric_valid_but_guard_distribution_misses"
        if all(invariants.values()) and useful_validation and not guard_exposed
        else "selection_consequence_not_discriminative"
        if all(invariants.values()) and not useful_validation
        else "invariant_failure"
        if not all(invariants.values())
        else "inconclusive"
    )
    follow_up = {
        "selection_consequence_metric_generalizes": "use parent_q_regret_auc as the PR #234 Adam line-search trust statistic with a preregistered causal budget.",
        "causal_metric_valid_but_guard_distribution_misses": "select a new TRAINING-ONLY guard set by high parent-Q selection regret across eligible replay states, then freeze it before any training.",
        "selection_consequence_not_discriminative": "measure counterfactual trajectory divergence under short parent-Q intervention windows.",
        "invariant_failure": "repair reference timing, eligibility, action tie-breaking, or first-flip reconstruction.",
        "inconclusive": "audit the frozen evaluation coverage before choosing a new statistic.",
    }[classification]
    summary = {
        "schema": "azlite_parent_q_selection_consequence_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "puct_modified": False,
            "simulations": SIMULATIONS,
            "c_puct": 1.25,
            "root_noise": False,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "artifacts": {
            name: {
                "path": str(path),
                "weights_sha256": sha256_file(path / "weights.json"),
            }
            for name, path in artifacts.items()
        },
        "classification": classification,
        "recommended_follow_up": follow_up,
        "hashes": {
            "replay": sha256_file(replay),
            "sensitivity_cache": sha256_file(cache_path),
            "p1_pre_simulation_root_q_cache": sha256_file(cache_path_out),
        },
        "exclusions": exclusions,
        "guard_manifest": guard_manifest,
        "invariants": invariants,
        "results": results,
        "first_flip_causal_validation": first_flip_validation,
        "amplified_vs_washed_predictors": predictors,
        "primary_vs_secondary_guard": {"parent_q_regret_auc": guard_difference},
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(_report(summary))
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        "invariant_failure"
        if not all(invariants.values())
        else "observational_complete"
    )


if __name__ == "__main__":
    main()
