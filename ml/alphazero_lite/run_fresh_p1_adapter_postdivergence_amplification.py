#!/usr/bin/env python3
"""Audit offline post-divergence Q/visit amplification for the PR #221 roots."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.margin_sensitivity import decision_sensitivity
from ml.alphazero_lite.postdivergence_amplification import (
    paired_postdivergence_metrics,
    reconstruct_root_trajectory,
    validate_final_root_trajectory,
    visit_js,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_margin_sensitivity import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    C_PUCT,
    SIMULATIONS,
    _auroc,
    _average_precision,
    _percentile,
    _root_policy_metrics,
    _seed,
    _spearman,
    _trace_prefix,
    decode_kalah_v3_base_state,
    first_divergence,
    select_sample,
)
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.kalah_rules import KalahGame

REPO_ROOT = Path(__file__).resolve().parents[2]
PR221_SUMMARY = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-summary.json"
)
PR221_MANIFEST = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json"
)
PR220_SUMMARY = (
    REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
)
PR221_MANIFEST_SHA = "20b0aa432b1ee978c9da22d030a6e51e1cc1753989cfd6bf6957129915b22078"
WINDOWS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
PRIMARY = "q_divergence_auc_32"
SECONDARY = ("backup_gap_auc_32", "path_divergence_fraction_32", "visit_js_auc_32")
_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


def _init_worker(p1_artifact: str, a16_artifact: str) -> None:
    global _P1, _A16
    _P1, _A16 = (
        ArtifactEvaluator(Path(p1_artifact)),
        ArtifactEvaluator(Path(a16_artifact)),
    )


def _search(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    seed: int,
    simulations: int = SIMULATIONS,
) -> tuple[list[dict], dict]:
    trace: list[dict] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=C_PUCT,
        rng=random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        selection_trace=trace,
        trace_checkpoints={384, 1200},
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return trace, search.root_summary()


def _snapshot(summary: dict, simulation: int) -> dict:
    return next(
        item
        for item in summary["trace_root_snapshots"]
        if item["simulation"] == simulation
    )


def _event_details(
    state: dict[str, Any], p1_trace: list[dict], divergence: dict
) -> dict[str, Any]:
    """Re-evaluate only the already selected first-divergence parent for PR #221 alignment."""
    assert _A16 is not None
    simulation = int(divergence["simulation"])
    depth = int(divergence["depth"])
    game = KalahGame.from_state(state)
    for decision in p1_trace[simulation - 1]["selection_path"][:depth]:
        if not game.move(game.pit_index(int(decision["chosen_move"]))):
            raise RuntimeError("cannot reconstruct first-divergence parent")
    decision = p1_trace[simulation - 1]["selection_path"][depth]
    if canonical_game_state_hash(game) != decision["state_hash"]:
        raise RuntimeError("first-divergence parent hash mismatch")
    score = decision_sensitivity(decision, _A16.evaluate(game)[0], c_puct=C_PUCT)
    selected = int(decision["chosen_move"])
    children = {int(item["move"]): item for item in decision["children"]}
    parent_margin = min(
        float(children[selected]["selection_score"]) - float(item["selection_score"])
        for move, item in children.items()
        if move != selected
    )
    return {
        **divergence,
        "parent_puct_margin": parent_margin,
        "delta_u": {
            str(move): float(value) for move, value in score["delta_u"].items()
        },
        "flip_excess": float(score["max_flip_excess"]),
    }


def _audit_root(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    state, metadata, old = task
    assert _P1 is not None and _A16 is not None
    seed = _seed(metadata["state_hash"])
    p1_trace, p1_summary = _search(_P1, state, seed)
    a16_trace, a16_summary = _search(_A16, state, seed)
    p1_trajectory, a16_trajectory = (
        reconstruct_root_trajectory(p1_trace),
        reconstruct_root_trajectory(a16_trace),
    )
    divergence = first_divergence(a16_trace, p1_trace)
    invariant_failure = divergence is not None and "invariant_failure" in divergence
    if invariant_failure:
        return {**metadata, "invariant_failure": divergence["invariant_failure"]}
    actual = None if divergence is None else _event_details(state, p1_trace, divergence)
    first_matches = (
        actual is None
        and old["actual_first_divergence"] is None
        or actual is not None
        and all(
            actual[key] == old["actual_first_divergence"][key]
            for key in ("simulation", "depth", "state_hash", "action_pair")
        )
    )
    result = {
        **metadata,
        "search_seed": seed,
        "actual_first_divergence": actual,
        "first_flip_matches_pr221": first_matches,
        "root_trajectory_matches_summary": {
            "p1": validate_final_root_trajectory(p1_trajectory, p1_summary),
            "a16": validate_final_root_trajectory(a16_trajectory, a16_summary),
        },
        "root_move_difference_384": _snapshot(p1_summary, 384)["selected_move"]
        != _snapshot(a16_summary, 384)["selected_move"],
        "root_move_difference_1200": p1_summary["selected_move"]
        != a16_summary["selected_move"],
        "visit_js_384": visit_js(
            _snapshot(p1_summary, 384)["visits"], _snapshot(a16_summary, 384)["visits"]
        ),
        "visit_js_1200": visit_js(
            p1_trajectory[-1]["visit_distribution"],
            a16_trajectory[-1]["visit_distribution"],
        ),
        "baseline": {
            "policy_l1": old["root_policy_l1"],
            "max_action_delta": old["root_max_abs_action_delta"],
            "max_pressure_ratio": old["through_1200"]["max_pressure_ratio"],
            "minimum_parent_puct_margin": old["through_1200"][
                "minimum_parent_puct_winner_runner_up_margin"
            ],
        },
    }
    if actual is not None:
        actual["pr221_predicted_same_event"] = first_matches
        result.update(
            paired_postdivergence_metrics(
                p1_trace, a16_trace, p1_trajectory, a16_trajectory, actual["simulation"]
            )
        )
    return result


def _predictors(records: list[dict]) -> dict[str, np.ndarray]:
    names = (
        "policy_l1",
        "max_action_delta",
        "max_pressure_ratio",
        "minimum_parent_puct_margin",
    )
    result = {
        name: np.asarray([record["baseline"][name] for record in records])
        for name in names
    }
    for width in (32, 64):
        for name in (
            "backup_gap_auc",
            "q_divergence_auc",
            "visit_js_auc",
            "path_divergence_fraction",
        ):
            key = f"{name}_{width}"
            result[key] = np.asarray(
                [record.get("early_metrics", {}).get(key, 0.0) for record in records]
            )
    return result


def _table(records: list[dict]) -> dict[str, Any]:
    table = {}
    for name, scores in _predictors(records).items():
        table[name] = {}
        for budget in (384, 1200):
            labels = np.asarray(
                [record[f"root_move_difference_{budget}"] for record in records],
                dtype=bool,
            )
            js = np.asarray([record[f"visit_js_{budget}"] for record in records])
            order = np.argsort(-scores, kind="stable")
            quartiles = np.array_split(order, 4)
            table[name][str(budget)] = {
                "auroc": _auroc(scores, labels),
                "auprc": _average_precision(scores, labels),
                "top10_capture": float(labels[order[: len(records) // 10]].mean()),
                "top25_capture": float(labels[order[: len(records) // 4]].mean()),
                "spearman_final_visit_js": _spearman(scores, js),
                "final_visit_js_by_quartile": [
                    float(js[index].mean()) for index in quartiles
                ],
            }
    return table


def _interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "estimate": float(array.mean()),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
        "samples": len(values),
    }


def _paired_bootstrap(records: list[dict]) -> dict[str, Any]:
    """Fixed-primary paired bootstrap; no post-hoc metric selection enters the CI."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, len(records), size=(BOOTSTRAP_SAMPLES, len(records)))
    scores = _predictors(records)
    label = np.asarray(
        [record["root_move_difference_1200"] for record in records], dtype=bool
    )
    js = np.asarray([record["visit_js_1200"] for record in records])
    differences = {"auprc": [], "spearman_final_visit_js": []}
    for draw in indexes:
        primary_ap, policy_ap = (
            _average_precision(scores[PRIMARY][draw], label[draw]),
            _average_precision(scores["policy_l1"][draw], label[draw]),
        )
        primary_rho, policy_rho = (
            _spearman(scores[PRIMARY][draw], js[draw]),
            _spearman(scores["policy_l1"][draw], js[draw]),
        )
        if primary_ap is not None and policy_ap is not None:
            differences["auprc"].append(primary_ap - policy_ap)
        if primary_rho is not None and policy_rho is not None:
            differences["spearman_final_visit_js"].append(primary_rho - policy_rho)
    return {
        f"{PRIMARY}_minus_policy_l1": {
            name: _interval(values) for name, values in differences.items()
        }
    }


def _window_summary(records: list[dict]) -> dict[str, Any]:
    result = {}
    fields = (
        "path_differs",
        "common_prefix_depth",
        "leaf_hash_mismatch",
        "terminal_mismatch",
        "backup_difference",
        "backup_absolute_difference",
        "backup_cumulative_absolute_difference",
        "backup_rms_difference",
        "backup_opposite_sign",
        "root_q_l1",
        "root_q_max",
        "q_rank_disagreement",
        "best_q_disagreement",
        "candidate_selected_root_q_difference",
        "visit_js",
        "visit_l1",
        "visit_leader_disagreement",
        "root_move_disagreement",
        "visit_margin_difference",
    )
    for offset in WINDOWS:
        rows = [
            record["windows"][str(offset)] for record in records if "windows" in record
        ]
        result[str(offset)] = {
            field: float(np.mean([row[field] for row in rows])) for field in fields
        }
    return result


def _lead_lag_summary(records: list[dict]) -> dict[str, Any]:
    names = (
        "path_divergence",
        "backup_value_difference",
        "q_ranking_difference",
        "visit_leader_difference",
        "root_move_difference",
    )
    result = {name: {} for name in names}
    orderings = Counter()
    for record in records:
        lead = record["lead_lag"]
        for name in names:
            for field in (
                "first_relative_simulation",
                "disappears_again",
                "longest_consecutive_run",
                "fraction_remaining",
            ):
                result[name].setdefault(field, []).append(lead[name][field])
        orderings[
            " -> ".join(
                sorted(
                    names,
                    key=lambda name: (
                        lead[name]["first_relative_simulation"] is None,
                        lead[name]["first_relative_simulation"] or 10**9,
                    ),
                )
            )
        ] += 1
    return {
        "events": {
            name: {
                "mean_first_relative_simulation": float(
                    np.mean(
                        [
                            value
                            for value in values["first_relative_simulation"]
                            if value is not None
                        ]
                    )
                )
                if any(
                    value is not None for value in values["first_relative_simulation"]
                )
                else None,
                "disappears_again_rate": float(np.mean(values["disappears_again"])),
                "mean_longest_consecutive_run": float(
                    np.mean(values["longest_consecutive_run"])
                ),
                "mean_fraction_remaining": float(np.mean(values["fraction_remaining"])),
            }
            for name, values in result.items()
        },
        "dominant_ordering": orderings.most_common(1)[0][0],
        "ordering_counts": dict(orderings),
    }


def _search_budget_summary(records: list[dict]) -> dict[str, Any]:
    late = [
        record
        for record in records
        if not record["root_move_difference_384"]
        and record["root_move_difference_1200"]
    ]
    at_384 = [record["at_384"] for record in late if record.get("at_384") is not None]
    return {
        "agrees_384_differs_1200_count": len(late),
        "q_divergence_visible_by_384_rate": float(
            np.mean([row["root_q_l1"] > 1e-12 for row in at_384])
        )
        if at_384
        else None,
        "visit_divergence_visible_by_384_rate": float(
            np.mean([row["visit_js"] > 1e-12 for row in at_384])
        )
        if at_384
        else None,
        "mean_q_l1_at_384": float(np.mean([row["root_q_l1"] for row in at_384]))
        if at_384
        else None,
        "mean_visit_js_at_384": float(np.mean([row["visit_js"] for row in at_384]))
        if at_384
        else None,
        "amplification_begins_after_384_count": sum(
            record.get("at_384") is None for record in late
        ),
    }


def _report(summary: dict) -> str:
    primary = summary["paired_bootstrap"][f"{PRIMARY}_minus_policy_l1"]
    lead = summary["lead_lag"]
    held = [
        "| State hash | Q AUC32 pct | Backup AUC32 pct | Path pct | Visit-JS AUC32 pct | Policy L1 pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["held_out_pr220"]:
        p = row["percentiles"]
        held.append(
            f"| `{row['state_hash']}` | {p[PRIMARY]:.2f} | {p['backup_gap_auc_32']:.2f} | {p['path_divergence_fraction_32']:.2f} | {p['visit_js_auc_32']:.2f} | {p['policy_l1']:.2f} |"
        )
    return "\n".join(
        [
            "# Post-Divergence Q/Visit Amplification",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended follow-up:** {summary['recommended_follow_up']}",
            "",
            "Unvisited root children use value sum and Q of zero, matching `PUCT.root_summary()`.",
            "",
            "## Primary Comparison",
            "",
            "```json",
            json.dumps(primary, indent=2, sort_keys=True),
            "```",
            "",
            "## Lead/Lag",
            "",
            f"Dominant ordering: `{lead['dominant_ordering']}`",
            "",
            "```json",
            json.dumps(lead["events"], indent=2, sort_keys=True),
            "```",
            "",
            "## Held-Out PR #220 States",
            "",
            *held,
            "",
            "## Validation",
            "",
            "```json",
            json.dumps(summary["invariants"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-postdivergence-amplification-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-postdivergence-amplification-results.md",
    )
    args = parser.parse_args()
    old_summary, manifest = (
        json.loads(PR221_SUMMARY.read_text()),
        json.loads(PR221_MANIFEST.read_text()),
    )
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    rows = read_jsonl(replay)
    selected, expected_manifest = select_sample(rows)
    invariant_manifest = (
        expected_manifest == manifest["rows"]
        and sha256_file(PR221_MANIFEST) == PR221_MANIFEST_SHA
    )
    old_by_hash = {record["state_hash"]: record for record in old_summary["records"]}
    tasks = [
        (
            decode_kalah_v3_base_state(list(rows[index]["state"])),
            metadata,
            old_by_hash[metadata["state_hash"]],
        )
        for index, metadata in zip(selected, manifest["rows"], strict=True)
    ]
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(p1_artifact), str(a16_artifact)),
    ) as executor:
        prefix_tasks = tasks[:8]
        prefix_ok = all(
            _trace_prefix(
                _search(
                    ArtifactEvaluator(p1_artifact), state, _seed(metadata["state_hash"])
                )[0],
                384,
            )
            == _trace_prefix(
                _search(
                    ArtifactEvaluator(p1_artifact),
                    state,
                    _seed(metadata["state_hash"]),
                    384,
                )[0],
                384,
            )
            for state, metadata, _old in prefix_tasks
        )
        records = list(executor.map(_audit_root, tasks, chunksize=1))
    divergent = [
        record for record in records if record["actual_first_divergence"] is not None
    ]
    washed_out_threshold = float(
        np.quantile([record["visit_js_1200"] for record in divergent], 0.75)
    )
    for record in records:
        record["descriptive_class"] = (
            "no_actual_divergence"
            if record["actual_first_divergence"] is None
            else "amplified"
            if record["root_move_difference_1200"]
            else "washed_out"
            if record["visit_js_1200"] <= washed_out_threshold
            else "non_amplified_high_visit_js"
        )
    invariants = {
        "pr221_artifact_invariants": all(old_summary["invariants"].values()),
        "manifest_hash_and_order": invariant_manifest,
        "replay_hash": sha256_file(replay) == old_summary["hashes"]["replay"],
        "continuous_1200_prefix_equals_standalone_384": prefix_ok,
        "first_divergence_alignment": all(
            record["first_flip_matches_pr221"] for record in records
        ),
        "root_trajectory_matches_puct_summary": all(
            all(record["root_trajectory_matches_summary"].values())
            for record in records
        ),
    }
    comparison = _table(records)
    conditional_records = [
        record
        for record in divergent
        if record["actual_first_divergence"]["simulation"] <= 384
    ]
    conditional = _table(conditional_records)
    bootstrap = _paired_bootstrap(records)
    distributions = _predictors(records)
    p1, a16 = ArtifactEvaluator(p1_artifact), ArtifactEvaluator(a16_artifact)
    held = []
    for row in {
        item["state_hash"]: item
        for item in json.loads(PR220_SUMMARY.read_text())["first_game_divergences"]
    }.values():
        state, seed = row["state"], _seed(row["state_hash"])
        pt, _p1_summary, at, _a16_summary = (
            *_search(p1, state, seed),
            *_search(a16, state, seed),
        )
        actual = first_divergence(at, pt)
        metrics = paired_postdivergence_metrics(
            pt,
            at,
            reconstruct_root_trajectory(pt),
            reconstruct_root_trajectory(at),
            actual["simulation"],
        )
        values = {
            **metrics["early_metrics"],
            "policy_l1": _root_policy_metrics(p1, a16, KalahGame.from_state(state))[
                "root_policy_l1"
            ],
        }
        held.append(
            {
                "state_hash": row["state_hash"],
                "values": values,
                "percentiles": {
                    key: _percentile(value, distributions[key])
                    for key, value in values.items()
                    if key in distributions
                },
            }
        )
    amplified = [record for record in divergent if record["root_move_difference_1200"]]
    q_leads = sum(
        record["lead_lag"]["q_ranking_difference"]["first_relative_simulation"]
        is not None
        and (
            record["lead_lag"]["root_move_difference"]["first_relative_simulation"]
            is None
            or record["lead_lag"]["q_ranking_difference"]["first_relative_simulation"]
            < record["lead_lag"]["root_move_difference"]["first_relative_simulation"]
        )
        for record in amplified
    )
    primary_ci = bootstrap[f"{PRIMARY}_minus_policy_l1"]["auprc"]
    classification = (
        "invariant_failure"
        if not all(invariants.values())
        else "q_feedback_predicts_amplification"
        if amplified and q_leads / len(amplified) > 0.5 and primary_ci["lower_95"] > 0
        else "inconclusive"
    )
    follow_up = (
        "Perform a diagnostic shared-backup-value / Q-feedback counterfactual to test causality."
        if classification == "q_feedback_predicts_amplification"
        else "Use the frozen audit results to choose a causal post-divergence intervention without changing search behavior."
    )
    summary = {
        "schema": "azlite_pr221_postdivergence_amplification_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "arena_derived_calibration": False,
            "promotion": False,
            "c_puct": C_PUCT,
            "simulations": SIMULATIONS,
        },
        "hashes": {
            "a16_state": old_summary["hashes"]["a16_state"],
            "p1_checkpoint": old_summary["hashes"]["p1_checkpoint"],
            "pr220_parent": old_summary["hashes"]["pr220_parent"],
            "pr220_commit": old_summary["hashes"]["pr220_commit"],
            "manifest": sha256_file(PR221_MANIFEST),
            "replay": sha256_file(replay),
            "p1_artifact_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_artifact_weights": sha256_file(a16_artifact / "weights.json"),
        },
        "invariants": invariants,
        "conventions": {
            "unvisited_root_child_q": 0.0,
            "unvisited_root_child_value_sum": 0.0,
            "early_window": "includes d through d+31 or d+63, clipped at simulation 1200",
        },
        "records": records,
        "outcomes": {
            "washed_out": sum(
                record["descriptive_class"] == "washed_out" for record in records
            ),
            "amplified": sum(
                record["descriptive_class"] == "amplified" for record in records
            ),
            "non_amplified_high_visit_js": sum(
                record["descriptive_class"] == "non_amplified_high_visit_js"
                for record in records
            ),
            "no_actual_divergence": sum(
                record["descriptive_class"] == "no_actual_divergence"
                for record in records
            ),
        },
        "window_summary": _window_summary(divergent),
        "lead_lag": _lead_lag_summary(divergent),
        "search_budget": _search_budget_summary(divergent),
        "predictor_comparison": comparison,
        "conditional_first_divergence_before_384": {
            "count": len(conditional_records),
            "predictor_comparison": conditional,
        },
        "paired_bootstrap": bootstrap,
        "held_out_pr220": held,
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
