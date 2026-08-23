#!/usr/bin/env python3
# ruff: noqa: E402
"""Offline-only P1/A16 root-Q confidence audit with frozen held-out roots.

This runner observes normal deterministic PUCT searches. Counterfactual scores
are computed after the searches from matched snapshots and never feed PUCT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_fresh_p1_adapter_q_feedback_necessity import (
    C_PUCT,
    SIMULATIONS,
    _seed,
    decode_kalah_v3_base_state,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_synchronized_backup import _control_subset
from ml.alphazero_lite.self_play import PUCT

MANIFEST = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json"
)
FROZEN = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-frozen-amplified-roots.json"
)
PR222 = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-postdivergence-amplification-summary.json"
)
PR220 = (
    REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
)
CHECKPOINTS = (32, 64, 128, 256, 384, 512, 768, 1024, 1200)
ROOT_BOOTSTRAP_SAMPLES = 200
ROOT_BOOTSTRAP_SEED = 227


def _search_config() -> dict[str, Any]:
    return {
        "simulations": SIMULATIONS,
        "c_puct": C_PUCT,
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
        "root_noise": False,
        "root_snapshot_checkpoints": list(CHECKPOINTS),
        "root_backup_history": True,
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


FROZEN_CONFIG_HASH = _canonical_hash(_search_config())


def _action_values(
    history: list[dict[str, Any]], checkpoint: int
) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in history:
        if int(row["simulation"]) <= checkpoint:
            values[int(row["action"])].append(float(row["root_value"]))
    return values


def _history_stats(
    history: list[dict[str, Any]], checkpoint: int
) -> dict[int, dict[str, float | int]]:
    """Reconstruct root-child counts and Q values from committed root backups."""
    return {
        action: {
            "visit_count": len(values),
            "value_sum": float(sum(values)),
            "q_value": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "mad": float(np.median(np.abs(np.asarray(values) - np.median(values)))),
        }
        for action, values in _action_values(history, checkpoint).items()
    }


def reconstruct_backup_stats(summary: dict[str, Any]) -> dict[str, Any]:
    """Verify every root snapshot against opt-in PUCT root backup history."""
    history = list(summary["root_backup_history"])
    checks = []
    for snapshot in summary["root_snapshots"]:
        checkpoint = int(snapshot["simulation"])
        reconstructed = _history_stats(history, checkpoint)
        observed = {int(row["move"]): row for row in snapshot["moves"]}
        actions = sorted(set(reconstructed) | set(observed))
        checks.append(
            {
                "simulation": checkpoint,
                "history_entries": sum(
                    int(row["simulation"]) <= checkpoint for row in history
                ),
                "counts_match": all(
                    int(reconstructed.get(action, {}).get("visit_count", 0))
                    == int(observed.get(action, {}).get("visit_count", 0))
                    for action in actions
                ),
                "q_match": all(
                    np.isclose(
                        float(reconstructed.get(action, {}).get("q_value", 0.0)),
                        float(observed.get(action, {}).get("stored_q_value", 0.0)),
                        atol=1e-12,
                    )
                    for action in actions
                ),
                "actions": reconstructed,
            }
        )
    return {
        "history_length": len(history),
        "continuous_simulations": [int(row["simulation"]) for row in history]
        == list(range(1, SIMULATIONS + 1)),
        "checkpoints": checks,
        "valid": len(history) == SIMULATIONS
        and all(row["counts_match"] and row["q_match"] for row in checks),
    }


def _recent_features(
    values: list[float], checkpoint: int, previous_q: float | None
) -> dict[str, float | int | None]:
    q_value = 0.0 if not values else float(np.mean(values))
    last_visit = None
    return {
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "mad": float(np.median(np.abs(np.asarray(values) - np.median(values))))
        if values
        else None,
        "recent_q_8": None if not values else float(np.mean(values[-8:])),
        "recent_q_32": None if not values else float(np.mean(values[-32:])),
        "recent_drift_8": None
        if not values
        else abs(float(np.mean(values[-8:])) - float(q_value)),
        "recent_drift_32": None
        if not values
        else abs(float(np.mean(values[-32:])) - float(q_value)),
        "q_change_from_previous_checkpoint": None
        if previous_q is None or q_value is None
        else abs(q_value - previous_q),
        # Filled from the history row's simulation index by the caller.
        "last_visit_age": last_visit,
    }


def _rank_by_q(moves: dict[int, dict[str, Any]]) -> dict[int, int]:
    return {
        int(row["move"]): rank
        for rank, row in enumerate(
            sorted(
                moves.values(),
                key=lambda row: (-float(row["stored_q_value"]), int(row["move"])),
            ),
            start=1,
        )
    }


def snapshot_features(a16: dict[str, Any], p1: dict[str, Any]) -> list[dict[str, Any]]:
    """Build all-action features and post-search counterfactuals for each A16 snapshot."""
    p1_by_time = {
        int(snapshot["simulation"]): snapshot for snapshot in p1["root_snapshots"]
    }
    a16_snapshots = {
        int(snapshot["simulation"]): snapshot for snapshot in a16["root_snapshots"]
    }
    checkpoints = sorted(set(a16_snapshots) & set(p1_by_time))
    final_moves = {
        int(row["move"]): row for row in a16_snapshots[max(checkpoints)]["moves"]
    }
    final_q_rank = _rank_by_q(final_moves)
    previous_stats: dict[int, dict[str, float | int]] = {}
    rows = []
    for checkpoint in checkpoints:
        snapshot, matched = a16_snapshots[checkpoint], p1_by_time[checkpoint]
        stats = _history_stats(a16["root_backup_history"], checkpoint)
        values = _action_values(a16["root_backup_history"], checkpoint)
        a16_moves = {int(row["move"]): row for row in snapshot["moves"]}
        p1_moves = {int(row["move"]): row for row in matched["moves"]}
        stats_by_action = {
            action: stats.get(
                action,
                {
                    "visit_count": 0,
                    "value_sum": 0.0,
                    "q_value": 0.0,
                    "std": 0.0,
                    "mad": 0.0,
                },
            )
            for action in a16_moves
        }
        policy_l1 = float(
            sum(
                abs(
                    float(row["prior"])
                    - float(p1_moves.get(action, {"prior": 0.0})["prior"])
                )
                for action, row in a16_moves.items()
            )
        )
        actions = []
        for action, row in sorted(a16_moves.items()):
            other = max(
                (candidate for candidate in stats_by_action if candidate != action),
                key=lambda candidate: float(stats_by_action[candidate]["q_value"]),
                default=None,
            )
            action_stats = stats_by_action[action]
            q_gap = (
                None
                if other is None
                else float(
                    float(action_stats["q_value"])
                    - float(stats_by_action[other]["q_value"])
                )
            )
            denominator = None
            if other is not None:
                action_std = float(action_stats["std"])
                other_std = float(stats_by_action[other]["std"])
                action_visits = int(action_stats["visit_count"])
                other_visits = int(stats_by_action[other]["visit_count"])
                denominator = float(
                    np.hypot(
                        action_std / np.sqrt(max(1, action_visits)),
                        other_std / np.sqrt(max(1, other_visits)),
                    )
                )
            p1_row = p1_moves.get(action)
            eligible = (
                int(row["visit_count"]) > 0
                and p1_row is not None
                and int(p1_row["visit_count"]) > 0
            )
            actions.append(
                {
                    "move": action,
                    "visit_count": int(row["visit_count"]),
                    "stored_q_value": float(row["stored_q_value"]),
                    "selection_score": float(row["selection_score"]),
                    "u_component": float(row["u_component"]),
                    "prior": float(row["prior"]),
                    "q_gap": q_gap,
                    "q_gap_z": None
                    if denominator is None or denominator == 0.0
                    else float(q_gap) / denominator,
                    **_recent_features(
                        values.get(action, []),
                        checkpoint,
                        None
                        if action not in previous_stats
                        else float(previous_stats[action]["q_value"]),
                    ),
                    "matched_p1_q_value": None
                    if p1_row is None
                    else float(p1_row["stored_q_value"]),
                    "matched_p1_visit_count": None
                    if p1_row is None
                    else int(p1_row["visit_count"]),
                    "counterfactual_eligible": eligible,
                    "p1_q_counterfactual_score": None
                    if not eligible
                    else float(
                        float(p1_row["stored_q_value"]) + float(row["u_component"])
                    ),
                    "policy_l1_vs_p1": policy_l1,
                    "future_selected_1200": action == int(a16["selected_move"]),
                    "future_q_value_1200": float(final_moves[action]["stored_q_value"]),
                    "future_q_order_1200": int(final_q_rank[action]),
                    "variance": float(action_stats["std"]) ** 2,
                }
            )
            actions[-1]["last_visit_age"] = checkpoint - max(
                (
                    int(item["simulation"])
                    for item in a16["root_backup_history"]
                    if int(item["action"]) == action
                    and int(item["simulation"]) <= checkpoint
                ),
                default=0,
            )
        eligible_actions = [
            action for action in actions if action["counterfactual_eligible"]
        ]
        counterfactual_move = (
            None
            if not eligible_actions
            else int(
                max(
                    eligible_actions,
                    key=lambda row: (row["p1_q_counterfactual_score"], -row["move"]),
                )["move"]
            )
        )
        selection_pair = sorted(
            actions, key=lambda row: (-row["selection_score"], row["move"])
        )[:2]
        if len(selection_pair) == 2:
            left, right = selection_pair
            left_se = float(left["std"]) / np.sqrt(max(1, int(left["visit_count"])))
            right_se = float(right["std"]) / np.sqrt(max(1, int(right["visit_count"])))
            combined_se = float(np.hypot(left_se, right_se))
            pair_q_gap = abs(
                float(left["stored_q_value"]) - float(right["stored_q_value"])
            )
            pair_features = {
                "min_visits": min(int(left["visit_count"]), int(right["visit_count"])),
                "harmonic_visits": 2.0
                * int(left["visit_count"])
                * int(right["visit_count"])
                / (int(left["visit_count"]) + int(right["visit_count"])),
                "q_gap": pair_q_gap,
                "combined_heuristic_se": combined_se,
                "q_gap_z": pair_q_gap / (combined_se + 1e-12),
                "max_recent_drift_32": max(
                    float(left["recent_drift_32"] or 0.0),
                    float(right["recent_drift_32"] or 0.0),
                ),
                "max_q_change": max(
                    float(left["q_change_from_previous_checkpoint"] or 0.0),
                    float(right["q_change_from_previous_checkpoint"] or 0.0),
                ),
                "visit_margin": abs(
                    int(left["visit_count"]) - int(right["visit_count"])
                ),
                "policy_l1": policy_l1,
            }
        else:
            pair_features = None
        rows.append(
            {
                "simulation": checkpoint,
                "a16_next_selection_move": int(selection_pair[0]["move"]),
                "candidate_selection_pair": selection_pair,
                "counterfactual_eligible_actions": [
                    row["move"] for row in eligible_actions
                ],
                "p1_q_counterfactual_move": counterfactual_move,
                "p1_next_selection_move": int(
                    max(
                        matched["moves"],
                        key=lambda row: (row["selection_score"], -row["move"]),
                    )["move"]
                ),
                "actions": actions,
                "pair_features": pair_features,
            }
        )
        previous_stats = stats
    return rows


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    # Average ranks handle tied scores without the quadratic pair expansion.
    rank_sum = 0.0
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(
            labels[row_index] for row_index, _ in ordered[index:end]
        )
        index = end
    return float(
        (rank_sum - (positive_count * (positive_count + 1) / 2.0))
        / (positive_count * negative_count)
    )


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    ordered = sorted(zip(labels, scores, strict=True), key=lambda row: -row[1])
    hits = 0
    precision_sum = 0.0
    for rank, (label, _score) in enumerate(ordered, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


PREDICTORS = {
    "q_gap_z": lambda row: row["q_gap_z"],
    "stored_q_value": lambda row: row["stored_q_value"],
    "selection_score": lambda row: row["selection_score"],
    "p1_q_counterfactual_score": lambda row: row["p1_q_counterfactual_score"],
    "negative_std": lambda row: -float(row["std"]),
    "negative_mad": lambda row: None if row["mad"] is None else -float(row["mad"]),
    "negative_recent_drift_32": lambda row: (
        None if row["recent_drift_32"] is None else -float(row["recent_drift_32"])
    ),
    "negative_q_change": lambda row: (
        None
        if row["q_change_from_previous_checkpoint"] is None
        else -float(row["q_change_from_previous_checkpoint"])
    ),
    "visit_age": lambda row: float(row["last_visit_age"]),
}


def _metric_rows(
    records: list[dict[str, Any]], predictor: str
) -> list[tuple[bool, float]]:
    rows = []
    for record in records:
        for snapshot in record["snapshots"]:
            for action in snapshot["actions"]:
                value = PREDICTORS[predictor](action)
                if value is not None:
                    rows.append((bool(action["future_selected_1200"]), float(value)))
    return rows


def _root_bootstrap(records: list[dict[str, Any]], predictor: str) -> dict[str, Any]:
    observed = _metric_rows(records, predictor)
    labels, scores = zip(*observed) if observed else ([], [])
    rng = np.random.default_rng(ROOT_BOOTSTRAP_SEED)
    auc_draws, auprc_draws = [], []
    if records:
        for indexes in rng.integers(
            0, len(records), size=(ROOT_BOOTSTRAP_SAMPLES, len(records))
        ):
            sample = [records[int(index)] for index in indexes]
            draw = _metric_rows(sample, predictor)
            if draw:
                draw_labels, draw_scores = zip(*draw)
                auc = _auc(list(draw_labels), list(draw_scores))
                auprc = _average_precision(list(draw_labels), list(draw_scores))
                if auc is not None and auprc is not None:
                    auc_draws.append(auc)
                    auprc_draws.append(auprc)
    return {
        "rows": len(observed),
        "positive_outcomes": int(sum(labels)),
        "auroc": _auc(list(labels), list(scores)),
        "auprc": _average_precision(list(labels), list(scores)),
        "root_bootstrap": {
            "unit": "root",
            "samples": ROOT_BOOTSTRAP_SAMPLES,
            "auroc_95": None
            if not auc_draws
            else [
                float(np.quantile(auc_draws, 0.025)),
                float(np.quantile(auc_draws, 0.975)),
            ],
            "auprc_95": None
            if not auprc_draws
            else [
                float(np.quantile(auprc_draws, 0.025)),
                float(np.quantile(auprc_draws, 0.975)),
            ],
        },
    }


def _bin(value: float, boundaries: tuple[float, ...]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"<={boundary:g}"
    return f">{boundaries[-1]:g}"


def _outcome_bins(
    records: list[dict[str, Any]], feature: str, boundaries: tuple[float, ...]
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        for snapshot in record["snapshots"]:
            for action in snapshot["actions"]:
                grouped[_bin(float(action[feature]), boundaries)].append(
                    bool(action["future_selected_1200"])
                )
    return {
        name: {
            "rows": len(outcomes),
            "future_selected_1200_rate": float(np.mean(outcomes)),
        }
        for name, outcomes in sorted(grouped.items())
    }


def aggregate_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate final-1200 action selection predictors with root bootstrapping."""
    return {
        "outcome": "future_selected_1200",
        "root_level_bootstrap_predictor_comparisons": {
            name: _root_bootstrap(records, name) for name in PREDICTORS
        },
        "visit_bins": _outcome_bins(records, "visit_count", (0, 1, 4, 16, 64, 256)),
        "variance_bins": _outcome_bins(
            records, "variance", (0.0, 0.001, 0.01, 0.05, 0.2)
        ),
        "roots": len(records),
        "action_rows": sum(
            len(snapshot["actions"])
            for record in records
            for snapshot in record["snapshots"]
        ),
    }


def _cache_path(workdir: Path, state_hash: str, artifact: str) -> Path:
    return workdir / "searches" / state_hash / f"{artifact}.json"


def _run_or_reuse(
    *,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    state_hash: str,
    artifact: str,
    artifact_weights_sha256: str,
    workdir: Path,
) -> tuple[dict[str, Any], bool]:
    path = _cache_path(workdir, state_hash, artifact)
    contract = {
        "state_hash": state_hash,
        "artifact": artifact,
        "artifact_weights_sha256": artifact_weights_sha256,
        "seed": _seed(state_hash),
        "frozen_config_hash": FROZEN_CONFIG_HASH,
    }
    if path.is_file():
        cached = json.loads(path.read_text())
        if (
            cached.get("contract") == contract
            and reconstruct_backup_stats(cached["summary"])["valid"]
        ):
            return cached["summary"], True
    history: list[dict[str, int | float]] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        rng=random.Random(_seed(state_hash)),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        root_snapshot_checkpoints=set(CHECKPOINTS),
        root_backup_history=history,
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    summary = search.root_summary()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"contract": contract, "summary": summary}, indent=2, sort_keys=True)
        + "\n"
    )
    return summary, False


def _classification(invariants: dict[str, bool]) -> tuple[str, str]:
    if not all(invariants.values()):
        return (
            "invariant_failure",
            "Do not interpret Q-confidence evidence; repair the frozen search or reconstruction contract.",
        )
    return (
        "inconclusive",
        "Run a preregistered small independent root verification search / ensemble rather than a confidence-weighted scalar Q rule.",
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Candidate Q Confidence Audit",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommendation:** {summary['recommendation']}",
            "",
            "## Frozen Configuration",
            "",
            "```json",
            json.dumps(
                {"hash": summary["frozen_config_hash"], "config": summary["config"]},
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Discovery Predictors",
            "",
            "```json",
            json.dumps(summary["discovery_analysis"], indent=2, sort_keys=True),
            "```",
            "",
            "## Held Roots",
            "",
            "```json",
            json.dumps(summary["held_analysis"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_adapter_candidate_q_confidence"),
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development cap for discovery roots only.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-candidate-q-confidence-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-candidate-q-confidence-results.md",
    )
    args = parser.parse_args()
    manifest, frozen, prior, pr220 = (
        json.loads(path.read_text()) for path in (MANIFEST, FROZEN, PR222, PR220)
    )
    if (
        len(manifest["rows"]) != 4096
        or len({row["state_hash"] for row in manifest["rows"]}) != 4096
    ):
        raise RuntimeError("candidate audit requires the immutable 4096-root manifest")
    manifest_by_hash = {row["state_hash"]: row for row in manifest["rows"]}
    prior_by_hash = {row["state_hash"]: row for row in prior["records"]}
    frozen_hashes = list(frozen["full_amplified_1200"])
    controls = _control_subset(
        prior["records"], [prior_by_hash[key] for key in frozen_hashes]
    )
    control_hashes = [row["state_hash"] for row in controls]
    pr220_rows = {row["state_hash"]: row for row in pr220["first_game_divergences"]}
    held_hashes = set(frozen_hashes) | set(control_hashes) | set(pr220_rows)
    discovery_metadata = [
        row for row in manifest["rows"] if row["state_hash"] not in held_hashes
    ]
    if args.limit is not None:
        discovery_metadata = discovery_metadata[: args.limit]
    replay = read_jsonl(args.adapter_workdir / "fresh_p1_self_play.jsonl")
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = args.adapter_workdir / "artifacts/step_0016/artifact"
    evaluators = {
        "p1": ArtifactEvaluator(p1_artifact),
        "a16": ArtifactEvaluator(a16_artifact),
    }
    artifact_hashes = {
        "p1": sha256_file(p1_artifact / "weights.json"),
        "a16": sha256_file(a16_artifact / "weights.json"),
    }

    def audit(metadata: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        state_hash = str(metadata["state_hash"])
        if canonical_game_state_hash(KalahGame.from_state(state)) != state_hash:
            raise RuntimeError(f"replay state hash mismatch: {state_hash}")
        p1, p1_reused = _run_or_reuse(
            evaluator=evaluators["p1"],
            state=state,
            state_hash=state_hash,
            artifact="p1",
            artifact_weights_sha256=artifact_hashes["p1"],
            workdir=args.workdir,
        )
        a16, a16_reused = _run_or_reuse(
            evaluator=evaluators["a16"],
            state=state,
            state_hash=state_hash,
            artifact="a16",
            artifact_weights_sha256=artifact_hashes["a16"],
            workdir=args.workdir,
        )
        return {
            **metadata,
            "p1_reused": p1_reused,
            "a16_reused": a16_reused,
            "p1_reconstruction": reconstruct_backup_stats(p1),
            "a16_reconstruction": reconstruct_backup_stats(a16),
            "p1_final_move": p1["selected_move"],
            "a16_final_move": a16["selected_move"],
            "snapshots": snapshot_features(a16, p1),
        }

    def replay_state(metadata: dict[str, Any]) -> dict[str, Any]:
        return decode_kalah_v3_base_state(
            list(replay[int(metadata["replay_index"])]["state"])
        )

    discovery_records = [
        audit(metadata, replay_state(metadata)) for metadata in discovery_metadata
    ]
    discovery_analysis = aggregate_analysis(discovery_records)
    # Held roots are scored only after all discovery calculations are complete.
    held_records = {
        "frozen_amplified_40": [
            audit(manifest_by_hash[key], replay_state(manifest_by_hash[key]))
            for key in frozen_hashes
        ],
        "matched_controls_40": [
            audit(manifest_by_hash[key], replay_state(manifest_by_hash[key]))
            for key in control_hashes
        ],
        "pr220": [
            audit({"state_hash": key, "held_group": "pr220"}, row["state"])
            for key, row in sorted(pr220_rows.items())
        ],
    }
    all_records = discovery_records + [
        record for group in held_records.values() for record in group
    ]
    invariants = {
        "frozen_40": len(frozen_hashes) == 40,
        "matched_controls_40": len(control_hashes) == 40,
        "discovery_excludes_all_held_roots": not (
            {row["state_hash"] for row in discovery_records} & held_hashes
        ),
        "all_a16_reconstructions_valid": all(
            row["a16_reconstruction"]["valid"] for row in all_records
        ),
        "all_p1_reconstructions_valid": all(
            row["p1_reconstruction"]["valid"] for row in all_records
        ),
    }
    classification, recommendation = _classification(invariants)
    summary = {
        "schema": "azlite_fresh_p1_adapter_candidate_q_confidence_v2",
        "guardrails": {
            "training": False,
            "self_play": False,
            "search_intervention": False,
            "counterfactual_is_post_search_only": True,
        },
        "config": _search_config(),
        "frozen_config_hash": FROZEN_CONFIG_HASH,
        "hashes": {
            "manifest": sha256_file(MANIFEST),
            "frozen": sha256_file(FROZEN),
            "pr220": sha256_file(PR220),
            "artifact_weights": artifact_hashes,
        },
        "population": {
            "manifest_rows": 4096,
            "discovery_roots": len(discovery_records),
            "held_frozen_amplified": len(frozen_hashes),
            "held_matched_controls": len(control_hashes),
            "held_pr220_unique_roots": len(pr220_rows),
        },
        "discovery_analysis": discovery_analysis,
        "held_analysis": {
            name: aggregate_analysis(records) for name, records in held_records.items()
        },
        "invariants": invariants,
        "classification": classification,
        "recommendation": recommendation,
    }
    for path in (args.out_summary, args.out_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
