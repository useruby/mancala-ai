#!/usr/bin/env python3
"""Run opening PUCT disagreement replay-weight ablation from promoted current."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    EXPECTED_INIT_CHECKPOINT_SHA256,
    EXPECTED_PROMOTED_WEIGHTS_SHA256,
    artifact_forward_details,
    build_input_summary,
    kl_divergence,
    margin_bucket,
    read_jsonl,
    require_existing_file,
    top_policy_move,
    verify_expected_hash,
    write_json,
)
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    DEFAULT_BUDGET_PAIRS,
    budget_results_by_pair,
    candidate_standard_ds,
    compute_param_delta_norm,
    export_checkpoint,
    find_candidate_report,
    heldout_summary,
    run_default_gate,
    run_opening_suite_benchmark,
    run_train,
    sha256_file,
)

STANDARD_BUDGET = "384:256"
EXPECTED_MINED_ROWS = 2000
PR122_WORKDIR = Path("/tmp/azlite_promoted_current_opening_puct_disagreement")


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("at least one mined weight is required")
    if any(value <= 0 for value in values):
        raise ValueError("mined weights must be positive integers")
    return values


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_sampled_fraction(
    *, generic_rows: int, random_rows: int, mined_rows: int, mined_weight: int
) -> float:
    total = (4 * generic_rows) + random_rows + (mined_weight * mined_rows)
    return (mined_weight * mined_rows) / total if total > 0 else 0.0


def load_selected_audit_states(
    *, audit_path: Path, mined_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    audit = load_json(audit_path)
    audit_states = audit.get("states")
    if not isinstance(audit_states, list):
        raise RuntimeError(f"disagreement audit is missing states: {audit_path}")

    replay_hashes = [str(row["state_hash"]) for row in mined_rows]
    if len(set(replay_hashes)) != len(replay_hashes):
        raise RuntimeError("mined replay contains duplicate state_hash rows")

    replay_hash_set = set(replay_hashes)
    selected_by_hash: dict[str, dict[str, Any]] = {}
    for state in audit_states:
        state_hash = str(state.get("state_hash", ""))
        if state_hash in replay_hash_set and state_hash not in selected_by_hash:
            selected_by_hash[state_hash] = state

    missing_hashes = [
        state_hash for state_hash in replay_hashes if state_hash not in selected_by_hash
    ]
    if missing_hashes:
        raise RuntimeError(
            "disagreement audit is missing mined states: "
            f"{len(missing_hashes)} missing, first={missing_hashes[0]}"
        )

    selected_states = [selected_by_hash[state_hash] for state_hash in replay_hashes]
    return selected_states, selected_by_hash


def build_mined_state_rows(
    mined_replay_rows: list[dict[str, Any]],
    audit_rows_by_hash: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replay_row in mined_replay_rows:
        state_hash = str(replay_row["state_hash"])
        audit_row = audit_rows_by_hash[state_hash]
        rows.append(
            {
                "state_hash": state_hash,
                "state": audit_row["state"],
                "legal_moves": [int(move) for move in replay_row["legal_moves"]],
                "phase": str(replay_row["phase"]),
                "raw_margin": float(replay_row["raw_margin"]),
                "raw_top1": replay_row.get("raw_top1"),
                "search_top1": replay_row.get("search_top1"),
                "search_policy": audit_row["search_policy"],
            }
        )
    return rows


def summarize_group_counts(changed: int, matches: int, total: int) -> dict[str, Any]:
    return {
        "states": total,
        "top1_changed": changed,
        "top1_changed_rate": (changed / total) if total else 0.0,
        "top1_matches_search": matches,
        "top1_matches_search_rate": (matches / total) if total else 0.0,
    }


def mined_state_policy_shift(
    candidate_artifact: Path, mined_state_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(candidate_artifact)
    changed = 0
    matches = 0
    kl_search_to_candidate: list[float] = []
    kl_candidate_to_search: list[float] = []
    by_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"changed": 0, "matches": 0, "total": 0}
    )
    by_phase: dict[str, dict[str, int]] = defaultdict(
        lambda: {"changed": 0, "matches": 0, "total": 0}
    )

    for row in mined_state_rows:
        game = KalahGame.from_state(row["state"])
        _logits, raw_policy, _value = artifact_forward_details(evaluator, game)
        top1 = top_policy_move(raw_policy, row["legal_moves"])
        row_changed = top1 != row["raw_top1"]
        row_matches = top1 == row["search_top1"]
        search_policy = np.asarray(row["search_policy"], dtype=np.float64)
        candidate_policy = np.asarray(raw_policy, dtype=np.float64)
        kl_search_to_candidate.append(kl_divergence(search_policy, candidate_policy))
        kl_candidate_to_search.append(kl_divergence(candidate_policy, search_policy))

        bucket = margin_bucket(float(row["raw_margin"]))
        phase = str(row["phase"])
        by_bucket[bucket]["total"] += 1
        by_phase[phase]["total"] += 1
        if row_changed:
            changed += 1
            by_bucket[bucket]["changed"] += 1
            by_phase[phase]["changed"] += 1
        if row_matches:
            matches += 1
            by_bucket[bucket]["matches"] += 1
            by_phase[phase]["matches"] += 1

    total = len(mined_state_rows)
    return {
        "states": total,
        "top1_changed_rate_vs_promoted_current": (changed / total) if total else 0.0,
        "top1_matches_search_rate": (matches / total) if total else 0.0,
        "mean_kl_search_to_candidate_raw": statistics.fmean(kl_search_to_candidate)
        if kl_search_to_candidate
        else 0.0,
        "mean_kl_candidate_raw_to_search": statistics.fmean(kl_candidate_to_search)
        if kl_candidate_to_search
        else 0.0,
        "top1_movement_by_raw_margin_bucket": {
            bucket: summarize_group_counts(
                counts["changed"], counts["matches"], counts["total"]
            )
            for bucket, counts in sorted(by_bucket.items())
        },
        "top1_movement_by_phase": {
            phase: summarize_group_counts(
                counts["changed"], counts["matches"], counts["total"]
            )
            for phase, counts in sorted(by_phase.items())
        },
    }


def train_or_reuse_lane(
    *,
    lane_dir: Path,
    lane_name: str,
    replay_weights: str,
    init_checkpoint: Path,
    generic_bootstrap: Path,
    random_teacher: Path,
    mined_replay: Path,
    epochs: int,
    seed: int,
    skip_training: bool,
) -> tuple[Path, Path, dict[str, Any] | None]:
    lane_dir.mkdir(parents=True, exist_ok=True)
    epoch_checkpoint_path = lane_dir / f"checkpoint_epoch{epochs}.npz"
    export_dir = lane_dir / f"artifact_{lane_name}"
    train_metrics_path = lane_dir / "train_metrics.json"

    if (
        export_dir.joinpath("weights.json").is_file()
        and epoch_checkpoint_path.is_file()
    ):
        train_metrics = (
            load_json(train_metrics_path) if train_metrics_path.is_file() else None
        )
        return epoch_checkpoint_path, export_dir, train_metrics

    if skip_training:
        require_existing_file(epoch_checkpoint_path, f"checkpoint for {lane_name}")
        require_existing_file(export_dir / "weights.json", f"artifact for {lane_name}")
        return epoch_checkpoint_path, export_dir, None

    checkpoint_out = lane_dir / "checkpoint.npz"
    train_metrics = run_train(
        data_files=f"{generic_bootstrap},{random_teacher},{mined_replay}",
        replay_weights=replay_weights,
        init_checkpoint=str(init_checkpoint),
        out=str(checkpoint_out),
        top_k_dir=str(lane_dir),
        epochs=epochs,
        seed=seed,
    )
    export_checkpoint(
        checkpoint_path=str(epoch_checkpoint_path),
        out_dir=str(export_dir),
        version=lane_name,
        policy_loss=float((train_metrics or {}).get("policy_loss", 0.0)),
        value_loss=float((train_metrics or {}).get("value_loss", 0.0)),
    )
    write_json(train_metrics_path, train_metrics)
    return epoch_checkpoint_path, export_dir, train_metrics


def classify_run(
    *,
    promoted_row: dict[str, Any],
    weighted_rows: list[dict[str, Any]],
    heldout_reports_available: bool,
) -> str:
    promoted_large = promoted_row.get("large_budget_results", {})
    promoted_384 = float(
        promoted_large.get(STANDARD_BUDGET, {}).get("ds", float("-inf"))
    )
    promoted_1200_1200 = float(
        promoted_large.get("1200:1200", {}).get("ds", float("-inf"))
    )
    promoted_1200_256 = float(
        promoted_large.get("1200:256", {}).get("ds", float("-inf"))
    )
    promoted_heldout = promoted_row.get("heldout_summary", {})
    promoted_heldout_mean = promoted_heldout.get("mean_ds_384_256")

    best_weighted = max(
        weighted_rows,
        key=lambda row: float(
            row.get("large_budget_results", {})
            .get(STANDARD_BUDGET, {})
            .get("ds", float("-inf"))
        ),
    )
    best_large = best_weighted.get("large_budget_results", {})
    best_384 = float(best_large.get(STANDARD_BUDGET, {}).get("ds", float("-inf")))
    best_heldout_mean = best_weighted.get("heldout_summary", {}).get("mean_ds_384_256")
    heldout_delta = None
    if promoted_heldout_mean is not None and best_heldout_mean is not None:
        heldout_delta = float(best_heldout_mean) - float(promoted_heldout_mean)

    weighted_large_scores = [
        float(
            row.get("large_budget_results", {})
            .get(STANDARD_BUDGET, {})
            .get("ds", float("-inf"))
        )
        for row in weighted_rows
    ]
    weighted_heldout_scores = [
        row.get("heldout_summary", {}).get("mean_ds_384_256") for row in weighted_rows
    ]
    max_shift = max(
        float(
            row.get("mined_state_policy_shift", {}).get(
                "top1_changed_rate_vs_promoted_current", 0.0
            )
        )
        for row in weighted_rows
    )

    for row in weighted_rows:
        row_large = row.get("large_budget_results", {})
        row_384 = float(row_large.get(STANDARD_BUDGET, {}).get("ds", float("-inf")))
        row_1200_1200 = float(row_large.get("1200:1200", {}).get("ds", float("-inf")))
        row_1200_256 = float(row_large.get("1200:256", {}).get("ds", float("-inf")))
        row_shift = float(
            row.get("mined_state_policy_shift", {}).get(
                "top1_changed_rate_vs_promoted_current", 0.0
            )
        )
        row_heldout_mean = row.get("heldout_summary", {}).get("mean_ds_384_256")
        row_heldout_delta = None
        if promoted_heldout_mean is not None and row_heldout_mean is not None:
            row_heldout_delta = float(row_heldout_mean) - float(promoted_heldout_mean)
        if (
            row_shift > 0.10
            and row_384 - promoted_384 >= 0.03
            and row_heldout_delta is not None
            and row_heldout_delta >= 0.02
            and row_1200_1200 - promoted_1200_1200 >= -0.03
            and row_1200_256 - promoted_1200_256 >= -0.03
        ):
            return "replay_weight_underpowered"

    if best_384 > promoted_384:
        p0 = float(best_large.get(STANDARD_BUDGET, {}).get("p0_score", 0.0))
        p1 = float(best_large.get(STANDARD_BUDGET, {}).get("p1_score", 0.0))
        promoted_p0 = float(
            promoted_large.get(STANDARD_BUDGET, {}).get("p0_score", 0.0)
        )
        promoted_p1 = float(
            promoted_large.get(STANDARD_BUDGET, {}).get("p1_score", 0.0)
        )
        imbalance_worse = abs(p0 - p1) > (abs(promoted_p0 - promoted_p1) + 0.05)
        if (heldout_delta is None or heldout_delta <= 0.0) or imbalance_worse:
            return "weighted_replay_overfit"

    if max_shift > 0.10:
        by_weight_e2 = [
            (
                int(row.get("mined_weight", 0)),
                float(
                    row.get("large_budget_results", {})
                    .get(STANDARD_BUDGET, {})
                    .get("ds", float("-inf"))
                ),
            )
            for row in weighted_rows
            if int(row.get("epochs", 0)) == 2
        ]
        by_weight_e2.sort(key=lambda item: item[0])
        if by_weight_e2 and all(
            score < promoted_384 for _weight, score in by_weight_e2
        ):
            non_improving = all(
                right_score <= left_score
                for (_left_weight, left_score), (_right_weight, right_score) in zip(
                    by_weight_e2, by_weight_e2[1:]
                )
            )
            if non_improving:
                return "mined_targets_harmful"

    weight16_rows = [
        row for row in weighted_rows if int(row.get("mined_weight", 0)) == 16
    ]
    weight16_best_shift = max(
        (
            float(
                row.get("mined_state_policy_shift", {}).get(
                    "top1_changed_rate_vs_promoted_current", 0.0
                )
            )
            for row in weight16_rows
        ),
        default=0.0,
    )
    if weight16_best_shift < 0.10 and best_384 <= promoted_384:
        return "update_still_too_weak"

    if all(score < promoted_384 for score in weighted_large_scores):
        if not heldout_reports_available:
            return "no_weighted_candidate"
        non_null_heldout = [
            score for score in weighted_heldout_scores if score is not None
        ]
        if (
            non_null_heldout
            and promoted_heldout_mean is not None
            and all(
                float(score) < float(promoted_heldout_mean)
                for score in non_null_heldout
            )
        ):
            return "no_weighted_candidate"

    return "inconclusive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default="/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default=EXPECTED_PROMOTED_WEIGHTS_SHA256,
    )
    parser.add_argument(
        "--init-checkpoint",
        default="/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz",
    )
    parser.add_argument(
        "--expected-init-checkpoint-sha256",
        default=EXPECTED_INIT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--generic-bootstrap",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl",
    )
    parser.add_argument(
        "--random-teacher",
        default="/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
    )
    parser.add_argument(
        "--mined-replay",
        default=(
            "/tmp/azlite_promoted_current_opening_puct_disagreement/"
            "opening_puct_disagreement_replay.jsonl"
        ),
    )
    parser.add_argument(
        "--disagreement-audit",
        default=(
            "/tmp/azlite_promoted_current_opening_puct_disagreement/"
            "opening_state_disagreement_audit.json"
        ),
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument(
        "--fixed-large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument(
        "--heldout-suites",
        default=(
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl,"
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl,"
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl"
        ),
    )
    parser.add_argument("--mined-weights", default="4,8,16")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_artifact = Path(args.current)
    init_checkpoint = Path(args.init_checkpoint)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    mined_replay = Path(args.mined_replay)
    disagreement_audit = Path(args.disagreement_audit)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    mined_weights = parse_int_list(args.mined_weights)
    heldout_suite_paths = [
        Path(item.strip())
        for item in str(args.heldout_suites).split(",")
        if item.strip()
    ]
    heldout_suite_paths = [path for path in heldout_suite_paths if path.is_file()]
    pr122_ref_checkpoint = (
        PR122_WORKDIR / "opening_disagreement_policy_head_e2" / "checkpoint_epoch2.npz"
    )
    pr122_ref_artifact = (
        PR122_WORKDIR
        / "opening_disagreement_policy_head_e2"
        / "artifact_opening_disagreement_policy_head_e2"
    )
    pr122_summary_path = PR122_WORKDIR / "summary_metrics.json"

    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(
        current_artifact / "metadata.json", "current artifact metadata"
    )
    require_existing_file(init_checkpoint, "init checkpoint")
    require_existing_file(generic_bootstrap, "generic bootstrap replay")
    require_existing_file(random_teacher, "random teacher replay")
    require_existing_file(mined_replay, "mined disagreement replay")
    require_existing_file(disagreement_audit, "disagreement audit")
    require_existing_file(fixed_large_suite, "fixed large suite")
    require_existing_file(medium_suite, "medium suite")

    mined_replay_rows = read_jsonl(mined_replay)
    if len(mined_replay_rows) != EXPECTED_MINED_ROWS:
        raise RuntimeError(
            "mined disagreement replay must contain exactly "
            f"{EXPECTED_MINED_ROWS} rows, got {len(mined_replay_rows)}"
        )

    selected_audit_states, audit_rows_by_hash = load_selected_audit_states(
        audit_path=disagreement_audit,
        mined_rows=mined_replay_rows,
    )
    mined_state_rows = build_mined_state_rows(mined_replay_rows, audit_rows_by_hash)
    audit = load_json(disagreement_audit)
    selection_rows = (
        audit.get("tables", {}).get("selection_summary", {}).get("selected_rows")
    )
    if selection_rows != EXPECTED_MINED_ROWS:
        raise RuntimeError(
            "disagreement audit selected_rows must equal mined replay rows: "
            f"expected {EXPECTED_MINED_ROWS}, got {selection_rows}"
        )

    generic_rows = int(build_input_summary(generic_bootstrap).get("rows", 0))
    random_rows = int(build_input_summary(random_teacher).get("rows", 0))
    mined_rows = len(mined_replay_rows)

    input_summary = {
        "current_artifact_weights": verify_expected_hash(
            current_artifact / "weights.json",
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact_metadata": build_input_summary(
            current_artifact / "metadata.json"
        ),
        "init_checkpoint": verify_expected_hash(
            init_checkpoint,
            args.expected_init_checkpoint_sha256,
            "init checkpoint",
        ),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "mined_replay": build_input_summary(mined_replay),
        "disagreement_audit": build_input_summary(disagreement_audit),
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suite_paths
        },
    }

    pr122_train_metrics: dict[str, Any] | None = None
    if pr122_summary_path.is_file():
        pr122_summary = load_json(pr122_summary_path)
        for row in pr122_summary.get("candidates", []):
            if row.get("candidate") == "opening_disagreement_policy_head_e2":
                pr122_train_metrics = row
                break

    lanes: list[dict[str, Any]] = [
        {
            "name": "promoted_current_ref",
            "report_candidate_name": current_artifact.name,
            "epochs": 0,
            "mined_weight": 0,
            "effective_mined_fraction": 0.0,
            "checkpoint_path": str(init_checkpoint),
            "artifact_dir": str(current_artifact),
            "trainable_scope": "none",
            "source": "promoted_current_ref",
            "replay_weights": None,
            "train_metrics": None,
        }
    ]

    if (
        pr122_ref_checkpoint.is_file()
        and not pr122_ref_artifact.joinpath("weights.json").is_file()
    ):
        export_checkpoint(
            checkpoint_path=str(pr122_ref_checkpoint),
            out_dir=str(pr122_ref_artifact),
            version="pr122_weight1_e2_ref",
            policy_loss=float(
                (pr122_train_metrics or {}).get("policy_loss", 0.0) or 0.0
            ),
            value_loss=float((pr122_train_metrics or {}).get("value_loss", 0.0) or 0.0),
        )

    if (
        pr122_ref_checkpoint.is_file()
        and pr122_ref_artifact.joinpath("weights.json").is_file()
    ):
        lanes.append(
            {
                "name": "pr122_weight1_e2_ref",
                "report_candidate_name": pr122_ref_artifact.name,
                "epochs": 2,
                "mined_weight": 1,
                "effective_mined_fraction": effective_sampled_fraction(
                    generic_rows=generic_rows,
                    random_rows=random_rows,
                    mined_rows=mined_rows,
                    mined_weight=1,
                ),
                "checkpoint_path": str(pr122_ref_checkpoint),
                "artifact_dir": str(pr122_ref_artifact),
                "trainable_scope": "policy_head",
                "source": "pr122_weight1_reference",
                "replay_weights": "4,1,1",
                "train_metrics": pr122_train_metrics,
            }
        )
    elif not args.skip_training:
        ref_dir = workdir / "pr122_weight1_e2_ref"
        checkpoint_path, artifact_dir, train_metrics = train_or_reuse_lane(
            lane_dir=ref_dir,
            lane_name="pr122_weight1_e2_ref",
            replay_weights="4,1,1",
            init_checkpoint=init_checkpoint,
            generic_bootstrap=generic_bootstrap,
            random_teacher=random_teacher,
            mined_replay=mined_replay,
            epochs=2,
            seed=args.seed,
            skip_training=args.skip_training,
        )
        lanes.append(
            {
                "name": "pr122_weight1_e2_ref",
                "report_candidate_name": artifact_dir.name,
                "epochs": 2,
                "mined_weight": 1,
                "effective_mined_fraction": effective_sampled_fraction(
                    generic_rows=generic_rows,
                    random_rows=random_rows,
                    mined_rows=mined_rows,
                    mined_weight=1,
                ),
                "checkpoint_path": str(checkpoint_path),
                "artifact_dir": str(artifact_dir),
                "trainable_scope": "policy_head",
                "source": "pr122_weight1_retrained_fallback",
                "replay_weights": "4,1,1",
                "train_metrics": train_metrics,
            }
        )

    for mined_weight in mined_weights:
        for epochs in (1, 2):
            lane_name = f"disagreement_w{mined_weight}_policy_head_e{epochs}"
            checkpoint_path, artifact_dir, train_metrics = train_or_reuse_lane(
                lane_dir=workdir / lane_name,
                lane_name=lane_name,
                replay_weights=f"4,1,{mined_weight}",
                init_checkpoint=init_checkpoint,
                generic_bootstrap=generic_bootstrap,
                random_teacher=random_teacher,
                mined_replay=mined_replay,
                epochs=epochs,
                seed=args.seed,
                skip_training=args.skip_training,
            )
            lanes.append(
                {
                    "name": lane_name,
                    "report_candidate_name": artifact_dir.name,
                    "epochs": epochs,
                    "mined_weight": mined_weight,
                    "effective_mined_fraction": effective_sampled_fraction(
                        generic_rows=generic_rows,
                        random_rows=random_rows,
                        mined_rows=mined_rows,
                        mined_weight=mined_weight,
                    ),
                    "checkpoint_path": str(checkpoint_path),
                    "artifact_dir": str(artifact_dir),
                    "trainable_scope": "policy_head",
                    "source": "opening_puct_disagreement_weight_ablation",
                    "replay_weights": f"4,1,{mined_weight}",
                    "train_metrics": train_metrics,
                }
            )

    candidate_paths = ",".join(str(lane["artifact_dir"]) for lane in lanes)
    medium_report: dict[str, Any] | None = None
    large_report: dict[str, Any] | None = None
    heldout_reports: dict[str, dict[str, Any]] = {}
    if not args.skip_eval:
        medium_report_path = (
            workdir / "eval_medium" / "temperature_benchmark_report.json"
        )
        if medium_report_path.is_file():
            medium_report = load_json(medium_report_path)
        else:
            medium_report = run_opening_suite_benchmark(
                workdir=str(workdir / "eval_medium"),
                suite=str(medium_suite),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.timeout,
            )
        large_report_path = workdir / "eval_large" / "temperature_benchmark_report.json"
        if large_report_path.is_file():
            large_report = load_json(large_report_path)
        else:
            large_report = run_opening_suite_benchmark(
                workdir=str(workdir / "eval_large"),
                suite=str(fixed_large_suite),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.timeout,
            )
        for suite_path in heldout_suite_paths:
            report_path = (
                workdir
                / f"eval_{suite_path.stem}"
                / "temperature_benchmark_report.json"
            )
            if report_path.is_file():
                heldout_reports[suite_path.stem] = load_json(report_path)
            else:
                heldout_reports[suite_path.stem] = run_opening_suite_benchmark(
                    workdir=str(workdir / f"eval_{suite_path.stem}"),
                    suite=str(suite_path),
                    current=str(current_artifact),
                    candidates=candidate_paths,
                    budget_pairs=args.budget_pairs,
                    games_per_opening=args.games_per_opening,
                    seed=args.seed,
                    workers=args.workers,
                    timeout=args.timeout,
                )

    gate_reports: dict[str, dict[str, Any]] = {}
    gate_targets = ["promoted_current_ref"]
    if large_report is not None:
        promoted_ref_score = candidate_standard_ds(large_report, current_artifact.name)
        for lane in lanes:
            if lane["name"] == "promoted_current_ref":
                continue
            if not str(lane["name"]).startswith("disagreement_w"):
                continue
            score = candidate_standard_ds(
                large_report, str(lane["report_candidate_name"])
            )
            if score >= promoted_ref_score + 0.01:
                gate_targets.append(str(lane["name"]))
    if not args.skip_gate and large_report is not None:
        gate_dir = workdir / "eval_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        for lane in lanes:
            lane_name = str(lane["name"])
            if lane_name not in gate_targets:
                continue
            gate_report_path = gate_dir / f"{lane_name}_default_gate.json"
            if gate_report_path.is_file():
                gate_reports[lane_name] = load_json(gate_report_path)
            else:
                gate_reports[lane_name] = run_default_gate(
                    candidate_path=str(lane["artifact_dir"]),
                    current_path=str(current_artifact),
                    out=str(gate_report_path),
                    seed=args.seed,
                    workers=args.workers,
                )

    summary_candidates: list[dict[str, Any]] = []
    for lane in lanes:
        checkpoint_path = Path(str(lane["checkpoint_path"]))
        artifact_dir = Path(str(lane["artifact_dir"]))
        delta_norm, relative_delta_pct = compute_param_delta_norm(
            checkpoint_path, init_checkpoint
        )
        mined_shift = mined_state_policy_shift(artifact_dir, mined_state_rows)
        row: dict[str, Any] = {
            "candidate": lane["name"],
            "report_candidate_name": lane["report_candidate_name"],
            "source": lane["source"],
            "epochs": lane["epochs"],
            "mined_weight": lane["mined_weight"],
            "effective_mined_fraction": lane["effective_mined_fraction"],
            "checkpoint_path": str(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
            "delta_norm_vs_promoted_e1": delta_norm,
            "relative_delta_pct_vs_promoted_e1": relative_delta_pct,
            "mined_state_policy_shift": mined_shift,
        }
        train_metrics = lane.get("train_metrics")
        if isinstance(train_metrics, dict):
            row["policy_loss"] = train_metrics.get("policy_loss")
            row["value_loss"] = train_metrics.get("value_loss")
            row["validation_loss"] = train_metrics.get(
                "best_val_loss"
            ) or train_metrics.get("validation_loss")
        if medium_report is not None:
            candidate_report = find_candidate_report(
                medium_report, str(lane["report_candidate_name"])
            )
            if candidate_report is not None:
                row["medium_budget_results"] = budget_results_by_pair(candidate_report)
        if large_report is not None:
            candidate_report = find_candidate_report(
                large_report, str(lane["report_candidate_name"])
            )
            if candidate_report is not None:
                row["large_budget_results"] = budget_results_by_pair(candidate_report)
        if heldout_reports:
            row["heldout_summary"] = heldout_summary(
                heldout_reports, str(lane["report_candidate_name"])
            )
        if lane["name"] in gate_reports:
            row["default_gate"] = {
                "classification": gate_reports[lane["name"]].get("classification")
            }
        summary_candidates.append(row)

    promoted_row = next(
        row for row in summary_candidates if row["candidate"] == "promoted_current_ref"
    )
    weighted_rows = [
        row
        for row in summary_candidates
        if str(row["candidate"]).startswith("disagreement_w")
    ]
    classification = classify_run(
        promoted_row=promoted_row,
        weighted_rows=weighted_rows,
        heldout_reports_available=bool(heldout_reports),
    )

    summary = {
        "schema": "azlite_promoted_current_opening_puct_disagreement_weight_ablation_v1",
        "status": "completed",
        "classification": classification,
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "budget_pairs": args.budget_pairs.split(","),
        "games_per_opening": args.games_per_opening,
        "mined_weights": mined_weights,
        "inputs": input_summary,
        "guardrails": {
            "new_self_play": False,
            "new_disagreement_mining": False,
            "promotion": False,
            "overwrite_current": False,
            "lr_change": False,
            "full_network_training": False,
            "last_block_policy_training": False,
            "architecture_change": False,
            "residual_v4": False,
            "seed_sweep": False,
            "threshold_change": False,
        },
        "reused_disagreement_audit": {
            "path": str(disagreement_audit),
            "selected_state_rows": len(selected_audit_states),
            "duplicate_trajectory_count": audit.get("tables", {})
            .get("collection_metadata", {})
            .get("duplicate_trajectory_count"),
            "duplicate_trajectory_rate": audit.get("tables", {})
            .get("collection_metadata", {})
            .get("duplicate_trajectory_rate"),
        },
        "mined_replay_row_count": len(mined_replay_rows),
        "effective_replay_fractions": {
            f"weight_{weight}": effective_sampled_fraction(
                generic_rows=generic_rows,
                random_rows=random_rows,
                mined_rows=mined_rows,
                mined_weight=weight,
            )
            for weight in [1, *mined_weights]
        },
        "gate_targets": gate_targets,
        "candidates": summary_candidates,
    }
    write_json(workdir / "summary_metrics.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
