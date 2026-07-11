#!/usr/bin/env python3
"""Current-init policy-only distillation preflight.

Evaluates whether current-init policy-only teacher distillation is useful under
runtime search even when the raw teacher top-1 gate is below 0.60.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.export_artifact import sha256_file  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    benchmark_budget_results,
    candidate_gate_preserved,
    find_candidate_report,
    run_default_gate,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    policy_entropy,
    read_jsonl,
    search_policy_from_visits,
    top_policy_move,
)
from ml.alphazero_lite.run_search_teacher_distillation_value_ablation import (  # noqa: E402
    aggregate_suite_summary,
    build_ds_orientation_audit,
)
from ml.alphazero_lite.run_search_teacher_student_preflight import (  # noqa: E402
    DEFAULT_GATE_BUDGETS,
    StudentSpec,
    aggregate_suite_metrics,
    compute_bootstrap_comparisons,
    count_jsonl_rows,
    current_checkpoint_path,
    load_json,
    masked_policy,
    parse_csv_paths,
    require_existing_file,
    train_student_candidate,
    verify_expected_hash,
    write_json,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.train import select_device  # noqa: E402


SUMMARY_SCHEMA = "azlite_current_init_policy_only_distillation_preflight_v1"
SUMMARY_FILENAME = "summary_metrics.json"
DS_AUDIT_FILENAME = "ds_orientation_audit.json"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-current-init-policy-only-distillation-preflight-results.md"
)
INPUT_ENCODING = "kalah_v3"
MODEL_TYPE = "residual_v3"
PRIMARY_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
SUITE_BUDGETS = ("384:256", "768:256", "768:768", "1200:1200", "1200:256", "256:768")
SEARCH_PROBE_BUCKET_CAP = 64
BOOTSTRAP_SAMPLES = 10000
ENTROPY_FLOOR = 1.0


@dataclass(frozen=True)
class LaneSpec:
    lane_key: str
    candidate_name: str
    epochs: int


LANE_SPECS = {
    "pr152_e2": LaneSpec("pr152_e2", "pr152_current_init_policy_only_e2", 2),
    "e1_repro": LaneSpec("e1_repro", "current_init_policy_only_e1_repro", 1),
    "e2_repro": LaneSpec("e2_repro", "current_init_policy_only_e2_repro", 2),
    "e4": LaneSpec("e4", "current_init_policy_only_e4", 4),
}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def log_progress(message: str) -> None:
    print(f"[current-init-policy-only] {message}", flush=True)


def parse_csv_values(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_input_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if path.suffix == ".jsonl":
        summary["rows"] = count_jsonl_rows(path)
    return summary


def matching_candidate_report(
    *,
    report: dict[str, Any],
    candidate_name: str,
    artifact_path: Path,
    current_artifact: Path,
) -> dict[str, Any] | None:
    expected_sha = sha256_file(artifact_path / "weights.json")
    for lookup_name in (candidate_name, artifact_path.name, "current", "current_ref"):
        candidate_report = find_candidate_report(report, lookup_name)
        if candidate_report is None:
            continue
        candidate_path = str(candidate_report.get("candidate_path", ""))
        candidate_sha = str(candidate_report.get("candidate_sha256", ""))
        if candidate_sha and candidate_sha != expected_sha:
            continue
        if candidate_path and candidate_path not in {
            str(artifact_path),
            str(current_artifact),
        }:
            continue
        return candidate_report
    return None


def run_suite_evaluations(
    *,
    workdir: Path,
    current_artifact: Path,
    candidate_artifacts: list[tuple[str, Path]],
    suite_paths: list[Path],
    budget_pairs: str,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    suite_rows: dict[str, Any] = {}
    candidates_csv = ",".join(str(path) for _name, path in candidate_artifacts)
    for suite_path in suite_paths:
        suite_name = suite_path.stem
        suite_workdir = workdir / "suite_eval" / suite_name
        suite_workdir.mkdir(parents=True, exist_ok=True)
        report_path = suite_workdir / "temperature_benchmark_report.json"
        report = None
        if report_path.is_file():
            cached = load_json(report_path)
            all_present = True
            for candidate_name, artifact_path in candidate_artifacts:
                if (
                    matching_candidate_report(
                        report=cached,
                        candidate_name=candidate_name,
                        artifact_path=artifact_path,
                        current_artifact=current_artifact,
                    )
                    is None
                ):
                    all_present = False
                    break
            if all_present:
                report = cached
                log_progress(f"reusing suite benchmark suite={suite_name}")
        if report is None:
            log_progress(f"running suite benchmark suite={suite_name}")
            report = run_opening_suite_benchmark(
                workdir=str(suite_workdir),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidates_csv,
                budget_pairs=budget_pairs,
                games_per_opening=2,
                seed=seed,
                workers=workers,
                timeout=7200,
            )
        suite_rows[suite_name] = {
            "suite_name": suite_name,
            "suite_path": str(suite_path),
            "suite_sha256": sha256_file(suite_path),
            "suite_size": count_jsonl_rows(suite_path),
            "candidates": {},
        }
        for candidate_name, artifact_path in candidate_artifacts:
            candidate_report = matching_candidate_report(
                report=report,
                candidate_name=candidate_name,
                artifact_path=artifact_path,
                current_artifact=current_artifact,
            )
            if candidate_report is None:
                raise RuntimeError(
                    f"missing benchmark report for {candidate_name} in {suite_name}"
                )
            suite_rows[suite_name]["candidates"][candidate_name] = {
                "candidate_path": str(artifact_path),
                "candidate_sha256": candidate_report.get("candidate_sha256"),
                "budget_results": benchmark_budget_results(candidate_report),
            }
    return suite_rows


def load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def preservation_keys(checkpoint: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    policy_keys = [
        key
        for key in checkpoint
        if key in {"w_policy", "b_policy", "w_policy_hidden", "b_policy_hidden"}
    ]
    preserved = sorted(key for key in checkpoint if key not in policy_keys)
    return preserved, sorted(policy_keys)


def compare_preserved_weights(
    current_checkpoint: Path, candidate_checkpoint: Path
) -> dict[str, Any]:
    current_arrays = load_checkpoint_arrays(current_checkpoint)
    candidate_arrays = load_checkpoint_arrays(candidate_checkpoint)
    preserved_keys, policy_keys = preservation_keys(current_arrays)
    if set(current_arrays) != set(candidate_arrays):
        return {
            "passes": False,
            "reason": "checkpoint key mismatch",
            "checked_keys": preserved_keys,
            "policy_keys": policy_keys,
        }
    max_abs_diff = 0.0
    changed_keys: list[str] = []
    for key in preserved_keys:
        diff = float(np.max(np.abs(candidate_arrays[key] - current_arrays[key])))
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 0.0:
            changed_keys.append(key)
    return {
        "passes": max_abs_diff == 0.0,
        "max_abs_diff": max_abs_diff,
        "changed_keys": changed_keys,
        "checked_keys": preserved_keys,
        "policy_keys": policy_keys,
    }


def stratified_probe_rows(
    rows: list[dict[str, Any]], *, seed: int, cap_per_bucket: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["budget_context"]),
                str(row["phase"]),
                str(row["seat_context"]),
            )
        ].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:cap_per_bucket])
    return selected


def evaluate_raw_outputs(
    *, artifact_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    outputs: list[dict[str, Any]] = []
    for row in rows:
        game = KalahGame.from_state(row["raw_state"])
        raw_policy, raw_value = evaluator.evaluate(game)
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        masked = masked_policy(raw_policy.tolist(), legal_moves)
        outputs.append(
            {
                "policy": masked,
                "value": float(raw_value),
                "top_move": top_policy_move(masked, legal_moves),
            }
        )
    return {"artifact": str(artifact_path), "outputs": outputs}


def evaluate_policy_probe_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate_outputs: list[dict[str, Any]],
    current_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    top1_matches = 0
    legal_failures = 0
    kls: list[float] = []
    entropies: list[float] = []
    value_diffs: list[float] = []
    by_budget: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_phase: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_seat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row, candidate, current in zip(
        rows, candidate_outputs, current_outputs, strict=True
    ):
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        candidate_policy = candidate["policy"]
        candidate_top = candidate["top_move"]
        teacher_top = top_policy_move(row["teacher_puct_policy"], legal_moves)
        current_top = current["top_move"]
        if any(
            candidate_policy[move] > 1e-7
            for move in range(6)
            if move not in legal_moves
        ):
            legal_failures += 1
        top1_match = 1.0 if candidate_top == teacher_top else 0.0
        raw_change = 1.0 if candidate_top != current_top else 0.0
        if top1_match:
            top1_matches += 1
        kl_value = float(
            np.sum(
                np.asarray(row["teacher_puct_policy"], dtype=np.float64)[legal_moves]
                * np.log(
                    np.clip(
                        np.asarray(row["teacher_puct_policy"], dtype=np.float64)[
                            legal_moves
                        ],
                        1e-8,
                        1.0,
                    )
                    / np.clip(
                        np.asarray(candidate_policy, dtype=np.float64)[legal_moves],
                        1e-8,
                        1.0,
                    )
                )
            )
        )
        kls.append(kl_value)
        entropy = policy_entropy(candidate_policy, legal_moves)
        entropies.append(entropy)
        value_diffs.append(abs(float(candidate["value"]) - float(current["value"])))
        for bucket_name, bucket_value, bucket_store in (
            ("budget", str(row["budget_context"]), by_budget),
            ("phase", str(row["phase"]), by_phase),
            ("seat", str(row["seat_context"]), by_seat),
        ):
            del bucket_name
            bucket_store[bucket_value]["top1_match"].append(top1_match)
            bucket_store[bucket_value]["raw_change"].append(raw_change)
    mean_entropy = float(statistics.fmean(entropies)) if entropies else 0.0
    return {
        "rows": len(rows),
        "teacher_puct_top1_agreement": top1_matches / max(len(rows), 1),
        "policy_kl": float(statistics.fmean(kls)) if kls else 0.0,
        "candidate_entropy": mean_entropy,
        "legal_failures": legal_failures,
        "changed_raw_top1_rate_vs_current": float(
            statistics.fmean(
                1.0 if candidate["top_move"] != current["top_move"] else 0.0
                for candidate, current in zip(
                    candidate_outputs, current_outputs, strict=True
                )
            )
        )
        if rows
        else 0.0,
        "max_value_diff_vs_current": max(value_diffs) if value_diffs else 0.0,
        "mean_value_diff_vs_current": float(statistics.fmean(value_diffs))
        if value_diffs
        else 0.0,
        "root_mae_vs_current": float(statistics.fmean(value_diffs))
        if value_diffs
        else 0.0,
        "top1_changes_by_budget_context": {
            key: {
                "rows": len(values["raw_change"]),
                "changed_rate": float(statistics.fmean(values["raw_change"])),
                "teacher_top1_agreement": float(statistics.fmean(values["top1_match"])),
            }
            for key, values in sorted(by_budget.items())
        },
        "top1_changes_by_phase": {
            key: {
                "rows": len(values["raw_change"]),
                "changed_rate": float(statistics.fmean(values["raw_change"])),
                "teacher_top1_agreement": float(statistics.fmean(values["top1_match"])),
            }
            for key, values in sorted(by_phase.items())
        },
        "top1_changes_by_seat": {
            key: {
                "rows": len(values["raw_change"]),
                "changed_rate": float(statistics.fmean(values["raw_change"])),
                "teacher_top1_agreement": float(statistics.fmean(values["top1_match"])),
            }
            for key, values in sorted(by_seat.items())
        },
    }


def search_probe_cache_path(workdir: Path, candidate_name: str) -> Path:
    return workdir / "search_probe_cache" / f"{candidate_name}.jsonl"


def evaluate_search_outputs(
    *,
    workdir: Path,
    candidate_name: str,
    artifact_path: Path,
    rows: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> list[dict[str, Any]]:
    cache_path = search_probe_cache_path(workdir, candidate_name)
    if cache_path.is_file():
        cached = read_jsonl(cache_path)
        if len(cached) == len(rows):
            return cached
    evaluator = ArtifactEvaluator(artifact_path)
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    outputs: list[dict[str, Any]] = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            game = KalahGame.from_state(row["raw_state"])
            effective_c_puct = resolve_budget_cpuct(
                schedule=cpuct_schedule,
                challenger_simulations=int(row["challenger_simulations"]),
                current_simulations=int(row["current_simulations"]),
                default_c_puct=default_c_puct,
            )
            result = evaluate_artifact_position(
                evaluator=evaluator,
                state=game.to_state(),
                simulations=int(row["challenger_simulations"]),
                seed=seed + index,
                c_puct=float(effective_c_puct),
                search_options=search_options,
                ablation_mode="full",
            )
            legal_moves = [int(move) for move in result["legal_moves"]]
            search_policy = search_policy_from_visits(result["visits"], legal_moves)
            selected_move = int(result["selected_move"])
            selected_visit_share = (
                float(search_policy[selected_move])
                if selected_move in legal_moves
                else 0.0
            )
            payload = {
                "state_hash": str(row["state_hash"]),
                "budget_context": str(row["budget_context"]),
                "seat_context": str(row["seat_context"]),
                "selected_move": selected_move,
                "search_policy": search_policy,
                "selected_visit_share": selected_visit_share,
            }
            outputs.append(payload)
            handle.write(json.dumps(payload) + "\n")
    return outputs


def evaluate_search_probe_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate_outputs: list[dict[str, Any]],
    current_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    agreement_with_teacher: list[float] = []
    changed_vs_current: list[float] = []
    visit_kls: list[float] = []
    selected_share_deltas: list[float] = []
    by_budget: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    p0_p1_384 = defaultdict(list)
    for row, candidate, current in zip(
        rows, candidate_outputs, current_outputs, strict=True
    ):
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        candidate_policy = candidate["search_policy"]
        current_policy = current["search_policy"]
        teacher_agree = (
            1.0
            if int(candidate["selected_move"]) == int(row["teacher_selected_move"])
            else 0.0
        )
        changed_rate = (
            1.0
            if int(candidate["selected_move"]) != int(current["selected_move"])
            else 0.0
        )
        agreement_with_teacher.append(teacher_agree)
        changed_vs_current.append(changed_rate)
        visit_kls.append(
            float(
                np.sum(
                    np.asarray(current_policy, dtype=np.float64)[legal_moves]
                    * np.log(
                        np.clip(
                            np.asarray(current_policy, dtype=np.float64)[legal_moves],
                            1e-8,
                            1.0,
                        )
                        / np.clip(
                            np.asarray(candidate_policy, dtype=np.float64)[legal_moves],
                            1e-8,
                            1.0,
                        )
                    )
                )
            )
        )
        selected_share_deltas.append(
            float(candidate["selected_visit_share"])
            - float(current["selected_visit_share"])
        )
        budget = str(row["budget_context"])
        by_budget[budget]["agreement"].append(teacher_agree)
        by_budget[budget]["changed"].append(changed_rate)
        by_budget[budget]["visit_kl"].append(visit_kls[-1])
        if budget == "384:256":
            p0_p1_384[str(row["seat_context"])].append(teacher_agree)
    return {
        "rows": len(rows),
        "search_selected_move_agreement_with_teacher": float(
            statistics.fmean(agreement_with_teacher)
        )
        if agreement_with_teacher
        else 0.0,
        "search_selected_move_changed_rate_vs_current": float(
            statistics.fmean(changed_vs_current)
        )
        if changed_vs_current
        else 0.0,
        "root_visit_kl_vs_current_ref": float(statistics.fmean(visit_kls))
        if visit_kls
        else 0.0,
        "root_selected_visit_share_delta_vs_current_ref": float(
            statistics.fmean(selected_share_deltas)
        )
        if selected_share_deltas
        else 0.0,
        "changed_move_rate_by_budget_context": {
            key: {
                "rows": len(values["changed"]),
                "search_agreement_with_teacher": float(
                    statistics.fmean(values["agreement"])
                ),
                "changed_rate_vs_current": float(statistics.fmean(values["changed"])),
                "root_visit_kl_vs_current_ref": float(
                    statistics.fmean(values["visit_kl"])
                ),
            }
            for key, values in sorted(by_budget.items())
        },
        "p0_p1_split_384_256": {
            "challenger_player_0": float(
                statistics.fmean(p0_p1_384["challenger_player_0"])
            )
            if p0_p1_384["challenger_player_0"]
            else 0.0,
            "challenger_player_1": float(
                statistics.fmean(p0_p1_384["challenger_player_1"])
            )
            if p0_p1_384["challenger_player_1"]
            else 0.0,
        },
    }


def search_probe_gate(
    *,
    policy_metrics: dict[str, Any],
    search_metrics: dict[str, Any],
    preservation_metrics: dict[str, Any],
    current_policy_metrics: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(policy_metrics.get("legal_failures", 0)) != 0:
        reasons.append("legal failures != 0")
    if not preservation_metrics.get("passes", False):
        reasons.append("value/trunk preservation failed")
    if (
        float(search_metrics.get("search_selected_move_agreement_with_teacher", 0.0))
        < 0.75
    ):
        reasons.append("search agreement with teacher < 0.75")
    if (
        float(search_metrics.get("search_selected_move_changed_rate_vs_current", 0.0))
        > 0.20
    ):
        reasons.append("overall changed search move rate > 0.20")
    for budget in ("768:768", "1200:1200"):
        budget_metrics = search_metrics.get(
            "changed_move_rate_by_budget_context", {}
        ).get(budget, {})
        if float(budget_metrics.get("changed_rate_vs_current", 0.0)) > 0.12:
            reasons.append(f"{budget} changed search move rate > 0.12")
    if (
        float(policy_metrics.get("policy_kl", 0.0))
        > float(current_policy_metrics.get("policy_kl", 0.0)) + 1e-12
    ):
        reasons.append("KL worse than current_ref")
    current_entropy = float(current_policy_metrics.get("candidate_entropy", 0.0))
    min_entropy = max(ENTROPY_FLOOR, 0.75 * current_entropy)
    if float(policy_metrics.get("candidate_entropy", 0.0)) < min_entropy:
        reasons.append("entropy collapsed")
    return {"passed": not reasons, "reasons": reasons}


def build_suite_orientation_table(
    *, suite_rows: dict[str, Any], suite_name: str, candidate_names: list[str]
) -> dict[str, Any]:
    suite = suite_rows[suite_name]
    table: dict[str, Any] = {"suite": suite_name, "budgets": {}}
    for budget in SUITE_BUDGETS:
        current_budget = suite["candidates"]["current_ref"]["budget_results"][budget]
        current_ds = float(current_budget["ds"])
        budget_rows: dict[str, Any] = {}
        for candidate_name in candidate_names:
            budget_result = suite["candidates"][candidate_name]["budget_results"][
                budget
            ]
            candidate_ds = float(budget_result["ds"])
            cand_minus = candidate_ds - current_ds
            cur_minus = current_ds - candidate_ds
            budget_rows[candidate_name] = {
                "raw_ds_mean": candidate_ds,
                "candidate_minus_current": cand_minus,
                "current_minus_candidate": cur_minus,
                "disadvantaged_seat_score": float(
                    budget_result.get("disadvantaged_seat_score", 0.0)
                ),
                "orientation_consistent": abs(cand_minus + cur_minus) <= 1e-12,
            }
        table["budgets"][budget] = budget_rows
    return table


def build_orientation_audit(
    *,
    medium_rows: dict[str, Any] | None,
    fixed_large_rows: dict[str, Any] | None,
    heldout_rows: dict[str, Any] | None,
    candidate_names: list[str],
    ds_audit_reference: dict[str, Any],
    bootstrap_tables: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "seed": seed,
        "reference_reuse": ds_audit_reference,
        "tables": {},
        "bootstrap_orientation": {},
        "passed": True,
        "errors": [],
    }
    if medium_rows is not None:
        medium_name = next(iter(medium_rows))
        audit["tables"]["medium_eval"] = build_suite_orientation_table(
            suite_rows=medium_rows,
            suite_name=medium_name,
            candidate_names=sorted(medium_rows[medium_name]["candidates"]),
        )
    if fixed_large_rows is not None:
        fixed_large_name = next(iter(fixed_large_rows))
        audit["tables"]["large_eval"] = build_suite_orientation_table(
            suite_rows=fixed_large_rows,
            suite_name=fixed_large_name,
            candidate_names=sorted(fixed_large_rows[fixed_large_name]["candidates"]),
        )
    if heldout_rows:
        heldout_candidate_names = sorted(
            next(iter(heldout_rows.values()))["candidates"]
        )
        heldout_summary = aggregate_suite_summary(heldout_rows, heldout_candidate_names)
        audit["tables"]["heldout_mean"] = {
            "budgets": {
                budget: {
                    candidate_name: {
                        "raw_ds_mean": float(
                            heldout_summary[candidate_name]["heldout"][budget][
                                "mean_ds"
                            ]
                        ),
                        "candidate_minus_current": float(
                            heldout_summary[candidate_name]["heldout"][budget][
                                "mean_ds"
                            ]
                        )
                        - float(
                            heldout_summary["current_ref"]["heldout"][budget]["mean_ds"]
                        ),
                        "current_minus_candidate": float(
                            heldout_summary["current_ref"]["heldout"][budget]["mean_ds"]
                        )
                        - float(
                            heldout_summary[candidate_name]["heldout"][budget][
                                "mean_ds"
                            ]
                        ),
                        "disadvantaged_seat_score": None,
                    }
                    for candidate_name in heldout_candidate_names
                }
                for budget in PRIMARY_BUDGETS
            }
        }
    for candidate_name, budgets in bootstrap_tables.items():
        audit["bootstrap_orientation"][candidate_name] = {}
        for budget, metrics in budgets.items():
            ds_metrics = metrics["ds"]
            score_metrics = metrics["disadvantaged_seat_score"]
            ds_ok = (
                abs(
                    float(ds_metrics["candidate_minus_current"]["mean"])
                    + float(ds_metrics["current_minus_candidate"]["mean"])
                )
                <= 1e-12
            )
            score_ok = (
                abs(
                    float(score_metrics["candidate_minus_current"]["mean"])
                    + float(score_metrics["current_minus_candidate"]["mean"])
                )
                <= 1e-12
            )
            audit["bootstrap_orientation"][candidate_name][budget] = {
                "ds_orientation": "candidate_minus_current and current_minus_candidate",
                "disadvantaged_seat_orientation": "candidate_minus_current and current_minus_candidate",
                "ds_consistent": ds_ok,
                "disadvantaged_seat_consistent": score_ok,
                "candidate_minus_current": ds_metrics["candidate_minus_current"],
                "current_minus_candidate": ds_metrics["current_minus_candidate"],
                "candidate_minus_current_disadvantaged_seat": score_metrics[
                    "candidate_minus_current"
                ],
                "current_minus_candidate_disadvantaged_seat": score_metrics[
                    "current_minus_candidate"
                ],
            }
            if not ds_ok or not score_ok:
                audit["passed"] = False
                audit["errors"].append(
                    f"bootstrap orientation mismatch candidate={candidate_name} budget={budget}"
                )
    for table_name, table in audit["tables"].items():
        for budget, candidate_entries in table.get("budgets", {}).items():
            for candidate_name, entry in candidate_entries.items():
                cand_minus = float(entry["candidate_minus_current"])
                cur_minus = float(entry["current_minus_candidate"])
                if abs(cand_minus + cur_minus) > 1e-12:
                    audit["passed"] = False
                    audit["errors"].append(
                        f"table orientation mismatch table={table_name} budget={budget} candidate={candidate_name}"
                    )
    return audit


def current_ref_candidate(
    current_artifact: Path, current_hash: dict[str, Any]
) -> dict[str, Any]:
    return {
        "name": "current_ref",
        "kind": "reference",
        "artifact_dir": str(current_artifact),
        "artifact_weights_sha256": current_hash["actual_sha256"],
        "training": {"provenance": "checked_in_current"},
        "training_metrics": [],
        "checkpoints": [],
    }


def discover_pr152_candidate(pr152_workdir: Path) -> dict[str, Any]:
    lane_dir = pr152_workdir / "current_init_96x3_policy_only_e2"
    checkpoint_path = lane_dir / "checkpoint_epoch2.npz"
    artifact_dir = lane_dir / "artifact_epoch2"
    require_existing_file(
        checkpoint_path, "PR #152 current-init policy-only e2 checkpoint"
    )
    require_existing_file(
        artifact_dir / "weights.json", "PR #152 current-init policy-only e2 artifact"
    )
    return {
        "name": "pr152_current_init_policy_only_e2",
        "kind": "student",
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
        "training": {
            "provenance": "reused_pr152_artifact",
            "epochs": 2,
            "trainable_scope": "policy_head",
            "init_checkpoint": "model-artifact/current",
        },
        "training_metrics": [],
        "checkpoints": [
            {
                "epoch": 2,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "artifact_dir": str(artifact_dir),
                "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
            }
        ],
        "checkpoint_path": str(checkpoint_path),
    }


def train_policy_only_candidate(
    *,
    workdir: Path,
    candidate_name: str,
    epochs: int,
    train_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    current_checkpoint: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    spec = StudentSpec(
        name=candidate_name,
        model_type=MODEL_TYPE,
        trunk_size=96,
        residual_block_count=3,
    )
    return train_student_candidate(
        spec=spec,
        train_rows=train_rows,
        probe_rows=probe_rows,
        workdir=workdir,
        input_encoding=INPUT_ENCODING,
        epochs=epochs,
        batch_size=512,
        lr=3e-4,
        grad_clip=1.0,
        value_loss_weight=1.0,
        seed=seed,
        device=device,
        init_checkpoint=current_checkpoint,
        trainable_scope="policy_head",
        disable_probe_early_stop=True,
    )


def candidate_artifact_path(candidate: dict[str, Any]) -> Path:
    return Path(str(candidate["artifact_dir"]))


def candidate_checkpoint_path(candidate: dict[str, Any]) -> Path:
    if candidate.get("checkpoint_path"):
        return Path(str(candidate["checkpoint_path"]))
    checkpoints = candidate.get("checkpoints", [])
    if not checkpoints:
        raise RuntimeError(f"candidate has no checkpoint path: {candidate['name']}")
    return Path(str(checkpoints[-1]["checkpoint_path"]))


def medium_candidate_passes(
    medium_suite_summary: dict[str, Any], candidate_name: str
) -> bool:
    current_fixed = medium_suite_summary["current_ref"]["fixed_large"]
    candidate_fixed = medium_suite_summary[candidate_name]["fixed_large"]
    gain_384 = float(candidate_fixed["384:256"]["ds"]) - float(
        current_fixed["384:256"]["ds"]
    )
    reg_768 = float(candidate_fixed["768:768"]["ds"]) - float(
        current_fixed["768:768"]["ds"]
    )
    reg_1200 = float(candidate_fixed["1200:1200"]["ds"]) - float(
        current_fixed["1200:1200"]["ds"]
    )
    return gain_384 >= 0.03 and reg_768 >= -0.03 and reg_1200 >= -0.03


def heldout_gate_passes(
    *,
    suite_summary: dict[str, Any],
    bootstrap_tables: dict[str, Any],
    candidate_name: str,
) -> bool:
    current = suite_summary["current_ref"]["heldout"]
    candidate = suite_summary[candidate_name]["heldout"]
    if (
        float(candidate["384:256"]["mean_ds"]) - float(current["384:256"]["mean_ds"])
        < 0.05
        or float(
            bootstrap_tables[candidate_name]["384:256"]["ds"][
                "candidate_minus_current"
            ]["lower"]
        )
        <= 0.01
        or float(candidate["768:768"]["mean_ds"]) - float(current["768:768"]["mean_ds"])
        < -0.05
        or float(candidate["1200:1200"]["mean_ds"])
        - float(current["1200:1200"]["mean_ds"])
        < -0.03
        or float(candidate["1200:256"]["mean_ds"])
        - float(current["1200:256"]["mean_ds"])
        < -0.03
    ):
        return False
    return True


def classify_run(
    *,
    orientation_audit: dict[str, Any],
    best_candidate: str | None,
    heldout_summary: dict[str, Any] | None,
    bootstrap_tables: dict[str, Any],
    gate_results: dict[str, Any],
    probe_gate_candidates: list[str],
    fixed_large_summary: dict[str, Any] | None,
) -> str:
    if (
        best_candidate
        and heldout_summary is not None
        and heldout_gate_passes(
            suite_summary=heldout_summary,
            bootstrap_tables=bootstrap_tables,
            candidate_name=best_candidate,
        )
        and gate_results.get(best_candidate, {}).get("preserved") is True
    ):
        return "policy_only_runtime_candidate"
    if heldout_summary is not None:
        current = heldout_summary["current_ref"]["heldout"]
        for candidate_name, candidate_summary in heldout_summary.items():
            if candidate_name == "current_ref":
                continue
            gain = float(candidate_summary["heldout"]["384:256"]["mean_ds"]) - float(
                current["384:256"]["mean_ds"]
            )
            reg_768 = float(candidate_summary["heldout"]["768:768"]["mean_ds"]) - float(
                current["768:768"]["mean_ds"]
            )
            reg_1200 = float(
                candidate_summary["heldout"]["1200:1200"]["mean_ds"]
            ) - float(current["1200:1200"]["mean_ds"])
            reg_1200_256 = float(
                candidate_summary["heldout"]["1200:256"]["mean_ds"]
            ) - float(current["1200:256"]["mean_ds"])
            if (
                gain > 0.0
                and reg_768 >= -0.05
                and reg_1200 >= -0.03
                and reg_1200_256 >= -0.03
            ):
                if gain < 0.05:
                    return "policy_only_small_safe_but_weak"
            if gain > 0.0 and (reg_768 < -0.05 or reg_1200 < -0.03):
                return "policy_only_tradeoff_confirmed"
    if probe_gate_candidates and fixed_large_summary is not None:
        return "raw_top1_gate_was_correct"
    if orientation_audit.get("passed", False):
        return "reporting_pipeline_fixed"
    return "policy_only_tradeoff_confirmed"


def build_markdown_report(summary: dict[str, Any]) -> str:
    candidates = summary["candidates"]
    candidate_names = list(candidates)
    training_rows = []
    candidate_rows = []
    provenance_rows = []
    policy_probe_rows = []
    preservation_rows = []
    search_probe_rows = []
    probe_gate_rows = []
    medium_rows = []
    fixed_large_rows = []
    heldout_rows = []
    bootstrap_rows = []
    p0_p1_rows = []
    runtime_rows = []
    duplicate_rows = []
    for candidate_name in candidate_names:
        candidate = candidates[candidate_name]
        candidate_rows.append(
            [
                candidate_name,
                candidate.get("kind"),
                MODEL_TYPE,
                96,
                3,
                candidate.get("artifact_weights_sha256"),
            ]
        )
        provenance_rows.append(
            [
                candidate_name,
                candidate.get("training", {}).get("provenance", "trained_in_this_run"),
                candidate.get("training", {}).get("epochs"),
                candidate.get("training", {}).get("trainable_scope"),
                candidate.get("training", {}).get("init_checkpoint"),
            ]
        )
        for metric in candidate.get("training_metrics", []):
            training_rows.append(
                [
                    candidate_name,
                    metric.get("epoch"),
                    fmt(metric.get("policy_loss")),
                    fmt(metric.get("value_loss")),
                    fmt(metric.get("total_loss")),
                    fmt(metric.get("gradient_norm")),
                ]
            )
        policy_probe = candidate.get("policy_probe", {})
        preservation = candidate.get("value_trunk_preservation", {})
        search_probe = candidate.get("search_probe", {})
        policy_probe_rows.append(
            [
                candidate_name,
                fmt(policy_probe.get("teacher_puct_top1_agreement")),
                fmt(policy_probe.get("policy_kl")),
                fmt(policy_probe.get("candidate_entropy")),
                int(policy_probe.get("legal_failures", 0)),
                fmt(policy_probe.get("changed_raw_top1_rate_vs_current")),
            ]
        )
        preservation_rows.append(
            [
                candidate_name,
                fmt(policy_probe.get("max_value_diff_vs_current")),
                fmt(policy_probe.get("mean_value_diff_vs_current")),
                fmt(policy_probe.get("root_mae_vs_current")),
                preservation.get("passes"),
                ", ".join(preservation.get("changed_keys", [])) or "none",
            ]
        )
        search_probe_rows.append(
            [
                candidate_name,
                fmt(search_probe.get("search_selected_move_agreement_with_teacher")),
                fmt(search_probe.get("search_selected_move_changed_rate_vs_current")),
                fmt(search_probe.get("root_visit_kl_vs_current_ref")),
                fmt(search_probe.get("root_selected_visit_share_delta_vs_current_ref")),
                summary["probe_gate_results"].get(candidate_name, {}).get("passed"),
            ]
        )
        probe_gate_rows.append(
            [
                candidate_name,
                summary["probe_gate_results"].get(candidate_name, {}).get("passed"),
                "; ".join(
                    summary["probe_gate_results"]
                    .get(candidate_name, {})
                    .get("reasons", [])
                )
                or "passed",
            ]
        )
    medium_summary = summary.get("medium_suite_summary") or {}
    if medium_summary:
        for candidate_name in medium_summary:
            fixed = medium_summary[candidate_name]["fixed_large"]
            medium_rows.append(
                [
                    candidate_name,
                    *[fmt(fixed[budget]["ds"]) for budget in SUITE_BUDGETS],
                ]
            )
    fixed_large_summary = summary.get("fixed_large_suite_summary") or {}
    if fixed_large_summary:
        for candidate_name in fixed_large_summary:
            fixed = fixed_large_summary[candidate_name]["fixed_large"]
            fixed_large_rows.append(
                [
                    candidate_name,
                    *[fmt(fixed[budget]["ds"]) for budget in SUITE_BUDGETS],
                ]
            )
    heldout_summary = summary.get("heldout_suite_summary") or {}
    if heldout_summary:
        for candidate_name in heldout_summary:
            heldout = heldout_summary[candidate_name]["heldout"]
            heldout_rows.append(
                [
                    candidate_name,
                    fmt(heldout["384:256"]["mean_ds"]),
                    fmt(heldout["384:256"]["worst_suite_ds"]),
                    fmt(heldout["768:768"]["mean_ds"]),
                    fmt(heldout["1200:1200"]["mean_ds"]),
                    fmt(heldout["1200:256"]["mean_ds"]),
                ]
            )
            p0_p1 = heldout_summary[candidate_name]["p0_p1_384_256"]
            p0_p1_rows.append(
                [
                    candidate_name,
                    fmt(p0_p1["mean_p0"]),
                    fmt(p0_p1["mean_p1"]),
                    fmt(p0_p1["gap"]),
                ]
            )
            runtime = heldout_summary[candidate_name]["runtime_cost"]
            runtime_rows.append(
                [
                    candidate_name,
                    fmt(runtime.get("mean_move_time_ms"), 2),
                    fmt(runtime.get("mean_p95_move_time_ms"), 2),
                ]
            )
            duplicate_rows.append(
                [
                    candidate_name,
                    fmt(
                        heldout_summary[candidate_name].get(
                            "duplicate_trajectory_count"
                        )
                    ),
                ]
            )
    for candidate_name, budgets in summary.get("bootstrap_tables", {}).items():
        for budget in PRIMARY_BUDGETS:
            ds = budgets[budget]["ds"]
            bootstrap_rows.append(
                [
                    candidate_name,
                    budget,
                    "candidate_minus_current",
                    fmt(ds["candidate_minus_current"]["mean"]),
                    fmt(ds["candidate_minus_current"]["lower"]),
                    fmt(ds["candidate_minus_current"]["upper"]),
                ]
            )
            bootstrap_rows.append(
                [
                    candidate_name,
                    budget,
                    "current_minus_candidate",
                    fmt(ds["current_minus_candidate"]["mean"]),
                    fmt(ds["current_minus_candidate"]["lower"]),
                    fmt(ds["current_minus_candidate"]["upper"]),
                ]
            )
    gate_lines = []
    for candidate_name, result in summary.get("gate_results", {}).items():
        if not result.get("ran"):
            gate_lines.append(
                f"- {candidate_name}: `not_run` reason=`{result.get('reason')}`"
            )
        else:
            gate_lines.append(
                f"- {candidate_name}: `{result.get('classification')}` preserved=`{result.get('preserved')}`"
            )
    return "\n".join(
        [
            "# AlphaZero-Lite Current-Init Policy-Only Distillation Preflight Results",
            "",
            f"**Classification**: `{summary['classification']}`",
            "",
            "## Current Artifact Hash",
            "",
            f"- current weights SHA256: `{summary['current_hash']['actual_sha256']}`",
            "",
            "## PR #152 Artifact Hashes",
            "",
            f"- PR #152 current-init policy-only e2 weights SHA256: `{summary['pr152_candidate']['artifact_weights_sha256']}`",
            f"- PR #152 current-init policy-only e2 checkpoint SHA256: `{summary['pr152_candidate']['checkpoints'][0]['checkpoint_sha256']}`",
            "",
            "## DS Orientation Audit",
            "",
            f"- audit passed: `{summary['ds_orientation_audit']['passed']}`",
            f"- reused PR #152 DS audit helper: `{True}`",
            f"- bootstrap orientations are explicit: `{True}`",
            *[
                f"- orientation error: `{error}`"
                for error in summary["ds_orientation_audit"].get("errors", [])
            ],
            "",
            "## Candidate Table",
            "",
            markdown_table(
                ["Candidate", "Kind", "Model", "Trunk", "Blocks", "Weights SHA256"],
                candidate_rows,
            ),
            "",
            "## Training/Provenance Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Provenance",
                    "Epochs",
                    "Trainable scope",
                    "Init checkpoint",
                ],
                provenance_rows,
            ),
            "",
            "## Training Loss Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Epoch",
                    "Policy loss",
                    "Value loss",
                    "Total loss",
                    "Grad norm",
                ],
                training_rows or [["none", "", "", "", "", ""]],
            ),
            "",
            "## Policy Probe Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Teacher PUCT top-1",
                    "KL",
                    "Entropy",
                    "Legal fails",
                    "Changed raw top-1 vs current",
                ],
                policy_probe_rows,
            ),
            "",
            "## Value/Trunk Preservation Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Max value diff",
                    "Mean value diff",
                    "Root MAE vs current",
                    "Preserved",
                    "Changed keys",
                ],
                preservation_rows,
            ),
            "",
            "## Search-Aware Probe Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Search agree with teacher",
                    "Changed search move rate",
                    "Root visit KL vs current",
                    "Selected visit-share delta",
                    "Probe pass",
                ],
                search_probe_rows,
            ),
            "",
            "## Probe Gate Result Table",
            "",
            markdown_table(["Candidate", "Passed", "Reasons"], probe_gate_rows),
            "",
            "## Medium DS Table",
            "",
            markdown_table(
                ["Candidate", *SUITE_BUDGETS],
                medium_rows or [["not_run", "", "", "", "", "", ""]],
            ),
            "",
            "## Fixed-Large DS Table",
            "",
            markdown_table(
                ["Candidate", *SUITE_BUDGETS],
                fixed_large_rows or [["not_run", "", "", "", "", "", ""]],
            ),
            "",
            "## Held-Out Mean/Worst-Suite DS Table",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Mean 384:256",
                    "Worst 384:256",
                    "Mean 768:768",
                    "Mean 1200:1200",
                    "Mean 1200:256",
                ],
                heldout_rows or [["not_run", "", "", "", "", ""]],
            ),
            "",
            "## Bootstrap CIs",
            "",
            "Every bootstrap row below names its orientation explicitly.",
            "",
            markdown_table(
                [
                    "Candidate",
                    "Budget",
                    "Orientation",
                    "Mean",
                    "Lower 95%",
                    "Upper 95%",
                ],
                bootstrap_rows or [["not_run", "", "", "", "", ""]],
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(
                ["Candidate", "Mean P0", "Mean P1", "Gap"],
                p0_p1_rows or [["not_run", "", "", ""]],
            ),
            "",
            "## Duplicate Trajectory Count",
            "",
            markdown_table(
                ["Candidate", "Mean duplicates"], duplicate_rows or [["not_run", ""]]
            ),
            "",
            "## Runtime Cost",
            "",
            markdown_table(
                ["Candidate", "Mean move latency ms", "Mean p95 latency ms"],
                runtime_rows or [["not_run", "", ""]],
            ),
            "",
            "## Gate Result",
            "",
            *(gate_lines or ["- gate not run"]),
            "",
            "## Final Classification",
            "",
            f"- result: `{summary['classification']}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--pr152-workdir", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--teacher-default-c-puct", type=float, default=1.25)
    parser.add_argument("--teacher-cpuct-schedule", required=True)
    parser.add_argument("--teacher-tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--lanes", default="pr152_e2,e1_repro,e2_repro,e4")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_artifact = Path(args.current)
    pr152_workdir = Path(args.pr152_workdir)
    teacher_dataset = Path(args.teacher_dataset)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    lanes = parse_csv_values(args.lanes)

    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(
        current_artifact / "metadata.json", "current artifact metadata"
    )
    require_existing_file(teacher_dataset, "teacher dataset")
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(fixed_large_suite, "fixed large suite")
    for path in heldout_suites:
        require_existing_file(path, f"heldout suite {path.name}")

    current_hash = verify_expected_hash(
        current_artifact / "weights.json",
        args.expected_current_weights_sha256,
        "current artifact weights",
    )
    current_checkpoint = current_checkpoint_path(current_artifact, workdir)
    device = select_device(args.device)
    cpuct_schedule = parse_cpuct_schedule_json(args.teacher_cpuct_schedule)

    teacher_rows = read_jsonl(teacher_dataset)
    raw_probe_rows = teacher_rows
    search_probe_rows = stratified_probe_rows(
        teacher_rows,
        seed=int(args.seed),
        cap_per_bucket=SEARCH_PROBE_BUCKET_CAP,
    )
    training_probe_rows = stratified_probe_rows(
        teacher_rows,
        seed=int(args.seed) + 17,
        cap_per_bucket=16,
    )
    log_progress(
        f"teacher dataset ready rows={len(teacher_rows)} raw_probe_rows={len(raw_probe_rows)} search_probe_rows={len(search_probe_rows)}"
    )

    candidates: dict[str, dict[str, Any]] = {
        "current_ref": current_ref_candidate(current_artifact, current_hash)
    }
    if "pr152_e2" in lanes:
        candidates["pr152_current_init_policy_only_e2"] = discover_pr152_candidate(
            pr152_workdir
        )

    if "e1_repro" in lanes:
        candidates["current_init_policy_only_e1_repro"] = train_policy_only_candidate(
            workdir=workdir,
            candidate_name="current_init_policy_only_e1_repro",
            epochs=1,
            train_rows=teacher_rows,
            probe_rows=training_probe_rows,
            current_checkpoint=current_checkpoint,
            device=device,
            seed=int(args.seed),
        )
        candidates["current_init_policy_only_e1_repro"]["training"]["provenance"] = (
            "trained_in_this_run"
        )

    if "e2_repro" in lanes:
        candidates["current_init_policy_only_e2_repro"] = train_policy_only_candidate(
            workdir=workdir,
            candidate_name="current_init_policy_only_e2_repro",
            epochs=2,
            train_rows=teacher_rows,
            probe_rows=training_probe_rows,
            current_checkpoint=current_checkpoint,
            device=device,
            seed=int(args.seed),
        )
        candidates["current_init_policy_only_e2_repro"]["training"]["provenance"] = (
            "trained_in_this_run"
        )

    current_raw_outputs = evaluate_raw_outputs(
        artifact_path=current_artifact, rows=raw_probe_rows
    )["outputs"]
    current_search_outputs = evaluate_search_outputs(
        workdir=workdir,
        candidate_name="current_ref",
        artifact_path=current_artifact,
        rows=search_probe_rows,
        default_c_puct=float(args.teacher_default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.teacher_tactical_root_bias),
        seed=int(args.seed),
    )

    probe_gate_results: dict[str, Any] = {}
    for candidate_name, candidate in list(candidates.items()):
        artifact_path = candidate_artifact_path(candidate)
        checkpoint_path = (
            candidate_checkpoint_path(candidate)
            if candidate_name != "current_ref"
            else current_checkpoint
        )
        candidate_raw_outputs = (
            current_raw_outputs
            if candidate_name == "current_ref"
            else evaluate_raw_outputs(artifact_path=artifact_path, rows=raw_probe_rows)[
                "outputs"
            ]
        )
        policy_probe = evaluate_policy_probe_metrics(
            rows=raw_probe_rows,
            candidate_outputs=candidate_raw_outputs,
            current_outputs=current_raw_outputs,
        )
        preservation = compare_preserved_weights(current_checkpoint, checkpoint_path)
        candidate_search_outputs = (
            current_search_outputs
            if candidate_name == "current_ref"
            else evaluate_search_outputs(
                workdir=workdir,
                candidate_name=candidate_name,
                artifact_path=artifact_path,
                rows=search_probe_rows,
                default_c_puct=float(args.teacher_default_c_puct),
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=float(args.teacher_tactical_root_bias),
                seed=int(args.seed),
            )
        )
        search_probe = evaluate_search_probe_metrics(
            rows=search_probe_rows,
            candidate_outputs=candidate_search_outputs,
            current_outputs=current_search_outputs,
        )
        candidate["policy_probe"] = policy_probe
        candidate["value_trunk_preservation"] = preservation
        candidate["search_probe"] = search_probe
    current_policy_probe = candidates["current_ref"]["policy_probe"]
    stable_probe_candidates: list[str] = []
    for candidate_name, candidate in candidates.items():
        probe_result = search_probe_gate(
            policy_metrics=candidate["policy_probe"],
            search_metrics=candidate["search_probe"],
            preservation_metrics=candidate["value_trunk_preservation"],
            current_policy_metrics=current_policy_probe,
        )
        probe_gate_results[candidate_name] = probe_result
        if candidate_name != "current_ref" and probe_result["passed"]:
            stable_probe_candidates.append(candidate_name)

    if "e4" in lanes and {
        "current_init_policy_only_e1_repro",
        "current_init_policy_only_e2_repro",
    }.issubset(candidates):
        e1_ok = (
            probe_gate_results.get("current_init_policy_only_e1_repro", {}).get(
                "passed"
            )
            is True
        )
        e2_ok = (
            probe_gate_results.get("current_init_policy_only_e2_repro", {}).get(
                "passed"
            )
            is True
        )
        if e1_ok and e2_ok:
            candidates["current_init_policy_only_e4"] = train_policy_only_candidate(
                workdir=workdir,
                candidate_name="current_init_policy_only_e4",
                epochs=4,
                train_rows=teacher_rows,
                probe_rows=training_probe_rows,
                current_checkpoint=current_checkpoint,
                device=device,
                seed=int(args.seed),
            )
            candidates["current_init_policy_only_e4"]["training"]["provenance"] = (
                "trained_in_this_run"
            )
            artifact_path = candidate_artifact_path(
                candidates["current_init_policy_only_e4"]
            )
            raw_outputs = evaluate_raw_outputs(
                artifact_path=artifact_path, rows=raw_probe_rows
            )["outputs"]
            search_outputs = evaluate_search_outputs(
                workdir=workdir,
                candidate_name="current_init_policy_only_e4",
                artifact_path=artifact_path,
                rows=search_probe_rows,
                default_c_puct=float(args.teacher_default_c_puct),
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=float(args.teacher_tactical_root_bias),
                seed=int(args.seed),
            )
            candidates["current_init_policy_only_e4"]["policy_probe"] = (
                evaluate_policy_probe_metrics(
                    rows=raw_probe_rows,
                    candidate_outputs=raw_outputs,
                    current_outputs=current_raw_outputs,
                )
            )
            candidates["current_init_policy_only_e4"]["value_trunk_preservation"] = (
                compare_preserved_weights(
                    current_checkpoint,
                    candidate_checkpoint_path(
                        candidates["current_init_policy_only_e4"]
                    ),
                )
            )
            candidates["current_init_policy_only_e4"]["search_probe"] = (
                evaluate_search_probe_metrics(
                    rows=search_probe_rows,
                    candidate_outputs=search_outputs,
                    current_outputs=current_search_outputs,
                )
            )
            probe_gate_results["current_init_policy_only_e4"] = search_probe_gate(
                policy_metrics=candidates["current_init_policy_only_e4"][
                    "policy_probe"
                ],
                search_metrics=candidates["current_init_policy_only_e4"][
                    "search_probe"
                ],
                preservation_metrics=candidates["current_init_policy_only_e4"][
                    "value_trunk_preservation"
                ],
                current_policy_metrics=current_policy_probe,
            )
            if probe_gate_results["current_init_policy_only_e4"]["passed"]:
                stable_probe_candidates.append("current_init_policy_only_e4")

    medium_candidate_artifacts = [
        (name, candidate_artifact_path(candidate))
        for name, candidate in candidates.items()
        if name == "current_ref"
        or probe_gate_results.get(name, {}).get("passed") is True
    ]
    medium_rows = None
    medium_suite_summary = None
    if medium_candidate_artifacts:
        medium_rows = run_suite_evaluations(
            workdir=workdir / "medium_eval",
            current_artifact=current_artifact,
            candidate_artifacts=medium_candidate_artifacts,
            suite_paths=[medium_suite],
            budget_pairs=",".join(SUITE_BUDGETS),
            workers=int(args.workers),
            seed=int(args.seed),
        )
        medium_suite_summary = aggregate_suite_metrics(
            suite_rows={"large_eval": next(iter(medium_rows.values()))},
            candidate_names=[name for name, _path in medium_candidate_artifacts],
        )

    fixed_large_candidate_names = ["current_ref"]
    if medium_suite_summary is not None:
        eligible = [
            name
            for name, _path in medium_candidate_artifacts
            if name != "current_ref"
            and medium_candidate_passes(medium_suite_summary, name)
        ]
        eligible = sorted(
            eligible,
            key=lambda name: (
                -(
                    float(medium_suite_summary[name]["fixed_large"]["384:256"]["ds"])
                    - float(
                        medium_suite_summary["current_ref"]["fixed_large"]["384:256"][
                            "ds"
                        ]
                    )
                ),
                name,
            ),
        )[:3]
        fixed_large_candidate_names.extend(eligible)
    fixed_large_candidate_artifacts = [
        (name, candidate_artifact_path(candidates[name]))
        for name in fixed_large_candidate_names
    ]
    fixed_large_rows = run_suite_evaluations(
        workdir=workdir / "fixed_large_eval",
        current_artifact=current_artifact,
        candidate_artifacts=fixed_large_candidate_artifacts,
        suite_paths=[fixed_large_suite],
        budget_pairs=",".join(SUITE_BUDGETS),
        workers=int(args.workers),
        seed=int(args.seed),
    )
    fixed_large_suite_summary = aggregate_suite_metrics(
        suite_rows={"large_eval": next(iter(fixed_large_rows.values()))},
        candidate_names=fixed_large_candidate_names,
    )

    heldout_rows = None
    heldout_suite_summary = None
    bootstrap_tables: dict[str, Any] = {}
    gate_results: dict[str, Any] = {}
    heldout_candidate_names = ["current_ref"]
    best_candidate = None
    fixed_large_candidates = [
        name for name in fixed_large_candidate_names if name != "current_ref"
    ]
    if fixed_large_candidates:
        best_candidate = max(
            fixed_large_candidates,
            key=lambda name: (
                float(fixed_large_suite_summary[name]["fixed_large"]["384:256"]["ds"])
                - float(
                    fixed_large_suite_summary["current_ref"]["fixed_large"]["384:256"][
                        "ds"
                    ]
                )
            ),
        )
        heldout_candidate_names.append(best_candidate)
        heldout_candidate_artifacts = [
            (name, candidate_artifact_path(candidates[name]))
            for name in heldout_candidate_names
        ]
        heldout_rows = run_suite_evaluations(
            workdir=workdir / "heldout_eval",
            current_artifact=current_artifact,
            candidate_artifacts=heldout_candidate_artifacts,
            suite_paths=heldout_suites,
            budget_pairs=",".join(SUITE_BUDGETS),
            workers=int(args.workers),
            seed=int(args.seed),
        )
        heldout_suite_summary = aggregate_suite_metrics(
            suite_rows=heldout_rows,
            candidate_names=heldout_candidate_names,
        )
        bootstrap_tables = compute_bootstrap_comparisons(
            suite_rows=heldout_rows,
            candidate_names=heldout_candidate_names,
            seed=int(args.seed),
        )
        if heldout_gate_passes(
            suite_summary=heldout_suite_summary,
            bootstrap_tables=bootstrap_tables,
            candidate_name=best_candidate,
        ):
            out_path = workdir / "gate" / f"{best_candidate}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report = run_default_gate(
                candidate_path=str(candidate_artifact_path(candidates[best_candidate])),
                current_path=str(current_artifact),
                out=str(out_path),
                seed=int(args.seed),
                workers=int(args.workers),
                games=120,
                budget_pairs=DEFAULT_GATE_BUDGETS,
            )
            gate_results[best_candidate] = {
                "ran": True,
                "report_path": str(out_path),
                "classification": report.get("classification"),
                "preserved": candidate_gate_preserved(report),
            }
        elif best_candidate is not None:
            gate_results[best_candidate] = {
                "ran": False,
                "reason": "did not clear held-out gate for explicit deterministic gate run",
            }

    ds_audit_reference = build_ds_orientation_audit(
        suite_rows=heldout_rows or fixed_large_rows,
        summary_candidate_name=best_candidate or fixed_large_candidate_names[-1],
        report_doc_path=REPO_ROOT
        / "docs/alphazero-lite-search-teacher-student-preflight-results.md",
        seed=int(args.seed),
    )
    orientation_audit = build_orientation_audit(
        medium_rows=medium_rows,
        fixed_large_rows=fixed_large_rows,
        heldout_rows=heldout_rows,
        candidate_names=list(candidates),
        ds_audit_reference=ds_audit_reference,
        bootstrap_tables=bootstrap_tables,
        seed=int(args.seed),
    )
    write_json(workdir / DS_AUDIT_FILENAME, orientation_audit)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": {
            "workdir": str(workdir),
            "current": str(current_artifact),
            "pr152_workdir": str(pr152_workdir),
            "teacher_dataset": build_input_summary(teacher_dataset),
            "medium_suite": build_input_summary(medium_suite),
            "fixed_large_suite": build_input_summary(fixed_large_suite),
            "heldout_suites": [build_input_summary(path) for path in heldout_suites],
            "teacher_runtime_profile": {
                "tactical_root_bias": float(args.teacher_tactical_root_bias),
                "default_c_puct": float(args.teacher_default_c_puct),
                "c_puct_schedule": schedule_definition(
                    default_c_puct=float(args.teacher_default_c_puct),
                    schedule=cpuct_schedule,
                ),
                "root_policy_mode": "deterministic",
                "root_prior_transform": None,
                "value_transform": None,
            },
            "lanes": lanes,
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "current_hash": current_hash,
        "pr152_candidate": candidates.get("pr152_current_init_policy_only_e2"),
        "teacher_dataset_sha256": sha256_file(teacher_dataset),
        "probe_dataset_summary": {
            "raw_probe_rows": len(raw_probe_rows),
            "search_probe_rows": len(search_probe_rows),
            "search_probe_bucket_cap": SEARCH_PROBE_BUCKET_CAP,
        },
        "candidates": candidates,
        "probe_gate_results": probe_gate_results,
        "medium_suite_summary": medium_suite_summary,
        "fixed_large_suite_summary": fixed_large_suite_summary,
        "heldout_suite_summary": heldout_suite_summary,
        "bootstrap_tables": bootstrap_tables,
        "ds_orientation_audit": orientation_audit,
        "gate_results": gate_results,
    }
    summary["classification"] = classify_run(
        orientation_audit=orientation_audit,
        best_candidate=best_candidate,
        heldout_summary=heldout_suite_summary,
        bootstrap_tables=bootstrap_tables,
        gate_results=gate_results,
        probe_gate_candidates=stable_probe_candidates,
        fixed_large_summary=fixed_large_suite_summary,
    )
    write_json(workdir / SUMMARY_FILENAME, summary)
    REPORT_PATH.write_text(build_markdown_report(summary) + "\n", encoding="utf-8")
    log_progress(f"wrote summary={workdir / SUMMARY_FILENAME}")
    log_progress(f"wrote report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
