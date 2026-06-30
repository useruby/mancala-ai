#!/usr/bin/env python3
"""Post-schedule pairwise PUCT strength calibration experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_balanced_opening_puct_replay import (  # noqa: E402
    ASYM_1200_256_BUDGET,
    EQ_768_BUDGET,
    EQ_1200_BUDGET,
    PRIMARY_BUDGET,
    markdown_table,
    parse_budget_pairs,
    parse_csv_paths,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    benchmark_budget_results,
    bootstrap_ci,
    find_candidate_report,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_post_schedule_pairwise_puct_update import (  # noqa: E402
    ANCHOR_FILENAME,
    DEFAULT_BUDGETS,
    PAIRWISE_FILENAME,
    PROBE_FILENAME,
    TARGET_BROAD_ANCHOR_PROBE_ROWS,
    TARGET_STABILITY_PROBE_ROWS,
    aggregate_budget_summary,
    build_anchor_training_row,
    build_dataset,
    build_pairwise_training_row,
    build_probe_row,
    build_search_records,
    group_records_by_state_hash,
    load_suite_rows,
    run_default_gate_schedule,
    run_opening_suite_benchmark_schedule,
    suite_rows_from_report,
    unique_state_rows,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    build_input_summary,
    read_jsonl,
    require_existing_file,
    sha256_file,
    top_policy_move,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    compute_param_delta_norm,
    export_checkpoint,
    heldout_summary,
)

SUMMARY_SCHEMA = "azlite_post_schedule_pairwise_strength_calibration_v1"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-post-schedule-pairwise-strength-calibration-results.md"
)
EXPECTED_PAIRWISE_ROWS = 1130
EXPECTED_ANCHOR_ROWS = 8000
EXPECTED_PROBE_COMPOSITION = {
    "pairwise": 1130,
    "stability": 2000,
    "broad_anchor": 4000,
}
ABSOLUTE_ANCHOR_KL_LIMIT = 0.01
RELATIVE_ANCHOR_KL_FACTOR = 2.0


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


@dataclass(frozen=True)
class LaneSpec:
    pairwise_weight: int
    behavior_anchor_weight: int
    margin: float
    epochs: tuple[int, ...]

    def candidate_name(self, epoch: int) -> str:
        margin_token = int(round(self.margin * 100))
        return (
            f"pairwise_w{self.pairwise_weight}_kl{self.behavior_anchor_weight}"
            f"_margin{margin_token:03d}_e{epoch}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--pr138-workdir", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument(
        "--pairwise-lanes",
        default="w4:kl2:m010,w8:kl2:m010,w8:kl1:m010,w16:kl2:m010:e1",
    )
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", required=True)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    return parser.parse_args()


def parse_pairwise_lanes(text: str) -> list[LaneSpec]:
    lanes: list[LaneSpec] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        pairwise_weight = None
        behavior_anchor_weight = None
        margin = None
        epochs: tuple[int, ...] = (1, 2)
        for part in token.split(":"):
            if part.startswith("w"):
                pairwise_weight = int(part[1:])
            elif part.startswith("kl"):
                behavior_anchor_weight = int(part[2:])
            elif part.startswith("m"):
                margin = int(part[1:]) / 100.0
            elif part.startswith("e"):
                epochs = (int(part[1:]),)
            else:
                raise ValueError(f"unsupported lane token: {part}")
        if pairwise_weight is None or behavior_anchor_weight is None or margin is None:
            raise ValueError(f"invalid lane spec: {token}")
        lanes.append(
            LaneSpec(
                pairwise_weight=pairwise_weight,
                behavior_anchor_weight=behavior_anchor_weight,
                margin=margin,
                epochs=epochs,
            )
        )
    if not lanes:
        raise ValueError("at least one pairwise lane is required")
    return lanes


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_jsonl(path)]


def probe_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("probe_kind", "unknown")) for row in rows)
    return {
        "pairwise": int(counts.get("pairwise", 0)),
        "stability": int(counts.get("stability", 0)),
        "broad_anchor": int(counts.get("broad_anchor", 0)),
    }


def summarize_probe_verification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    composition = probe_composition(rows)
    return {
        "actual": composition,
        "expected": dict(EXPECTED_PROBE_COMPOSITION),
        "matches_expected": composition == EXPECTED_PROBE_COMPOSITION,
    }


def copy_input_if_needed(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def prepare_reused_or_rebuilt_dataset(
    *,
    workdir: Path,
    pr138_workdir: Path,
    current_dir: Path,
    fixed_large_suite: Path,
    heldout_suites: list[Path],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    pairwise_source = pr138_workdir / PAIRWISE_FILENAME
    anchor_source = pr138_workdir / ANCHOR_FILENAME
    probe_source = pr138_workdir / PROBE_FILENAME
    pairwise_target_path = workdir / PAIRWISE_FILENAME
    anchor_target_path = workdir / ANCHOR_FILENAME
    probe_target_path = workdir / PROBE_FILENAME

    if pairwise_source.is_file() and anchor_source.is_file() and probe_source.is_file():
        pairwise_path = copy_input_if_needed(pairwise_source, pairwise_target_path)
        anchor_path = copy_input_if_needed(anchor_source, anchor_target_path)
        probe_path = copy_input_if_needed(probe_source, probe_target_path)
        pairwise_rows = jsonl_rows(pairwise_path)
        anchor_rows = jsonl_rows(anchor_path)
        probe_rows = jsonl_rows(probe_path)
        return {
            "source": "reused_pr138",
            "pairwise_file": pairwise_path,
            "anchor_file": anchor_path,
            "probe_file": probe_path,
            "pairwise_rows": pairwise_rows,
            "anchor_rows": anchor_rows,
            "probe_rows": probe_rows,
            "reuse_inputs": {
                "pairwise_targets": build_input_summary(pairwise_source),
                "behavior_anchors": build_input_summary(anchor_source),
                "probe_rows": build_input_summary(probe_source),
            },
        }

    evaluator = ArtifactEvaluator(current_dir)
    suite_rows = load_suite_rows([fixed_large_suite, *heldout_suites])
    unique_states = unique_state_rows(suite_rows)
    records = build_search_records(
        state_rows=unique_states,
        budget_pairs=parse_budget_pairs(DEFAULT_BUDGETS),
        evaluator=evaluator,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        seed=seed,
    )
    dataset = build_dataset(
        records=records,
        grouped=group_records_by_state_hash(records),
    )
    pairwise_rows = [
        build_pairwise_training_row(row) for row in dataset["pairwise_rows"]
    ]
    anchor_rows = [build_anchor_training_row(row) for row in dataset["anchor_rows"]]
    probe_rows = [
        *(
            build_probe_row(row, "pairwise")
            for row in dataset["pairwise_rows"][:EXPECTED_PAIRWISE_ROWS]
        ),
        *(
            build_probe_row(row, "stability")
            for row in dataset["stability_rows"][:TARGET_STABILITY_PROBE_ROWS]
        ),
        *(
            build_probe_row(row, "broad_anchor")
            for row in dataset["broad_anchor_probe_rows"][
                :TARGET_BROAD_ANCHOR_PROBE_ROWS
            ]
        ),
    ]
    write_jsonl(pairwise_target_path, pairwise_rows)
    write_jsonl(anchor_target_path, anchor_rows)
    write_jsonl(probe_target_path, probe_rows)
    return {
        "source": "rebuilt_from_current",
        "pairwise_file": pairwise_target_path,
        "anchor_file": anchor_target_path,
        "probe_file": probe_target_path,
        "pairwise_rows": pairwise_rows,
        "anchor_rows": anchor_rows,
        "probe_rows": probe_rows,
        "reuse_inputs": {
            "pairwise_targets": build_input_summary(pairwise_target_path),
            "behavior_anchors": build_input_summary(anchor_target_path),
            "probe_rows": build_input_summary(probe_target_path),
        },
    }


def run_train_command(
    *,
    pairwise_file: Path,
    anchor_file: Path,
    init_checkpoint: Path,
    out_path: Path,
    pairwise_weight: int,
    anchor_weight: int,
    margin: float,
    epochs: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--pairwise-target-files",
        str(pairwise_file),
        "--pairwise-loss-weight",
        str(pairwise_weight),
        "--pairwise-margin",
        str(margin),
        "--behavior-anchor-files",
        str(anchor_file),
        "--behavior-loss-weight",
        str(anchor_weight),
        "--init-checkpoint",
        str(init_checkpoint),
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--hidden-sizes",
        "96,3",
        "--epochs",
        str(epochs),
        "--batch-size",
        "512",
        "--lr",
        "1e-5",
        "--value-loss-weight",
        "0.0",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "0",
        "--out",
        str(out_path),
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--lr-scheduler",
        "none",
        "--seed",
        str(seed),
        "--trainable-scope",
        "policy_head",
        "--save-epochs",
        str(epochs),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed: {result.stderr[-2000:]}")
    metrics: dict[str, Any] = {"stdout": result.stdout, "stderr": result.stderr}
    for line in result.stdout.splitlines():
        line = line.strip()
        for key in (
            "policy_loss",
            "value_loss",
            "pairwise_loss",
            "behavior_anchor_loss",
            "total_loss",
            "best_val_loss",
        ):
            token = f"{key}="
            if line.startswith(token):
                metrics[key] = float(line.split("=", 1)[1])
    return metrics


def candidate_spec(
    *,
    name: str,
    artifact_dir: Path,
    checkpoint_path: Path | None,
    epochs: int,
    pairwise_weight: int | None,
    margin: float | None,
    anchor_weight: int | None,
    report_name: str | None = None,
    source: str,
) -> dict[str, Any]:
    return {
        "candidate": name,
        "artifact_dir": str(artifact_dir),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "epochs": epochs,
        "pairwise_weight": pairwise_weight,
        "pairwise_margin": margin,
        "behavior_anchor_weight": anchor_weight,
        "report_candidate_name": report_name or name,
        "source": source,
    }


def candidate_row_from_pr138_summary(
    pr138_summary: dict[str, Any], candidate_name: str
) -> dict[str, Any] | None:
    for row in pr138_summary.get("candidates", []):
        if row.get("candidate") == candidate_name:
            return dict(row)
    return None


def choose_pr138_reference_name(pr138_workdir: Path) -> str | None:
    for candidate_name in ("pairwise_margin010_kl4_e2", "pairwise_margin005_kl4_e2"):
        artifact_path = pr138_workdir / candidate_name / "artifact" / "weights.json"
        if artifact_path.is_file():
            return candidate_name
    return None


def build_current_forward_cache(
    *, current_artifact: Path, probe_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(current_artifact)
    outputs: list[dict[str, Any]] = []
    for row in probe_rows:
        game = KalahGame.from_state(row["state"])
        logits, policy, _value = artifact_forward_details(evaluator, game)
        legal_moves = [int(move) for move in row["legal_moves"]]
        outputs.append(
            {
                "logits": np.asarray(logits, dtype=np.float64),
                "policy": np.asarray(policy, dtype=np.float64),
                "top1": top_policy_move(policy, legal_moves),
            }
        )
    return outputs


def evaluate_probe_candidate(
    *,
    artifact_path: Path,
    probe_rows: list[dict[str, Any]],
    current_forwards: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    pairwise_successes = 0
    pairwise_margins: list[float] = []
    pairwise_margin_improvements: list[float] = []
    pairwise_total = 0
    stability_total = 0
    stability_preserved = 0
    broad_total = 0
    broad_changed = 0
    anchor_kls: list[float] = []
    changed_by_context: dict[str, int] = defaultdict(int)
    total_by_context: dict[str, int] = defaultdict(int)
    changed_384_256_p0 = 0
    total_384_256_p0 = 0
    for row, current in zip(probe_rows, current_forwards, strict=True):
        game = KalahGame.from_state(row["state"])
        logits, candidate_policy, _value = artifact_forward_details(evaluator, game)
        logits_array = np.asarray(logits, dtype=np.float64)
        candidate_policy_array = np.asarray(candidate_policy, dtype=np.float64)
        legal_moves = [int(move) for move in row["legal_moves"]]
        candidate_top1 = top_policy_move(candidate_policy, legal_moves)
        current_top1 = int(current["top1"])
        puct_top1 = int(row["puct_top1"])

        if row["probe_kind"] == "pairwise":
            pairwise_total += 1
            margin = float(logits_array[puct_top1] - logits_array[current_top1])
            current_margin = float(
                current["logits"][puct_top1] - current["logits"][current_top1]
            )
            if margin > 0.0:
                pairwise_successes += 1
            pairwise_margins.append(margin)
            pairwise_margin_improvements.append(margin - current_margin)
            continue

        if row["probe_kind"] == "stability":
            stability_total += 1
            if candidate_top1 == current_top1:
                stability_preserved += 1
            continue

        if row["probe_kind"] != "broad_anchor":
            continue

        broad_total += 1
        context_key = f"{row['phase']}::{row['seat_context']}"
        total_by_context[context_key] += 1
        if row["budget_pair"] == PRIMARY_BUDGET and row["seat_context"] == "P0":
            total_384_256_p0 += 1
        safe_current = np.clip(current["policy"], 1e-12, 1.0)
        safe_current /= np.sum(safe_current)
        safe_candidate = np.clip(candidate_policy_array, 1e-12, 1.0)
        safe_candidate /= np.sum(safe_candidate)
        anchor_kls.append(
            float(np.sum(safe_current * np.log(safe_current / safe_candidate)))
        )
        if candidate_top1 != current_top1:
            broad_changed += 1
            changed_by_context[context_key] += 1
            if row["budget_pair"] == PRIMARY_BUDGET and row["seat_context"] == "P0":
                changed_384_256_p0 += 1

    context_rates = {
        key: (changed_by_context.get(key, 0) / total)
        for key, total in sorted(total_by_context.items())
        if total > 0
    }
    return {
        "pairwise_target_success_rate": pairwise_successes / max(pairwise_total, 1),
        "mean_target_margin": statistics.fmean(pairwise_margins)
        if pairwise_margins
        else 0.0,
        "mean_target_margin_improvement": statistics.fmean(pairwise_margin_improvements)
        if pairwise_margin_improvements
        else 0.0,
        "stability_top1_preservation": stability_preserved / max(stability_total, 1),
        "broad_anchor_top1_changed_rate": broad_changed / max(broad_total, 1),
        "mean_anchor_kl_current_to_candidate": statistics.fmean(anchor_kls)
        if anchor_kls
        else 0.0,
        "changed_top1_rate_by_phase_and_seat_context": context_rates,
        "max_phase_seat_context_changed_rate": max(context_rates.values(), default=0.0),
        "changed_top1_rate_384_256_p0_context": changed_384_256_p0
        / max(total_384_256_p0, 1),
    }


def apply_probe_gates(candidates: list[dict[str, Any]]) -> None:
    current_probe = next(
        row["probe_metrics"] for row in candidates if row["candidate"] == "current_ref"
    )
    noncurrent = [row for row in candidates if row["candidate"] != "current_ref"]
    for row in noncurrent:
        probe = row["probe_metrics"]
        row["pairwise_success_gain_vs_current"] = float(
            probe["pairwise_target_success_rate"]
        ) - float(current_probe["pairwise_target_success_rate"])

    for row in noncurrent:
        probe = row["probe_metrics"]
        changed_rate = float(probe["broad_anchor_top1_changed_rate"])
        lower_drift_kls = [
            float(other["probe_metrics"]["mean_anchor_kl_current_to_candidate"])
            for other in noncurrent
            if other["candidate"] != row["candidate"]
            and float(other["probe_metrics"]["broad_anchor_top1_changed_rate"])
            <= changed_rate + 1e-12
        ]
        relative_reference = min(lower_drift_kls) if lower_drift_kls else None
        row["anchor_kl_monotonic_reference"] = relative_reference
        row["anchor_kl_gate_threshold"] = max(
            ABSOLUTE_ANCHOR_KL_LIMIT,
            (
                ABSOLUTE_ANCHOR_KL_LIMIT
                if relative_reference is None
                else (relative_reference * RELATIVE_ANCHOR_KL_FACTOR)
            ),
        )
        reasons: list[str] = []
        if float(row["pairwise_success_gain_vs_current"]) < 0.03:
            reasons.append("pairwise success gain < +0.03")
        if float(probe["stability_top1_preservation"]) < 0.99:
            reasons.append("stability top1 preservation < 0.99")
        if float(probe["broad_anchor_top1_changed_rate"]) > 0.03:
            reasons.append("broad-anchor top1 changed rate > 0.03")
        if float(probe["max_phase_seat_context_changed_rate"]) > 0.08:
            reasons.append("phase/seat context changed rate > 0.08")
        if float(probe["mean_anchor_kl_current_to_candidate"]) > float(
            row["anchor_kl_gate_threshold"]
        ):
            reasons.append("anchor KL exceeds small/monotonic drift envelope")
        row["aborted"] = bool(reasons)
        row["abort_reasons"] = reasons


def classification(summary_rows: list[dict[str, Any]]) -> str:
    current_row = next(row for row in summary_rows if row["candidate"] == "current_ref")
    current_heldout_mean = float(
        current_row.get("heldout_summary", {}).get("mean_ds_384_256", 0.0)
    )
    moved_any = False
    successful = False
    moved_but_harmful = False
    w16_lane = None
    for row in summary_rows:
        if row["candidate"] == "current_ref":
            continue
        if row.get("pairwise_weight") == 16:
            w16_lane = row
        probe_gain = float(row.get("pairwise_success_gain_vs_current", 0.0))
        if probe_gain >= 0.03:
            moved_any = True
        heldout_mean = float(row.get("heldout_summary", {}).get("mean_ds_384_256", 0.0))
        improvement = heldout_mean - current_heldout_mean
        row["heldout_mean_384_256_delta_vs_current_ref"] = improvement
        ci = row.get("bootstrap_cis", {}).get(
            f"{row['candidate']}_minus_current_ref_384_256",
            {},
        )
        robust = (
            float(
                row.get("heldout_aggregate", {})
                .get(EQ_768_BUDGET, {})
                .get("mean_delta_vs_current_ref", 0.0)
            )
            >= -0.05
            and float(
                row.get("heldout_aggregate", {})
                .get(EQ_1200_BUDGET, {})
                .get("mean_delta_vs_current_ref", 0.0)
            )
            >= -0.03
            and float(
                row.get("heldout_aggregate", {})
                .get(ASYM_1200_256_BUDGET, {})
                .get("mean_delta_vs_current_ref", 0.0)
            )
            >= -0.03
        )
        if (
            not row.get("aborted")
            and improvement >= 0.05
            and float(ci.get("lower", 0.0)) > 0.01
            and robust
            and row.get("gate_classification") in {None, "high_search_breakthrough"}
        ):
            successful = True
        elif probe_gain >= 0.03:
            moved_but_harmful = True

    if successful:
        return "pairwise_strength_success"
    if moved_any:
        return "pairwise_now_moves_but_harmful"
    if w16_lane is not None:
        w16_probe = w16_lane.get("probe_metrics", {})
        within_drift_limits = (
            float(w16_lane.get("pairwise_success_gain_vs_current", 0.0)) < 0.03
            and float(w16_probe.get("stability_top1_preservation", 0.0)) >= 0.99
            and float(w16_probe.get("broad_anchor_top1_changed_rate", 0.0)) <= 0.03
            and float(w16_probe.get("max_phase_seat_context_changed_rate", 0.0)) <= 0.08
            and not any(
                reason == "anchor KL exceeds small/monotonic drift envelope"
                for reason in w16_lane.get("abort_reasons", [])
            )
        )
        if within_drift_limits:
            return "pairwise_still_too_weak"
    if moved_but_harmful:
        return "pairwise_now_moves_but_harmful"
    return "policy_head_learning_blocked"


def write_report(*, summary: dict[str, Any], report_path: Path) -> None:
    input_rows = []
    for label, payload in summary["inputs"].items():
        if isinstance(payload, dict) and "actual_sha256" in payload:
            input_rows.append(
                [
                    label,
                    payload["path"],
                    payload["actual_sha256"],
                    payload.get("rows", "n/a"),
                ]
            )
        elif isinstance(payload, dict) and "path" in payload and "sha256" in payload:
            input_rows.append(
                [label, payload["path"], payload["sha256"], payload.get("rows", "n/a")]
            )
        elif isinstance(payload, dict):
            for nested_label, nested_payload in payload.items():
                input_rows.append(
                    [
                        f"{label}:{nested_label}",
                        nested_payload["path"],
                        nested_payload.get(
                            "actual_sha256", nested_payload.get("sha256", "n/a")
                        ),
                        nested_payload.get("rows", "n/a"),
                    ]
                )

    weight_rows = []
    probe_rows = []
    training_rows = []
    aborted_rows = []
    evaluated_rows = []
    fixed_large_rows = []
    heldout_rows = []
    bootstrap_rows = []
    p0p1_rows = []
    duplicate_rows = []
    for row in summary["candidates"]:
        probe = row["probe_metrics"]
        weight_rows.append(
            [
                row["candidate"],
                row.get("pairwise_weight", "n/a"),
                row.get("behavior_anchor_weight", "n/a"),
                fmt(row.get("pairwise_margin")),
                row.get("epochs", "n/a"),
                row.get("source", "n/a"),
            ]
        )
        probe_rows.append(
            [
                row["candidate"],
                fmt(row.get("pairwise_success_gain_vs_current", 0.0)),
                fmt(probe["pairwise_target_success_rate"]),
                fmt(probe["mean_target_margin_improvement"]),
                fmt(probe["stability_top1_preservation"]),
                fmt(probe["broad_anchor_top1_changed_rate"]),
                fmt(probe["mean_anchor_kl_current_to_candidate"]),
                fmt(probe["max_phase_seat_context_changed_rate"]),
                fmt(probe["changed_top1_rate_384_256_p0_context"]),
                str(row.get("aborted", False)),
            ]
        )
        training_rows.append(
            [
                row["candidate"],
                fmt(row.get("pairwise_loss")),
                fmt(row.get("behavior_anchor_loss")),
                fmt(row.get("total_loss")),
                fmt(row.get("validation_loss")),
                row.get("checkpoint_sha256", "n/a"),
                row.get("artifact_weights_sha256", "n/a"),
                fmt(row.get("delta_norm_vs_current_checkpoint")),
            ]
        )
        if row.get("aborted"):
            aborted_rows.append(
                [row["candidate"], "; ".join(row.get("abort_reasons", []))]
            )
        if row.get("evaluated"):
            evaluated_rows.append(
                [
                    row["candidate"],
                    str(row.get("evaluated")),
                    str(row.get("gate_ran", False)),
                    fmt(row.get("heldout_mean_384_256_delta_vs_current_ref")),
                ]
            )
        if "fixed_large_budget_results" in row:
            fixed_large_rows.append(
                [
                    row["candidate"],
                    fmt(row["fixed_large_budget_results"][PRIMARY_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"]["768:256"]["ds"]),
                    fmt(row["fixed_large_budget_results"][EQ_768_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"][EQ_1200_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"][ASYM_1200_256_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"]["256:768"]["ds"]),
                ]
            )
        if "heldout_summary" in row:
            heldout_rows.append(
                [
                    row["candidate"],
                    fmt(row["heldout_summary"].get("mean_ds_384_256")),
                    fmt(row.get("heldout_mean_384_256_delta_vs_current_ref")),
                    fmt(row["heldout_summary"].get("worst_suite_ds_384_256")),
                ]
            )
        if "heldout_aggregate" in row:
            p0p1_rows.append(
                [
                    row["candidate"],
                    fmt(row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p0_score"]),
                    fmt(row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p1_score"]),
                    fmt(
                        row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p1_score"]
                        - row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p0_score"]
                    ),
                ]
            )
            duplicate_rows.append(
                [
                    row["candidate"],
                    fmt(
                        row["heldout_aggregate"][PRIMARY_BUDGET][
                            "mean_duplicate_trajectory_count"
                        ]
                    ),
                ]
            )
        for budget_pair in (
            PRIMARY_BUDGET,
            EQ_768_BUDGET,
            EQ_1200_BUDGET,
            ASYM_1200_256_BUDGET,
        ):
            ci = row.get("bootstrap_cis", {}).get(
                f"{row['candidate']}_minus_current_ref_{budget_pair.replace(':', '_')}"
            )
            if ci is None:
                continue
            bootstrap_rows.append(
                [
                    f"{row['candidate']}_minus_current_ref_{budget_pair}",
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )

    dataset = summary["dataset"]
    lines = [
        "# AlphaZero-Lite Post-Schedule Pairwise Strength Calibration Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Input Hashes And Row Counts",
        "",
        markdown_table(["Input", "Path", "SHA256", "Rows"], input_rows),
        "",
        f"- Pairwise target rows: `{dataset['pairwise_target_rows_actual']}` expected `{dataset['pairwise_target_rows_expected']}` match=`{dataset['pairwise_target_rows_match']}`",
        f"- Behavior anchor rows: `{dataset['anchor_rows_actual']}` expected `{dataset['anchor_rows_expected']}` match=`{dataset['anchor_rows_match']}`",
        f"- Probe composition: `{json.dumps(dataset['probe_composition_actual'], sort_keys=True)}` expected `{json.dumps(dataset['probe_composition_expected'], sort_keys=True)}` match=`{dataset['probe_composition_match']}`",
        f"- Dataset source: `{dataset['source']}`",
        "",
        "## Promoted Search Schedule Confirmation",
        "",
        f"- Schedule manifest: `{json.dumps(summary['search_schedule'], sort_keys=True)}`",
        f"- Root policy mode: `{summary['search_options']['root_policy_mode']}`",
        f"- Tactical root bias: `{summary['search_options']['tactical_root_bias']}`",
        "",
        "## Pairwise/Anchor Weight Table",
        "",
        markdown_table(
            ["Candidate", "Pairwise w", "Anchor w", "Margin", "Epochs", "Source"],
            weight_rows,
        ),
        "",
        "## Training Loss Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Pairwise loss",
                "Behavior loss",
                "Total loss",
                "Validation loss",
                "Checkpoint SHA256",
                "Artifact weights SHA256",
                "Delta norm",
            ],
            training_rows,
        ),
        "",
        "## Probe Metrics Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Success gain",
                "Success rate",
                "Margin improve",
                "Stability preserve",
                "Broad changed",
                "Anchor KL",
                "Max ctx drift",
                "384:256 P0 drift",
                "Aborted",
            ],
            probe_rows,
        ),
        "",
        "## Aborted-Candidate Table",
        "",
        markdown_table(["Candidate", "Reasons"], aborted_rows or [["none", "n/a"]]),
        "",
        "## Evaluated-Candidate Table",
        "",
        markdown_table(
            ["Candidate", "Evaluated", "Gate ran", "Held-out delta 384:256"],
            evaluated_rows or [["current_ref", "True", "True", fmt(0.0)]],
        ),
        "",
        "## Fixed Large DS Table",
        "",
        markdown_table(
            [
                "Candidate",
                "384:256",
                "768:256",
                "768:768",
                "1200:1200",
                "1200:256",
                "256:768",
            ],
            fixed_large_rows,
        ),
        "",
        "## Held-Out Mean/Worst-Suite Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Held-out mean 384:256",
                "Delta vs current",
                "Held-out worst-suite 384:256",
            ],
            heldout_rows,
        ),
        "",
        "## Bootstrap CI For Candidate Minus Current",
        "",
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%"], bootstrap_rows
        ),
        "",
        "## P0/P1 Split For 384:256",
        "",
        markdown_table(["Candidate", "Mean P0", "Mean P1", "Gap"], p0p1_rows),
        "",
        "## Duplicate Trajectory Count",
        "",
        markdown_table(["Candidate", "Mean duplicates"], duplicate_rows),
        "",
        "## Gate Classification",
        "",
        *[
            f"- {row['candidate']}: `{row.get('gate_classification', 'not_run')}`"
            for row in summary["candidates"]
            if row.get("evaluated")
        ],
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_dir = Path(args.current)
    current_weights_path = current_dir / "weights.json"
    init_checkpoint = Path(args.init_checkpoint)
    pr138_workdir = Path(args.pr138_workdir)
    pr138_summary_path = pr138_workdir / SUMMARY_FILENAME
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    budget_pairs = parse_budget_pairs(DEFAULT_BUDGETS)
    lane_specs = parse_pairwise_lanes(args.pairwise_lanes)

    for path, label in (
        (current_weights_path, "current weights"),
        (init_checkpoint, "init checkpoint"),
        (medium_suite, "medium suite"),
        (fixed_large_suite, "fixed large suite"),
    ):
        require_existing_file(path, label)
    if not pr138_workdir.is_dir():
        raise FileNotFoundError(f"missing pr138 workdir: {pr138_workdir}")
    require_existing_file(pr138_summary_path, "pr138 summary")
    for suite in heldout_suites:
        require_existing_file(suite, f"heldout suite {suite.name}")

    dataset_info = prepare_reused_or_rebuilt_dataset(
        workdir=workdir,
        pr138_workdir=pr138_workdir,
        current_dir=current_dir,
        fixed_large_suite=fixed_large_suite,
        heldout_suites=heldout_suites,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        seed=args.seed,
    )
    pr138_summary = json.loads(pr138_summary_path.read_text(encoding="utf-8"))

    probe_rows = dataset_info["probe_rows"]
    probe_verify = summarize_probe_verification(probe_rows)
    pairwise_input_summary = build_input_summary(dataset_info["pairwise_file"])
    anchor_input_summary = build_input_summary(dataset_info["anchor_file"])
    probe_input_summary = build_input_summary(dataset_info["probe_file"])

    inputs = {
        "current_weights": verify_expected_hash(
            current_weights_path,
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact": build_input_summary(current_weights_path),
        "init_checkpoint": build_input_summary(init_checkpoint),
        "pr138_pairwise_targets": dataset_info["reuse_inputs"]["pairwise_targets"],
        "pr138_behavior_anchors": dataset_info["reuse_inputs"]["behavior_anchors"],
        "pr138_probe_rows": dataset_info["reuse_inputs"]["probe_rows"],
        "local_pairwise_targets": pairwise_input_summary,
        "local_behavior_anchors": anchor_input_summary,
        "local_probe_rows": probe_input_summary,
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suites
        },
    }

    current_candidate = candidate_spec(
        name="current_ref",
        artifact_dir=current_dir,
        checkpoint_path=init_checkpoint,
        epochs=0,
        pairwise_weight=None,
        margin=None,
        anchor_weight=None,
        report_name=current_dir.name,
        source="checked_in_current",
    )
    current_candidate["artifact_weights_sha256"] = sha256_file(current_weights_path)

    candidates: list[dict[str, Any]] = [current_candidate]
    pr138_reference_name = choose_pr138_reference_name(pr138_workdir)
    if pr138_reference_name is not None:
        summary_row = candidate_row_from_pr138_summary(
            pr138_summary, pr138_reference_name
        )
        checkpoint_path = (
            pr138_workdir / pr138_reference_name / f"{pr138_reference_name}.npz"
        )
        artifact_dir = pr138_workdir / pr138_reference_name / "artifact"
        pr138_candidate = candidate_spec(
            name="pr138_best_ref",
            artifact_dir=artifact_dir,
            checkpoint_path=checkpoint_path if checkpoint_path.is_file() else None,
            epochs=2,
            pairwise_weight=1,
            margin=0.10 if "margin010" in pr138_reference_name else 0.05,
            anchor_weight=4,
            report_name=pr138_reference_name,
            source=f"reused_{pr138_reference_name}",
        )
        if summary_row is not None:
            for key in (
                "pairwise_loss",
                "behavior_anchor_loss",
                "total_loss",
                "validation_loss",
                "delta_norm_vs_current_checkpoint",
            ):
                if key in summary_row:
                    pr138_candidate[key] = summary_row[key]
        if checkpoint_path.is_file():
            pr138_candidate["checkpoint_sha256"] = sha256_file(checkpoint_path)
        artifact_weights = artifact_dir / "weights.json"
        if artifact_weights.is_file():
            pr138_candidate["artifact_weights_sha256"] = sha256_file(artifact_weights)
        candidates.append(pr138_candidate)

    for lane in lane_specs:
        for epoch in lane.epochs:
            candidate_name = lane.candidate_name(epoch)
            candidate_dir = workdir / candidate_name
            checkpoint_path = candidate_dir / f"{candidate_name}.npz"
            artifact_dir = candidate_dir / "artifact"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            metrics = run_train_command(
                pairwise_file=Path(dataset_info["pairwise_file"]),
                anchor_file=Path(dataset_info["anchor_file"]),
                init_checkpoint=init_checkpoint,
                out_path=checkpoint_path,
                pairwise_weight=lane.pairwise_weight,
                anchor_weight=lane.behavior_anchor_weight,
                margin=lane.margin,
                epochs=epoch,
                seed=args.seed,
                timeout=args.timeout,
            )
            export_checkpoint(
                checkpoint_path=str(checkpoint_path),
                out_dir=str(artifact_dir),
                version=candidate_name,
                policy_loss=float(metrics.get("pairwise_loss", 0.0)),
                value_loss=0.0,
            )
            delta_norm, _relative = compute_param_delta_norm(
                checkpoint_path, init_checkpoint
            )
            candidates.append(
                {
                    **candidate_spec(
                        name=candidate_name,
                        artifact_dir=artifact_dir,
                        checkpoint_path=checkpoint_path,
                        epochs=epoch,
                        pairwise_weight=lane.pairwise_weight,
                        margin=lane.margin,
                        anchor_weight=lane.behavior_anchor_weight,
                        source="trained_strength_calibration",
                    ),
                    **metrics,
                    "validation_loss": float(metrics.get("best_val_loss", 0.0)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "artifact_weights_sha256": sha256_file(
                        artifact_dir / "weights.json"
                    ),
                    "delta_norm_vs_current_checkpoint": delta_norm,
                }
            )

    current_forwards = build_current_forward_cache(
        current_artifact=current_dir,
        probe_rows=probe_rows,
    )
    current_probe: dict[str, Any] | None = None
    for row in candidates:
        row["probe_metrics"] = evaluate_probe_candidate(
            artifact_path=Path(row["artifact_dir"]),
            probe_rows=probe_rows,
            current_forwards=current_forwards,
        )
        if row["candidate"] == "current_ref":
            current_probe = row["probe_metrics"]
    if current_probe is None:
        raise RuntimeError("missing current_ref probe metrics")
    for row in candidates:
        row["current_probe_reference"] = current_probe
    apply_probe_gates(candidates)

    eval_candidates = [
        row
        for row in candidates
        if row["candidate"] == "current_ref" or not row.get("aborted")
    ]
    for row in candidates:
        row["evaluated"] = row in eval_candidates

    eval_candidate_dirs = [Path(row["artifact_dir"]) for row in eval_candidates]
    suite_reports: dict[str, dict[str, Any]] = {}
    suite_rows_map: dict[str, dict[str, Any]] = {}
    for suite_path in [medium_suite, fixed_large_suite, *heldout_suites]:
        report = run_opening_suite_benchmark_schedule(
            workdir=workdir / f"bench_{suite_path.stem}",
            suite=suite_path,
            current=current_dir,
            candidate_dirs=eval_candidate_dirs,
            budget_pairs=DEFAULT_BUDGETS,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
        suite_reports[suite_path.stem] = report
        suite_rows_map[suite_path.stem] = suite_rows_from_report(
            report=report,
            eval_candidates=eval_candidates,
            suite_name=suite_path.stem,
        )

    heldout_suite_rows = {
        name: rows
        for name, rows in suite_rows_map.items()
        if name not in {medium_suite.stem, fixed_large_suite.stem}
    }
    fixed_large_rows = {fixed_large_suite.stem: suite_rows_map[fixed_large_suite.stem]}
    current_ref_row = next(
        row for row in candidates if row["candidate"] == "current_ref"
    )

    for row in candidates:
        if not row.get("evaluated"):
            row["gate_classification"] = "not_run"
            row["gate_ran"] = False
            continue
        candidate_name = row["candidate"]
        fixed_report = find_candidate_report(
            suite_reports[fixed_large_suite.stem], row["report_candidate_name"]
        )
        if fixed_report is None:
            raise RuntimeError(
                f"missing candidate {row['report_candidate_name']} in fixed large report"
            )
        row["fixed_large_budget_results"] = benchmark_budget_results(fixed_report)
        row["heldout_summary"] = heldout_summary(
            suite_reports, row["report_candidate_name"]
        )
        row["heldout_summary"] = {
            "available": bool(row["heldout_summary"].get("available", True)),
            "mean_ds_384_256": float(
                row["heldout_summary"].get("mean_ds_384_256", 0.0)
            ),
            "worst_suite_ds_384_256": float(
                row["heldout_summary"].get("worst_suite_ds_384_256", 0.0)
            ),
        }
        row["heldout_aggregate"] = aggregate_budget_summary(
            suite_rows=heldout_suite_rows,
            candidate_name=candidate_name,
            current_name="current_ref",
            budget_pairs=budget_pairs,
        )
        row["fixed_large_aggregate"] = aggregate_budget_summary(
            suite_rows=fixed_large_rows,
            candidate_name=candidate_name,
            current_name="current_ref",
            budget_pairs=budget_pairs,
        )
        row["bootstrap_cis"] = {}
        if candidate_name != "current_ref":
            for budget_pair in (
                PRIMARY_BUDGET,
                EQ_768_BUDGET,
                EQ_1200_BUDGET,
                ASYM_1200_256_BUDGET,
            ):
                diffs = pooled_per_opening_differences(
                    suite_rows=heldout_suite_rows,
                    candidate_a=candidate_name,
                    candidate_b="current_ref",
                    budget_pair=budget_pair,
                    metric_key="disadvantaged_seat_score",
                )
                row["bootstrap_cis"][
                    f"{candidate_name}_minus_current_ref_{budget_pair.replace(':', '_')}"
                ] = bootstrap_ci(diffs, seed=args.seed, samples=args.bootstrap_samples)

    current_heldout_mean = float(
        current_ref_row.get("heldout_summary", {}).get("mean_ds_384_256", 0.0)
    )
    for row in candidates:
        if not row.get("evaluated"):
            continue
        row["heldout_mean_384_256_delta_vs_current_ref"] = (
            float(row.get("heldout_summary", {}).get("mean_ds_384_256", 0.0))
            - current_heldout_mean
        )

    for row in candidates:
        if not row.get("evaluated"):
            continue
        if row["candidate"] == "current_ref":
            gate_report = run_default_gate_schedule(
                candidate_path=Path(row["artifact_dir"]),
                current_path=current_dir,
                out_path=workdir / "gate_current_ref.json",
                default_c_puct=float(args.default_c_puct),
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=float(args.tactical_root_bias),
                seed=args.seed,
                workers=args.workers,
            )
            row["gate_classification"] = gate_report.get("classification")
            row["gate_ran"] = True
            continue
        robust = (
            float(row["heldout_mean_384_256_delta_vs_current_ref"]) >= 0.05
            and float(
                row["heldout_aggregate"][EQ_768_BUDGET]["mean_delta_vs_current_ref"]
            )
            >= -0.05
            and float(
                row["heldout_aggregate"][EQ_1200_BUDGET]["mean_delta_vs_current_ref"]
            )
            >= -0.03
            and float(
                row["heldout_aggregate"][ASYM_1200_256_BUDGET][
                    "mean_delta_vs_current_ref"
                ]
            )
            >= -0.03
        )
        if not robust:
            row["gate_classification"] = "not_run"
            row["gate_ran"] = False
            continue
        gate_report = run_default_gate_schedule(
            candidate_path=Path(row["artifact_dir"]),
            current_path=current_dir,
            out_path=workdir / f"gate_{row['candidate']}.json",
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=args.seed,
            workers=args.workers,
        )
        row["gate_classification"] = gate_report.get("classification")
        row["gate_ran"] = True

    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": inputs,
        "search_schedule": schedule_definition(
            default_c_puct=float(args.default_c_puct),
            schedule=cpuct_schedule,
        ),
        "search_options": {
            "root_policy_mode": "deterministic",
            "tactical_root_bias": float(args.tactical_root_bias),
            "search_mode": "full",
        },
        "dataset": {
            "source": dataset_info["source"],
            "pairwise_target_rows_expected": EXPECTED_PAIRWISE_ROWS,
            "pairwise_target_rows_actual": len(dataset_info["pairwise_rows"]),
            "pairwise_target_rows_match": len(dataset_info["pairwise_rows"])
            == EXPECTED_PAIRWISE_ROWS,
            "anchor_rows_expected": EXPECTED_ANCHOR_ROWS,
            "anchor_rows_actual": len(dataset_info["anchor_rows"]),
            "anchor_rows_match": len(dataset_info["anchor_rows"])
            == EXPECTED_ANCHOR_ROWS,
            "probe_composition_expected": EXPECTED_PROBE_COMPOSITION,
            "probe_composition_actual": probe_verify["actual"],
            "probe_composition_match": probe_verify["matches_expected"],
        },
        "candidates": candidates,
    }
    summary["classification"] = classification(candidates)
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(summary=summary, report_path=REPORT_PATH)


if __name__ == "__main__":
    main()
